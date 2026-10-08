"""Optional local SFT backend. Importing this module does not import PyTorch.

Training consumes an immutable prepared job, applies the base model's own chat
template, and saves a PEFT adapter rather than overwriting the base weights.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import sys
from types import SimpleNamespace
from typing import Callable


_REQUIREMENTS = {
    "torch": ((2, 6, 0), (3, 0, 0)),
    "transformers": ((5, 19, 0), (5, 20, 0)),
    "trl": ((1, 14, 2), (1, 15, 0)),
    "peft": ((0, 21, 2), (0, 22, 0)),
    "datasets": ((4, 7, 0), (5, 0, 0)),
    "accelerate": ((1, 4, 0), (2, 0, 0)),
    "bitsandbytes": ((0, 50, 2), (0, 51, 0)),
}
_INSTALL_HINT = "Установите зависимости в выбранный Python: python -m pip install -r requirements-training.txt."
_MAX_DATASET_BYTES = 64 * 1024 * 1024


class TrainingError(ValueError):
    """An actionable training configuration or environment error."""


def _package_info() -> dict:
    packages = {}
    for package, (minimum, maximum) in _REQUIREMENTS.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\+.*)?", version or "")
        release = tuple(int(part or 0) for part in match.groups()) if match else None
        packages[package] = {
            "version": version,
            "required": ">=" + ".".join(map(str, minimum)) + ",<" + ".".join(map(str, maximum)),
            "supported": release is not None and minimum <= release < maximum,
        }
    return packages


def check_training_environment() -> dict:
    """Check the selected Python and GPU offline, without fetching any model."""
    packages = _package_info()
    required = [package for package in packages if package != "bitsandbytes"]
    result = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "missing": [name for name in required if packages[name]["version"] is None],
        "unsupported": [name for name in required if packages[name]["version"] and not packages[name]["supported"]],
        "cuda_available": False,
        "bf16_supported": False,
        "gpu_name": None,
        "gpu_count": 0,
    }
    if packages["torch"]["version"]:
        try:
            import torch

            result["cuda_available"] = bool(torch.cuda.is_available())
            if result["cuda_available"]:
                result["gpu_count"] = torch.cuda.device_count()
                result["gpu_name"] = torch.cuda.get_device_name(0)
                result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
                result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            result["torch_cuda_version"] = torch.version.cuda
        except Exception as exc:
            result["torch_error"] = str(exc)
    result["ready"] = result["lora_ready"] = not (
        result["missing"] or result["unsupported"] or result.get("torch_error")
    )
    result["qlora_ready"] = bool(
        result["ready"] and result["cuda_available"] and packages["bitsandbytes"]["supported"]
    )
    result["install_hint"] = _INSTALL_HINT
    return result


def _load_backend(method: str) -> SimpleNamespace:
    packages = _package_info()
    required = [name for name in packages if method == "qlora" or name != "bitsandbytes"]
    missing = [name for name in required if packages[name]["version"] is None]
    unsupported = [f"{name}=={packages[name]['version']} ({packages[name]['required']})" for name in required
                   if packages[name]["version"] and not packages[name]["supported"]]
    if missing:
        raise TrainingError(f"Нет зависимостей для SFT: {', '.join(missing)}. {_INSTALL_HINT}")
    if unsupported:
        raise TrainingError(f"Неподдерживаемые версии SFT: {', '.join(unsupported)}. {_INSTALL_HINT}")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
        from trl import SFTConfig, SFTTrainer

        if method == "qlora":
            import bitsandbytes  # noqa: F401 -- fail before fetching weights if the native backend is broken
    except Exception as exc:
        raise TrainingError(f"Не удалось загрузить движок SFT: {exc}. {_INSTALL_HINT}") from exc
    return SimpleNamespace(
        torch=torch, Dataset=Dataset, LoraConfig=LoraConfig,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
        AutoConfig=AutoConfig, AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer, BitsAndBytesConfig=BitsAndBytesConfig,
        TrainerCallback=TrainerCallback, SFTConfig=SFTConfig, SFTTrainer=SFTTrainer, set_seed=set_seed,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_relative_path(root: Path, name, field: str) -> Path:
    if not isinstance(name, str) or not name.strip() or Path(name).is_absolute():
        raise TrainingError(f"{field}: требуется относительный путь внутри каталога job.")
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root):
        raise TrainingError(f"{field}: путь выходит за пределы каталога job.")
    return path


def _number(value, field: str, minimum, maximum, *, integer=False, inclusive_min=True, inclusive_max=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TrainingError(f"{field}: требуется конечное число.")
    if integer and not isinstance(value, int):
        raise TrainingError(f"{field}: требуется целое число.")
    if (value < minimum if inclusive_min else value <= minimum) or (value > maximum if inclusive_max else value >= maximum):
        raise TrainingError(f"{field}: значение вне допустимого диапазона.")


def _read_rows(path: Path, expected_hash) -> list[dict]:
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise TrainingError(f"{path.name}: отсутствует корректная SHA256 в job.")
    if path.stat().st_size > _MAX_DATASET_BYTES:
        raise TrainingError(f"{path.name}: файл превышает лимит 64 MiB.")
    with path.open("rb") as source:
        data = source.read(_MAX_DATASET_BYTES + 1)
    if len(data) > _MAX_DATASET_BYTES:
        raise TrainingError(f"{path.name}: файл превышает лимит 64 MiB.")
    # Hash and parse the same bounded snapshot, even if another process edits
    # the file between stat/open or while we are reading it.
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise TrainingError(f"SHA256 не совпадает для {path.name}; подготовьте job заново.")
    rows = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingError(f"{path.name}:{line_number}: некорректный JSON.") from exc
        if not isinstance(row, dict) or set(row) != {"prompt", "completion"}:
            raise TrainingError(f"{path.name}:{line_number}: нужны только prompt и completion.")
        prompt, completion = row["prompt"], row["completion"]
        if not isinstance(prompt, list) or not prompt or not isinstance(completion, list) or len(completion) != 1:
            raise TrainingError(f"{path.name}:{line_number}: нужен контекст и ровно один целевой ответ.")
        for message in prompt + completion:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str) or not message["content"].strip()):
                raise TrainingError(f"{path.name}:{line_number}: некорректное текстовое сообщение.")
        if prompt[-1]["role"] != "user" or completion[0]["role"] != "assistant":
            raise TrainingError(f"{path.name}:{line_number}: prompt должен завершаться user, completion — assistant.")
        rows.append(row)
    return rows


def _load_job(path: Path) -> tuple[dict, Path, list[dict], list[dict]]:
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Не удалось прочитать job: {exc}") from exc
    if not isinstance(job, dict) or job.get("version") != 1:
        raise TrainingError("Неподдерживаемая версия SFT job; подготовьте job заново.")
    if not isinstance(job.get("base_model"), str) or not job["base_model"].strip():
        raise TrainingError("Нужна явная исходная Hugging Face модель (base_model).")
    if job["base_model"].lower().endswith(".gguf"):
        raise TrainingError("GGUF используется для инференса; SFT нужны исходные обучаемые Hugging Face веса.")
    if job.get("revision") is not None and (not isinstance(job["revision"], str) or not job["revision"].strip()):
        raise TrainingError("revision должна быть непустой строкой или null.")
    if job.get("method") not in {"lora", "qlora"}:
        raise TrainingError("method должна быть lora или qlora.")
    _number(job.get("max_length"), "max_length", 16, 131072, integer=True)
    _number(job.get("epochs"), "epochs", 0, 100, inclusive_min=False)
    _number(job.get("learning_rate"), "learning_rate", 0, 1, inclusive_min=False)
    _number(job.get("seed"), "seed", 0, 2**32 - 1, integer=True)
    lora = job.get("lora")
    if not isinstance(lora, dict):
        raise TrainingError("Отсутствуют параметры lora.")
    _number(lora.get("rank"), "lora.rank", 1, 256, integer=True)
    _number(lora.get("alpha"), "lora.alpha", 1, 1024, integer=True)
    _number(lora.get("dropout"), "lora.dropout", 0, 1, inclusive_max=False)
    root = path.parent
    output = _job_relative_path(root, job.get("output_dir"), "output_dir")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise TrainingError(f"Каталог адаптера не пуст: {output}. Подготовьте новый job, чтобы сохранить существующий результат.")
    train_path = _job_relative_path(root, job.get("train_file"), "train_file")
    validation_path = (_job_relative_path(root, job["validation_file"], "validation_file")
                       if job.get("validation_file") is not None else None)
    if train_path.is_relative_to(output) or (validation_path and validation_path.is_relative_to(output)):
        raise TrainingError("Каталог адаптера пересекается с файлами датасета.")
    train = _read_rows(train_path, job.get("train_sha256"))
    validation = _read_rows(validation_path, job.get("validation_sha256")) if validation_path else []
    if not train:
        raise TrainingError("В train.jsonl нет примеров для обучения.")
    return job, output, train, validation


def _tokenize_preflight(tokenizer, rows: list[dict], max_length: int, split: str) -> list[dict]:
    expected = []
    for index, row in enumerate(rows, 1):
        prefix = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        tokens = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"], tokenize=True, add_generation_prompt=False, return_dict=False,
        )
        if not isinstance(prefix, list) or not isinstance(tokens, list):
            raise TrainingError("Токенизатор вернул неподдерживаемый результат chat_template.")
        if tokens[:len(prefix)] != prefix:
            raise TrainingError(
                f"{split}:{index}: chat_template меняет границу prompt/ответа. "
                "Этот шаблон нельзя безопасно использовать для completion-only SFT; выберите совместимую instruct-модель."
            )
        if len(tokens) > max_length:
            raise TrainingError(
                f"{split}:{index}: {len(tokens)} токенов превышают max_length={max_length}. "
                "Сократите пример или увеличьте max_length; ответы автоматически не обрезаются."
            )
        if not prefix or len(tokens) <= len(prefix):
            raise TrainingError(f"{split}:{index}: chat_template не оставил токенов целевого ответа.")
        expected.append({"input_ids": tokens, "labels": [-100] * len(prefix) + tokens[len(prefix):]})
    return expected


def _verify_prepared_dataset(dataset, expected: list[dict], split: str) -> None:
    if dataset is None or len(dataset) != len(expected):
        raise TrainingError(f"{split}: движок изменил число подготовленных примеров.")
    for index, (actual, reference) in enumerate(zip(dataset, expected), 1):
        if actual.get("input_ids") != reference["input_ids"] or actual.get("labels") != reference["labels"]:
            raise TrainingError(
                f"{split}:{index}: токены или completion-only маска движка не совпадают с проверкой. "
                "Обучение остановлено до первого шага; проверьте версии TRL и токенизатора."
            )


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _acquire_run_lock(path: Path):
    """Take a nonblocking OS lock; the OS also releases it if Python is killed.

    Keep the empty lock file after closing it. Unlinking an advisory lock file
    would let another process lock a different inode while the old one is held.
    """
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise TrainingError(f"Этот job уже обучается или недоступен для блокировки: {path}. Дождитесь завершения процесса.") from exc
    return handle


def _local_source_metadata(path: Path) -> dict:
    names = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "tokenizer.model", "vocab.json", "vocab.txt", "merges.txt",
             "chat_template.jinja", "added_tokens.json"}
    files = {item.name: _sha256(item) for item in sorted(path.iterdir()) if item.is_file() and
             (item.name in names or item.suffix in {".safetensors", ".model"}
              or item.name.endswith(".index.json") or item.name.startswith("pytorch_model"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"local_directory": str(path), "local_files_sha256": files, "local_source_sha256": digest}


def run_training(job_path: str | Path, *, emit: Callable[[str], None] = print) -> int:
    """Train one job in the current Python; return 0, 1 or 130 on cancellation."""
    path = Path(job_path).expanduser().resolve()
    root = path.parent
    status_path = root / "status.json"
    status = {"state": "preflight", "job": str(path)}
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "python_executable": sys.executable}
    lock_path = root / ".training.lock"
    run_lock = None
    run_started = False

    def update(state: str, **values):
        status.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        _write_json(status_path, status)

    def log(event: dict):
        with (root / "logs.jsonl").open("a", encoding="utf-8") as destination:
            destination.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **event},
                                         ensure_ascii=False, default=str) + "\n")

    def failure(state: str, message: str, code: int) -> int:
        if run_started:
            try:
                update(state, error=message)
                metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), state=state, error=message)
                _write_json(root / "metadata.json", metadata)
                log({"event": state, "error": message})
            except OSError:
                pass  # A missing or inaccessible directory still produces an actionable terminal error.
        emit(message)
        return code

    try:
        job, output, train_rows, validation_rows = _load_job(path)
        run_lock = _acquire_run_lock(lock_path)
        # A concurrent run could have completed between the first validation
        # and lock acquisition. Revalidate while exclusively owning this job.
        job, output, train_rows, validation_rows = _load_job(path)
        run_started = True
        update("preflight", output_dir=str(output))
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise TrainingError("Первая версия SFT поддерживает один процесс и одну CUDA GPU; запускайте без torchrun.")
        # Restrict the dedicated training subprocess before PyTorch initializes CUDA.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",", 1)[0]
        backend = _load_backend(job["method"])
        # TRL creates the random LoRA matrices before Trainer.__init__ seeds
        # the runtime, so seed before constructing both model and adapter.
        backend.set_seed(job["seed"])
        torch = backend.torch
        cuda = bool(torch.cuda.is_available())
        if cuda and torch.cuda.device_count() != 1:
            raise TrainingError("Выберите одну GPU через CUDA_VISIBLE_DEVICES до запуска Python.")
        if job["method"] == "qlora" and not cuda:
            raise TrainingError("QLoRA требует CUDA GPU и CUDA-сборку PyTorch. Для проверки на CPU выберите method=lora.")
        bf16 = cuda and bool(torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if bf16 else torch.float16 if cuda else torch.float32
        metadata.update(
            job=job, job_sha256=_sha256(path), packages=_package_info(),
            device="cuda:0" if cuda else "cpu", dtype=str(dtype),
            gpu_name=torch.cuda.get_device_name(0) if cuda else None,
        )
        base = job["base_model"]
        candidate = (root / base).resolve() if not Path(base).is_absolute() else Path(base).resolve()
        if candidate.is_dir():
            base = str(candidate)
            metadata.update(_local_source_metadata(candidate))
        elif Path(base).is_absolute() or base.startswith(("./", "../", ".\\", "..\\")):
            raise TrainingError(f"Каталог исходной модели не найден: {candidate}")
        emit(f"SFT: загрузка конфигурации {base}; {job['method'].upper()}, {metadata['device']}.")
        config = backend.AutoConfig.from_pretrained(base, revision=job.get("revision"), trust_remote_code=False)
        if getattr(config, "is_encoder_decoder", False):
            raise TrainingError("SFT поддерживает только causal language models.")
        if getattr(config, "quantization_config", None):
            raise TrainingError("Выберите исходные неквантованные Hugging Face веса; QLoRA выполнит NF4-квантизацию при загрузке.")
        resolved_revision = getattr(config, "_commit_hash", None)
        if not candidate.is_dir() and not resolved_revision:
            raise TrainingError("Не удалось определить точную ревизию исходной модели Hugging Face.")
        revision = resolved_revision or job.get("revision")
        metadata.update(base_model=base, requested_revision=job.get("revision"), resolved_revision=resolved_revision)
        tokenizer = backend.AutoTokenizer.from_pretrained(base, revision=revision, trust_remote_code=False)
        if not getattr(tokenizer, "chat_template", None):
            raise TrainingError("У исходной модели нет chat_template. Выберите instruct-модель с собственным шаблоном диалога.")
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise TrainingError("Токенизатор не содержит pad_token и eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        expected_train = _tokenize_preflight(tokenizer, train_rows, job["max_length"], "train")
        expected_validation = _tokenize_preflight(tokenizer, validation_rows, job["max_length"], "validation")
        longest = max(len(row["input_ids"]) for row in expected_train + expected_validation)
        context_limit = getattr(config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and longest > context_limit:
            raise TrainingError(f"Пример содержит {longest} токенов, контекст исходной модели ограничен {context_limit}.")
        metadata.update(train_examples=len(train_rows), validation_examples=len(validation_rows), max_example_tokens=longest,
                        chat_template_sha256=hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest())
        load_kwargs = {"config": config, "revision": revision, "trust_remote_code": False,
                       "dtype": dtype, "device_map": {"": 0 if cuda else "cpu"}}
        if job["method"] == "qlora":
            load_kwargs["quantization_config"] = backend.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        emit(f"SFT: примеры проверены ({len(train_rows)} train, {len(validation_rows)} validation); загрузка весов.")
        model = backend.AutoModelForCausalLM.from_pretrained(base, **load_kwargs)
        model.config.use_cache = False
        if job["method"] == "qlora":
            model = backend.prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        lora = backend.LoraConfig(
            r=job["lora"]["rank"], lora_alpha=job["lora"]["alpha"], lora_dropout=job["lora"]["dropout"],
            target_modules="all-linear", task_type="CAUSAL_LM", bias="none", revision=revision,
            base_model_name_or_path=base,
        )
        training_args = backend.SFTConfig(
            output_dir=str(output), num_train_epochs=job["epochs"], learning_rate=job["learning_rate"],
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
            max_length=job["max_length"], completion_only_loss=True, assistant_only_loss=False,
            packing=False, eval_packing=False, padding_free=False, loss_type="nll",
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=bool(bf16), fp16=bool(cuda and not bf16), use_cpu=not cuda,
            seed=job["seed"], data_seed=job["seed"], optim="adamw_torch", report_to="none",
            logging_steps=1, save_strategy="no", eval_strategy="epoch" if validation_rows else "no",
            dataloader_num_workers=0, dataloader_pin_memory=cuda,
        )

        class ProgressCallback(backend.TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                metrics = dict(logs or {})
                log({"event": "metrics", "global_step": state.global_step, "metrics": metrics})
                update("running", global_step=state.global_step, epoch=state.epoch, metrics=metrics)
                emit("SFT: " + json.dumps(metrics, ensure_ascii=False, default=str))

        trainer = backend.SFTTrainer(
            model=model, args=training_args, processing_class=tokenizer, peft_config=lora,
            train_dataset=backend.Dataset.from_list(train_rows),
            eval_dataset=backend.Dataset.from_list(validation_rows) if validation_rows else None,
            callbacks=[ProgressCallback()],
        )
        _verify_prepared_dataset(trainer.train_dataset, expected_train, "train")
        if validation_rows:
            _verify_prepared_dataset(trainer.eval_dataset, expected_validation, "validation")
        # TRL 1.14 casts QLoRA adapters to bf16 unconditionally. fp16 AMP needs
        # fp32 trainable weights, otherwise GradScaler cannot unscale gradients.
        if job["method"] == "qlora" and not bf16:
            for parameter in trainer.model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(torch.float32)
        update("running")
        _write_json(root / "metadata.json", metadata)
        log({"event": "started", "base_model": base, "resolved_revision": resolved_revision})
        result = trainer.train()
        metrics = dict(getattr(result, "metrics", {}) or {})
        if validation_rows:
            metrics.update(trainer.evaluate())
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
        metadata.update(completed_at=datetime.now(timezone.utc).isoformat(), metrics=metrics)
        _write_json(root / "metadata.json", metadata)
        _write_json(output / "training_metadata.json", metadata)
        update("completed", metrics=metrics)
        log({"event": "completed", "metrics": metrics})
        emit(f"SFT завершено. Адаптер и токенизатор: {output}")
        return 0
    except KeyboardInterrupt:
        return failure("canceled", "SFT отменено пользователем.", 130)
    except Exception as exc:
        return failure("failed", f"SFT не выполнено: {exc}", 1)
    finally:
        if run_lock is not None:
            run_lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local LoRA/QLoRA supervised fine-tuning")
    parser.add_argument("job", nargs="?", help="Path to a prepared job.json")
    parser.add_argument("--check", action="store_true", help="Check training packages and CUDA without downloading models")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(check_training_environment(), ensure_ascii=False, indent=2))
        return 0
    if not args.job:
        parser.error("Укажите job.json или --check.")
    if hasattr(signal, "SIGBREAK"):
        def cancel(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGBREAK, cancel)
    return run_training(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
