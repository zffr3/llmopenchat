"""Reviewed, model-independent text datasets and reproducible SFT job preparation.

This module has no training dependencies. Training examples contain a prompt and
one assistant completion, so the trainer can mask every context token from loss.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

from .history import load_session

MAX_DATASET_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_MESSAGES = 1000


class SftError(ValueError):
    """An actionable dataset or job configuration error."""


def _json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise SftError(f"Нельзя сохранить JSON примера: {error}") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _invalid_constant(value: str) -> None:
    raise SftError(f"JSON содержит недопустимое число: {value}.")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SftError(f"JSON содержит недопустимое число: {value}.")
    return result


def _parse(text: str, location: str) -> object:
    try:
        return json.loads(text, parse_constant=_invalid_constant, parse_float=_finite_float)
    except (ValueError, RecursionError) as error:
        raise SftError(f"{location}: неверный JSON: {error}") from error


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SftError(f"{name} должен быть непустой строкой.")
    return value


def _validate_record(value: object, number: int) -> dict:
    label = f"Пример {number}"
    if not isinstance(value, dict):
        raise SftError(f"{label}: нужен JSON-объект.")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise SftError(f"{label}: schema_version должен быть 1.")
    if "tools" in value or "tool_calls" in value:
        raise SftError(f"{label}: SFT v1 поддерживает только текст; tools/tool_calls пока не поддерживаются.")
    unknown = value.keys() - {"schema_version", "id", "task", "status", "group_id", "metadata", "messages"}
    if unknown:
        raise SftError(f"{label}: неизвестные поля {', '.join(sorted(unknown))}. "
                       "Оценку задавайте через status, дополнительные сведения — через metadata.")
    record = {"schema_version": 1,
              "id": _nonempty_string(value.get("id"), f"{label}: id"),
              "task": _nonempty_string(value.get("task"), f"{label}: task"),
              "status": value.get("status", "draft")}
    if record["status"] not in ("approved", "draft", "rejected"):
        raise SftError(f"{label}: status должен быть approved, draft или rejected.")
    if "group_id" in value:
        record["group_id"] = _nonempty_string(value["group_id"], f"{label}: group_id")
    if "metadata" in value:
        if not isinstance(value["metadata"], dict):
            raise SftError(f"{label}: metadata должен быть JSON-объектом.")
        record["metadata"] = value["metadata"]
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES:
        raise SftError(f"{label}: messages должен содержать от 1 до {MAX_MESSAGES} сообщений.")
    clean = []
    has_user = False
    for index, message in enumerate(messages, 1):
        location = f"{label}, сообщение {index}"
        if not isinstance(message, dict):
            raise SftError(f"{location}: нужен объект role/content.")
        role = message.get("role")
        if role == "tool" or "tool_calls" in message or "tool_call_id" in message:
            raise SftError(f"{location}: SFT v1 поддерживает только текст; диалоги с инструментами "
                           "пока не поддерживаются. Подготовьте отдельный текстовый пример без tool/tool_calls.")
        unknown = message.keys() - {"role", "content", "train"}
        if unknown:
            raise SftError(f"{location}: неизвестные поля {', '.join(sorted(unknown))}. "
                           "Текстовый пример поддерживает role, content и assistant.train.")
        if role not in ("system", "user", "assistant"):
            raise SftError(f"{location}: допустимые роли — system, user, assistant.")
        content = message.get("content")
        if not isinstance(content, str):
            raise SftError(f"{location}: content должен быть строкой; мультимодальные сообщения пока не поддерживаются.")
        if not content.strip():
            raise SftError(f"{location}: content не должен быть пустым; удалите пустое сообщение или добавьте текст.")
        entry = {"role": role, "content": content}
        if "train" in message:
            if role != "assistant" or type(message["train"]) is not bool:
                raise SftError(f"{location}: train допускается только как bool у assistant.")
            entry["train"] = message["train"]
        if role == "user" and content.strip():
            has_user = True
        if role == "assistant" and entry.get("train", True):
            if not has_user or not content.strip() or not clean or clean[-1]["role"] != "user":
                raise SftError(f"{location}: обучаемый ответ требует непустого запроса user перед ним "
                               "(последнее сообщение prompt должно быть user) и непустого эталонного ответа. "
                               "Для контекста без обучения используйте train:false.")
        clean.append(entry)
    record["messages"] = clean
    if record["status"] == "approved" and not any(
            message["role"] == "assistant" and message.get("train", True) for message in clean):
        raise SftError(f"{label}: approved требует хотя бы одного эталонного assistant с train:true.")
    return record


def read_dataset(path: Path) -> list[dict]:
    """Read bounded UTF-8 JSONL or a JSON array; validate every record."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_DATASET_BYTES:
            raise SftError("Датасет слишком большой: максимум 64 МиБ.")
        with path.open("rb") as source:
            raw = source.read(MAX_DATASET_BYTES + 1)
        if len(raw) > MAX_DATASET_BYTES:
            raise SftError("Датасет слишком большой: максимум 64 МиБ.")
        text = raw.decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise SftError(f"Не удалось прочитать UTF-8 датасет {path}: {error}") from error
    if not text.strip():
        raise SftError("Датасет пуст.")
    if text.lstrip().startswith("["):
        values = _parse(text, str(path))
        if not isinstance(values, list):
            raise SftError("JSON-датасет должен быть массивом примеров.")
    else:
        values = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                raise SftError(f"Строка {number}: пример превышает 2 МиБ.")
            values.append(_parse(line, f"Строка {number}"))
            if len(values) > MAX_RECORDS:
                raise SftError(f"Слишком много примеров: максимум {MAX_RECORDS}.")
    if not values or len(values) > MAX_RECORDS:
        raise SftError(f"Датасет должен содержать от 1 до {MAX_RECORDS} примеров.")
    records = []
    identifiers = set()
    for number, value in enumerate(values, 1):
        if len(_json(value).encode("utf-8")) > MAX_RECORD_BYTES:
            raise SftError(f"Пример {number}: превышает 2 МиБ.")
        record = _validate_record(value, number)
        if record["id"] in identifiers:
            raise SftError(f"Пример {number}: повторный id {record['id']!r}.")
        identifiers.add(record["id"])
        records.append(record)
    return records


def _targets(record: dict) -> list[dict]:
    prompt = []
    rows = []
    for message in record["messages"]:
        clean = {"role": message["role"], "content": message["content"]}
        if message["role"] == "assistant" and message.get("train", True):
            rows.append({"prompt": list(prompt), "completion": [clean]})
        prompt.append(clean)
    return rows


def _approved_rows(records: list[dict]) -> tuple[list[dict], list[list[int]], int]:
    """Deduplicate targets after joining all conversation and prompt groups."""
    approved = [record for record in records if record["status"] == "approved"]
    parents = list(range(len(approved)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        parents[find(right)] = find(left)

    groups = {}
    prompts = {}
    candidates = []
    for number, record in enumerate(approved):
        # An explicit group unites records; otherwise the complete record is one group.
        if "group_id" in record:
            group = record["group_id"]
            if group in groups:
                join(number, groups[group])
            groups[group] = number
        for row in _targets(record):
            prompt_key = _digest(row["prompt"])
            if prompt_key in prompts:
                join(number, prompts[prompt_key])
            prompts[prompt_key] = number
            candidates.append((number, row))
    rows = []
    by_group = {}
    seen = set()
    duplicates = 0
    for number, row in candidates:
        key = _digest(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        index = len(rows)
        rows.append(row)
        by_group.setdefault(find(number), []).append(index)
    return rows, list(by_group.values()), duplicates


def _stats(records: list[dict]) -> dict:
    counts = Counter(record["status"] for record in records)
    rows, groups, duplicates = _approved_rows(records)
    return {"records": len(records), "approved": counts["approved"], "draft": counts["draft"],
            "rejected": counts["rejected"], "skipped": counts["draft"] + counts["rejected"],
            "targets": sum(len(_targets(record)) for record in records),
            "exportable_targets": len(rows), "duplicate_targets": duplicates,
            "groups": len(groups), "tasks": dict(Counter(record["task"] for record in records)),
            "dataset_sha256": _digest(records)}


def validate_dataset(path: Path) -> dict:
    return _stats(read_dataset(path))


def _write_new(path: Path, content: str) -> Path:
    path = Path(path)
    created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as destination:
            created = True
            destination.write(content)
    except OSError as error:
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise SftError(f"Не удалось создать {path}; существующие файлы не перезаписываются: {error}") from error
    return path


def write_template(path: Path) -> Path:
    """A draft demonstrates the schema without accidentally approving an example."""
    record = {"schema_version": 1, "id": "example-001", "task": "classification", "status": "draft",
              "messages": [{"role": "system", "content": "Определи тональность текста. Ответь: positive, neutral или negative."},
                           {"role": "user", "content": "Мне понравилось обслуживание."},
                           {"role": "assistant", "content": "positive", "train": True}]}
    return _write_new(path, _json(record) + "\n")


def from_session(session_path: Path, output: Path, task: str = "chat") -> Path:
    """Import a conversation for review; no generated answer is auto-approved."""
    _nonempty_string(task, "task")
    try:
        messages = load_session(Path(session_path))
    except (OSError, ValueError, UnicodeError) as error:
        raise SftError(f"Не удалось импортировать диалог: {error}") from error
    digest = _digest(messages)
    record = {"schema_version": 1, "id": "session-" + digest[:24],
              "group_id": "session-" + digest, "task": task, "status": "draft", "messages": messages}
    record = _validate_record(record, 1)
    if not _targets(record):
        raise SftError("В диалоге нет ответов assistant для проверки и дообучения.")
    content = _json(record)
    if len(content.encode("utf-8")) > MAX_RECORD_BYTES:
        raise SftError("Диалог превышает 2 МиБ на пример. Подготовьте более короткие независимые текстовые примеры.")
    return _write_new(output, content + "\n")


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise SftError(f"{name} должен быть целым числом от {low} до {high}.")
    return value


def _number(value: object, name: str, low: float, high: float, *, low_open=False, high_open=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SftError(f"{name} должен быть конечным числом.")
    if value < low or value > high or (low_open and value == low) or (high_open and value == high):
        raise SftError(f"{name} вне допустимого диапазона {low}..{high}.")
    return float(value)


def _base_model(value: object) -> str:
    model = _nonempty_string(value, "base_model")
    if model != model.strip() or model.startswith("-") or any(ord(char) < 32 for char in model):
        raise SftError("base_model содержит недопустимые символы.")
    if ".gguf" in model.lower() or model.lower().rstrip("/\\").endswith("gguf"):
        raise SftError("Для обучения нужен исходный Transformers/Hugging Face checkpoint, а не GGUF. Укажите base_model явно.")
    candidate = Path(model)
    if candidate.exists():
        if not candidate.is_dir() or not (candidate / "config.json").is_file():
            raise SftError("Локальный base_model должен быть каталогом Transformers с config.json, не файлом весов.")
        return str(candidate.resolve())
    elif not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model):
        raise SftError("base_model должен быть HF ID вида автор/модель или существующим каталогом Transformers.")
    return model


def prepare_dataset(dataset_path: Path, output_dir: Path, base_model: str, revision: str | None = None,
                    method: str = "qlora", validation_ratio: float = 0.1, seed: int = 42,
                    max_length: int = 2048, epochs: float = 3.0, learning_rate: float = 0.0001,
                    lora_rank: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05) -> Path:
    """Export only reviewed targets and prepare a versioned job without loading a model."""
    model = _base_model(base_model)
    if revision is not None:
        _nonempty_string(revision, "revision")
        if revision != revision.strip() or revision.startswith("-") or any(ord(char) < 32 for char in revision):
            raise SftError("revision содержит недопустимые символы.")
    if method not in ("lora", "qlora"):
        raise SftError("method должен быть lora или qlora.")
    ratio = _number(validation_ratio, "validation_ratio", 0, 1, high_open=True)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    max_length = _integer(max_length, "max_length", 16, 131072)
    epochs = _number(epochs, "epochs", 0, 100, low_open=True)
    rate = _number(learning_rate, "learning_rate", 0, 1, low_open=True)
    rank = _integer(lora_rank, "lora_rank", 1, 256)
    alpha = _integer(lora_alpha, "lora_alpha", 1, 1024)
    dropout = _number(lora_dropout, "lora_dropout", 0, 1, high_open=True)
    records = read_dataset(dataset_path)
    rows, groups, _ = _approved_rows(records)
    if not rows:
        raise SftError("Нет одобренных эталонных ответов. Проверьте примеры, исправьте ответы и задайте status:approved.")
    random.Random(seed).shuffle(groups)
    validation_groups = 0 if ratio == 0 or len(groups) < 2 else min(len(groups) - 1, max(1, round(len(groups) * ratio)))
    validation_indices = {index for group in groups[:validation_groups] for index in group}
    train = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation = [row for index, row in enumerate(rows) if index in validation_indices]
    stats = _stats(records)
    stats.update(train=len(train), validation=len(validation), train_groups=len(groups) - validation_groups,
                 validation_groups=validation_groups)
    stats["warnings"] = (["Все примеры образуют одну связанную группу; validation пуст. Добавьте независимые диалоги."]
                         if ratio > 0 and len(groups) < 2 else [])
    train_content = "".join(_json(row) + "\n" for row in train)
    validation_content = "".join(_json(row) + "\n" for row in validation)
    job = {"version": 1, "base_model": model, "revision": revision, "method": method,
           "train_file": "train.jsonl", "validation_file": "validation.jsonl" if validation else None,
           "output_dir": "adapter", "max_length": max_length, "epochs": epochs, "learning_rate": rate,
           "seed": seed, "lora": {"rank": rank, "alpha": alpha, "dropout": dropout},
           "dataset_sha256": stats["dataset_sha256"],
           "train_sha256": hashlib.sha256(train_content.encode("utf-8")).hexdigest(),
           "validation_sha256": hashlib.sha256(validation_content.encode("utf-8")).hexdigest() if validation else None,
           "stats": stats}
    directory = Path(output_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise SftError("Каталог задания должен быть новым или пустым; существующие результаты не перезаписываются.")
    created_directory = not directory.exists()
    written = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in (("train.jsonl", train_content), ("validation.jsonl", validation_content),
                              ("job.json", json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False) + "\n")):
            path = _write_new(directory / name, content)
            written.append(path)
    except (OSError, SftError) as error:
        for path in written:
            path.unlink(missing_ok=True)
        if created_directory and directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
        if isinstance(error, SftError):
            raise
        raise SftError(f"Не удалось сохранить задание: {error}") from error
    return directory / "job.json"
