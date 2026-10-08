"""Training orchestration and fail-closed masking tests without heavy dependencies."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llmopenchat import sft_train


class FakeTokenizer:
    chat_template = "test-template"
    pad_token = None
    eos_token = "</assistant>"
    padding_side = "left"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False, return_dict=True):
        text = "".join(f"<{item['role']}>{item['content']}</{item['role']}>" for item in messages)
        if add_generation_prompt:
            text += "<assistant>"
        tokens = list(map(ord, text))
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)} if return_dict else tokens

    def save_pretrained(self, destination):
        Path(destination, "tokenizer_config.json").write_text("{}", encoding="utf-8")


class TrainerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="llmopenchat-sft-train-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.emit = Mock()
        self.rows = [{"prompt": [{"role": "user", "content": "Q"}],
                      "completion": [{"role": "assistant", "content": "A"}]}]
        self.job = {
            "version": 1, "base_model": "test/model", "revision": "requested-branch", "method": "lora",
            "train_file": "train.jsonl", "validation_file": None, "output_dir": "adapter",
            "max_length": 256, "epochs": 1.0, "learning_rate": 0.0002, "seed": 42,
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.05}, "dataset_sha256": "a" * 64,
            "validation_sha256": None,
        }
        self.write_rows()

    def write_rows(self, *, validation=False):
        name = "validation.jsonl" if validation else "train.jsonl"
        source = self.root / name
        source.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        self.job["validation_sha256" if validation else "train_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        if validation:
            self.job["validation_file"] = name
        self.write_job()

    def write_job(self):
        self.path = self.root / "job.json"
        self.path.write_text(json.dumps(self.job), encoding="utf-8")

    def backend(self, *, cuda=False, bf16=False, train_error=None, bad_masks=False):
        self.tokenizer = FakeTokenizer()
        self.model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
        self.parameter = SimpleNamespace(requires_grad=True, data=Mock())
        self.parameter.data.to.return_value = self.parameter.data
        self.model.parameters = lambda: [self.parameter]
        self.fake_trainer = None
        torch = SimpleNamespace(
            float32="float32", float16="float16", bfloat16="bfloat16",
            cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 1,
                                 is_bf16_supported=lambda: bf16, get_device_name=lambda _: "Test GPU"),
        )
        backend = SimpleNamespace(
            torch=torch, Dataset=SimpleNamespace(from_list=lambda rows: rows),
            AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(
                _commit_hash="f" * 40, max_position_embeddings=4096, is_encoder_decoder=False))),
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer)),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=self.model)),
            LoraConfig=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            SFTConfig=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            BitsAndBytesConfig=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            prepare_model_for_kbit_training=Mock(return_value=self.model), TrainerCallback=object,
            set_seed=Mock(),
        )

        class FakeTrainer:
            def __init__(trainer, **kwargs):
                backend.set_seed.assert_called_once_with(self.job["seed"])
                trainer.kwargs = kwargs
                trainer.model = kwargs["model"]
                trainer.train_dataset = sft_train._tokenize_preflight(self.tokenizer, kwargs["train_dataset"], 10000, "train")
                trainer.eval_dataset = (sft_train._tokenize_preflight(self.tokenizer, kwargs["eval_dataset"], 10000, "validation")
                                        if kwargs["eval_dataset"] else None)
                if bad_masks:
                    trainer.train_dataset[0]["labels"][0] = trainer.train_dataset[0]["input_ids"][0]
                trainer.train = Mock(side_effect=train_error, return_value=SimpleNamespace(metrics={"train_loss": 0.5}))
                trainer.evaluate = Mock(return_value={"eval_loss": 0.7})
                trainer.save_model = Mock(side_effect=lambda destination: (
                    Path(destination).mkdir(exist_ok=True),
                    Path(destination, "adapter_model.safetensors").write_bytes(b"adapter"),
                ))
                self.fake_trainer = trainer

        backend.SFTTrainer = FakeTrainer
        return backend

    def train_job(self, backend):
        with patch.object(sft_train, "_load_backend", return_value=backend):
            return sft_train.run_training(self.path, emit=self.emit)

    def state(self):
        return json.loads((self.root / "status.json").read_text(encoding="utf-8"))

    def test_import_does_not_load_training_libraries(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import llmopenchat.sft_train; "
             "assert not any(x in sys.modules for x in ['torch','trl','peft','datasets','transformers'])"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_hash_drift_stops_before_import_or_download(self):
        (self.root / "train.jsonl").write_text("{}\n", encoding="utf-8")
        with patch.object(sft_train, "_load_backend") as loader:
            self.assertEqual(sft_train.run_training(self.path, emit=self.emit), 1)
        loader.assert_not_called()
        self.assertFalse((self.root / "status.json").exists())
        self.assertIn("SHA256", self.emit.call_args.args[0])

    def test_nonempty_adapter_is_preserved(self):
        output = self.root / "adapter"
        output.mkdir()
        original = output / "adapter_model.safetensors"
        original.write_bytes(b"previous")
        with patch.object(sft_train, "_load_backend") as loader:
            self.assertEqual(sft_train.run_training(self.path, emit=self.emit), 1)
        loader.assert_not_called()
        self.assertEqual(original.read_bytes(), b"previous")
        self.assertFalse((self.root / "status.json").exists())
        self.assertIn("не пуст", self.emit.call_args.args[0])

    def test_retry_completed_preserves_existing_status_metadata_and_logs(self):
        self.assertEqual(self.train_job(self.backend()), 0)
        saved = {name: (self.root / name).read_bytes() for name in ("status.json", "metadata.json", "logs.jsonl")}
        with patch.object(sft_train, "_load_backend") as loader:
            self.assertEqual(sft_train.run_training(self.path, emit=self.emit), 1)
        loader.assert_not_called()
        self.assertEqual({name: (self.root / name).read_bytes() for name in saved}, saved)
        self.assertEqual(self.state()["state"], "completed")

    def test_concurrent_job_lock_preserves_previous_run_artifacts(self):
        saved = {name: b"previous run" for name in ("status.json", "metadata.json", "logs.jsonl")}
        for name, data in saved.items():
            (self.root / name).write_bytes(data)
        with sft_train._acquire_run_lock(self.root / ".training.lock"):
            with patch.object(sft_train, "_load_backend") as loader:
                self.assertEqual(sft_train.run_training(self.path, emit=self.emit), 1)
            loader.assert_not_called()
            self.assertEqual({name: (self.root / name).read_bytes() for name in saved}, saved)
            with self.assertRaises(sft_train.TrainingError):
                sft_train._acquire_run_lock(self.root / ".training.lock")
        # The advisory file remains; its existence does not block a new owner.
        with sft_train._acquire_run_lock(self.root / ".training.lock"):
            pass

    def test_process_death_releases_advisory_lock(self):
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", "import sys,time; from pathlib import Path; "
             "from llmopenchat.sft_train import _acquire_run_lock; "
             "lock=_acquire_run_lock(Path(sys.argv[1])); print('locked',flush=True); time.sleep(60)",
             str(self.root / ".training.lock")],
            cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), "locked")
            with self.assertRaises(sft_train.TrainingError):
                sft_train._acquire_run_lock(self.root / ".training.lock")
        finally:
            child.terminate()
            child.communicate(timeout=10)
        with sft_train._acquire_run_lock(self.root / ".training.lock"):
            pass

    def test_dataset_limit_is_checked_before_read(self):
        source = self.root / "train.jsonl"
        with source.open("wb") as destination:
            destination.truncate(sft_train._MAX_DATASET_BYTES + 1)
        with patch.object(sft_train, "_load_backend") as loader:
            self.assertEqual(sft_train.run_training(self.path, emit=self.emit), 1)
        loader.assert_not_called()
        self.assertIn("64 MiB", self.emit.call_args.args[0])
        self.assertFalse((self.root / "status.json").exists())

    def test_traversal_and_gguf_fail_before_backend(self):
        for field, value in (("train_file", "../outside.jsonl"), ("output_dir", "../outside"),
                             ("base_model", "existing.gguf")):
            with self.subTest(field=field):
                old = self.job[field]
                self.job[field] = value
                self.write_job()
                with patch.object(sft_train, "_load_backend") as loader:
                    self.assertEqual(sft_train.run_training(self.path, emit=self.emit), 1)
                loader.assert_not_called()
                self.job[field] = old

    def test_missing_dependencies_are_actionable(self):
        packages = {name: {"version": None, "required": ">=1,<2", "supported": False}
                    for name in sft_train._REQUIREMENTS}
        with patch.object(sft_train, "_package_info", return_value=packages):
            self.assertEqual(sft_train.run_training(self.path, emit=self.emit), 1)
        self.assertEqual(self.state()["state"], "failed")
        self.assertIn("requirements-training.txt", self.state()["error"])

    def test_unsupported_versions_fail_before_import(self):
        packages = {name: {"version": "0.1", "required": ">=1,<2", "supported": False}
                    for name in sft_train._REQUIREMENTS}
        with patch.object(sft_train, "_package_info", return_value=packages):
            with self.assertRaisesRegex(sft_train.TrainingError, "Неподдерживаемые версии"):
                sft_train._load_backend("lora")

    def test_missing_chat_template_stops_before_weights(self):
        backend = self.backend()
        self.tokenizer.chat_template = None
        self.assertEqual(self.train_job(backend), 1)
        backend.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertIn("chat_template", self.state()["error"])

    def test_long_target_is_never_silently_truncated(self):
        backend = self.backend()
        self.rows[0]["completion"][0]["content"] = "x" * 300
        self.write_rows()
        self.assertEqual(self.train_job(backend), 1)
        backend.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertIn("превышают max_length", self.state()["error"])

    def test_incompatible_template_boundary_stops_before_weights(self):
        backend = self.backend()
        apply = self.tokenizer.apply_chat_template
        self.tokenizer.apply_chat_template = lambda *args, **kwargs: apply(*args, **kwargs) + ([0] if kwargs.get("add_generation_prompt") else [])
        self.assertEqual(self.train_job(backend), 1)
        backend.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertIn("границу prompt", self.state()["error"])

    def test_template_removing_completion_is_rejected(self):
        backend = self.backend()
        prompt = self.tokenizer.apply_chat_template(
            self.rows[0]["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        self.tokenizer.apply_chat_template = lambda *args, **kwargs: prompt
        self.assertEqual(self.train_job(backend), 1)
        backend.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertIn("не оставил токенов", self.state()["error"])

    def test_preflight_explicitly_requests_lists_from_transformers_5_tokenizer(self):
        tokenizer = FakeTokenizer()
        self.assertIsInstance(tokenizer.apply_chat_template(self.rows[0]["prompt"], tokenize=True), dict)
        result = sft_train._tokenize_preflight(tokenizer, self.rows, 256, "train")
        self.assertIsInstance(result[0]["input_ids"], list)
        self.assertTrue(any(label != -100 for label in result[0]["labels"]))

    def test_actual_trl_masks_are_verified_before_optimizer_step(self):
        self.assertEqual(self.train_job(self.backend(bad_masks=True)), 1)
        self.fake_trainer.train.assert_not_called()
        self.assertIn("completion-only маска", self.state()["error"])

    def test_lora_cpu_saves_adapter_tokenizer_metadata_with_pinned_revision(self):
        backend = self.backend()
        self.write_rows(validation=True)
        self.assertEqual(self.train_job(backend), 0)
        self.assertEqual(self.state()["state"], "completed")
        model_kwargs = backend.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        tokenizer_kwargs = backend.AutoTokenizer.from_pretrained.call_args.kwargs
        self.assertEqual(model_kwargs["revision"], "f" * 40)
        self.assertEqual(tokenizer_kwargs["revision"], model_kwargs["revision"])
        self.assertFalse(model_kwargs["trust_remote_code"])
        self.assertFalse(tokenizer_kwargs["trust_remote_code"])
        self.assertEqual(model_kwargs["device_map"], {"": "cpu"})
        self.assertEqual(model_kwargs["dtype"], "float32")
        self.assertFalse(self.model.config.use_cache)
        args = backend.SFTConfig.call_args.kwargs
        self.assertTrue(args["completion_only_loss"])
        self.assertFalse(args["assistant_only_loss"])
        self.assertFalse(args["packing"])
        self.assertTrue(args["use_cpu"])
        self.assertTrue(args["gradient_checkpointing"])
        self.assertEqual(args["gradient_accumulation_steps"], 8)
        self.assertEqual(backend.LoraConfig.call_args.kwargs["target_modules"], "all-linear")
        self.assertEqual(backend.LoraConfig.call_args.kwargs["revision"], "f" * 40)
        backend.prepare_model_for_kbit_training.assert_not_called()
        self.assertTrue((self.root / "adapter" / "tokenizer_config.json").exists())
        metadata = json.loads((self.root / "adapter" / "training_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["resolved_revision"], "f" * 40)
        self.assertEqual(metadata["metrics"]["eval_loss"], 0.7)

    def test_qlora_requires_cuda_before_any_model_download(self):
        self.job["method"] = "qlora"
        self.write_job()
        backend = self.backend()
        self.assertEqual(self.train_job(backend), 1)
        backend.AutoConfig.from_pretrained.assert_not_called()
        self.assertIn("CUDA", self.state()["error"])

    def test_qlora_config_and_kbit_gradient_preparation(self):
        self.job["method"] = "qlora"
        self.write_job()
        backend = self.backend(cuda=True, bf16=False)
        self.assertEqual(self.train_job(backend), 0)
        quantization = backend.BitsAndBytesConfig.call_args.kwargs
        self.assertEqual(quantization, {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                      "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "float16"})
        backend.prepare_model_for_kbit_training.assert_called_once_with(
            self.model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
        self.parameter.data.to.assert_called_once_with("float32")
        self.assertTrue(backend.SFTConfig.call_args.kwargs["fp16"])
        self.assertFalse(backend.SFTConfig.call_args.kwargs["bf16"])

    def test_bf16_gpu_uses_native_dtype(self):
        backend = self.backend(cuda=True, bf16=True)
        self.assertEqual(self.train_job(backend), 0)
        self.assertEqual(backend.AutoModelForCausalLM.from_pretrained.call_args.kwargs["dtype"], "bfloat16")
        self.assertTrue(backend.SFTConfig.call_args.kwargs["bf16"])
        self.assertFalse(backend.SFTConfig.call_args.kwargs["fp16"])

    def test_cancel_and_failure_status_are_persisted(self):
        for error, state, code in ((KeyboardInterrupt(), "canceled", 130), (RuntimeError("out of memory"), "failed", 1)):
            with self.subTest(state=state):
                backend = self.backend(train_error=error)
                self.assertEqual(self.train_job(backend), code)
                self.assertEqual(self.state()["state"], state)
                self.fake_trainer.save_model.assert_not_called()
                self.assertTrue((self.root / "metadata.json").exists())

    def test_local_base_files_have_source_fingerprint(self):
        base = self.root / "local-model"
        base.mkdir()
        (base / "config.json").write_text("{}", encoding="utf-8")
        (base / "model.safetensors").write_bytes(b"local weights")
        self.job["base_model"] = "local-model"
        self.write_job()
        backend = self.backend()
        backend.AutoConfig.from_pretrained.return_value._commit_hash = None
        self.assertEqual(self.train_job(backend), 0)
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["local_directory"], str(base))
        self.assertEqual(metadata["local_files_sha256"]["model.safetensors"], hashlib.sha256(b"local weights").hexdigest())

    def test_check_environment_does_not_fetch_model(self):
        packages = {name: {"version": None, "required": ">=1,<2", "supported": False}
                    for name in sft_train._REQUIREMENTS}
        with patch.object(sft_train, "_package_info", return_value=packages), patch.object(sft_train, "_load_backend") as loader:
            result = sft_train.check_training_environment()
        loader.assert_not_called()
        self.assertFalse(result["ready"])
        self.assertFalse(result["qlora_ready"])
        self.assertEqual(result["python_executable"], sys.executable)
        self.assertIn("torch", result["missing"])


if __name__ == "__main__":
    unittest.main()

