"""Dataset entry points must stay independent of inference and its config."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console

from llmopenchat import cli, sft_cli


class SftCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="llmopenchat-sft-cli-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = io.StringIO()
        self.console = Console(file=self.output, color_system=None, width=200)

    def invoke(self, *args):
        with (patch.object(cli, "ROOT", self.root), patch.object(cli, "console", self.console),
              patch.object(cli, "load_config") as config,
              patch.object(cli, "ManagedServer") as server):
            result = cli.main(["--config", str(self.root / "missing.json"), "sft", *args])
        config.assert_not_called()
        server.assert_not_called()
        return result

    def test_init_validate_and_no_overwrite(self):
        self.assertEqual(self.invoke("init", "примеры.jsonl"), 0)
        content = (self.root / "примеры.jsonl").read_bytes()
        self.assertEqual(self.invoke("validate", "примеры.jsonl"), 0)
        self.assertEqual(self.invoke("init", "примеры.jsonl"), 1)
        self.assertEqual((self.root / "примеры.jsonl").read_bytes(), content)

    def test_prepare_requires_approved_answers_and_explicit_trainable_base(self):
        path = self.root / "data.jsonl"
        records = [{"schema_version": 1, "id": str(i), "task": "test", "status": "approved",
                    "messages": [{"role": "user", "content": f"Question {i}"},
                                 {"role": "assistant", "content": f"Answer {i}"}]}
                   for i in range(3)]
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        self.assertEqual(self.invoke("prepare", "data.jsonl", "--output", "job",
                                    "--base-model", "example/model", "--method", "lora"), 0)
        manifest = json.loads((self.root / "job/job.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["base_model"], "example/model")
        self.assertEqual(manifest["method"], "lora")
        self.assertEqual(self.invoke("prepare", "data.jsonl", "--output", "wrong",
                                    "--base-model", "example/model-GGUF"), 1)
        self.assertFalse((self.root / "wrong").exists())

    def test_check_and_train_delegate_without_importing_training_stack(self):
        with patch.object(sft_cli, "launch_training", return_value=7) as launch:
            self.assertEqual(self.invoke("check", "--python", "training/python.exe"), 7)
        self.assertEqual(launch.call_args.kwargs["python"], Path("training/python.exe"))
        with patch.object(sft_cli, "launch_training", return_value=0) as launch:
            self.assertEqual(self.invoke("train", "job/job.json"), 0)
        self.assertEqual(launch.call_args.kwargs["job"], Path("job/job.json"))

    def test_subprocess_uses_argv_and_streams_logs_with_space_in_paths(self):
        job = self.root / "job folder/job.json"
        job.parent.mkdir()
        job.write_text("{}", encoding="utf-8")
        process = MagicMock()
        process.stdout = io.StringIO("training progress\n")
        process.wait.return_value = 0
        output = []
        with (patch.object(sft_cli.subprocess, "Popen", return_value=process) as popen,
              patch.object(sft_cli, "_WindowsJob") as guard):
            self.assertEqual(sft_cli.launch_training(self.root, job=job,
                                                     python=Path("env folder/python.exe"), emit=output.append), 0)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], str(self.root / "env folder/python.exe"))
        self.assertEqual(argv[-1], str(job))
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertEqual(output, ["training progress"])
        self.assertIn(str(self.root), popen.call_args.kwargs["env"]["PYTHONPATH"])
        if sft_cli.os.name == "nt":
            guard.return_value.assign.assert_called_once_with(process.pid)
            guard.return_value.close.assert_called_once()

    def test_cancel_signals_child_and_releases_owner(self):
        process = MagicMock()
        process.stdout.__iter__.side_effect = KeyboardInterrupt
        process.poll.return_value = 130
        with (patch.object(sft_cli.subprocess, "Popen", return_value=process),
              patch.object(sft_cli, "_WindowsJob") as guard):
            self.assertEqual(sft_cli.launch_training(self.root, emit=lambda text: None), 130)
        process.send_signal.assert_called_once()
        process.stdout.close.assert_called_once()
        if sft_cli.os.name == "nt":
            guard.return_value.close.assert_called_once()

    def test_sft_menu_returns_to_main_menu_without_loading_model(self):
        from llmopenchat import menu
        with (patch.object(menu, "read_choice", side_effect=["6", "0", "0"]),
              patch.object(menu, "ModelManager"), patch.object(cli, "ManagedServer") as server):
            result = menu.main_menu({"target_model": "test"}, self.root / "config.json",
                                    self.root, self.console, lambda *args: 0)
        self.assertEqual(result, 0)
        server.assert_not_called()
        self.assertIn("Дообучение (SFT)", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
