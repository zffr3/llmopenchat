from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llmopenchat import setup
from llmopenchat.download import DownloadError


class OllamaSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.runtime = self.root / ".local" / "runtime" / "llama-b11445-vulkan" / "llama-server.exe"
        self.model = self.root / "models" / "huihui_ai--glm-4.7-flash-abliterated" / "Huihui-GLM-4.7-Flash-abliterated-ollama.Q4_K_M.gguf"
        self.converted = self.model.with_name(self.model.stem + ".llamacpp.gguf")
        self.template = self.root / "llmopenchat" / "templates" / "GLM-4.7-Flash.jinja"
        self.template.parent.mkdir(parents=True)
        shutil.copyfile(Path(setup.__file__).parent / "templates" / self.template.name, self.template)
        self.compatibility = {
            "source_filename": self.model.name,
            "filename": self.converted.name,
            "source_architecture": "glm4moelite",
            "architecture": "deepseek2",
            "source_sha256": "db7192ff754ada80a81f7f8cd4704a273f79d43a3160b1b74f3b5988b09f238c",
            "source_size": 18765923488,
            "sha256": "12" * 32,
            "size": 18765923968,
            "tensor_payload_sha256": "34" * 32,
            "tensor_descriptors_unchanged": True,
            "tensor_payload_unchanged": True,
            "chat_template_sha256": "d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375",
        }
        self.messages = []
        self.emit = self.messages.append
        self.download_patch = patch("llmopenchat.download.download_file", return_value=self.model)
        self.runtime_patch = patch("llmopenchat.setup._install_runtime", return_value=self.runtime)
        self.converter_patch = patch("llmopenchat.gguf_compat.convert_glm4moelite", return_value=self.compatibility)
        self.disk_patch = patch("llmopenchat.setup.shutil.disk_usage", return_value=SimpleNamespace(free=100 * 1024**3))
        self.download = self.download_patch.start()
        self.install_runtime = self.runtime_patch.start()
        self.converter = self.converter_patch.start()
        self.disk = self.disk_patch.start()
        self.addCleanup(self.download_patch.stop)
        self.addCleanup(self.runtime_patch.stop)
        self.addCleanup(self.converter_patch.stop)
        self.addCleanup(self.disk_patch.stop)

    def install(self):
        return setup.install_ollama_local(self.root, emit=self.emit)

    def test_windows_x64_guard_runs_before_disk_checks_or_transfers(self):
        for system, machine in (("posix", "x86_64"), ("nt", "arm64")):
            with self.subTest(system=system, machine=machine):
                with patch("llmopenchat.setup.os.name", system), patch("llmopenchat.setup.platform.machine", return_value=machine):
                    with self.assertRaisesRegex(setup.SetupError, "Windows x64"):
                        self.install()
        self.disk.assert_not_called()
        self.install_runtime.assert_not_called()
        self.download.assert_not_called()
        self.converter.assert_not_called()

    def test_missing_project_directory_fails_before_transfers(self):
        with self.assertRaisesRegex(setup.SetupError, "не существует"):
            setup.install_ollama_local(self.root / "absent", emit=self.emit)
        self.disk.assert_not_called()
        self.install_runtime.assert_not_called()
        self.download.assert_not_called()

    def test_insufficient_disk_stops_before_runtime_or_model_download(self):
        self.disk.return_value = SimpleNamespace(free=1)
        with self.assertRaisesRegex(setup.SetupError, "Недостаточно места"):
            self.install()
        self.install_runtime.assert_not_called()
        self.download.assert_not_called()
        self.assertFalse((self.root / ".local" / "install.json").exists())

    def test_download_error_preserves_cause_and_resume_instruction(self):
        original = DownloadError("Соединение прервано")
        self.download.side_effect = original
        with self.assertRaises(setup.SetupError) as raised:
            self.install()
        self.assertIs(raised.exception.__cause__, original)
        self.assertIn("Соединение прервано", str(raised.exception))
        self.assertIn("Повторный запуск продолжит загрузку", str(raised.exception))
        self.converter.assert_not_called()
        self.assertFalse((self.model.parent / "download.json").exists())
        self.assertFalse((self.root / ".local" / "install.json").exists())

    def test_runtime_setup_error_is_preserved_and_model_not_downloaded(self):
        original = setup.SetupError("Повреждён официальный архив")
        self.install_runtime.side_effect = original
        with self.assertRaises(setup.SetupError) as raised:
            self.install()
        self.assertIs(raised.exception, original)
        self.download.assert_not_called()

    def test_keyboard_interrupt_is_not_wrapped_or_recorded_as_installed(self):
        self.download.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.install()
        self.assertFalse((self.root / ".local" / "install.json").exists())

    def test_changed_source_hash_stops_without_recording_success(self):
        self.compatibility["source_sha256"] = "00" * 32
        with self.assertRaisesRegex(setup.SetupError, "Исходный GGUF изменился"):
            self.install()
        self.converter.assert_called_once_with(self.model, self.converted, emit=self.emit)
        self.assertFalse((self.model.parent / "download.json").exists())
        self.assertFalse((self.root / ".local" / "install.json").exists())

    def test_changed_or_missing_template_stops_without_recording_success(self):
        self.template.write_text("altered template", encoding="utf-8")
        for state in ("changed", "missing"):
            with self.subTest(state=state):
                if state == "missing":
                    self.template.unlink()
                with self.assertRaisesRegex(setup.SetupError, "Шаблон GLM-4.7-Flash"):
                    self.install()
                self.assertFalse((self.model.parent / "download.json").exists())
                self.assertFalse((self.root / ".local" / "install.json").exists())

    def test_success_records_exact_pinned_model_and_runtime_provenance(self):
        result = self.install()
        expected_hash = "db7192ff754ada80a81f7f8cd4704a273f79d43a3160b1b74f3b5988b09f238c"
        expected_blob = "https://registry.ollama.ai/v2/huihui_ai/glm-4.7-flash-abliterated/blobs/sha256:" + expected_hash
        self.install_runtime.assert_called_once_with(self.root, self.emit)
        self.download.assert_called_once_with(expected_blob, self.model, 18765923488, expected_hash, emit=self.emit, workers=32)
        self.converter.assert_called_once_with(self.model, self.converted, emit=self.emit)
        self.assertEqual(
            result,
            {
                "executable": ".local/runtime/llama-b11445-vulkan/llama-server.exe",
                "model_path": "models/huihui_ai--glm-4.7-flash-abliterated/Huihui-GLM-4.7-Flash-abliterated-ollama.Q4_K_M.llamacpp.gguf",
                "repo_id": "huihui_ai/glm-4.7-flash-abliterated",
                "filename": "Huihui-GLM-4.7-Flash-abliterated-ollama.Q4_K_M.gguf",
                "chat_template_file": "llmopenchat/templates/GLM-4.7-Flash.jinja",
            },
        )
        provenance = json.loads((self.model.parent / "download.json").read_text(encoding="utf-8"))
        self.assertEqual(
            provenance,
            {
                "model": "huihui_ai/glm-4.7-flash-abliterated",
                "source": "https://ollama.com/huihui_ai/glm-4.7-flash-abliterated",
                "blob": expected_blob,
                "sha256": expected_hash,
                "size": 18765923488,
                "quantization": "Q4_K_M",
                "llama_cpp_compatibility": self.compatibility,
            },
        )
        installed = json.loads((self.root / ".local" / "install.json").read_text(encoding="utf-8"))
        self.assertEqual(installed["model"], provenance)
        self.assertEqual(installed["runtime_release"], "b11445")
        self.assertEqual(installed["runtime_backend"], "vulkan")
        self.assertEqual(
            installed["runtime_assets"],
            [
                {
                    "name": "llama-b11445-bin-win-vulkan-x64.zip",
                    "size": 33337869,
                    "sha256": "975a788f5e5a55410fb5b72660f3f5673ad9fccbe99d66a2a2034f775a0bd432",
                }
            ],
        )
        for key, value in result.items():
            self.assertEqual(installed[key], value)


if __name__ == "__main__":
    unittest.main()
