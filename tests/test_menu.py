"""Console navigation and model actions without downloading large weights."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console

from llmopenchat import cli, menu
from llmopenchat.config import DEFAULT_CONFIG, write_config
from llmopenchat.history import save_session
from llmopenchat.models import ModelError
from llmopenchat.runtime import RuntimeErrorDetail


class MenuTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-menu-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "config.json"
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.output = io.StringIO()
        self.console = Console(file=self.output, color_system=None, width=160)
        write_config(self.path, self.config)

    def run_main(self, arguments, choices):
        with (
            patch.object(cli, "ROOT", self.root),
            patch.object(cli, "console", self.console),
            patch.object(cli.sys.stdin, "isatty", return_value=False),
            patch("builtins.input", side_effect=choices),
            patch.object(cli, "ManagedServer") as server,
            patch.object(cli, "generate", side_effect=lambda client, messages, shown: "Ответ: " + messages[-1]["content"]) as generate,
        ):
            result = cli.main(["--config", str(self.path), *arguments])
        return result, server, generate

    def test_default_menu_and_catalogue_do_not_start_server(self):
        result, server, generate = self.run_main([], ["0"])
        self.assertEqual(result, 0)
        server.assert_not_called()
        generate.assert_not_called()
        self.assertIn("Новый чат", self.output.getvalue())
        self.assertIn("История чатов", self.output.getvalue())
        result, server, generate = self.run_main(["models", "catalog"], [])
        self.assertEqual(result, 0)
        server.assert_not_called()
        self.assertIn("Каталог GGUF", self.output.getvalue())

    def test_legacy_install_clears_previous_manager_identity(self):
        config = copy.deepcopy(self.config)
        config.update(model_package="old-package", model="old-api-alias")
        write_config(self.path, config)
        installed = {"repo_id": "author/new-model", "executable": "server.exe", "model_path": "new.gguf"}
        with patch("llmopenchat.setup.install_ollama_local", return_value=installed):
            result, server, generate = self.run_main(["install"], [])
        self.assertEqual(result, 0)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertNotIn("model_package", saved)
        self.assertEqual(saved["model"], "glm-local")
        server.assert_not_called()

    def test_return_to_menu_releases_server_and_new_chat_has_fresh_context(self):
        result, server, generate = self.run_main([], ["1", "Первый вопрос", "/menu", "1", "Другой вопрос", "/quit"])
        self.assertEqual(result, 0)
        self.assertEqual(server.call_count, 2)
        self.assertEqual(server.return_value.__exit__.call_count, 2)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args_list[1].args[1], [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": "Другой вопрос"},
        ])
        saved = list((self.root / ".local" / "sessions").glob("*.json"))
        self.assertEqual(len(saved), 2)
        for path in saved:
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["target_model"], self.config["target_model"])

    def test_history_resumes_context_in_separate_snapshot(self):
        seed = self.root / ".local" / "sessions" / "original.json"
        messages = [{"role": "user", "content": "Старый вопрос"}, {"role": "assistant", "content": "Старый ответ"}]
        save_session(seed, messages, "old-model", target_model="old-repository")
        original = seed.read_bytes()
        result, server, generate = self.run_main(["history"], ["1", "Продолжение", "/menu", "0"])
        self.assertEqual(result, 0)
        self.assertEqual(server.call_count, 1)
        self.assertEqual(generate.call_args.args[1], messages + [{"role": "user", "content": "Продолжение"}])
        self.assertEqual(seed.read_bytes(), original)
        self.assertEqual(len(list(seed.parent.glob("*.json"))), 2)
        self.assertIn("old-repository", self.output.getvalue())

    def test_startup_error_returns_to_usable_menu(self):
        with (
            patch.object(cli, "ROOT", self.root),
            patch.object(cli, "console", self.console),
            patch("builtins.input", side_effect=["1", "0"]),
            patch.object(cli, "ManagedServer") as server,
        ):
            server.return_value.__enter__.side_effect = RuntimeErrorDetail("Модель не установлена")
            self.assertEqual(cli.main(["--config", str(self.path)]), 0)
        self.assertIn("Модель не установлена", self.output.getvalue())
        self.assertIn("До встречи.", self.output.getvalue())

    def test_explicit_chat_session_and_menu_shortcuts(self):
        seed = self.root / "seed.json"
        save_session(seed, [{"role": "user", "content": "Старый"}], "model")
        result, server, generate = self.run_main(["chat", "--session", "seed.json"], ["Продолжение", "/models", "0", "0"])
        self.assertEqual(result, 0)
        self.assertEqual(server.call_count, 1)
        self.assertEqual(generate.call_args.args[1][0], {"role": "user", "content": "Старый"})
        self.assertIn("Каталог и установка", self.output.getvalue())

    def manager(self):
        manager = MagicMock()
        manager.catalog.return_value = [SimpleNamespace(
            id="glm-coder", name="GLM Coder", parameters_b=30, repo_id="author/model-GGUF",
            description="Модель для кода", focus="код / программирование", censorship="по карточке автора",
            source_url="https://huggingface.co/author/model", default_quantization="Q4_K_M",
            default_size=20 * 2**30,
        )]
        manager.available_quantizations.return_value = [{"quantization": "Q4_K_M", "size": 20 * 2**30, "files": ["model.gguf"]}]
        manager.install.return_value = {"id": "glm-coder-q4", "name": "GLM Coder"}
        manager.installed.return_value = [{
            "id": "glm-coder-q4", "name": "GLM Coder", "model_id": "glm-coder", "status": "ready",
            "quantization": "Q4_K_M", "total_size": 20 * 2**30, "active": False, "managed": True,
        }]
        return manager

    def test_catalog_search_finds_small_models_and_shows_purpose_and_download_size(self):
        manager = self.manager()
        small = SimpleNamespace(**vars(manager.catalog.return_value[0]))
        small.id, small.name, small.parameters_b = "tiny-text", "Tiny Text", 3
        small.description, small.focus = "Сводки и извлечение фактов.", "анализ текста"
        small.default_size = 2 * 2**30
        manager.catalog.return_value.append(small)
        for query in ("МАЛЕНЬКИЕ", "анализ текста", "3B"):
            with self.subTest(query=query):
                self.assertEqual(menu.print_catalog(manager, self.console, query), [small])
        self.assertIn("Назначение: анализ текста", self.output.getvalue())
        self.assertIn("Q4_K_M", self.output.getvalue())
        self.assertIn("2.00 ГиБ", self.output.getvalue())
        manager.available_quantizations.assert_not_called()

    def test_catalog_size_groups_use_total_parameters_and_include_boundaries(self):
        manager = self.manager()
        template = vars(manager.catalog.return_value[0])
        specs = []
        for parameters in (3, 10, 14, 19.9, 20, 30, 60, 60.1, 61):
            spec = SimpleNamespace(**template)
            spec.id, spec.name, spec.parameters_b = f"model-{parameters}", f"Model {parameters}", parameters
            specs.append(spec)
        specs[2].description = "Небольшие скрипты."
        mistral = SimpleNamespace(**template)
        mistral.id, mistral.name, mistral.parameters_b = "mistral-small", "Mistral Small 24B", 24
        specs.append(mistral)
        manager.catalog.return_value = specs
        expected_large = [specs[4], specs[5], specs[6], mistral]
        for query in ("БОЛЬШИЕ", " LaRgE ", "20-60B", "20–60b"):
            with self.subTest(query=query):
                self.assertEqual(menu.print_catalog(manager, self.console, query), expected_large)
        for query in ("МАЛЕНЬКИЕ", "КОМПАКТНЫЕ", "small"):
            with self.subTest(query=query):
                self.assertEqual(menu.print_catalog(manager, self.console, query), specs[:2])
        self.assertEqual(menu.print_catalog(manager, self.console, "mistral small"), [mistral])
        self.assertEqual(menu.print_catalog(manager, self.console, "небольшие"), [specs[2]])
        self.assertEqual(menu.print_catalog(manager, self.console), specs)
        manager.available_quantizations.assert_not_called()

    def test_install_selection_reports_size_and_does_not_activate(self):
        manager = self.manager()
        with patch("builtins.input", side_effect=["1", "код", "1", "1", "0", "0"]):
            result = menu.models_menu(self.config, self.path, manager, self.console)
        manager.install.assert_called_once_with("glm-coder", "Q4_K_M")
        manager.activate.assert_not_called()
        self.assertIs(result, self.config)
        self.assertIn("20.00 ГиБ", self.output.getvalue())

    def test_failed_download_does_not_change_config_and_menu_remains_usable(self):
        manager = self.manager()
        manager.install.side_effect = ModelError("Недостаточно места")
        before = self.path.read_bytes()
        with patch("builtins.input", side_effect=["1", "1", "1", "0"]):
            menu.models_menu(self.config, self.path, manager, self.console)
        self.assertEqual(before, self.path.read_bytes())
        manager.activate.assert_not_called()
        self.assertIn("Недостаточно места", self.output.getvalue())

    def test_activation_updates_next_chat_config(self):
        manager = self.manager()
        updated = copy.deepcopy(self.config)
        updated["target_model"] = "author/model"
        manager.activate.return_value = updated
        with patch("builtins.input", side_effect=["2", "1", "0"]):
            result = menu.models_menu(self.config, self.path, manager, self.console)
        manager.activate.assert_called_once_with("glm-coder-q4", self.config, self.path)
        self.assertIs(result, updated)

    def test_deletion_requires_confirmation_and_rejects_active_package(self):
        manager = self.manager()
        with patch("builtins.input", side_effect=["2", "у 1", "", "0"]):
            menu.models_menu(self.config, self.path, manager, self.console)
        manager.remove.assert_not_called()
        with patch("builtins.input", side_effect=["2", "у 1", "удалить", "0"]):
            menu.models_menu(self.config, self.path, manager, self.console)
        manager.remove.assert_called_once_with("glm-coder-q4", self.config)
        manager.remove.reset_mock()
        manager.installed.return_value[0]["active"] = True
        with patch("builtins.input", side_effect=["2", "у 1", "0"]):
            menu.models_menu(self.config, self.path, manager, self.console)
        manager.remove.assert_not_called()

    def test_eof_exits_without_loading_model(self):
        result, server, generate = self.run_main([], [EOFError()])
        self.assertEqual(result, 0)
        server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
