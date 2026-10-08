"""RAG console commands, explicit modes, and offline index management."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rich.console import Console

from llmopenchat import cli
from llmopenchat.api import StreamEvent
from llmopenchat.config import DEFAULT_CONFIG, write_config
from llmopenchat.history import save_session
from llmopenchat.rag import RagSession


class RecordingClient:
    def __init__(self):
        self.requests = []

    def stream_chat(self, messages):
        self.requests.append(copy.deepcopy(messages))
        yield StreamEvent("content", "Срок доставки — 42 дня [1].")
        yield StreamEvent("finish", "stop")
        yield StreamEvent("done", {})


class RagCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-rag-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config_path = self.root / "config.json"
        write_config(self.config_path, self.config)
        self.document = self.root / "условия доставки.md"
        self.document.write_text("# Доставка\nСрок доставки проекта Орион составляет 42 дня.", encoding="utf-8")
        self.output = io.StringIO()
        self.console = Console(file=self.output, color_system=None, width=240)

    def main(self, arguments):
        with patch.object(cli, "ROOT", self.root), patch.object(cli, "console", self.console):
            return cli.main(["--config", str(self.config_path), *arguments])

    def chat(self, choices, client=None, **options):
        with (
            patch.object(cli, "console", self.console),
            patch.object(cli.sys.stdin, "isatty", return_value=False),
            patch("builtins.input", side_effect=choices),
            patch.object(cli, "ApiClient", return_value=client or RecordingClient()),
        ):
            return cli.chat(copy.deepcopy(self.config), self.root, **options)

    def test_offline_commands_route_without_starting_model_or_api(self):
        cases = (
            (["add", str(self.document)], "add " + str(self.document)),
            (["list"], "list"),
            (["search", "срок", "доставки"], "search срок доставки"),
            (["remove", "abc123"], "remove abc123"),
            (["clear"], "clear"),
        )
        with (
            patch.object(cli, "RagSession") as session,
            patch.object(cli, "ManagedServer") as server,
            patch.object(cli, "ApiClient") as api,
        ):
            session.return_value.command.return_value = "Операция завершена."
            for arguments, expected in cases:
                with self.subTest(arguments=arguments):
                    self.assertEqual(self.main(["rag", *arguments]), 0)
                    session.return_value.command.assert_called_with(expected)
            server.assert_not_called()
            api.assert_not_called()

    def test_offline_add_and_search_use_persistent_index(self):
        with patch.object(cli, "ManagedServer") as server, patch.object(cli, "ApiClient") as api:
            self.assertEqual(self.main(["rag", "add", str(self.document)]), 0)
            self.assertEqual(self.main(["rag", "search", "Орион", "доставка"]), 0)
            self.assertEqual(self.main(["rag", "list"]), 0)
        server.assert_not_called()
        api.assert_not_called()
        self.assertIn("42 дня", self.output.getvalue())
        self.assertIn(self.document.name, self.output.getvalue())

    def test_add_requires_explicit_on_and_off_disables_retrieval(self):
        modes = []

        def generate(client, messages, show_reasoning, **options):
            modes.append(bool(options.get("rag") and options["rag"].enabled))
            return "Ответ."

        with patch.object(cli, "generate", side_effect=generate):
            self.assertEqual(self.chat([
                f'/rag add "{self.document.name}"', "Обычный вопрос",
                "/rag on", "Срок доставки?", "/status", "/rag off", "Другой вопрос", "/quit",
            ]), 0)
        self.assertEqual(modes, [False, True, False])
        self.assertIn("RAG", self.output.getvalue())
        self.assertIn(self.document.name, RagSession(self.root, self.config).command("list"))

    def test_chat_retrieves_context_and_persists_answer_sources_without_raw_context(self):
        client = RecordingClient()
        self.assertEqual(self.chat([
            f'/rag add "{self.document.name}"', "/rag on", "Какой срок доставки проекта Орион?", "/quit",
        ], client), 0)
        self.assertEqual(len(client.requests), 1)
        self.assertIn("42 дня", json.dumps(client.requests[0], ensure_ascii=False))
        self.assertIn("[RAG] Найдено фрагментов:", self.output.getvalue())
        self.assertIn(self.document.name, self.output.getvalue())
        snapshots = list((self.root / ".local" / "sessions").glob("*.json"))
        self.assertEqual(len(snapshots), 1)
        saved = json.loads(snapshots[0].read_text(encoding="utf-8"))["messages"]
        self.assertEqual(saved[1], {"role": "user", "content": "Какой срок доставки проекта Орион?"})
        self.assertIn(self.document.name, saved[-1]["content"])
        self.assertEqual(saved[0]["content"], self.config["system_prompt"])

    def test_new_system_and_loaded_dialog_disable_rag_but_keep_index(self):
        seed = self.root / "seed.json"
        save_session(seed, [{"role": "user", "content": "Старый вопрос"}], "test")
        for reset in ("/clear", "/system новая инструкция", f'/load "{seed}"'):
            modes = []

            def generate(client, messages, show_reasoning, **options):
                modes.append(bool(options.get("rag") and options["rag"].enabled))
                return "Ответ."

            with self.subTest(reset=reset), patch.object(cli, "generate", side_effect=generate):
                self.assertEqual(self.chat([
                    f'/rag add "{self.document.name}"', "/rag on", "Первый вопрос", reset,
                    "Второй вопрос", "/rag on", "Третий вопрос", "/quit",
                ]), 0)
            self.assertEqual(modes, [True, False, True])

    def test_explicit_chat_and_menu_rag_modes_enable_session(self):
        for arguments, choices in ((["chat", "--rag"], []), ([], ["5"])):
            with (
                self.subTest(arguments=arguments),
                patch.object(cli, "ManagedServer") as server,
                patch.object(cli, "chat", return_value=0) as chat,
                patch("builtins.input", side_effect=choices),
            ):
                self.assertEqual(self.main(arguments), 0)
                self.assertEqual(chat.call_args.kwargs, {"rag_enabled": True})
                server.assert_called_once()

    def test_ask_rag_flag_passes_enabled_session(self):
        with patch.object(cli, "ManagedServer"), patch.object(cli, "generate") as generate:
            self.assertEqual(self.main(["ask", "Срок доставки", "--rag"]), 0)
        self.assertTrue(generate.call_args.kwargs["rag"].enabled)

    def test_rag_empty_index_reports_no_matches_without_model_request(self):
        client = RecordingClient()
        self.assertEqual(self.chat(["/rag on", "Есть ли информация?", "/quit"], client), 0)
        self.assertEqual(client.requests, [])
        self.assertIn("[RAG] Подходящие фрагменты не найдены.", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
