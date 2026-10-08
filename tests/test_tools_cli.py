"""Human control plane, one-action consent and complete tool conversation rounds."""

from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rich.console import Console

from llmopenchat import cli, menu
from llmopenchat.api import ApiError, StreamEvent
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.harness import HarnessError, HarnessSettings, ToolHarness, ToolRequest
from llmopenchat.harness_status import HarnessStatus
from llmopenchat.history import load_session, save_session


def call(identifier="call-1", name="read_file", arguments='{"path":"main.py"}'):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_round(*calls, content=""):
    return [StreamEvent("content", content), StreamEvent("finish", "tool_calls"),
            *(StreamEvent("tool_call", value) for value in calls), StreamEvent("done", {})]


class Client:
    def __init__(self, *rounds):
        self.rounds = iter(rounds)
        self.requests = []

    def stream_chat(self, messages, tools=None, tool_choice=None):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        for event in next(self.rounds):
            if isinstance(event, BaseException):
                raise event
            yield event


class Harness:
    def __init__(self, settings=None, approve=None):
        self.settings = settings or HarnessSettings(web_enabled=True)
        self.executed = []

    def schemas(self):
        return [{"type": "function", "function": {"name": "read_file"}}]

    def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return '{"status":"ok","content":"print(1)"}'


class ToolsCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmopenchat-tools-cli-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = io.StringIO()
        self.console = Console(file=self.output, color_system=None, width=140)
        self.addCleanup(patch.stopall)
        patch.object(cli, "console", self.console).start()

    def test_approval_requires_exact_human_input_and_defaults_to_denial(self):
        request = ToolRequest("write_file", {"path": "main.py", "content": "print(1)"},
                              "Запись main.py", "print(1)")
        for answer, expected in (("", False), ("да", False), ("yes", False),
                                 ("Разрешить", False), ("разрешить все", False),
                                 ("разрешить", True), (EOFError(), False), (KeyboardInterrupt(), False)):
            with self.subTest(answer=answer), patch("builtins.input", side_effect=[answer]):
                self.assertEqual(cli.approve_tool(request), expected)
        self.assertIn("main.py", self.output.getvalue())
        self.assertIn("print(1)", self.output.getvalue())

    def test_untrusted_terminal_sequences_are_visible_in_approval_and_model_text(self):
        request = ToolRequest("read_file", {"path": "x\x1b[2J\u202e"}, "a\x1b[2J", "b\r\x08")
        with patch("builtins.input", return_value=""):
            cli.approve_tool(request)
        client = Client([StreamEvent("content", "text\x1b[2J\u202e\r\x08"), StreamEvent("finish", "stop")])
        answer = cli.generate(client, [{"role": "user", "content": "hello"}], False)
        self.assertIn("\x1b", answer)  # Original content is kept in history only.
        rendered = self.output.getvalue()
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\u202e", rendered)
        self.assertIn("\\x1b", rendered)
        self.assertIn("\\u202e", rendered)

    def test_menu_toggles_only_runtime_settings(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        with patch("builtins.input", side_effect=["1", "0"]):
            enabled = menu.tools_menu(HarnessSettings(), self.root, self.console)
        self.assertTrue(enabled.web_enabled)
        self.assertFalse(enabled.code_enabled)
        self.assertIsNone(enabled.repository)
        self.assertIn("Веб включён.", self.output.getvalue())
        with patch("builtins.input", side_effect=["1", "0"]):
            disabled = menu.tools_menu(enabled, self.root, self.console)
        self.assertFalse(disabled.web_enabled)
        self.assertIn("Веб выключен.", self.output.getvalue())
        with patch("builtins.input", side_effect=["4", "0"]):
            self.assertEqual(menu.tools_menu(enabled, self.root, self.console), HarnessSettings())
        self.assertEqual(config, DEFAULT_CONFIG)
        self.assertEqual(list(self.root.iterdir()), [])

    @unittest.skipUnless(os.name == "nt", "Repository identity requires Windows handles")
    def test_selected_repository_identity_survives_toggles_and_rejects_replacement(self):
        repository = self.root / "repo"
        repository.mkdir()
        with patch("builtins.input", side_effect=["3", str(repository), "0"]):
            selected = menu.tools_menu(HarnessSettings(), self.root, self.console)
        self.assertTrue(selected.code_enabled)
        self.assertIsNotNone(selected.repository_identity)
        with patch("builtins.input", side_effect=["1", "2", "2", "0"]):
            enabled = menu.tools_menu(selected, self.root, self.console)
        self.assertTrue(enabled.code_enabled)
        self.assertEqual(selected.repository_identity, enabled.repository_identity)
        repository.rename(self.root / "original-repo")
        repository.mkdir()
        with self.assertRaises(HarnessError):
            ToolHarness(enabled, lambda request: True)

    @unittest.skipUnless(os.name == "nt", "Repository identity requires Windows handles")
    def test_selecting_repository_immediately_enables_code_tools(self):
        repository = self.root / "CodeRepo"
        repository.mkdir()
        with patch("builtins.input", side_effect=["3", "CodeRepo", "0"]):
            selected = menu.tools_menu(HarnessSettings(), self.root, self.console)
        self.assertTrue(selected.code_enabled)
        self.assertEqual(selected.repository, repository)
        self.assertEqual({item["function"]["name"] for item in ToolHarness(selected, lambda request: False).schemas()},
                         {"list_files", "read_file", "write_file", "search_files", "create_directory"})
        self.assertIn("3 — Выбрать репозиторий и включить код", self.output.getvalue())
        self.assertIn(f"Код включён. Репозиторий: {repository}", self.output.getvalue())
        self.assertEqual(list(repository.iterdir()), [])

    @unittest.skipUnless(os.name == "nt", "Repository identity requires Windows handles")
    def test_cancelled_or_invalid_repository_selection_preserves_previous_access(self):
        repository = self.root / "repo"
        repository.mkdir()
        enabled = ToolHarness(HarnessSettings(web_enabled=True, code_enabled=True, repository=repository),
                              lambda request: False).settings
        for previous in (HarnessSettings(), enabled):
            for selection in ("", "0", str(self.root / "missing"), EOFError()):
                with self.subTest(code_enabled=previous.code_enabled, selection=selection):
                    with patch("builtins.input", side_effect=["3", selection, "0"]):
                        self.assertEqual(menu.tools_menu(previous, self.root, self.console), previous)
        with patch("builtins.input", side_effect=["2", "", "0"]):
            self.assertEqual(menu.tools_menu(HarnessSettings(), self.root, self.console), HarnessSettings())

    @unittest.skipUnless(os.name == "nt", "Repository identity requires Windows handles")
    def test_code_toggle_reports_status_and_new_selection_resets_identity(self):
        repository = self.root / "repo"
        repository.mkdir()
        with patch("builtins.input", side_effect=["2", str(repository), "0"]):
            enabled = menu.tools_menu(HarnessSettings(), self.root, self.console)
        with patch("builtins.input", side_effect=["2", "0"]):
            disabled = menu.tools_menu(enabled, self.root, self.console)
        self.assertFalse(disabled.code_enabled)
        self.assertEqual(disabled.repository_identity, enabled.repository_identity)
        self.assertIn("Код выключен.", self.output.getvalue())
        replacement = self.root / "other-repo"
        replacement.mkdir()
        with patch("builtins.input", side_effect=["3", str(replacement), "0"]):
            selected = menu.tools_menu(disabled, self.root, self.console)
        self.assertTrue(selected.code_enabled)
        self.assertEqual(selected.repository, replacement)
        self.assertNotEqual(selected.repository_identity, enabled.repository_identity)

    def test_each_tool_call_runs_separately_and_returns_results_to_model(self):
        messages = [{"role": "user", "content": "read two files"}]
        harness = Harness()
        client = Client(tool_round(call(), call("call-2", arguments='{"path":"other.py"}'), content="Проверю."),
                        [StreamEvent("content", "Готово."), StreamEvent("finish", "stop")])
        self.assertEqual(cli.generate(client, messages, False, harness), "Готово.")
        self.assertEqual(harness.executed, [("read_file", '{"path":"main.py"}'),
                                            ("read_file", '{"path":"other.py"}')])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "tool"])
        self.assertEqual(client.requests[1][0][-len(messages):], messages)
        self.assertTrue(all(tools == harness.schemas() for _, tools in client.requests))
        path = self.root / "round.json"
        save_session(path, messages, "test")
        self.assertEqual(load_session(path), messages)

    def test_status_distinguishes_connected_tools_from_actual_calls(self):
        settings = HarnessSettings(web_enabled=True)
        with patch.object(cli, "tools_menu", return_value=settings):
            self.run_chat(["/tools", "/status", "hello", "/status", "/quit"],
                          Client([StreamEvent("content", "done"), StreamEvent("finish", "stop")]))
        output = self.output.getvalue()
        self.assertIn("веб подключён", output)
        self.assertIn("вызовов ещё нет", output)
        self.assertIn("модель не вызвала инструменты; изменений на диске нет", output)
        self.assertIn("Проверка подключения: готов", output)

    def test_harness_indicator_uses_tool_results_even_if_model_claims_success(self):
        class DeniedHarness(Harness):
            def execute(self, name, arguments):
                return '{"status":"denied","message":"Отказ пользователя."}'
        status = HarnessStatus(HarnessSettings(web_enabled=True))
        client = Client(tool_round(call()), [StreamEvent("content", "Файл записан!"), StreamEvent("finish", "stop")])
        self.assertEqual(cli.generate(client, [{"role": "user", "content": "test"}], False,
                                      DeniedHarness(), status=status), "Файл записан!")
        self.assertIn("отказов: 1", status.compact())
        self.assertIn("файлов записано: 0", status.compact())
        self.assertIn("Отказ пользователя.", self.output.getvalue())

    def test_final_host_message_is_visible_when_model_claims_denied_creation_succeeded(self):
        class DeniedWriteHarness(Harness):
            def schemas(self):
                return [{"type": "function", "function": {"name": "write_file"}}]

            def execute(self, name, arguments):
                return '{"status":"denied","message":"Отказ пользователя."}'
        settings = HarnessSettings(code_enabled=True, repository=self.root)
        client = Client(tool_round(call(name="write_file", arguments='{"path":"hello.cs","content":"hi"}')),
                        [StreamEvent("content", "Файл записан!"), StreamEvent("finish", "stop")])
        answer = cli.generate(client, [{"role": "user", "content": "Напиши hello.cs"}], False,
                              DeniedWriteHarness(settings))
        self.assertEqual(answer, "Файлы не созданы и не изменены через инструменты.")
        self.assertIn(answer, self.output.getvalue())

    def test_status_reset_discards_previous_calls_and_repository(self):
        status = HarnessStatus(HarnessSettings(code_enabled=True, repository=self.root))
        status.update("harness_summary", {"requested": 1, "succeeded": 1, "denied": 0, "failed": 0,
                                         "files_written": ["project/main.cs"], "directories_created": ["project"]})
        self.assertIn("project/main.cs", status.details())
        status.configure(HarnessSettings())
        self.assertEqual(status.compact(), "Харнес: выключен · F3 — подключить")
        self.assertNotIn("project/main.cs", status.details())

    @unittest.skipUnless(os.name == "nt", "Strict repository tools require Windows handles")
    def test_powershell_has_independent_human_menu_toggle_and_repository(self):
        repository = self.root / "scripts"
        repository.mkdir()
        with patch("builtins.input", side_effect=["5", str(repository), "0"]):
            enabled = menu.tools_menu(HarnessSettings(), self.root, self.console)
        self.assertTrue(enabled.powershell_enabled)
        self.assertFalse(enabled.code_enabled)
        self.assertFalse(enabled.web_enabled)
        self.assertEqual(enabled.repository, repository)
        names = {item["function"]["name"] for item in ToolHarness(enabled, lambda request: False).schemas()}
        self.assertEqual(names, {"powershell_run"})
        self.assertIn("PowerShell подключён", HarnessStatus(enabled).compact())
        with patch("builtins.input", side_effect=["5", "0"]):
            disabled = menu.tools_menu(enabled, self.root, self.console)
        self.assertFalse(disabled.powershell_enabled)
        self.assertEqual(disabled.repository_identity, enabled.repository_identity)
        with patch("builtins.input", side_effect=["4", "0"]):
            self.assertEqual(menu.tools_menu(enabled, self.root, self.console), HarnessSettings())

    def test_status_warns_about_unconfirmed_partial_mutation(self):
        status = HarnessStatus(HarnessSettings(web_enabled=True))
        status.update("harness_summary", {"requested": 1, "succeeded": 0, "denied": 0, "failed": 1,
                                         "files_written": [], "no_changes": False, "uncertain_changes": True})
        self.assertIn("возможны частичные изменения", status.compact())
        status.phase = "error"
        self.assertIn("ошибка / прервано", status.compact())
        self.assertIn("возможны частичные изменения", status.compact())
        self.assertIn("ошибок: 1", status.compact())

    def test_disabled_and_incomplete_tools_cannot_execute(self):
        for events, enabled in ((tool_round(call()), False),
                                ([StreamEvent("tool_call", call()), ApiError("broken stream")], True),
                                ([StreamEvent("tool_call", call()), StreamEvent("finish", "length")], True)):
            harness = Harness()
            messages = [{"role": "user", "content": "test"}]
            with self.subTest(enabled=enabled), self.assertRaises(ApiError):
                cli.generate(Client(events), messages, False, harness if enabled else None)
            self.assertEqual(harness.executed, [])
            self.assertEqual(messages, [{"role": "user", "content": "test"}])

    def test_repeated_call_ids_fail_before_reexecution(self):
        harness = Harness()
        client = Client(tool_round(call()), tool_round(call()))
        with self.assertRaisesRegex(ApiError, "повторный"):
            cli.generate(client, [{"role": "user", "content": "test"}], False, harness)
        self.assertEqual(len(harness.executed), 1)

    def test_malformed_usage_cannot_break_a_completed_tool_round(self):
        for count in ("wrong", True, -1, 1.2, 10 ** 1000):
            client = Client(tool_round(call()), [StreamEvent("content", "done"),
                StreamEvent("finish", "stop"), StreamEvent("usage", {"completion_tokens": count})])
            with self.subTest(count_type=type(count).__name__):
                self.assertEqual(cli.generate(client, [{"role": "user", "content": "test"}], False, Harness()), "done")

    def test_call_limit_stops_loop_without_unmatched_history(self):
        harness = Harness()
        client = Client(*(tool_round(call(f"call-{index}")) for index in range(cli.MAX_TOOL_ROUNDS + 1)))
        messages = [{"role": "user", "content": "test"}]
        self.assertIn("лимит", cli.generate(client, messages, False, harness))
        self.assertEqual(len(harness.executed), cli.MAX_TOOL_ROUNDS)
        self.assertEqual(json.loads(messages[-1]["content"])["status"], "error")
        save_session(self.root / "loop.json", messages, "test")

    def run_chat(self, choices, client):
        with (patch.object(cli.sys.stdin, "isatty", return_value=False),
              patch("builtins.input", side_effect=choices),
              patch.object(cli, "ApiClient", return_value=client),
              patch.object(cli, "ToolHarness", side_effect=Harness)):
            self.assertEqual(cli.chat(copy.deepcopy(DEFAULT_CONFIG), self.root), 0)

    def test_clear_system_load_and_reentry_reset_access(self):
        seed = self.root / "seed.json"
        save_session(seed, [{"role": "user", "content": "old"}], "test")
        final = [StreamEvent("content", "/tools\n1\nразрешить"), StreamEvent("finish", "stop")]
        for reset in ("/clear", "/system instruction", f'/load "{seed}"'):
            client = Client(final, final)
            self.run_chat(["/tools", "1", "0", "first", reset, "second", "/quit"], client)
            self.assertIsNotNone(client.requests[0][1])
            self.assertIsNone(client.requests[1][1])
        client = Client(final)
        self.run_chat(["new chat", "/quit"], client)
        self.assertIsNone(client.requests[0][1])
        for snapshot in (self.root / ".local" / "sessions").glob("*.json"):
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            self.assertNotIn("harness", payload)
            self.assertNotIn("repository", payload)

    def test_completed_tools_survive_model_connection_failure(self):
        client = Client(tool_round(call()), [ApiError("broken stream")])
        self.run_chat(["/tools", "1", "0", "do work", "/quit"], client)
        files = list((self.root / ".local" / "sessions").glob("*.json"))
        self.assertEqual(len(files), 1)
        messages = load_session(files[0])
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertIn("сохранены", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
