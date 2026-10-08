from __future__ import annotations

import asyncio
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from llmopenchat import cli
from llmopenchat.api import StreamEvent
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.hotkeys import PromptAction, create_key_bindings


def read_prompt(session, pipe, keys, **kwargs):
    async def read():
        try:
            return await session.prompt_async(
                "Test > ",
                pre_run=lambda: pipe.send_text(keys),
                handle_sigint=False,
                **kwargs,
            )
        except KeyboardInterrupt as error:
            # Catch inside this task so asyncio does not treat Ctrl+C as a signal.
            return error

    return asyncio.run(asyncio.wait_for(read(), timeout=5))


class KeyDispatchTests(unittest.TestCase):
    def session(self, pipe):
        history = InMemoryHistory()
        return PromptSession(
            input=pipe,
            output=DummyOutput(),
            history=history,
            key_bindings=create_key_bindings(),
        ), history

    def test_command_shortcuts_return_actions_without_accepting_draft(self):
        for keys, command in (
            ("\x0e", "/clear"),
            ("\x13", "/save"),
            ("\x14", "/thinking"),
            ("\x11", "/quit"),
            ("\x1bOP", "/help"),
            ("\x1bOQ", "/menu"),
            ("\x1bOR", "/tools"),
        ):
            with self.subTest(command=command), create_pipe_input() as pipe:
                session, history = self.session(pipe)
                result = read_prompt(session, pipe, "Черновик\x1b[D\x1b[D" + keys)
                self.assertIsInstance(result, PromptAction)
                self.assertEqual(result.command, command)
                self.assertEqual(result.draft.text, "Черновик")
                self.assertEqual(result.draft.cursor_position, len("Черновик") - 2)
                self.assertEqual(history.get_strings(), [])

    def test_save_restores_cursor_and_records_only_the_eventually_sent_message(self):
        with create_pipe_input() as pipe:
            session, history = self.session(pipe)
            action = read_prompt(session, pipe, "Текст\x1b[D\x1b[D\x13")
            self.assertEqual(history.get_strings(), [])
            accepted = read_prompt(session, pipe, "!\r", default=action.draft)
            self.assertEqual(accepted, "Тек!ст")
            self.assertEqual(history.get_strings(), ["Тек!ст"])

    def test_backspace_keeps_its_editing_behavior(self):
        with create_pipe_input() as pipe:
            session, history = self.session(pipe)
            accepted = read_prompt(session, pipe, "Текст\x08\x7f\r")
            self.assertEqual(accepted, "Тек")
            self.assertEqual(history.get_strings(), ["Тек"])

    def test_alt_enter_adds_newline_before_enter_sends(self):
        with create_pipe_input() as pipe:
            session, history = self.session(pipe)
            accepted = read_prompt(session, pipe, "Первая\x1b\rВторая\r")
            self.assertEqual(accepted, "Первая\nВторая")
            self.assertEqual(history.get_strings(), ["Первая\nВторая"])

    def test_clear_screen_redraws_without_accepting_or_changing_draft(self):
        with create_pipe_input() as pipe:
            session, history = self.session(pipe)
            with patch.object(session.app.renderer, "clear", wraps=session.app.renderer.clear) as clear:
                accepted = read_prompt(session, pipe, "Текст\x1b[D\x0c!\r")
            clear.assert_called()
            self.assertEqual(accepted, "Текс!т")
            self.assertEqual(history.get_strings(), ["Текс!т"])

    def test_ctrl_c_interrupts_and_does_not_record_draft(self):
        with create_pipe_input() as pipe:
            session, history = self.session(pipe)
            interrupted = read_prompt(session, pipe, "Неотправленный\x03")
            self.assertIsInstance(interrupted, KeyboardInterrupt)
            self.assertEqual(history.get_strings(), [])


class RecordingClient:
    def __init__(self):
        self.requests = []

    def stream_chat(self, messages):
        self.requests.append(copy.deepcopy(messages))
        yield StreamEvent("reasoning", "Внутренняя мысль.")
        yield StreamEvent("content", "Ответ: " + messages[-1]["content"])
        yield StreamEvent("finish", "stop")
        yield StreamEvent("done", {})


class ChatHotkeyTests(unittest.TestCase):
    def run_chat(self, scripts):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        output = io.StringIO()
        history = InMemoryHistory()
        client = RecordingClient()
        calls = []
        real_session = None
        script_iterator = iter(scripts)

        with create_pipe_input() as pipe:
            class ScriptedSession:
                def prompt(self, *args, **kwargs):
                    calls.append(
                        {
                            "default": kwargs["default"],
                            "bottom_toolbar": kwargs["bottom_toolbar"],
                            "requests_before": len(client.requests),
                        }
                    )
                    result = read_prompt(real_session, pipe, next(script_iterator), **kwargs)
                    if isinstance(result, KeyboardInterrupt):
                        raise result
                    return result

            def create_session(*args, **kwargs):
                nonlocal real_session
                real_session = PromptSession(
                    input=pipe,
                    output=DummyOutput(),
                    history=history,
                    key_bindings=kwargs["key_bindings"],
                    auto_suggest=kwargs.get("auto_suggest"),
                )
                return ScriptedSession()

            with (
                patch("llmopenchat.cli.PromptSession", side_effect=create_session),
                patch("llmopenchat.cli.sys.stdin", SimpleNamespace(isatty=lambda: True)),
                patch("llmopenchat.cli.ApiClient", return_value=client),
                patch("llmopenchat.cli.console", Console(file=output, color_system=None, width=240)),
            ):
                status = cli.chat(copy.deepcopy(DEFAULT_CONFIG), root)

        self.assertEqual(status, 0)
        self.assertEqual(len(calls), len(scripts))
        saved = [json.loads(path.read_text(encoding="utf-8")) for path in (root / ".local" / "sessions").glob("*.json")]
        return client.requests, calls, history.get_strings(), saved, output.getvalue()

    def test_new_dialog_drops_old_context_and_the_unsent_draft(self):
        requests, calls, history, saved, output = self.run_chat(
            ["Первый вопрос\r", "Старый черновик\x0e", "После очистки\r", "Не отправлять\x11"]
        )
        system = {"role": "system", "content": DEFAULT_CONFIG["system_prompt"]}
        self.assertEqual(requests, [[system, {"role": "user", "content": "Первый вопрос"}], [system, {"role": "user", "content": "После очистки"}]])
        self.assertEqual(calls[2]["default"].text, "")
        self.assertEqual(history, ["Первый вопрос", "После очистки"])
        self.assertEqual(len(saved), 2)
        self.assertIn("Начат новый диалог.", output)

    def test_save_help_thinking_and_redraw_keep_draft_without_api_requests(self):
        requests, calls, history, saved, output = self.run_chat(
            [
                "Черновик\x1b[D\x14",
                "\x13",
                "\x1bOP",
                "\x0c!\r",
                "Неотправленный\x14",
                "\x11",
            ]
        )
        self.assertEqual([call["requests_before"] for call in calls], [0, 0, 0, 0, 1, 1])
        for call in calls[1:4]:
            self.assertEqual(call["default"], Document("Черновик", cursor_position=len("Черновик") - 1))
            self.assertIn("мысли:да", call["bottom_toolbar"])
        self.assertIn("мысли:нет", calls[5]["bottom_toolbar"])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][-1]["content"], "Чернови!к")
        self.assertEqual(history, ["Чернови!к"])
        self.assertEqual(saved[0]["messages"][-1]["content"], "Ответ: Чернови!к")
        self.assertIn("Сохранено:", output)
        self.assertIn("/help", output)
        self.assertIn("Внутренняя мысль.", output)

    def test_quit_does_not_send_or_save_unfinished_input(self):
        requests, calls, history, saved, output = self.run_chat(["Неотправленный вопрос\x11"])
        self.assertEqual(requests, [])
        self.assertEqual(history, [])
        self.assertEqual(saved, [])
        self.assertIn("До встречи.", output)

    def test_clear_screen_keeps_previous_conversation_and_the_current_draft(self):
        requests, calls, history, saved, output = self.run_chat(
            ["Первый вопрос\r", "Черновик\x0c\r", "\x11"]
        )
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            requests[1],
            [
                {"role": "system", "content": DEFAULT_CONFIG["system_prompt"]},
                {"role": "user", "content": "Первый вопрос"},
                {"role": "assistant", "content": "Ответ: Первый вопрос"},
                {"role": "user", "content": "Черновик"},
            ],
        )
        self.assertEqual(history, ["Первый вопрос", "Черновик"])
        self.assertEqual(len(saved[0]["messages"]), 5)

    def test_ctrl_c_discards_draft_and_chat_can_continue(self):
        requests, calls, history, saved, output = self.run_chat(
            ["Отменённый черновик\x03", "Новый запрос\r", "\x11"]
        )
        self.assertEqual(calls[1]["default"].text, "")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][-1]["content"], "Новый запрос")
        self.assertEqual(history, ["Новый запрос"])
        self.assertIn("Ввод отменен.", output)


if __name__ == "__main__":
    unittest.main()
