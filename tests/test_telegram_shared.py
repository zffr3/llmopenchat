"""Shared Telegram sessions, administrator permissions and request ownership."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from llmopenchat import telegram
from llmopenchat._files import supported
from llmopenchat.api import StreamEvent
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session

from tests.test_telegram import (FakeAPI, MISSING_USERNAME, OTHER, OTHER_NAME, StreamingClient,
                                USER, USER_NAME, callback_update, message_update, tool_call)


THIRD = 303
THIRD_NAME = "charlie_user"


class PausedClient(StreamingClient):
    """An HTTP stream that remains silent until released or cancelled."""

    def __init__(self, answer="Первый ответ"):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.answer = answer

    def cancel(self):
        super().cancel()
        self.release.set()

    def stream_chat(self, messages, tools=None, tool_choice=None):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        self.started.set()
        if not self.release.wait(timeout=4):
            raise AssertionError("The test did not release the silent HTTP stream")
        yield StreamEvent("content", self.answer)
        yield StreamEvent("finish", "stop")


class SharedTelegramTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmopenchat-shared-telegram-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repository = self.root / "project"
        self.repository.mkdir()
        self.whitelist = self.root / "whitelist.txt"
        self.whitelist.write_text(f"{OTHER_NAME},{THIRD_NAME}", encoding="utf-8")
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["backend"] = "external"
        self.config["telegram"] = {"admin_user_id": USER, "repository": "project"}
        self.api = FakeAPI()
        self.bot = self.make_bot(self.api)
        for user_id in (USER, OTHER, THIRD):
            self.bot.handle_update(message_update("/id", user_id))
        self.api.messages.clear()

    def make_bot(self, api):
        bot = telegram.TelegramBot(api, self.config, self.root, self.whitelist, emit=lambda _: None)
        self.addCleanup(bot.close)
        return bot

    def join_worker(self, user_id=OTHER):
        user = self.bot.users.get(user_id)
        if user is not None and user.worker is not None:
            user.worker.join(timeout=3)
            self.assertFalse(user.worker.is_alive(), "The shared worker did not finish")

    def send(self, text, user_id=OTHER, **kwargs):
        self.bot.handle_update(message_update(text, user_id, **kwargs))
        self.join_worker(user_id)
        return self.bot.users.get(user_id) or self.bot.user_chat(user_id)

    def approval(self):
        message = self.api.wait_for(lambda item: item["chat_id"] == USER
                                   and item["reply_markup"] is not None
                                   and item["reply_markup"]["inline_keyboard"][0][0]["callback_data"].startswith("approve:"))
        data = message["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
        token = data.split(":")[1]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.bot.control_lock:
                pending = self.bot.approvals.get(token)
                if pending is not None and pending.message_id is not None:
                    return message, data, pending
            threading.Event().wait(0.001)
        self.fail("The administrator approval was not registered")

    def test_administrator_id_survives_missing_username_and_missing_whitelist(self):
        self.whitelist.unlink()
        self.assertTrue(self.bot.allowed(USER))
        self.assertFalse(self.bot.allowed(OTHER))
        self.send("/system Общая инструкция", USER, username=MISSING_USERNAME)
        self.assertTrue(self.bot.allowed(USER))
        self.assertEqual(self.bot.users[USER].messages,
                         [{"role": "system", "content": "Общая инструкция"}])
        self.assertNotIn("Доступ запрещён", self.api.messages[-1]["text"])

    def test_all_participants_use_one_state_directory_and_configured_repository(self):
        client = self.send("/start")
        second = self.send("/start", THIRD)
        admin = self.send("/start", USER)
        self.assertIs(client, second)
        self.assertIs(client, admin)
        self.assertIs(self.bot.user_chat(OTHER), self.bot.user_chat(USER))
        self.assertEqual(client.user_id, USER)
        self.assertEqual(client.directory, self.root / ".local" / "telegram" / "shared")
        self.assertEqual(client.settings.repository, self.repository)
        self.assertFalse((self.root / ".local" / "telegram" / "users").exists())

    def test_client_commands_and_direct_command_calls_cannot_change_shared_configuration(self):
        user = self.send("/start")
        before = copy.deepcopy(user.config), copy.deepcopy(user.messages), user.settings
        commands = ("/tools web on", "/tools repo .", "/rag on", "/system forbidden",
                    "/clear", "/thinking", "/save stolen.json", "/load unknown.json",
                    "/history", "/models", "/cancel", "/quit", "/exit")
        with (patch.object(self.bot, "tools") as tools,
              patch.object(self.bot, "rag") as rag,
              patch.object(self.bot, "models") as models,
              patch.object(self.bot, "_reset") as reset):
            for command in commands:
                with self.subTest(command=command):
                    self.bot.command(user, OTHER, command)
                    self.assertEqual((user.config, user.messages, user.settings), before)
                    self.assertIn("админ", self.api.messages[-1]["text"].casefold())
            tools.assert_not_called()
            rag.assert_not_called()
            models.assert_not_called()
            reset.assert_not_called()
        self.assertEqual(self.api.documents, [])

    def test_client_document_is_rejected_without_download_or_index_changes(self):
        user = self.send("/start")
        document = {"file_id": "private-file", "file_name": "notes.txt", "file_size": 10}
        update = message_update("", OTHER)
        update["message"]["document"] = document
        self.api.download_document = Mock(return_value=b"not imported")
        with patch.object(self.bot, "add_document") as add:
            self.bot.handle_update(update)
            self.join_worker()
            add.assert_not_called()
        self.api.download_document.assert_not_called()
        self.assertFalse((user.workspace / "documents").exists())
        self.assertIn("админ", self.api.messages[-1]["text"].casefold())

    def test_client_menu_and_help_expose_only_client_actions(self):
        for command in ("/start", "/menu"):
            self.send(command)
            menu = self.api.messages[-1]
            self.assertIsNotNone(menu["reply_markup"])
            actions = [self.bot.buttons[button["callback_data"][4:]]
                       for row in menu["reply_markup"]["inline_keyboard"] for button in row]
            self.assertTrue(actions)
            self.assertTrue(all(action.user_id == OTHER for action in actions))
            self.assertTrue({action.command for action in actions}
                            <= {"/start", "/help", "/menu", "/status", "/id"})
            for command in ("/tools", "/models", "/clear", "/rag", "/save", "/system"):
                self.assertNotIn(command, menu["text"])
        self.send("/help")
        help_text = self.api.messages[-1]["text"]
        self.assertIn("/status", help_text)
        for command in ("/tools", "/models", "/clear", "/rag", "/save", "/system"):
            self.assertNotIn(command, help_text)

    def test_client_cannot_replay_administrator_menu_button(self):
        self.send("/tools", USER)
        menu = self.api.messages[-1]
        button = next(button for row in menu["reply_markup"]["inline_keyboard"] for button in row
                      if "web" in self.bot.buttons[button["callback_data"][4:]].command)
        user = self.bot.user_chat(USER)
        self.bot.handle_update(callback_update(button["callback_data"], menu["message_id"], OTHER))
        self.join_worker()
        self.assertFalse(user.settings.web_enabled)
        self.bot.handle_update(callback_update(button["callback_data"], menu["message_id"], USER,
                                              username=MISSING_USERNAME))
        self.join_worker(USER)
        self.assertTrue(user.settings.web_enabled)

    @unittest.skipUnless(supported(), "Secure repository access requires Windows or Linux")
    def test_admin_repository_tools_and_rag_survive_clear_and_restart(self):
        user = self.send("/start", USER)
        for command in ("/tools web on", "/tools code on", "/rag on"):
            self.send(command, USER)
        original = user.settings
        self.send("/clear", USER)
        self.assertEqual(user.settings, original)
        self.assertTrue(user.rag.enabled)
        self.assertTrue((user.directory / "settings.json").is_file())
        self.bot.close()
        restarted = self.make_bot(FakeAPI())
        restored = restarted.user_chat(OTHER)
        self.assertTrue(restored.settings.web_enabled)
        self.assertTrue(restored.settings.code_enabled)
        self.assertEqual(restored.settings.repository, self.repository)
        self.assertTrue(restored.rag.enabled)

    @unittest.skipUnless(supported(), "Secure repository access requires Windows or Linux")
    def test_admin_can_choose_root_relative_or_absolute_existing_repository(self):
        user = self.send("/start", USER)
        other_repo = self.root / "second-project"
        other_repo.mkdir()
        self.send("/tools repo second-project", USER)
        self.assertEqual(user.settings.repository, other_repo)
        with tempfile.TemporaryDirectory(prefix="llmopenchat-external-repo-") as folder:
            external = Path(folder).resolve()
            self.send('/tools repo "' + str(external) + '"', USER)
            self.assertEqual(user.settings.repository, external)
            current = user.settings
            with self.assertRaises(ValueError):
                self.bot.command(user, USER, "/tools repo missing-folder")
            self.assertEqual(user.settings, current)
            self.bot.close()
            self.config["telegram"].pop("repository")
            restarted = self.make_bot(FakeAPI())
            self.assertEqual(restarted.user_chat(THIRD).settings.repository, external)
            restarted.close()
            self.config["telegram"]["repository"] = "project"
            overridden = self.make_bot(FakeAPI())
            self.assertEqual(overridden.user_chat(THIRD).settings.repository, self.repository)

    def test_fifo_client_prompts_see_prior_turn_and_reply_to_initiating_chat(self):
        first = PausedClient()
        second = StreamingClient([StreamEvent("content", "Второй ответ"), StreamEvent("finish", "stop")])
        with patch.object(telegram, "ApiClient", side_effect=[first, second]) as clients:
            self.bot.handle_update(message_update("Первый вопрос", OTHER))
            self.assertTrue(first.started.wait(timeout=2))
            user = self.bot.users[OTHER]
            self.assertEqual(user.request_user_id, OTHER)
            self.bot.handle_update(message_update("Второй вопрос", THIRD))
            first.release.set()
            self.join_worker(OTHER)
            self.join_worker(THIRD)
            self.assertEqual(clients.call_count, 2)
        context = second.requests[0][0]
        self.assertEqual(context[-3:], [{"role": "user", "content": "Первый вопрос"},
                                       {"role": "assistant", "content": "Первый ответ"},
                                       {"role": "user", "content": "Второй вопрос"}])
        self.assertTrue(any(item["chat_id"] == OTHER and item["text"] == "Первый ответ"
                            for item in self.api.messages))
        self.assertTrue(any(item["chat_id"] == THIRD and item["text"] == "Второй ответ"
                            for item in self.api.messages))
        self.assertIs(self.bot.users[OTHER], self.bot.users[THIRD])
        self.assertEqual(load_session(user.directory / "current.json"), user.messages)
        self.bot.close()
        restarted = self.make_bot(FakeAPI())
        self.assertEqual(restarted.user_chat(USER).messages, user.messages)

    def test_client_status_callback_works_while_shared_generation_is_busy(self):
        self.send("/start")
        menu = self.api.messages[-1]
        data = next(button["callback_data"]
                    for row in menu["reply_markup"]["inline_keyboard"] for button in row
                    if self.bot.buttons[button["callback_data"][4:]].command == "/status")
        client = PausedClient()
        with patch.object(telegram, "ApiClient", return_value=client):
            self.bot.handle_update(message_update("Долгий вопрос", OTHER))
            self.assertTrue(client.started.wait(timeout=2))
            self.bot.handle_update(callback_update(data, menu["message_id"], OTHER))
            self.assertEqual(self.api.messages[-1]["chat_id"], OTHER)
            self.assertIn("Выполняется запрос", self.api.messages[-1]["text"])
            self.assertFalse(client.cancelled)
            self.assertTrue(self.bot.users[OTHER].worker.is_alive())
            client.release.set()
            self.join_worker()
        self.assertEqual(self.bot.users[OTHER].messages[-1]["content"], "Первый ответ")

    def test_username_transfer_cannot_displace_or_cancel_numeric_administrator(self):
        self.whitelist.write_text(f"{OTHER_NAME},{THIRD_NAME},{USER_NAME}", encoding="utf-8")
        client = PausedClient("Ответ администратору")
        with patch.object(telegram, "ApiClient", return_value=client):
            self.bot.handle_update(message_update("Запрос администратора", USER))
            self.assertTrue(client.started.wait(timeout=2))
            self.bot.handle_update(message_update("/status", OTHER, username=USER_NAME))
            self.assertTrue(self.bot.allowed(USER))
            self.assertTrue(self.bot.allowed(OTHER))
            self.assertFalse(client.cancelled)
            self.assertFalse(self.bot.users[USER].cancelled.is_set())
            client.release.set()
            self.join_worker(USER)
        self.assertEqual(self.bot.users[USER].messages[-1]["content"], "Ответ администратору")

    def test_admin_cancel_before_worker_starts_removes_only_first_queued_prompt(self):
        started, release = threading.Event(), threading.Event()
        real_work = self.bot._shared_work

        def delayed_work(user):
            started.set()
            release.wait(timeout=3)
            real_work(user)

        client = StreamingClient([StreamEvent("content", "Ответ второму клиенту"),
                                  StreamEvent("finish", "stop")])
        with (patch.object(self.bot, "_shared_work", side_effect=delayed_work),
              patch.object(telegram, "ApiClient", return_value=client) as clients):
            try:
                self.bot.handle_update(message_update("Отменённый первый вопрос", OTHER))
                self.assertTrue(started.wait(timeout=2))
                user = self.bot.users[OTHER]
                self.assertIsNone(user.request_user_id)
                self.bot.handle_update(message_update("Второй вопрос из очереди", THIRD))
                self.bot.handle_update(message_update("/cancel", USER))
            finally:
                release.set()
                self.join_worker(OTHER)
            self.assertEqual(clients.call_count, 1)
        self.assertEqual(client.requests[0][0][-1],
                         {"role": "user", "content": "Второй вопрос из очереди"})
        self.assertNotIn("Отменённый первый вопрос", json.dumps(user.messages, ensure_ascii=False))
        self.assertEqual(user.messages[-1]["content"], "Ответ второму клиенту")
        self.assertTrue(any(item["chat_id"] == THIRD and item["text"] == "Ответ второму клиенту"
                            for item in self.api.messages))

    def test_admin_quit_before_worker_starts_clears_history_and_next_answer_survives(self):
        user = self.bot.user_chat(OTHER)
        user.messages.extend([{"role": "user", "content": "Старая общая история"},
                              {"role": "assistant", "content": "Старый общий ответ"}])
        self.bot._save(user)
        started, release = threading.Event(), threading.Event()
        real_work = self.bot._shared_work

        def delayed_work(shared):
            started.set()
            release.wait(timeout=3)
            real_work(shared)

        client = StreamingClient([StreamEvent("content", "Новый ответ сохранён"),
                                  StreamEvent("finish", "stop")])
        with (patch.object(self.bot, "_shared_work", side_effect=delayed_work),
              patch.object(telegram, "ApiClient", return_value=client) as clients):
            try:
                self.bot.handle_update(message_update("Первый удалённый вопрос", OTHER))
                self.assertTrue(started.wait(timeout=2))
                self.assertIsNone(user.request_user_id)
                self.bot.handle_update(message_update("Второй удалённый вопрос", THIRD))
                self.bot.handle_update(message_update("/quit", USER))
                self.assertEqual([item["role"] for item in user.messages], ["system"])
                self.assertFalse(self.bot.shared_queue)
                self.assertFalse(user.close_requested.is_set())
                clients.assert_not_called()
            finally:
                release.set()
                self.join_worker(OTHER)
            self.send("Новый вопрос после закрытия", OTHER)
            self.assertEqual(clients.call_count, 1)
        self.assertEqual(user.messages[-2:], [
            {"role": "user", "content": "Новый вопрос после закрытия"},
            {"role": "assistant", "content": "Новый ответ сохранён"},
        ])
        self.assertEqual(load_session(user.directory / "current.json"), user.messages)
        self.assertNotIn("удалённый вопрос", json.dumps(user.messages, ensure_ascii=False))
        self.assertNotIn("Старая общая история", json.dumps(user.messages, ensure_ascii=False))
        self.assertFalse(user.close_requested.is_set())

    def test_admin_quit_during_stream_discards_queue_history_and_preserves_tools(self):
        user = self.send("/tools web on", USER)
        user.messages.extend([{"role": "user", "content": "Старая общая история"},
                              {"role": "assistant", "content": "Старый общий ответ"}])
        self.bot._save(user)
        original_settings = user.settings
        active = PausedClient()
        resumed = StreamingClient([StreamEvent("content", "Ответ после закрытия"),
                                   StreamEvent("finish", "stop")])
        with patch.object(telegram, "ApiClient", side_effect=[active, resumed]) as clients:
            self.bot.handle_update(message_update("Активный вопрос", OTHER))
            self.assertTrue(active.started.wait(timeout=2))
            self.bot.handle_update(message_update("Удалённый вопрос из очереди", THIRD))
            self.bot.handle_update(message_update("/quit", USER))
            self.join_worker(OTHER)
            self.assertTrue(active.cancelled)
            self.assertEqual(clients.call_count, 1)
            self.assertFalse(self.bot.shared_queue)
            self.assertFalse(user.close_requested.is_set())
            self.assertEqual([item["role"] for item in user.messages], ["system"])
            self.assertEqual(user.settings, original_settings)
            self.send("Новый вопрос", OTHER)
            self.assertEqual(clients.call_count, 2)
        self.assertEqual(user.settings, original_settings)
        self.assertEqual(user.messages[-1]["content"], "Ответ после закрытия")
        self.assertNotIn("Удалённый вопрос из очереди", json.dumps(user.messages, ensure_ascii=False))
        self.assertEqual(load_session(user.directory / "current.json"), user.messages)

    def test_unrelated_revoked_client_cannot_cancel_shared_generation(self):
        self.send("/start", THIRD)
        client = PausedClient()
        with patch.object(telegram, "ApiClient", return_value=client):
            self.bot.handle_update(message_update("Долгий вопрос", OTHER))
            self.assertTrue(client.started.wait(timeout=2))
            self.bot.handle_update(message_update("/cancel", THIRD, username="renamed_user"))
            self.assertFalse(client.cancelled)
            self.assertFalse(self.bot.users[OTHER].cancelled.is_set())
            self.bot.handle_update(message_update("/cancel", USER, username=MISSING_USERNAME))
            self.join_worker()
        self.assertTrue(client.cancelled)
        self.assertTrue(any(item["chat_id"] == OTHER and "отменена" in item["text"]
                            for item in self.api.messages))

    def test_queued_request_is_rechecked_when_its_client_loses_access(self):
        client = PausedClient()
        with patch.object(telegram, "ApiClient", return_value=client) as clients:
            self.bot.handle_update(message_update("Первый вопрос", OTHER))
            self.assertTrue(client.started.wait(timeout=2))
            self.bot.handle_update(message_update("Отозванный вопрос", THIRD))
            self.bot.handle_update(message_update("/id", THIRD, username=MISSING_USERNAME))
            self.assertFalse(client.cancelled)
            client.release.set()
            self.join_worker(OTHER)
            self.join_worker(THIRD)
            self.assertEqual(clients.call_count, 1)
        self.assertEqual(self.bot.users[OTHER].messages[-1]["content"], "Первый ответ")
        self.assertNotIn("Отозванный вопрос", json.dumps(self.bot.users[OTHER].messages, ensure_ascii=False))

    def test_client_cancel_command_does_not_interrupt_active_shared_http(self):
        client = PausedClient()
        with patch.object(telegram, "ApiClient", return_value=client):
            self.bot.handle_update(message_update("Долгий вопрос", OTHER))
            self.assertTrue(client.started.wait(timeout=2))
            self.bot.handle_update(message_update("/cancel", OTHER))
            self.assertFalse(client.cancelled)
            client.release.set()
            self.join_worker()
        self.assertEqual(self.bot.users[OTHER].messages[-1]["content"], "Первый ответ")

    def test_revoking_active_requester_cancels_the_shared_http_stream(self):
        client = PausedClient()
        with patch.object(telegram, "ApiClient", return_value=client):
            self.bot.handle_update(message_update("Долгий вопрос", OTHER))
            self.assertTrue(client.started.wait(timeout=2))
            self.bot.handle_update(message_update("/status", OTHER, username=MISSING_USERNAME))
            self.join_worker()
        self.assertTrue(client.cancelled)
        self.assertNotEqual(self.bot.users[OTHER].messages[-1]["content"], "Первый ответ")

    @unittest.skipUnless(supported(), "Secure repository access requires Windows or Linux")
    def test_client_tool_call_runs_real_harness_only_after_admin_approval(self):
        (self.repository / "notes.txt").write_text("Общие данные репозитория", encoding="utf-8")
        self.send("/tools code on", USER)
        read = tool_call("shared-read", "read_file", '{"path":"notes.txt"}')
        client = StreamingClient(
            [StreamEvent("tool_call", read), StreamEvent("finish", "tool_calls"), StreamEvent("done", {})],
            [StreamEvent("content", "Прочитал общие данные."), StreamEvent("finish", "stop")],
        )
        with patch.object(telegram, "ApiClient", return_value=client):
            self.bot.handle_update(message_update("Прочитай notes.txt", OTHER))
            message, data, pending = self.approval()
            self.assertEqual(message["chat_id"], USER)
            for update in (callback_update(data, message["message_id"], OTHER),
                           callback_update(data, message["message_id"], OTHER, chat_id=USER),
                           callback_update(data, message["message_id"] + 1, USER)):
                self.bot.handle_update(update)
                self.assertFalse(pending.event.is_set())
            self.bot.handle_update(callback_update(data, message["message_id"], USER,
                                                  username=MISSING_USERNAME))
            self.join_worker()
        self.assertEqual(len(client.requests), 2)
        result = next(item for item in client.requests[1][0] if item["role"] == "tool")
        self.assertEqual(json.loads(result["content"])["content"], "Общие данные репозитория")
        self.assertTrue(any(item["chat_id"] == OTHER and "Прочитал общие данные." in item["text"]
                            for item in self.api.messages))
        self.assertEqual(self.bot.approvals, {})

    @unittest.skipUnless(supported(), "Secure repository access requires Windows or Linux")
    def test_requester_revocation_denies_an_approval_owned_by_admin(self):
        (self.repository / "notes.txt").write_text("Общие данные", encoding="utf-8")
        self.send("/tools code on", USER)
        read = tool_call("shared-read", "read_file", '{"path":"notes.txt"}')
        client = StreamingClient(
            [StreamEvent("tool_call", read), StreamEvent("finish", "tool_calls"), StreamEvent("done", {})],
            [StreamEvent("content", "Не должно быть ответа"), StreamEvent("finish", "stop")],
        )
        with patch.object(telegram, "ApiClient", return_value=client):
            self.bot.handle_update(message_update("Прочитай notes.txt", OTHER))
            message, data, pending = self.approval()
            self.bot.handle_update(message_update("/id", OTHER, username="revoked_user"))
            self.join_worker()
            self.bot.handle_update(callback_update(data, message["message_id"], USER))
        self.assertTrue(pending.event.is_set())
        self.assertFalse(pending.allowed)
        self.assertTrue(client.cancelled)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(self.bot.approvals, {})


if __name__ == "__main__":
    unittest.main()
