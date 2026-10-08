"""Telegram users, commands, single-action approvals and persisted tool rounds."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llmopenchat import telegram
from llmopenchat.api import ApiError, StreamEvent
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.harness import HarnessSettings, ToolRequest
from llmopenchat.history import load_session, save_session
from llmopenchat.runtime import RuntimeErrorDetail
from llmopenchat.telegram_api import TelegramError


USER = 101
OTHER = 202
USER_NAME = "alice_user"
OTHER_NAME = "bob_user"
NAMES = {USER: USER_NAME, OTHER: OTHER_NAME}
DEFAULT_USERNAME = object()
MISSING_USERNAME = object()


def sender(user_id, username=DEFAULT_USERNAME, *, is_bot=False):
    result = {"id": user_id, "is_bot": is_bot}
    if username is DEFAULT_USERNAME:
        username = NAMES.get(user_id, "charlie_user")
    if username is not MISSING_USERNAME:
        result["username"] = username
    return result


def message_update(text, user_id=USER, *, chat_id=None, chat_type="private", is_bot=False,
                   username=DEFAULT_USERNAME):
    return {"message": {"from": sender(user_id, username, is_bot=is_bot),
                        "chat": {"id": user_id if chat_id is None else chat_id, "type": chat_type},
                        "text": text}}


def callback_update(data, message_id, user_id=USER, *, chat_id=None, chat_type="private",
                    username=DEFAULT_USERNAME):
    return {"callback_query": {"id": "callback-id", "from": sender(user_id, username), "data": data,
                              "message": {"message_id": message_id, "chat": {
                                  "id": user_id if chat_id is None else chat_id, "type": chat_type}}}}


def tool_call(identifier="call-web", name="web_fetch", arguments='{"url":"https://example.com/"}'):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


class FakeAPI:
    def __init__(self):
        self.messages = []
        self.documents = []
        self.answers = []
        self.edits = []
        self.condition = threading.Condition()

    def send_message(self, chat_id, text, reply_markup=None):
        with self.condition:
            message = {"chat_id": chat_id, "text": text, "reply_markup": copy.deepcopy(reply_markup),
                       "message_id": len(self.messages) + 1}
            self.messages.append(message)
            self.condition.notify_all()
            return message

    def send_document(self, chat_id, filename, content, caption=""):
        with self.condition:
            self.documents.append((chat_id, filename, content, caption))
            self.condition.notify_all()
        return {"message_id": 999}

    def answer_callback_query(self, callback_id, text="", show_alert=False):
        self.answers.append((callback_id, text, show_alert))

    def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
        self.edits.append((chat_id, message_id, reply_markup))

    def wait_for(self, predicate, timeout=2):
        with self.condition:
            if not self.condition.wait_for(lambda: any(predicate(item) for item in self.messages), timeout):
                raise AssertionError("Bot did not send the expected message")
            return next(item for item in reversed(self.messages) if predicate(item))


class StreamingClient:
    def __init__(self, *rounds):
        self.rounds = iter(rounds)
        self.requests = []
        self.tool_choices = []
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def stream_chat(self, messages, tools=None, tool_choice=None):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        self.tool_choices.append(tool_choice)
        for event in next(self.rounds):
            if isinstance(event, BaseException):
                raise event
            yield event


class TelegramBotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmopenchat-telegram-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.whitelist = self.root / "whitelist.txt"
        self.whitelist.write_text(f"@{USER_NAME}, {OTHER_NAME}", encoding="utf-8")
        self.api = FakeAPI()
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["backend"] = "external"
        self.emitted = []
        self.bot = telegram.TelegramBot(self.api, self.config, self.root, self.whitelist,
                                        emit=self.emitted.append)
        self.addCleanup(self.bot.close)
        self.bot.handle_update(message_update("/id", USER))
        self.bot.handle_update(message_update("/id", OTHER))
        self.api.messages.clear()

    def join_worker(self, user_id=USER):
        user = self.bot.users.get(user_id)
        if user is not None and user.worker is not None:
            user.worker.join(timeout=3)
            self.assertFalse(user.worker.is_alive(), "User worker did not finish")

    def send(self, text, user_id=USER):
        self.bot.handle_update(message_update(text, user_id))
        self.join_worker(user_id)
        return self.bot.users.get(user_id)

    def ready_approval(self, user_id=USER):
        message = self.api.wait_for(lambda item: item["chat_id"] == user_id
                                   and item["reply_markup"] is not None
                                   and item["reply_markup"]["inline_keyboard"][0][0]["callback_data"].startswith("approve:")
                                   and item["reply_markup"]["inline_keyboard"][0][0]["callback_data"].split(":")[1] in self.bot.approvals)
        data = message["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
        token = data.split(":")[1]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.bot.control_lock:
                pending = self.bot.approvals.get(token)
                if pending is not None and pending.message_id is not None:
                    return message, data, pending
            threading.Event().wait(0.001)
        self.fail("Approval message ID was not registered")

    def begin_approval(self, request=None, user_id=USER):
        user = self.bot.user_chat(user_id)
        results = []
        request = request or ToolRequest("read_file", {"path": "file.txt"}, "Чтение file.txt")
        thread = threading.Thread(target=lambda: results.append(self.bot.approve(user, user_id, request)), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 1)
        return thread, results, self.ready_approval(user_id)

    def test_username_whitelist_normalizes_at_and_case_and_fails_closed(self):
        self.whitelist.write_text("\ufeff@ALICE_User, bob_USER,@alice_user\n", encoding="utf-8")
        self.assertEqual(telegram.read_whitelist(self.whitelist), {USER_NAME, OTHER_NAME})
        self.assertTrue(self.bot.allowed(USER))
        self.assertTrue(self.bot.allowed(OTHER))
        for content in ("101", "@101", "alice_user,", "alice-user", "@@alice_user",
                        "alice user", "пользователь", "https://t.me/alice_user", "a" * 33, "a" * 65537):
            self.whitelist.write_text(content, encoding="utf-8")
            with self.subTest(content=content[:20]), self.assertRaises(ValueError):
                telegram.read_whitelist(self.whitelist)
            self.assertFalse(self.bot.allowed(USER))
        self.whitelist.write_text("", encoding="utf-8")
        self.assertEqual(telegram.read_whitelist(self.whitelist), set())
        self.assertFalse(self.bot.allowed(USER))

    def test_token_parsing_is_bounded_and_does_not_expose_invalid_token(self):
        credentials = self.root / "credits.txt"
        credentials.write_text("\ufeff12345:" + "secret_token_" * 3 + "\n", encoding="utf-8")
        self.assertEqual(telegram.read_token(credentials), "12345:" + "secret_token_" * 3)
        credentials.write_text("12345:secret/path", encoding="utf-8")
        with self.assertRaises(ValueError) as raised:
            telegram.read_token(credentials)
        self.assertNotIn("secret/path", str(raised.exception))

    def test_unknown_users_can_get_id_but_create_no_directory_or_chat(self):
        unknown = 303
        self.send("/id", unknown)
        self.assertIn(str(unknown), self.api.messages[-1]["text"])
        self.assertIn("charlie_user", self.api.messages[-1]["text"])
        self.send("/start", unknown)
        self.assertIn("Доступ запрещён", self.api.messages[-1]["text"])
        self.assertNotIn(unknown, self.bot.users)
        self.assertFalse((self.root / ".local" / "telegram" / "users" / str(unknown)).exists())

    def test_missing_or_invalid_current_username_denies_even_after_prior_valid_observation(self):
        self.assertTrue(self.bot.allowed(USER))
        for username in (MISSING_USERNAME, None, "", 123, "bad/name"):
            update = message_update("/start @alice_user", username=username)
            update["message"]["from"]["first_name"] = USER_NAME
            update["message"]["from"]["last_name"] = "@" + USER_NAME
            self.bot.handle_update(update)
            with self.subTest(username=username):
                self.assertFalse(self.bot.allowed(USER))
                self.assertNotIn(USER, self.bot.users)
                self.assertIn("Доступ запрещён", self.api.messages[-1]["text"])
        self.bot.handle_update(message_update("/start", username="ALICE_USER"))
        self.join_worker()
        self.assertTrue(self.bot.allowed(USER))
        self.assertIn(USER, self.bot.users)

    def test_whitelisted_name_does_not_authorize_an_id_before_observing_its_sender(self):
        unknown = 303
        self.whitelist.write_text(f"{USER_NAME},{OTHER_NAME},charlie_user", encoding="utf-8")
        self.assertFalse(self.bot.allowed(unknown))
        self.bot.dispatch(unknown, unknown, "/start")
        self.assertNotIn(unknown, self.bot.users)
        self.send("/start", unknown)
        self.assertTrue(self.bot.allowed(unknown))
        self.assertIn(unknown, self.bot.users)

    def test_id_reports_fresh_username_or_its_absence(self):
        self.bot.handle_update(message_update("/id"))
        self.assertIn(str(USER), self.api.messages[-1]["text"])
        self.assertIn(USER_NAME, self.api.messages[-1]["text"])
        self.bot.handle_update(message_update("/id", username=MISSING_USERNAME))
        self.assertIn(str(USER), self.api.messages[-1]["text"])
        self.assertNotIn(USER_NAME, self.api.messages[-1]["text"])
        self.assertFalse(self.bot.allowed(USER))

    def test_renamed_username_in_message_revokes_chat_commands_and_pending_approval(self):
        user = self.send("/system Приватная инструкция")
        snapshot = copy.deepcopy(user.messages)
        thread, results, (message, data, pending) = self.begin_approval()
        self.bot.handle_update(message_update("/system Заменить", username="renamed_user"))
        self.bot.handle_update(callback_update(data, message["message_id"], username="renamed_user"))
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.bot.allowed(USER))
        self.assertEqual(user.messages, snapshot)
        self.assertEqual(results, [False])
        self.assertFalse(pending.allowed)

    def test_renamed_or_missing_username_on_callback_cannot_use_cached_authorization(self):
        for username in ("renamed_user", MISSING_USERNAME, None):
            with self.subTest(username=username):
                self.send("/start")
                thread, results, (message, data, pending) = self.begin_approval()
                self.bot.handle_update(callback_update(data, message["message_id"], username=username))
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(results, [False])
                self.assertFalse(pending.allowed)
                self.assertFalse(self.bot.allowed(USER))

    def test_username_transfer_to_another_id_keeps_disk_history_and_old_approval_separate(self):
        first = self.send("/system Приватная инструкция первого")
        first.messages.append({"role": "user", "content": "Приватная история первого"})
        self.bot.command(first, USER, "/save private.json")
        thread, results, (message, data, pending) = self.begin_approval()
        self.bot.handle_update(message_update("/start", OTHER, username=USER_NAME))
        self.join_worker(OTHER)
        second = self.bot.users[OTHER]
        self.assertTrue(self.bot.allowed(OTHER))
        self.assertFalse(self.bot.allowed(USER))
        thread.join(timeout=2)
        self.assertEqual(results, [False])
        self.assertTrue(pending.event.is_set())
        self.assertFalse(pending.allowed)
        self.assertNotIn("Приватная", json.dumps(second.messages, ensure_ascii=False))
        self.assertEqual(first.directory.name, str(USER))
        self.assertEqual(second.directory.name, str(OTHER))
        self.assertNotEqual(first.workspace, second.workspace)
        with self.assertRaises(FileNotFoundError):
            self.bot.command(second, OTHER, "/load private.json")
        self.bot.handle_update(callback_update(data, message["message_id"], OTHER, username=USER_NAME))
        self.assertFalse(pending.allowed)
        self.assertIn("устарела", self.api.answers[-1][1])
        self.bot.handle_update(message_update("/id", USER, username="former_alice"))
        thread.join(timeout=2)
        self.assertEqual(results, [False])
        self.assertFalse(pending.allowed)
        self.assertFalse(self.bot.allowed(USER))
        self.assertEqual(load_session(first.sessions / "private.json"), first.messages)

    def test_only_real_users_in_their_private_chat_can_dispatch(self):
        updates = [message_update("/start", chat_type="group"),
                   message_update("/start", chat_id=OTHER),
                   message_update("/start", is_bot=True),
                   message_update("/start", user_id=True),
                   message_update("/start", user_id=-1)]
        for update in updates:
            self.bot.handle_update(update)
        self.assertEqual(self.api.messages, [])
        self.assertEqual(self.bot.users, {})

    def test_malformed_nested_updates_are_ignored_without_creating_a_chat(self):
        updates = [None, [], {"message": None}, {"message": {"from": None}},
                   {"message": {"from": [], "chat": {}}},
                   {"message": {"from": {"id": USER}, "chat": None}},
                   {"callback_query": {"id": "callback", "from": None}},
                   {"callback_query": {"id": "callback", "from": {"id": USER}, "message": None}},
                   {"callback_query": {"id": "callback", "from": {"id": USER}, "message": {"chat": None}}}]
        for update in updates:
            with self.subTest(update=update):
                self.bot.handle_update(update)
        self.assertEqual(self.bot.users, {})
        self.assertEqual(self.api.messages, [])

    def test_mutable_whitelist_revokes_and_can_restore_access_without_restart(self):
        user = self.send("/start")
        initial_messages = copy.deepcopy(user.messages)
        self.whitelist.write_text(OTHER_NAME, encoding="utf-8")
        self.send("/system forbidden")
        self.assertEqual(user.messages, initial_messages)
        self.assertIn("Доступ запрещён", self.api.messages[-1]["text"])
        self.whitelist.write_text(f"{USER_NAME}, {OTHER_NAME}", encoding="utf-8")
        self.send("/system allowed")
        self.assertEqual(user.messages, [{"role": "system", "content": "allowed"}])

    def test_each_user_has_separate_disk_workspace_context_and_preferences(self):
        first = self.send("/start")
        second = self.send("/start", OTHER)
        self.assertTrue(first.workspace.is_dir())
        self.assertTrue(second.workspace.is_dir())
        self.assertNotEqual(first.workspace, second.workspace)
        self.assertEqual(first.workspace.parent.name, str(USER))
        self.assertEqual(second.workspace.parent.name, str(OTHER))
        self.send("/system Первая инструкция")
        self.send("/thinking")
        self.assertEqual(first.config["system_prompt"], "Первая инструкция")
        self.assertNotEqual(second.config["system_prompt"], "Первая инструкция")
        self.assertNotEqual(first.config["show_reasoning"], second.config["show_reasoning"])
        self.assertEqual(self.config, {**DEFAULT_CONFIG, "backend": "external"})

    def test_save_and_load_can_only_address_own_sessions(self):
        first = self.send("/start")
        second = self.send("/start", OTHER)
        first.messages += [{"role": "user", "content": "Секрет первого"},
                           {"role": "assistant", "content": "Ответ первого"}]
        second.messages += [{"role": "user", "content": "Секрет второго"}]
        self.bot.command(second, OTHER, "/save second.json")
        self.bot.command(first, USER, "/save nested/first.json")
        self.assertEqual(load_session(first.sessions / "nested" / "first.json"), first.messages)
        self.assertEqual(self.api.documents[-1][0:2], (USER, "first.json"))
        self.assertIn("Секрет первого", self.api.documents[-1][2].decode("utf-8"))
        snapshot = copy.deepcopy(first.messages)
        for path in ("../second.json", str(second.sessions / "second.json"),
                     "../../../202/.local/sessions/second.json", "C:second.json", "\\\\server\\second.json"):
            for command in ("/save ", "/load "):
                with self.subTest(command=command, path=path), self.assertRaises(ValueError):
                    self.bot.command(first, USER, command + path)
        self.assertEqual(first.messages, snapshot)
        with self.assertRaises(FileNotFoundError):
            self.bot.command(first, USER, "/load second.json")
        self.bot.command(first, USER, "/history")
        self.assertNotIn("Секрет второго", self.api.messages[-1]["text"])

    def test_clear_system_load_history_and_quit_reset_harness_settings(self):
        user = self.send("/start")
        seed = [{"role": "system", "content": "Из файла"}, {"role": "user", "content": "История"}]
        save_session(user.sessions / "seed.json", seed, "test")
        for command in ("/clear", "/system Новая инструкция", "/load seed.json", "/history 1", "/quit", "/exit"):
            with self.subTest(command=command):
                user.settings = HarnessSettings(web_enabled=True, repository=user.workspace, powershell_enabled=True)
                self.bot.command(user, USER, command)
                self.assertEqual(user.settings, HarnessSettings())
        self.bot.command(user, USER, "/load seed.json")
        self.assertEqual(user.messages, seed)
        self.assertEqual(user.config["system_prompt"], "Из файла")
        self.assertEqual(load_session(user.directory / "current.json"), seed)

    @unittest.skipUnless(os.name == "nt", "Strict repository tools require Windows handles")
    def test_harness_toggle_uses_own_workspace_and_selects_only_existing_subfolders(self):
        user = self.send("/start")
        second = self.send("/start", OTHER)
        self.bot.command(user, USER, "/tools web on")
        self.bot.command(user, USER, "/tools code on")
        self.assertTrue(user.settings.web_enabled)
        self.assertTrue(user.settings.code_enabled)
        self.assertEqual(user.settings.repository, user.workspace)
        self.assertIsNotNone(user.settings.repository_identity)
        self.assertEqual(second.settings, HarnessSettings())
        folder = user.workspace / "project"
        folder.mkdir()
        self.bot.command(user, USER, "/tools repo project")
        self.assertEqual(user.settings.repository, folder)
        for value in ("..", str(second.workspace), "missing"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.bot.command(user, USER, "/tools repo " + value)
        self.bot.command(user, USER, "/tools off")
        self.assertEqual(user.settings, HarnessSettings())

    @unittest.skipUnless(os.name == "nt", "Strict repository tools require Windows handles")
    def test_powershell_menu_is_independent_and_uses_guarded_personal_repository(self):
        user = self.send("/start")
        second = self.send("/start", OTHER)
        self.bot.command(user, USER, "/tools")
        menu = self.api.messages[-1]
        button = next(button for row in menu["reply_markup"]["inline_keyboard"] for button in row
                      if button["text"] == "Включить PowerShell")
        self.bot.handle_update(callback_update(button["callback_data"], menu["message_id"], OTHER))
        self.assertFalse(user.settings.powershell_enabled)
        self.bot.handle_update(callback_update(button["callback_data"], menu["message_id"]))
        self.join_worker()
        self.assertTrue(user.settings.powershell_enabled)
        self.assertFalse(user.settings.code_enabled)
        self.assertFalse(user.settings.web_enabled)
        self.assertEqual(user.settings.repository, user.workspace)
        self.assertIsNotNone(user.settings.repository_identity)
        self.assertEqual(second.settings, HarnessSettings())
        self.assertIn("PowerShell: только вычисления и вывод", self.api.messages[-1]["text"])
        folder = user.workspace / "scripts"
        folder.mkdir()
        self.bot.command(user, USER, "/tools repo scripts")
        self.assertEqual(user.settings.repository, folder)
        self.assertTrue(user.settings.powershell_enabled)
        self.assertFalse(user.settings.code_enabled)
        for value in ("..", str(second.workspace), "missing"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.bot.command(user, USER, "/tools repo " + value)
        self.bot.command(user, USER, "/tools powershell off")
        self.assertFalse(user.settings.powershell_enabled)
        self.assertFalse(user.settings.code_enabled)
        self.bot.command(user, USER, "/tools powershell on")
        self.bot.command(user, USER, "/tools off")
        self.assertEqual(user.settings, HarnessSettings())

    def test_powershell_approval_explains_restricted_execution(self):
        request = ToolRequest("powershell_run", {"path": "hello.ps1", "timeout_seconds": 5},
                              "Запуск hello.ps1 в выбранном репозитории.")
        thread, results, (message, data, _) = self.begin_approval(request)
        self.assertIn("hello.ps1", message["text"])
        self.assertIn("Разрешены только вычисления и вывод", message["text"])
        self.assertIn("Файловые команды, сеть и запуск других программ запрещены", message["text"])
        self.assertNotIn("Полное предлагаемое содержимое", message["text"])
        self.bot.handle_update(callback_update(data, message["message_id"]))
        thread.join(timeout=1)
        self.assertEqual(results, [True])

    def test_command_buttons_are_bound_to_owner_and_message_and_single_use(self):
        user = self.send("/start")
        self.bot.command(user, USER, "/tools")
        menu = self.api.messages[-1]
        data = menu["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
        for update in (callback_update(data, menu["message_id"], OTHER),
                       callback_update(data, menu["message_id"] + 1),
                       callback_update(data, menu["message_id"], chat_id=OTHER),
                       callback_update(data, menu["message_id"], chat_type="group")):
            self.bot.handle_update(update)
            self.assertFalse(user.settings.web_enabled)
        self.bot.handle_update(callback_update(data, menu["message_id"]))
        self.join_worker()
        self.assertTrue(user.settings.web_enabled)
        self.bot.handle_update(callback_update(data, menu["message_id"]))
        self.assertIn("устарела", self.api.answers[-1][1])
        self.assertTrue(user.settings.web_enabled)

    def test_approval_buttons_bind_user_chat_message_and_accept_exactly_once(self):
        thread, results, (message, data, pending) = self.begin_approval()
        for update in (callback_update(data, message["message_id"], OTHER),
                       callback_update(data, message["message_id"] + 1),
                       callback_update(data, message["message_id"], chat_id=OTHER),
                       callback_update(data, message["message_id"], chat_type="group"),
                       callback_update(data[:-3] + "all", message["message_id"])):
            self.bot.handle_update(update)
            self.assertFalse(pending.event.is_set())
        self.bot.handle_update(callback_update(data, message["message_id"]))
        self.bot.handle_update(callback_update(data.rsplit(":", 1)[0] + ":no", message["message_id"]))
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [True])
        self.assertTrue(pending.allowed)
        self.assertEqual(self.bot.approvals, {})
        self.assertIn((USER, message["message_id"], None), self.api.edits)

    def test_denied_approval_never_becomes_allowed_after_a_second_click(self):
        thread, results, (message, data, pending) = self.begin_approval()
        self.bot.handle_update(callback_update(data.rsplit(":", 1)[0] + ":no", message["message_id"]))
        self.bot.handle_update(callback_update(data, message["message_id"]))
        thread.join(timeout=2)
        self.assertEqual(results, [False])
        self.assertFalse(pending.allowed)

    def test_expired_approval_denies_and_removes_buttons(self):
        with patch.object(telegram, "APPROVAL_TIMEOUT", 0):
            thread, results, (message, data, pending) = self.begin_approval()
            self.bot.handle_update(callback_update(data, message["message_id"]))
            thread.join(timeout=2)
        self.assertEqual(results, [False])
        self.assertEqual(self.bot.approvals, {})
        self.assertFalse(pending.allowed)
        self.assertIn((USER, message["message_id"], None), self.api.edits)

    def test_cancel_or_revocation_while_waiting_denies_approval(self):
        for revoke in (False, True):
            with self.subTest(revoke=revoke):
                self.whitelist.write_text(f"{USER_NAME},{OTHER_NAME}", encoding="utf-8")
                user = self.bot.user_chat(USER)
                user.cancelled.clear()
                user.lock.acquire()
                try:
                    thread, results, (message, data, pending) = self.begin_approval()
                    if revoke:
                        self.whitelist.write_text(OTHER_NAME, encoding="utf-8")
                        self.bot.handle_update(callback_update(data, message["message_id"]))
                    else:
                        self.bot.dispatch(USER, USER, "/cancel")
                        self.bot.handle_update(callback_update(data, message["message_id"]))
                    thread.join(timeout=2)
                    self.assertEqual(results, [False])
                    self.assertFalse(pending.allowed)
                finally:
                    user.lock.release()

    def test_large_write_approval_uploads_complete_arguments_and_preview(self):
        preview = "print('данные')\n" * 2000
        request = ToolRequest("write_file", {"path": "main.py", "content": preview}, "Перезапись main.py", preview)
        thread, results, (message, data, _) = self.begin_approval(request)
        self.assertEqual(self.api.documents[0][0:2], (USER, "approval.txt"))
        uploaded = self.api.documents[0][2].decode("utf-8")
        self.assertIn(preview, uploaded)
        self.assertIn(json.dumps(request.arguments, ensure_ascii=False, indent=2), uploaded)
        self.assertIn("approval.txt", message["text"])
        self.bot.handle_update(callback_update(data.rsplit(":", 1)[0] + ":no", message["message_id"]))
        thread.join(timeout=2)
        self.assertEqual(results, [False])

    def test_text_turns_use_only_the_calling_users_context_and_save_on_disk(self):
        clients = []

        class AnswerClient:
            def __init__(self, config):
                self.config = config
                self.requests = []
                clients.append(self)

            def stream_chat(self, messages):
                self.requests.append(copy.deepcopy(messages))
                yield StreamEvent("reasoning", "Мысль")
                yield StreamEvent("content", "Ответ: " + messages[-1]["content"])
                yield StreamEvent("finish", "stop")

        self.send("/system Первый")
        self.send("/system Второй", OTHER)
        with patch.object(telegram, "ApiClient", AnswerClient):
            first = self.send("Секрет A")
            second = self.send("Секрет B", OTHER)
            self.send("Продолжение A")
        self.assertEqual(clients[0].requests[0][0]["content"], "Первый")
        self.assertEqual(clients[1].requests[0][0]["content"], "Второй")
        self.assertNotIn("Секрет B", json.dumps(clients[2].requests, ensure_ascii=False))
        self.assertNotIn("Секрет A", json.dumps(second.messages, ensure_ascii=False))
        self.assertEqual(load_session(first.directory / "current.json"), first.messages)
        self.assertEqual(load_session(second.directory / "current.json"), second.messages)
        restarted = telegram.TelegramBot(self.api, self.config, self.root, self.whitelist, emit=self.emitted.append)
        self.addCleanup(restarted.close)
        resumed = restarted.user_chat(USER)
        self.assertEqual(resumed.messages, first.messages)
        self.assertEqual(resumed.config["system_prompt"], "Первый")
        self.assertEqual(resumed.settings, HarnessSettings())

    def test_real_core_and_harness_wait_for_inline_approval_then_persist_tool_result(self):
        user = self.send("/start")
        user.settings = HarnessSettings(web_enabled=True)
        client = StreamingClient(
            [StreamEvent("content", "Проверю страницу."), StreamEvent("finish", "tool_calls"),
             StreamEvent("tool_call", tool_call()), StreamEvent("done", {})],
            [StreamEvent("content", "Страница прочитана."), StreamEvent("finish", "stop")],
        )
        with (patch.object(telegram, "ApiClient", return_value=client),
              patch("llmopenchat.harness._web_text", return_value={"url": "https://example.com/", "text": "Пример"}) as web):
            self.bot.handle_update(message_update("Прочитай страницу"))
            message, data, _ = self.ready_approval()
            web.assert_not_called()
            self.assertIn("Проверю страницу.", "".join(item["text"] for item in self.api.messages))
            self.bot.handle_update(callback_update(data, message["message_id"]))
            self.join_worker()
        web.assert_called_once_with("https://example.com/")
        self.assertEqual([item["role"] for item in user.messages], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(json.loads(user.messages[-2]["content"])["status"], "ok")
        self.assertEqual(load_session(user.directory / "current.json"), user.messages)
        self.assertEqual(client.requests[1][0][-1]["role"], "tool")

    def test_cancelled_tool_round_remains_complete_and_saveable(self):
        user = self.send("/start")
        user.settings = HarnessSettings(web_enabled=True)
        client = StreamingClient([StreamEvent("finish", "tool_calls"), StreamEvent("tool_call", tool_call()), StreamEvent("done", {})])
        with (patch.object(telegram, "ApiClient", return_value=client),
              patch("llmopenchat.harness._web_text") as web):
            self.bot.handle_update(message_update("Прочитай страницу"))
            self.ready_approval()
            self.bot.handle_update(message_update("/cancel"))
            self.join_worker()
        web.assert_not_called()
        self.assertEqual([item["role"] for item in user.messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(json.loads(user.messages[-1]["content"])["status"], "denied")
        self.assertEqual(load_session(user.directory / "current.json"), user.messages)
        self.assertFalse(user.lock.locked())
        self.assertIn("отказов: 1", self.api.messages[-1]["text"])
        self.assertIn("файлов записано: 0", self.api.messages[-1]["text"])

    def test_enabled_harness_reports_no_calls_from_host_events(self):
        user = self.send("/start")
        user.settings = HarnessSettings(web_enabled=True)
        client = StreamingClient([StreamEvent("content", "Обычный ответ."), StreamEvent("finish", "stop")])
        with patch.object(telegram, "ApiClient", return_value=client):
            self.send("Расскажи историю")
        reply = self.api.messages[-1]["text"]
        self.assertIn("Обычный ответ.", reply)
        self.assertIn("веб подключён", reply)
        self.assertIn("модель не вызвала инструменты; изменений на диске нет", reply)
        self.assertEqual(user.messages[-1]["content"], "Обычный ответ.")
        self.assertNotIn("Харнес", json.dumps(user.messages, ensure_ascii=False))

    def test_creation_retry_clears_discarded_buffers_before_approval(self):
        user = self.send("/start")
        user.settings = HarnessSettings(code_enabled=True, repository=user.workspace)
        user.config["show_reasoning"] = True
        source = 'print("hello")\n'
        write = tool_call("call-write", "write_file", json.dumps({"path": "main.py", "content": source}))
        client = StreamingClient(
            [StreamEvent("reasoning", "Отброшенное рассуждение."),
             StreamEvent("content", "Отброшенный ответ без записи."), StreamEvent("finish", "stop")],
            [StreamEvent("content", "Теперь запишу файл."), StreamEvent("finish", "tool_calls"),
             StreamEvent("tool_call", write), StreamEvent("done", {})],
            [StreamEvent("content", "Готово."), StreamEvent("finish", "stop")],
        )

        def execute(harness, name, arguments):
            values = json.loads(arguments)
            approved = harness.approve(ToolRequest(name, values, "Запись main.py", values["content"]))
            return json.dumps({"status": "ok" if approved else "denied", "path": values["path"],
                               "bytes_written": len(values["content"].encode("utf-8"))})

        with (patch.object(telegram, "ApiClient", return_value=client),
              patch.object(telegram.ToolHarness, "execute", autospec=True, side_effect=execute) as run,
              patch.object(self.bot, "approve", return_value=True) as approve):
            self.send("Создай файл main.py")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(approve.call_count, 1)
        self.assertEqual(approve.call_args.args[-1].preview, source)
        sent = "\n".join(item["text"] for item in self.api.messages)
        self.assertNotIn("Отброшенн", sent)
        self.assertIn("Теперь запишу файл.", sent)
        self.assertIn("файлов записано: 1", self.api.messages[-1]["text"])
        self.assertIn("Записан файл: main.py", self.api.messages[-1]["text"])
        self.assertEqual(client.tool_choices, ["required", "required", None])
        self.assertFalse((user.workspace / "main.py").exists(), "This test mocks file writes")
        self.assertEqual([message["role"] for message in user.messages],
                         ["system", "user", "assistant", "tool", "assistant"])
        self.assertFalse(any(message["role"] == "system" and "selected repository root" in message["content"]
                             for message in user.messages))

    def test_denied_file_call_reports_refusal_without_corrective_retry(self):
        user = self.send("/start")
        user.settings = HarnessSettings(code_enabled=True, repository=user.workspace)
        write = tool_call("call-write", "write_file", '{"path":"main.py","content":"print(1)"}')
        client = StreamingClient(
            [StreamEvent("finish", "tool_calls"), StreamEvent("tool_call", write), StreamEvent("done", {})],
            [StreamEvent("content", "Запись отклонена."), StreamEvent("finish", "stop")],
        )
        with (patch.object(telegram, "ApiClient", return_value=client),
              patch.object(self.bot, "approve", return_value=False) as approve):
            self.send("Создай файл main.py")
        self.assertEqual(approve.call_count, 1)
        self.assertEqual(len(client.requests), 2)
        reply = self.api.messages[-1]["text"]
        self.assertIn("отказов: 1", reply)
        self.assertIn("файлов записано: 0", reply)
        self.assertNotIn("Записан файл: main.py", reply)
        self.assertFalse((user.workspace / "main.py").exists())
        self.assertEqual(json.loads(user.messages[-2]["content"])["status"], "denied")

    def test_powershell_only_generation_requires_approval_and_reports_host_result(self):
        user = self.send("/start")
        user.settings = HarnessSettings(repository=user.workspace, powershell_enabled=True)
        run = tool_call("call-ps", "powershell_run", '{"path":"hello.ps1","timeout_seconds":5}')
        client = StreamingClient(
            [StreamEvent("finish", "tool_calls"), StreamEvent("tool_call", run), StreamEvent("done", {})],
            [StreamEvent("content", "Скрипт вывел hello."), StreamEvent("finish", "stop")],
        )

        def execute(harness, name, arguments):
            values = json.loads(arguments)
            approved = harness.approve(ToolRequest(name, values, "Запуск hello.ps1 в выбранном репозитории."))
            return json.dumps({"status": "ok" if approved else "denied", "path": values["path"],
                               "stdout": "hello\n", "stderr": "", "exit_code": 0, "restricted": True})

        with (patch.object(telegram, "ApiClient", return_value=client),
              patch.object(telegram.ToolHarness, "execute", autospec=True, side_effect=execute) as runner,
              patch.object(self.bot, "approve", return_value=True) as approve):
            self.send("Запусти hello.ps1")
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(approve.call_count, 1)
        self.assertEqual(approve.call_args.args[-1].name, "powershell_run")
        self.assertEqual([item["function"]["name"] for item in client.requests[0][1]], ["powershell_run"])
        reply = self.api.messages[-1]["text"]
        self.assertIn("PowerShell подключён", reply)
        self.assertIn("успешно: 1", reply)
        self.assertIn("файлов записано: 0", reply)
        self.assertEqual(json.loads(user.messages[-2]["content"])["stdout"], "hello\n")
        self.assertEqual(user.messages[-1]["content"], "Скрипт вывел hello.")
        self.assertFalse((user.workspace / "hello.ps1").exists(), "This test mocks script execution")

    def test_error_reply_retains_written_file_status_without_persisting_ui_state(self):
        user = self.send("/start")
        user.settings = HarnessSettings(code_enabled=True, repository=user.workspace)
        write = tool_call("call-write", "write_file", '{"path":"main.py","content":"print(1)"}')
        client = StreamingClient(
            [StreamEvent("finish", "tool_calls"), StreamEvent("tool_call", write), StreamEvent("done", {})],
            [ApiError("Сбой ответа модели")],
        )

        def execute(harness, name, arguments):
            values = json.loads(arguments)
            allowed = harness.approve(ToolRequest(name, values, "Запись main.py", values["content"]))
            return json.dumps({"status": "ok" if allowed else "denied", "path": values["path"]})

        with (patch.object(telegram, "ApiClient", return_value=client),
              patch.object(telegram.ToolHarness, "execute", autospec=True, side_effect=execute),
              patch.object(self.bot, "approve", return_value=True)):
            self.send("Создай файл main.py")
        reply = self.api.messages[-1]["text"]
        self.assertTrue(reply.startswith("Ошибка: Сбой ответа модели"))
        self.assertIn("файлов записано: 1", reply)
        self.assertIn("Записан файл: main.py", reply)
        self.assertEqual(user.messages[-1]["role"], "tool")
        self.assertEqual(load_session(user.directory / "current.json"), user.messages)
        self.assertNotIn("Харнес", json.dumps(user.messages, ensure_ascii=False))
        self.assertFalse((user.workspace / "main.py").exists(), "This test mocks file writes")
        self.bot.command(user, USER, "/clear")
        self.assertIsNone(user.harness_status)

    def test_busy_user_can_cancel_queued_request_without_touching_model(self):
        self.bot.runtime.lock.acquire()
        try:
            with patch.object(telegram, "ApiClient") as client:
                self.bot.handle_update(message_update("Вопрос"))
                self.api.wait_for(lambda item: item["text"].startswith("Запрос принят"))
                self.bot.handle_update(message_update("Второй вопрос"))
                self.assertIn("Предыдущий запрос", self.api.messages[-1]["text"])
                self.bot.handle_update(message_update("/cancel"))
                self.join_worker()
                client.assert_not_called()
            self.assertEqual(self.bot.users[USER].messages, [{"role": "system", "content": self.config["system_prompt"]}])
        finally:
            self.bot.runtime.lock.release()

    def test_model_commands_cover_catalog_install_select_remove_with_approval(self):
        user = self.send("/start")
        manager = Mock()
        manager.catalog.return_value = [SimpleNamespace(id="example", name="Example", parameters_b=3,
                                                       description="Description", default_quantization="Q4_K_M")]
        manager.installed.return_value = [{"id": "pkg", "name": "Example", "status": "ready", "active": False}]
        manager.inspect.return_value = {"name": "Example", "total_size": 2**30}
        manager.install.return_value = {"id": "pkg"}
        with (patch.object(telegram, "ModelManager", return_value=manager),
              patch.object(self.bot, "approve", return_value=False) as approval,
              patch.object(self.bot.runtime, "select") as select):
            for command in ("/models", "/models catalog", "/models list"):
                self.bot.command(user, USER, command)
            approval.assert_not_called()
            for command in ("/models install example", "/models use pkg", "/models remove pkg"):
                self.bot.command(user, USER, command)
            manager.install.assert_not_called()
            manager.remove.assert_not_called()
            select.assert_not_called()
            approval.return_value = True
            self.bot.command(user, USER, "/models install example Q5_K_M")
            self.bot.command(user, USER, "/models use pkg")
            self.bot.command(user, USER, "/models remove pkg")
        manager.install.assert_called_once()
        self.assertEqual(manager.install.call_args.args, ("example", "Q5_K_M"))
        self.assertTrue(callable(manager.install.call_args.kwargs["emit"]))
        select.assert_called_once()
        self.assertEqual(select.call_args.args, (manager, "pkg"))
        self.assertTrue(callable(select.call_args.kwargs["cancelled"]))
        manager.remove.assert_called_once_with("pkg", self.bot.runtime.config)

    def test_model_change_waiting_for_other_inference_stops_on_cancel_or_revocation(self):
        user = self.send("/start")
        self.bot.runtime.lock.acquire()
        try:
            for command in ("/models use pkg", "/models remove pkg"):
                for revoke in (False, True):
                    with self.subTest(command=command, revoke=revoke):
                        self.whitelist.write_text(f"{USER_NAME},{OTHER_NAME}", encoding="utf-8")
                        approved = threading.Event()
                        manager = Mock()

                        def approve(*args):
                            approved.set()
                            return True

                        with (patch.object(telegram, "ModelManager", return_value=manager),
                              patch.object(self.bot, "approve", side_effect=approve),
                              patch.object(self.bot.runtime, "select") as select):
                            self.bot.handle_update(message_update(command))
                            self.assertTrue(approved.wait(timeout=2))
                            if revoke:
                                self.whitelist.write_text(OTHER_NAME, encoding="utf-8")
                            else:
                                self.bot.handle_update(message_update("/cancel"))
                            self.join_worker()
                            select.assert_not_called()
                            manager.remove.assert_not_called()
                        self.assertFalse(user.lock.locked())
        finally:
            self.whitelist.write_text(f"{USER_NAME},{OTHER_NAME}", encoding="utf-8")
            self.bot.runtime.lock.release()

    def test_model_download_progress_checks_cancelled_user(self):
        user = self.send("/start")
        manager = Mock()
        manager.inspect.return_value = {"name": "Example", "total_size": 2**30}

        def install(model_id, quant, *, emit):
            user.cancelled.set()
            emit("Downloading")
            self.fail("Cancelled installation continued after its progress callback")

        manager.install.side_effect = install
        with (patch.object(telegram, "ModelManager", return_value=manager),
              patch.object(self.bot, "approve", return_value=True)):
            self.send("/models install example")
        manager.install.assert_called_once()
        self.assertIn("Генерация отменена", self.api.messages[-1]["text"])
        self.assertFalse(user.lock.locked())

    def test_stop_denies_pending_approval_and_prevents_new_work(self):
        thread, results, _ = self.begin_approval()
        self.bot.close()
        thread.join(timeout=2)
        self.assertEqual(results, [False])
        count = len(self.api.messages)
        self.bot.dispatch(OTHER, OTHER, "/start")
        self.assertEqual(len(self.api.messages), count)
        self.assertNotIn(OTHER, self.bot.users)

    def test_poll_offsets_advance_for_valid_ids_and_terminal_errors_propagate(self):
        api = Mock()
        updates = [message_update("/help"), {"update_id": True}, {"update_id": "7"},
                   {"update_id": 7, **message_update("/help")}]
        api.get_updates.side_effect = [updates, [], TelegramError("Unauthorized", error_code=401)]
        self.bot.api = api
        with patch.object(self.bot, "handle_update") as handle:
            with self.assertRaises(TelegramError) as raised:
                self.bot.poll()
        self.assertEqual(raised.exception.error_code, 401)
        handle.assert_called_once_with(updates[-1])
        self.assertEqual([call.kwargs for call in api.get_updates.call_args_list],
                         [{"offset": None, "timeout": 25}, {"offset": 8, "timeout": 25},
                          {"offset": 8, "timeout": 25}])

    def test_poll_reconnect_backoff_is_bounded_and_resets_after_success(self):
        api = Mock()
        api.get_updates.side_effect = [TelegramError("offline") for _ in range(6)] + [[], TelegramError("offline"),
                                                                                  TelegramError("Conflict", error_code=409)]
        self.bot.api = api
        with patch.object(self.bot.stopped, "wait", return_value=False) as wait:
            with self.assertRaises(TelegramError) as raised:
                self.bot.poll()
        self.assertEqual(raised.exception.error_code, 409)
        self.assertEqual([call.args[0] for call in wait.call_args_list], [2, 4, 8, 16, 30, 30, 2])


class TelegramStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmopenchat-telegram-startup-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.token = "12345:" + "dummy_test_token_" * 3
        (self.root / "credits.txt").write_text(self.token, encoding="utf-8")
        (self.root / "whitelist.txt").write_text(f"@{USER_NAME},{OTHER_NAME}", encoding="utf-8")
        self.emitted = []

    def test_startup_reads_credentials_registers_commands_and_cleans_up_on_interrupt(self):
        runtime = Mock()
        api = Mock()
        api.get_me.return_value = {"username": "test_bot"}
        bot = Mock()
        bot.poll.side_effect = KeyboardInterrupt
        with (patch.object(telegram, "BotRuntime", return_value=runtime),
              patch.object(telegram, "TelegramAPI", return_value=api) as api_type,
              patch.object(telegram, "TelegramBot", return_value=bot) as bot_type):
            self.assertEqual(telegram.run_telegram(self.config, self.root, emit=self.emitted.append), 0)
        api_type.assert_called_once_with(self.token)
        api.set_commands.assert_called_once_with(telegram.COMMANDS)
        self.assertEqual(bot.username, "test_bot")
        self.assertEqual(bot_type.call_args.args[3], self.root / "whitelist.txt")
        runtime.start.assert_called_once()
        runtime.shutdown.assert_called_once()
        bot.close.assert_called_once()
        self.assertNotIn(self.token, "\n".join(self.emitted))

    def test_missing_or_invalid_credentials_and_whitelist_fail_before_api_or_server_start(self):
        cases = [("credits.txt", None), ("credits.txt", "bad token"), ("whitelist.txt", "@@unknown")]
        for filename, content in cases:
            with self.subTest(filename=filename, content=content):
                (self.root / "credits.txt").write_text(self.token, encoding="utf-8")
                (self.root / "whitelist.txt").write_text(USER_NAME, encoding="utf-8")
                if content is None:
                    (self.root / filename).unlink()
                else:
                    (self.root / filename).write_text(content, encoding="utf-8")
                runtime = Mock()
                with (patch.object(telegram, "BotRuntime", return_value=runtime),
                      patch.object(telegram, "TelegramAPI") as api_type):
                    self.assertEqual(telegram.run_telegram(self.config, self.root, emit=self.emitted.append), 1)
                api_type.assert_not_called()
                runtime.start.assert_not_called()
                runtime.shutdown.assert_called_once()

    def test_relative_credential_paths_resolve_against_client_root(self):
        directory = self.root / "settings"
        directory.mkdir()
        (directory / "token.txt").write_text(self.token, encoding="utf-8")
        (directory / "users.txt").write_text(USER_NAME, encoding="utf-8")
        runtime = Mock()
        api = Mock()
        api.get_me.return_value = {"username": "test_bot"}
        bot = Mock()
        with (patch.object(telegram, "BotRuntime", return_value=runtime),
              patch.object(telegram, "TelegramAPI", return_value=api) as api_type,
              patch.object(telegram, "TelegramBot", return_value=bot) as bot_type):
            self.assertEqual(telegram.run_telegram(self.config, self.root,
                                                  whitelist_path=Path("settings/users.txt"),
                                                  credits_path=Path("settings/token.txt"),
                                                  emit=self.emitted.append), 0)
        api_type.assert_called_once_with(self.token)
        self.assertEqual(bot_type.call_args.args[3], directory / "users.txt")
        bot.close.assert_called_once()
        runtime.shutdown.assert_called_once()


class BotRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.runtime = telegram.BotRuntime(self.config, Path("."), lambda text: None)

    def test_select_restores_previous_model_and_server_when_new_server_fails(self):
        previous_server = Mock()
        self.runtime.server = previous_server
        manager = Mock()
        selected = {**self.config, "model": "new-model", "target_model": "New Model"}
        manager.activate.return_value = selected
        failed = Mock()
        failed.__enter__ = Mock(side_effect=RuntimeErrorDetail("failed to load"))
        restored = Mock()
        restored.__enter__ = Mock(return_value=restored)
        with patch.object(telegram, "ManagedServer", side_effect=[failed, restored]) as server_type:
            with self.assertRaisesRegex(RuntimeErrorDetail, "failed to load"):
                self.runtime.select(manager, "package")
        previous_server.stop.assert_called_once()
        self.assertEqual(self.runtime.config, self.config)
        self.assertIs(self.runtime.server, restored)
        self.assertEqual(server_type.call_args_list[0].args[0], selected)
        self.assertEqual(server_type.call_args_list[1].args[0], self.config)
        restored.__enter__.assert_called_once()

    def test_shutdown_does_not_wait_for_worker_lock_and_rejects_later_start_or_select(self):
        server = Mock()
        self.runtime.server = server
        self.runtime.lock.acquire()
        shutdown = threading.Thread(target=self.runtime.shutdown, daemon=True)
        try:
            shutdown.start()
            shutdown.join(timeout=1)
            self.assertFalse(shutdown.is_alive())
        finally:
            self.runtime.lock.release()
            shutdown.join(timeout=1)
        self.assertTrue(self.runtime.closed.is_set())
        server.stop.assert_called_once()
        manager = Mock()
        with patch.object(telegram, "ManagedServer") as server_type:
            with self.assertRaises(RuntimeErrorDetail):
                self.runtime.start()
            with self.assertRaises(RuntimeErrorDetail):
                self.runtime.select(manager, "package")
        server_type.assert_not_called()
        manager.activate.assert_not_called()

    def test_shutdown_during_server_start_stops_the_new_server(self):
        server = Mock()
        server.__enter__ = Mock(side_effect=lambda: self.runtime.closed.set())
        with patch.object(telegram, "ManagedServer", return_value=server):
            with self.assertRaises(RuntimeErrorDetail):
                self.runtime.start()
        server.stop.assert_called_once()
        self.assertIsNone(self.runtime.server)


if __name__ == "__main__":
    unittest.main()
