"""Telegram RAG ownership, explicit imports, uploads and generation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llmopenchat.api import StreamEvent
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session
from llmopenchat.rag import MAX_FILE_BYTES
from llmopenchat.telegram import TelegramBot
from llmopenchat.telegram_api import TelegramError


def update(user_id, text=None, *, document=None, username=None):
    message = {"from": {"id": user_id, "username": username or {101: "alice", 202: "bob"}[user_id]},
               "chat": {"id": user_id, "type": "private"}}
    if text is not None:
        message["text"] = text
    if document is not None:
        message["document"] = document
    return {"message": message}


class FakeTelegram:
    def __init__(self):
        self.messages = []
        self.downloads = []
        self.content = b"Nebula token is cobalt-71. This is the trusted project token."

    def send_message(self, chat_id, text, reply_markup=None):
        message = {"chat_id": chat_id, "text": text, "reply_markup": reply_markup,
                   "message_id": len(self.messages) + 1}
        self.messages.append(message)
        return message

    def send_document(self, chat_id, filename, content, caption=""):
        return {"message_id": 1}

    def download_document(self, file_id, *, max_bytes):
        self.downloads.append((file_id, max_bytes))
        return self.content


class RecordingClient:
    def __init__(self):
        self.requests = []

    def stream_chat(self, messages, tools=None, tool_choice=None):
        self.requests.append(copy.deepcopy(messages))
        yield StreamEvent("content", "The Nebula token is cobalt-71 [1].")
        yield StreamEvent("finish", "stop")

    def cancel(self):
        pass


class TelegramRagTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-rag-telegram-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        whitelist = self.root / "whitelist.txt"
        whitelist.write_text("alice,bob", encoding="utf-8")
        self.api = FakeTelegram()
        self.bot = TelegramBot(self.api, copy.deepcopy(DEFAULT_CONFIG), self.root, whitelist,
                               emit=lambda value: None)
        self.addCleanup(self.bot.close)
        self.bot.handle_update(update(101, "/id"))
        self.bot.handle_update(update(202, "/id"))
        self.api.messages.clear()

    def send(self, text=None, user_id=101, *, document=None, username=None):
        self.bot.handle_update(update(user_id, text, document=document, username=username))
        user = self.bot.users.get(user_id)
        if user is not None and user.worker is not None:
            user.worker.join(3)
            self.assertFalse(user.worker.is_alive())
        return user

    def add(self, user_id=101, name="knowledge.txt", text="Nebula token is cobalt-71."):
        user = self.bot.user_chat(user_id)
        (user.workspace / name).write_text(text, encoding="utf-8")
        self.send("/rag add " + name, user_id)
        return user

    def test_users_have_separate_knowledge_bases_and_enabled_state(self):
        alice = self.add(name="alice.txt", text="Nebula token is cobalt-71.")
        bob = self.add(202, "bob.txt", "Nebula token is amber-42.")
        (self.root / "console-secret.txt").write_text("Nebula token is violet-93.", encoding="utf-8")
        self.send("/rag on")
        self.assertTrue(alice.rag.enabled)
        self.assertFalse(bob.rag.enabled)
        alice_list = alice.rag.command("list")
        self.assertIn("alice.txt", alice_list)
        self.assertNotIn("bob.txt", alice_list)
        self.assertNotIn("console-secret", alice_list)
        self.assertIn("bob.txt", bob.rag.command("list"))
        self.assertNotIn("amber-42", alice.rag.command("search Nebula token"))
        self.assertNotIn("cobalt-71", bob.rag.command("search Nebula token"))

    def test_add_rejects_paths_outside_workspace(self):
        alice = self.add()
        other = self.add(202, "secret.txt")
        for value in ("../current.json", str(other.workspace / "secret.txt"),
                      str(self.root / "whitelist.txt"), "C:\\Windows\\win.ini"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.bot.command(alice, 101, "/rag add " + value)
        self.assertNotIn("secret.txt", alice.rag.command("list"))

    def test_add_rejects_symlink_to_other_users_document(self):
        alice = self.bot.user_chat(101)
        bob = self.add(202, "secret.txt")
        link = alice.workspace / "linked.txt"
        try:
            link.symlink_to(bob.workspace / "secret.txt")
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        with self.assertRaises(ValueError):
            self.bot.command(alice, 101, "/rag add linked.txt")

    def test_add_rejects_hardlink_to_other_users_document(self):
        alice = self.bot.user_chat(101)
        bob = self.add(202, "secret.txt")
        link = alice.workspace / "linked.txt"
        try:
            link.hardlink_to(bob.workspace / "secret.txt")
        except OSError as error:
            self.skipTest(f"Hardlink creation unavailable: {error}")
        with self.assertRaises(ValueError):
            self.bot.command(alice, 101, "/rag add linked.txt")

    def test_menu_help_and_status_expose_rag(self):
        self.send("/menu")
        labels = [button["text"] for row in self.api.messages[-1]["reply_markup"]["inline_keyboard"]
                  for button in row]
        self.assertIn("База знаний RAG", labels)
        self.send("/help")
        self.assertIn("/rag", self.api.messages[-1]["text"])
        user = self.send("/rag")
        self.assertIn(str(user.workspace), self.api.messages[-1]["text"])
        self.send("/status")
        self.assertIn("RAG", self.api.messages[-1]["text"])

    def test_chat_reset_and_load_disable_rag_but_keep_documents(self):
        user = self.add()
        for command in ("/clear", "/system Reply in Russian", "/load saved.json"):
            if command.startswith("/load"):
                self.send("/save saved.json")
            self.send("/rag on")
            self.send(command)
            self.assertFalse(user.rag.enabled)
            self.assertIn("knowledge.txt", user.rag.command("list"))

    def test_enabled_generation_gets_sources_without_saving_retrieval_prompt(self):
        user = self.add(text="Nebula token is cobalt-71. This is a project secret.")
        self.send("/rag on")
        client = RecordingClient()
        with patch("llmopenchat.telegram.ApiClient", return_value=client):
            self.send("What is the Nebula token?")
        self.assertEqual(len(client.requests), 1)
        request = json.dumps(client.requests[0], ensure_ascii=False)
        self.assertIn("This is a project secret.", request)
        history = load_session(user.directory / "current.json")
        self.assertNotIn("This is a project secret.", json.dumps(history, ensure_ascii=False))
        self.assertIn("knowledge.txt", history[-1]["content"])
        self.assertIn("knowledge.txt", self.api.messages[-1]["text"])
        self.assertTrue(any("RAG: найдено фрагментов:" in item["text"] for item in self.api.messages))

    def test_disabled_generation_keeps_original_chat_prompt(self):
        self.add()
        client = RecordingClient()
        with patch("llmopenchat.telegram.ApiClient", return_value=client):
            self.send("What is the Nebula token?")
        self.assertNotIn("cobalt-71", json.dumps(client.requests[0], ensure_ascii=False))
        self.assertEqual(client.requests[0][-1]["content"], "What is the Nebula token?")

    def test_enabled_generation_without_matches_does_not_ask_model_to_guess(self):
        user = self.add()
        self.send("/rag on")
        client = RecordingClient()
        with patch("llmopenchat.telegram.ApiClient", return_value=client):
            self.send("unobtainium987")
        self.assertFalse(client.requests)
        self.assertEqual(user.messages[-2]["content"], "unobtainium987")
        self.assertTrue(any("не найдено" in item["text"] for item in self.api.messages))

    def test_upload_imports_into_only_sender_workspace_without_enabling_rag(self):
        document = {"file_id": "telegram-file", "file_name": "notes.txt", "file_size": 70}
        user = self.send(document=document)
        self.assertFalse(user.rag.enabled)
        self.assertIn("notes-", user.rag.command("list"))
        uploaded = list((user.workspace / "documents").glob("*.txt"))
        self.assertEqual(len(uploaded), 1)
        self.assertEqual(uploaded[0].read_bytes(), self.api.content)
        self.assertNotIn("notes-", self.bot.user_chat(202).rag.command("list"))
        self.assertEqual(self.api.downloads, [("telegram-file", MAX_FILE_BYTES)])
        self.assertIn("/rag on", self.api.messages[-1]["text"])
        self.send(document=document)
        self.assertEqual(len(list((user.workspace / "documents").glob("*.txt"))), 2)

    def test_upload_rejects_paths_unsupported_formats_and_size_before_downloading(self):
        base = {"file_id": "telegram-file", "file_name": "notes.txt", "file_size": 70}
        invalid = [{**base, "file_name": value} for value in
                   ("../notes.txt", "nested/notes.txt", "C:\\notes.txt", "notes.exe", "con.txt")]
        invalid += [{**base, "file_size": MAX_FILE_BYTES + 1}, {**base, "file_size": True},
                    {**base, "file_id": ""}]
        for document in invalid:
            with self.subTest(document=document):
                self.send(document=document)
                self.assertIn("Ошибка", self.api.messages[-1]["text"])
        self.assertFalse(self.api.downloads)

    def test_upload_unauthorized_sender_does_not_download_or_create_user(self):
        self.send(document={"file_id": "secret", "file_name": "notes.txt"}, username="outsider")
        self.assertFalse(self.api.downloads)
        self.assertNotIn(101, self.bot.users)

    def test_upload_respects_active_user_lock(self):
        user = self.bot.user_chat(101)
        user.lock.acquire()
        try:
            self.send(document={"file_id": "secret", "file_name": "notes.txt"})
        finally:
            user.lock.release()
        self.assertFalse(self.api.downloads)
        self.assertIn("Предыдущий запрос", self.api.messages[-1]["text"])

    def test_upload_rechecks_access_after_download(self):
        user = self.bot.user_chat(101)
        def download(file_id, *, max_bytes):
            self.bot.usernames[101] = None
            return self.api.content
        with patch.object(self.api, "download_document", side_effect=download):
            self.send(document={"file_id": "secret", "file_name": "notes.txt"})
        self.assertFalse(list((user.workspace / "documents").glob("*.txt")))
        self.assertNotIn("notes-", user.rag.command("list"))

    def test_upload_failed_import_removes_unindexed_copy(self):
        user = self.bot.user_chat(101)
        with patch.object(user.rag, "command", side_effect=ValueError("Invalid document")):
            self.send(document={"file_id": "secret", "file_name": "notes.txt"})
        self.assertFalse(list((user.workspace / "documents").glob("*.txt")))
        self.assertIn("Ошибка", self.api.messages[-1]["text"])

    def test_upload_download_failure_is_reported_without_transport_details(self):
        with patch.object(self.api, "download_document", side_effect=TelegramError("private transport detail")):
            user = self.send(document={"file_id": "secret", "file_name": "notes.txt"})
        self.assertIn("Не удалось скачать документ", self.api.messages[-1]["text"])
        self.assertNotIn("private transport detail", self.api.messages[-1]["text"])
        self.assertFalse(list((user.workspace / "documents").glob("*.txt")))


if __name__ == "__main__":
    unittest.main()
