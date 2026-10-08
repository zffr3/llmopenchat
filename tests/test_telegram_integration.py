"""Local HTTP/SSE through Telegram callbacks into the real per-user harness."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session
from llmopenchat.telegram import BotRuntime, TelegramBot


class FakeTelegramAPI:
    """No Telegram network access; callbacks are delivered through handle_update."""

    def __init__(self):
        self.condition = threading.Condition()
        self.messages = []
        self.documents = []
        self.acknowledgements = []
        self.edits = []
        self.next_message_id = 1

    def send_message(self, chat_id, text, reply_markup=None):
        with self.condition:
            message = {
                "message_id": self.next_message_id,
                "chat_id": chat_id,
                "text": text,
                "reply_markup": copy.deepcopy(reply_markup),
            }
            self.next_message_id += 1
            self.messages.append(message)
            self.condition.notify_all()
            return copy.deepcopy(message)

    def send_document(self, chat_id, filename, content, caption=""):
        with self.condition:
            document = {
                "message_id": self.next_message_id,
                "chat_id": chat_id,
                "filename": filename,
                "content": content,
                "caption": caption,
            }
            self.next_message_id += 1
            self.documents.append(document)
            self.condition.notify_all()
            return {"message_id": document["message_id"]}

    def answer_callback_query(self, callback_query_id, text="", show_alert=False):
        with self.condition:
            self.acknowledgements.append((callback_query_id, text, show_alert))
            self.condition.notify_all()
        return True

    def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
        with self.condition:
            self.edits.append((chat_id, message_id, reply_markup))
            self.condition.notify_all()
        return {"message_id": message_id}


@unittest.skipUnless(os.name == "nt", "Strict repository tools require Windows handles")
class TelegramIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmopenchat-telegram-http-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.whitelist = self.root / "whitelist.txt"
        self.usernames = {11: "alice_user", 22: "bob_user"}
        self.whitelist.write_text("alice_user,bob_user", encoding="utf-8")
        self.requests = []
        self.request_lock = threading.Lock()
        self.large_source = "Привет 🦊\n" * 3000
        self.slow_started = threading.Event()
        self.release_slow = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with owner.request_lock:
                    owner.requests.append(payload)
                    request_number = len(owner.requests)
                prompt = next(message["content"] for message in reversed(payload["messages"]) if message["role"] == "user")
                tool_name = None
                if payload["messages"][-1]["role"] == "user":
                    if prompt.startswith("write ") or prompt in {"deny", "cancel", "large"}:
                        tool_name = "write_file"
                        source = owner.large_source if prompt == "large" else f"Содержимое: {prompt}\n"
                        arguments = {"path": "main.py", "content": source}
                    elif prompt == "read":
                        tool_name, arguments = "read_file", {"path": "main.py"}
                    elif prompt == "escape":
                        tool_name, arguments = "read_file", {"path": "../../22/workspace/secret.txt"}
                if tool_name is not None:
                    arguments = json.dumps(arguments, ensure_ascii=False)
                    chunks = [
                        {"choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": 0, "id": f"call_http_{request_number}", "type": "function",
                             "function": {"name": tool_name, "arguments": arguments[:19]}}
                        ]}}]},
                        {"choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": 0, "function": {"arguments": arguments[19:]}}
                        ]}}]},
                        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                    ]
                else:
                    status = "plain"
                    if payload["messages"][-1]["role"] == "tool":
                        status = json.loads(payload["messages"][-1]["content"])["status"]
                    chunks = [{"choices": [{"index": 0, "delta": {"content": f"Ответ {status}: {prompt}"}, "finish_reason": "stop"}]}]
                body = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
                body += "data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                if prompt == "slow":
                    self.wfile.flush()
                    owner.slow_started.set()
                    owner.release_slow.wait(timeout=10)
                try:
                    encoded = body.encode("utf-8")
                    # Break UTF-8 characters as well as tool arguments across writes.
                    for start in range(0, len(encoded), 31):
                        self.wfile.write(encoded[start:start + 31])
                    self.wfile.flush()
                except ConnectionError:
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config.update(backend="external", base_url=f"http://127.0.0.1:{self.server.server_port}/v1", request_timeout=5)
        self.api = FakeTelegramAPI()
        self.bot = self.new_bot()
        self.environment = patch.dict(os.environ, {"LLMOPENCHAT_API_KEY": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def stop_server(self):
        self.release_slow.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def new_bot(self):
        runtime = BotRuntime(self.config, self.root, emit=lambda text: None)
        bot = TelegramBot(self.api, self.config, self.root, self.whitelist, runtime=runtime, emit=lambda text: None)
        self.addCleanup(bot.close)
        return bot

    def send(self, user_id, text, *, wait=True, bot=None):
        bot = bot or self.bot
        bot.handle_update({"message": {"from": self.sender(user_id), "chat": {"id": user_id, "type": "private"}, "text": text}})
        if wait:
            self.join_worker(bot, user_id)

    def sender(self, user_id):
        sender = {"id": user_id, "is_bot": False}
        username = self.usernames.get(user_id)
        if username is not None:
            sender["username"] = username
        return sender

    def join_worker(self, bot, user_id):
        worker = bot.users[user_id].worker
        self.assertIsNotNone(worker)
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), f"Worker {user_id} did not complete")
        self.assertFalse(bot.users[user_id].lock.locked())

    def wait_stream_open(self, user_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with self.bot.control_lock:
                client = self.bot.clients.get(user_id)
            if client is not None:
                with client._response_lock:
                    if client._response is not None:
                        return
            with self.api.condition:
                self.api.condition.wait(timeout=0.01)
        self.fail("No active HTTP response after SSE headers")

    def approval(self, user_id, *, bot=None):
        bot = bot or self.bot
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with bot.control_lock:
                pending = next((pending for pending in bot.approvals.values() if pending.user_id == user_id and pending.message_id is not None), None)
                message_id = pending.message_id if pending else None
            if message_id is not None:
                with self.api.condition:
                    return copy.deepcopy(next(message for message in self.api.messages if message["message_id"] == message_id))
            with self.api.condition:
                self.api.condition.wait(timeout=0.02)
        self.fail(f"No ready approval message for {user_id}")

    def callback(self, user_id, message, *, allow=True, message_id=None, bot=None):
        bot = bot or self.bot
        buttons = message["reply_markup"]["inline_keyboard"][0]
        data = buttons[0 if allow else 1]["callback_data"]
        callback_id = f"callback-{user_id}-{len(self.api.acknowledgements)}"
        bot.handle_update({"callback_query": {
            "id": callback_id, "from": self.sender(user_id), "data": data,
            "message": {"message_id": message["message_id"] if message_id is None else message_id,
                        "chat": {"id": user_id, "type": "private"}},
        }})
        return self.api.acknowledgements[-1][1]

    def tool_result(self, index=-1):
        return json.loads(self.requests[index]["messages"][-1]["content"])

    def test_approved_write_real_read_and_context_restore(self):
        self.send(11, "/tools code on")
        self.send(11, "write alpha", wait=False)
        approval = self.approval(11)
        self.assertIn("Содержимое: write alpha", approval["text"])
        self.assertIn(str(self.bot.users[11].workspace), approval["text"])
        self.assertIn("разрешён", self.callback(11, approval))
        self.join_worker(self.bot, 11)
        user = self.bot.users[11]
        self.assertEqual((user.workspace / "main.py").read_text(encoding="utf-8"), "Содержимое: write alpha\n")
        self.assertEqual(self.tool_result()["status"], "ok")
        self.assertEqual([message["role"] for message in load_session(user.directory / "current.json")], ["system", "user", "assistant", "tool", "assistant"])
        self.assertIn((11, approval["message_id"], None), self.api.edits)

        self.bot.close()
        restored = self.new_bot()
        restored_user = restored.user_chat(11)
        self.assertEqual(restored_user.messages, user.messages)
        self.assertFalse(restored_user.settings.code_enabled)
        self.assertFalse(restored_user.settings.web_enabled)
        self.send(11, "/tools code on", bot=restored)
        self.send(11, "read", wait=False, bot=restored)
        self.callback(11, self.approval(11, bot=restored), bot=restored)
        self.join_worker(restored, 11)
        self.assertEqual(self.tool_result()["content"], "Содержимое: write alpha\n")
        self.assertEqual(len(restored_user.messages), 9)
        outbound = list(self.requests[-2]["messages"])
        runtime_instruction = outbound.pop(1)
        self.assertEqual(runtime_instruction["role"], "system")
        self.assertIn(json.dumps(str(restored_user.workspace), ensure_ascii=False), runtime_instruction["content"])
        self.assertEqual(outbound[:len(user.messages)], user.messages)
        self.assertNotIn(runtime_instruction, restored_user.messages)

    def test_two_users_keep_independent_context_settings_and_files(self):
        self.send(11, "/system Только пользователь alpha")
        self.send(22, "/system Только пользователь beta")
        self.send(11, "/tools code on")
        self.assertFalse(self.bot.users[22].settings.code_enabled)
        self.send(22, "/tools code on")
        self.send(11, "write alpha", wait=False)
        self.send(22, "write beta", wait=False)
        self.callback(11, self.approval(11))
        self.callback(22, self.approval(22))
        self.join_worker(self.bot, 11)
        self.join_worker(self.bot, 22)
        for user_id, name in ((11, "alpha"), (22, "beta")):
            user = self.bot.users[user_id]
            self.assertEqual((user.workspace / "main.py").read_text(encoding="utf-8"), f"Содержимое: write {name}\n")
            self.assertEqual(user.messages[0]["content"], f"Только пользователь {name}")
            self.assertEqual([message["content"] for message in user.messages if message["role"] == "user"], [f"write {name}"])
        self.assertNotEqual(self.bot.users[11].workspace, self.bot.users[22].workspace)
        self.assertEqual(len(self.requests), 4)
        for request in self.requests:
            names = {tool["function"]["name"] for tool in request["tools"]}
            self.assertIn("write_file", names)
            self.assertNotIn("web_fetch", names)
            self.assertFalse(request["parallel_tool_calls"])

    def test_deny_callback_preserves_complete_tool_round_without_writing(self):
        self.send(11, "/tools code on")
        self.send(11, "deny", wait=False)
        message = self.approval(11)
        self.assertIn("отклонён", self.callback(11, message, allow=False))
        self.join_worker(self.bot, 11)
        user = self.bot.users[11]
        self.assertFalse((user.workspace / "main.py").exists())
        self.assertEqual(self.tool_result()["status"], "denied")
        self.assertEqual([message["role"] for message in load_session(user.directory / "current.json")], ["system", "user", "assistant", "tool", "assistant"])

    def test_other_user_wrong_message_and_replayed_approval_never_authorize(self):
        self.send(11, "/tools code on")
        self.send(11, "write protected", wait=False)
        message = self.approval(11)
        self.assertIn("другому пользователю", self.callback(22, message))
        self.assertIn("устарела", self.callback(11, message, message_id=message["message_id"] + 1))
        self.assertFalse((self.bot.users[11].workspace / "main.py").exists())
        self.assertFalse(next(iter(self.bot.approvals.values())).event.is_set())
        self.callback(11, message)
        self.join_worker(self.bot, 11)
        self.assertIn("устарела", self.callback(11, message))
        self.assertEqual(self.tool_result()["status"], "ok")

    def test_cancel_pending_approval_saves_valid_round_and_next_prompt_works(self):
        self.send(11, "/tools code on")
        self.send(11, "cancel", wait=False)
        self.approval(11)
        self.send(11, "/cancel", wait=False)
        self.join_worker(self.bot, 11)
        user = self.bot.users[11]
        self.assertFalse((user.workspace / "main.py").exists())
        messages = load_session(user.directory / "current.json")
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(json.loads(messages[-1]["content"])["status"], "denied")
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(self.bot.approvals)
        self.send(11, "plain after cancellation")
        self.assertFalse(user.cancelled.is_set())
        self.assertEqual(user.messages[-1]["content"], "Ответ plain: plain after cancellation")
        outbound = list(self.requests[-1]["messages"])
        runtime_instruction = outbound.pop(1)
        self.assertEqual(runtime_instruction["role"], "system")
        self.assertEqual(outbound[:len(messages)], messages)
        self.assertNotIn(runtime_instruction, user.messages)

    def test_cancellation_interrupts_silent_sse_without_waiting_for_next_event(self):
        self.bot.runtime.config["request_timeout"] = 30
        self.send(11, "slow", wait=False)
        self.assertTrue(self.slow_started.wait(timeout=3))
        self.wait_stream_open(11)
        self.send(11, "/cancel", wait=False)
        self.assertTrue(self.bot.users[11].cancelled.is_set())
        self.assertTrue(any(message["text"] == "Запрос отменяется." for message in self.api.messages))
        self.bot.users[11].worker.join(timeout=1)
        self.assertFalse(self.bot.users[11].worker.is_alive(), "Cancellation did not wake silent SSE read")
        self.assertFalse(self.release_slow.is_set())
        self.assertFalse(any(message["role"] == "assistant" for message in self.bot.users[11].messages))
        self.send(11, "plain after stream cancellation")
        self.assertEqual(self.bot.users[11].messages[-1]["content"], "Ответ plain: plain after stream cancellation")

    def test_shutdown_interrupts_silent_sse_and_does_not_wait_for_request_timeout(self):
        self.bot.runtime.config["request_timeout"] = 30
        self.send(11, "slow", wait=False)
        self.assertTrue(self.slow_started.wait(timeout=3))
        self.wait_stream_open(11)
        start = time.monotonic()
        self.bot.close()
        self.bot.runtime.shutdown()
        self.assertLess(time.monotonic() - start, 2.5)
        self.assertFalse(self.bot.users[11].worker.is_alive())
        self.assertFalse(self.bot.users[11].lock.locked())
        self.assertFalse(self.release_slow.is_set())
        self.assertTrue(self.bot.stopped.is_set())
        self.bot.dispatch(11, 11, "plain after shutdown")
        self.assertEqual(len(self.requests), 1)

    def test_quit_during_approval_automatically_resets_context_and_harness(self):
        self.send(11, "/tools code on")
        self.send(11, "write cancelled-by-quit", wait=False)
        self.approval(11)
        self.send(11, "/quit", wait=False)
        self.join_worker(self.bot, 11)
        user = self.bot.users[11]
        self.assertFalse(user.close_requested.is_set())
        self.assertFalse(user.settings.code_enabled)
        self.assertFalse(user.settings.web_enabled)
        self.assertFalse((user.workspace / "main.py").exists())
        self.assertEqual(user.messages, [{"role": "system", "content": self.config["system_prompt"]}])
        self.assertEqual(load_session(user.directory / "current.json"), user.messages)
        self.send(11, "/start")
        self.assertFalse(user.cancelled.is_set())
        self.assertTrue(any(message["text"].startswith("llmopenchat") for message in self.api.messages))

    def test_save_download_and_load_restore_only_the_users_complete_dialog(self):
        self.send(11, "plain saved dialog")
        expected = copy.deepcopy(self.bot.users[11].messages)
        self.send(11, "/save named.json")
        user = self.bot.users[11]
        document = self.api.documents[-1]
        self.assertEqual(document["filename"], "named.json")
        self.assertEqual(document["content"], (user.sessions / "named.json").read_bytes())
        self.assertEqual(json.loads(document["content"])["messages"], expected)
        self.send(11, "/clear")
        self.assertEqual(len(user.messages), 1)
        self.send(11, "/load named.json")
        self.assertEqual(user.messages, expected)
        self.assertFalse(user.settings.code_enabled)
        self.send(22, "/load named.json")
        self.assertEqual(len(self.bot.users[22].messages), 1)
        self.assertTrue(any(message["chat_id"] == 22 and message["text"].startswith("Ошибка:") for message in self.api.messages))

    def test_whitelist_revocation_during_approval_blocks_real_write(self):
        self.send(11, "/tools code on")
        self.send(11, "write revoked", wait=False)
        message = self.approval(11)
        self.whitelist.write_text("bob_user", encoding="utf-8")
        self.assertIn("устарела", self.callback(11, message))
        self.join_worker(self.bot, 11)
        self.assertFalse((self.bot.users[11].workspace / "main.py").exists())
        messages = load_session(self.bot.users[11].directory / "current.json")
        self.assertEqual(json.loads(messages[-1]["content"])["status"], "denied")
        self.assertEqual(len(self.requests), 1)

    def test_non_whitelisted_rename_callback_cannot_approve_existing_write(self):
        self.send(11, "/tools code on")
        self.send(11, "write renamed-denied", wait=False)
        message = self.approval(11)
        original = self.bot.users[11]
        self.usernames[11] = "not_allowed_user"
        self.assertIn("устарела", self.callback(11, message))
        self.join_worker(self.bot, 11)
        self.assertIs(self.bot.users[11], original)
        self.assertFalse((original.workspace / "main.py").exists())
        messages = load_session(original.directory / "current.json")
        self.assertEqual(json.loads(messages[-1]["content"])["status"], "denied")
        self.assertEqual(len(self.requests), 1)

    def test_whitelisted_rename_keeps_numeric_directory_and_existing_context(self):
        self.send(11, "plain original identity")
        user = self.bot.users[11]
        expected_history = copy.deepcopy(user.messages)
        workspace = user.workspace
        self.send(11, "/tools code on")
        self.send(11, "write allowed-rename", wait=False)
        message = self.approval(11)
        self.whitelist.write_text("alice_user,alice_renamed,bob_user", encoding="utf-8")
        self.usernames[11] = "alice_renamed"
        self.assertIn("разрешён", self.callback(11, message))
        self.join_worker(self.bot, 11)
        self.assertIs(self.bot.users[11], user)
        self.assertEqual(user.workspace, workspace)
        self.assertEqual(workspace.parent.name, "11")
        self.assertEqual(user.messages[:len(expected_history)], expected_history)
        self.assertEqual((workspace / "main.py").read_text(encoding="utf-8"), "Содержимое: write allowed-rename\n")
        self.whitelist.write_text("alice_renamed,bob_user", encoding="utf-8")
        self.send(11, "plain renamed identity")
        self.assertEqual(user.messages[-1]["content"], "Ответ plain: plain renamed identity")
        self.assertEqual([path.name for path in (self.root / ".local" / "telegram" / "users").iterdir()], ["11"])

    def test_model_traversal_cannot_read_another_users_workspace(self):
        other = self.bot.user_chat(22)
        (other.workspace / "secret.txt").write_text("private beta value", encoding="utf-8")
        self.send(11, "/tools code on")
        self.send(11, "escape")
        self.assertEqual(self.tool_result()["status"], "error")
        self.assertFalse(self.bot.approvals)
        self.assertFalse(any("Апрув одного вызова" in message["text"] for message in self.api.messages))
        self.assertNotIn("private beta value", json.dumps(self.requests, ensure_ascii=False))

    def test_large_approval_uploads_full_document_before_inline_decision(self):
        self.send(11, "/tools code on")
        self.send(11, "large", wait=False)
        message = self.approval(11)
        self.assertEqual(len(self.api.documents), 1)
        document = self.api.documents[0]
        self.assertEqual(document["filename"], "approval.txt")
        self.assertIn(self.large_source.encode("utf-8"), document["content"])
        self.assertIn("approval.txt", message["text"])
        self.assertLess(document["message_id"], message["message_id"])
        self.callback(11, message)
        self.join_worker(self.bot, 11)
        self.assertEqual((self.bot.users[11].workspace / "main.py").read_text(encoding="utf-8"), self.large_source)
        self.assertEqual(self.tool_result()["bytes_written"], len(self.large_source.encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
