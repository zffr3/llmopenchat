"""Business updates through the real HTTP/SSE client and Telegram serialization."""

from __future__ import annotations

import copy
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session
from llmopenchat.telegram import TelegramBot
from llmopenchat.telegram_api import TelegramAPI


OWNER, CONTACT = 101, 202
CONNECTION = "business-http-connection"


class RecordingTelegramAPI(TelegramAPI):
    """Exercise the real API payload builder without connecting to Telegram."""

    def __init__(self):
        super().__init__("12345:dummy_token_for_local_tests_only")
        self.sent = []
        self.lookups = []

    def call(self, method, payload=None, timeout=30):
        if method == "getBusinessConnection":
            self.lookups.append(copy.deepcopy(payload))
            return {"id": CONNECTION, "user": {"id": OWNER}, "is_enabled": True,
                    "rights": {"can_reply": True}}
        if method == "sendMessage":
            self.sent.append(copy.deepcopy(payload))
            return {"message_id": len(self.sent)}
        raise AssertionError("Unexpected Telegram method: " + method)


class TelegramBusinessIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-business-http-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.requests = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.addCleanup(self.release.set)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(payload)
                prompt = payload["messages"][-1]["content"]
                answer = "🙂" * 2500 if prompt == "Длинный ответ" else "Ответ: " + prompt
                events = [
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "PRIVATE REASONING"}}]},
                    {"choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": "stop"}]},
                ]
                body = "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events)
                body += "data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                if prompt == "Медленный ответ":
                    self.wfile.flush()
                    owner.started.set()
                    owner.release.wait(5)
                try:
                    self.wfile.write(body.encode("utf-8"))
                    self.wfile.flush()
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        thread = threading.Thread(target=self.server.serve_forever,
                                  kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config.update(backend="external", request_timeout=3, show_reasoning=True,
                           base_url=f"http://127.0.0.1:{self.server.server_port}/v1")
        self.config["telegram"].update(admin_user_id=OWNER, business_enabled=True)
        environment = patch.dict(os.environ, {"LLMOPENCHAT_API_KEY": ""})
        environment.start()
        self.addCleanup(environment.stop)
        self.api = RecordingTelegramAPI()
        self.bot = self.new_bot()

    def new_bot(self):
        bot = TelegramBot(self.api, self.config, self.root, self.root / "absent-whitelist.txt",
                          emit=lambda text: None)
        self.addCleanup(bot.close)
        return bot

    def send(self, text, message_id=1):
        self.bot.handle_update({"business_message": {
            "business_connection_id": CONNECTION, "message_id": message_id,
            "from": {"id": CONTACT, "is_bot": False},
            "chat": {"id": CONTACT, "type": "private"}, "text": text,
        }})

    def join_worker(self):
        chat = self.bot.business.chats[(CONNECTION, CONTACT)]
        worker = chat.worker
        if worker is not None:
            worker.join(4)
            self.assertFalse(worker.is_alive(), "Business worker did not finish")
        return chat

    def test_http_answer_uses_business_route_and_restores_context_without_tools(self):
        self.send("/clear")
        first = self.join_worker()
        self.assertEqual(self.api.sent, [{"chat_id": CONTACT, "text": "Ответ: /clear",
                                         "business_connection_id": CONNECTION}])
        self.assertEqual(load_session(first.directory / "current.json"), first.messages)
        self.bot.close()
        self.bot = self.new_bot()
        self.send("Продолжим", message_id=2)
        self.join_worker()
        self.assertEqual([m["content"] for m in self.requests[-1]["messages"] if m["role"] == "user"],
                         ["/clear", "Продолжим"])
        self.assertTrue(all("tools" not in request for request in self.requests))
        self.assertNotIn("PRIVATE REASONING", str(self.api.sent))
        self.assertEqual(self.bot.users, {})
        self.assertEqual(len(self.api.lookups), 2)

    def test_long_http_answer_keeps_business_connection_on_every_chunk(self):
        self.send("Длинный ответ")
        self.join_worker()
        self.assertEqual(len(self.api.sent), 2)
        self.assertEqual("".join(message["text"] for message in self.api.sent), "🙂" * 2500)
        self.assertTrue(all(message["business_connection_id"] == CONNECTION for message in self.api.sent))

    def test_disconnect_cancels_real_silent_http_stream_without_delivering_reply(self):
        self.send("Медленный ответ")
        self.assertTrue(self.started.wait(2), "Model HTTP stream did not open")
        self.bot.handle_update({"business_connection": {
            "id": CONNECTION, "user": {"id": OWNER}, "is_enabled": False,
            "rights": {"can_reply": True},
        }})
        self.join_worker()
        self.assertEqual(self.api.sent, [])
        self.release.set()


if __name__ == "__main__":
    unittest.main()
