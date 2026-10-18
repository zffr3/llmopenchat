"""Real SSE tool round and HTTP wire request through the human chat menu."""

from __future__ import annotations

import copy
import io
import json
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from llmopenchat import cli, harness
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session


class HttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-http-round-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_requests = []
        self.http_requests = []
        self.body = json.dumps({"message": "Привет, мир"}, ensure_ascii=False)
        self.arguments = {
            "url": "http://api.example.com/items?mode=test", "method": "POST",
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer explicit-test"},
            "body": self.body,
        }
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/v1/chat/completions":
                    payload = json.loads(body)
                    owner.model_requests.append(payload)
                    if len(owner.model_requests) == 1:
                        arguments = json.dumps(owner.arguments, ensure_ascii=False)
                        chunks = [
                            {"choices": [{"delta": {"tool_calls": [{
                                "index": 0, "id": "http-request-1", "type": "function",
                                "function": {"name": "http_request", "arguments": arguments[:27]},
                            }]}}]},
                            {"choices": [{"delta": {"tool_calls": [{
                                "index": 0, "function": {"arguments": arguments[27:]},
                            }]}, "finish_reason": "tool_calls"}]},
                        ]
                    else:
                        chunks = [{"choices": [{"delta": {"content": "Ответ сервера обработан."},
                                                 "finish_reason": "stop"}]}]
                    result = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
                    result = (result + "data: [DONE]\n\n").encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                else:
                    owner.http_requests.append((self.path, dict(self.headers.items()), body))
                    result = '{"id":7,"message":"Создано"}'.encode("utf-8")
                    self.send_response(201)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("X-Request-Id", "local-test-7")
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()

        def stop():
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop)

    def run_chat(self, decision):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(base_url=f"http://127.0.0.1:{self.server.server_port}/v1", request_timeout=5)
        output = io.StringIO()
        # Only the test resolver substitutes a local endpoint for the public
        # hostname. Production URL/DNS validation remains exercised separately.
        address = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", self.server.server_port))
        with (patch.object(cli, "console", Console(file=output, width=180, color_system=None)),
              patch.object(cli.sys.stdin, "isatty", return_value=False),
              patch("builtins.input", side_effect=["/tools", "6", "0", "Отправь HTTP POST", decision, "/quit"]),
              patch.object(harness, "_resolve_public", return_value=[address]) as resolve,
              patch.dict(os.environ, {"LLMOPENCHAT_API_KEY": "model-api-test-secret"})):
            self.assertEqual(cli.chat(config, self.root), 0)
        self.assertEqual(len(self.model_requests), 2)
        self.assertEqual([tool["function"]["name"] for tool in self.model_requests[0]["tools"]], ["http_request"])
        tool_result = self.model_requests[1]["messages"][-1]
        self.assertEqual(tool_result["role"], "tool")
        self.assertEqual(tool_result["tool_call_id"], "http-request-1")
        sessions = list((self.root / ".local" / "sessions").glob("*.json"))
        self.assertEqual(len(sessions), 1)
        messages = load_session(sessions[0])
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[-2], tool_result)
        self.assertIn("http_request", self.model_requests[0]["messages"][1]["content"])
        self.assertNotIn("request_started=true", messages[0]["content"])
        return json.loads(tool_result["content"]), output.getvalue(), resolve

    def test_approved_http_post_reaches_server_and_result_is_returned_and_saved(self):
        result, output, resolve = self.run_chat("разрешить")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["status_code"], 201)
        self.assertEqual(json.loads(result["body"]), {"id": 7, "message": "Создано"})
        self.assertEqual(result["headers"]["x-request-id"], "local-test-7")
        self.assertEqual(len(self.http_requests), 1)
        target, headers, body = self.http_requests[0]
        self.assertEqual(target, "/items?mode=test")
        self.assertEqual(headers["Host"], "api.example.com")
        self.assertEqual(headers["Authorization"], "Bearer explicit-test")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(body, self.body.encode("utf-8"))
        self.assertEqual(int(headers["Content-Length"]), len(body))
        resolve.assert_called_once()
        self.assertIn("POST", output)
        self.assertIn("Привет, мир", output)

    def test_denied_http_post_never_reaches_server_but_model_gets_denial(self):
        result, output, resolve = self.run_chat("")
        self.assertEqual(result["status"], "denied")
        self.assertEqual(self.http_requests, [])
        resolve.assert_not_called()
        self.assertIn("отклонён", output)


if __name__ == "__main__":
    unittest.main()
