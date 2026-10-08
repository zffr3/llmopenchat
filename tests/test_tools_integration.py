"""Real HTTP/SSE through the human menu into the Windows repository boundary."""

from __future__ import annotations

import copy
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from rich.console import Console

from llmopenchat import cli
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session


@unittest.skipUnless(os.name == "nt", "Strict repository tools require Windows handles")
class ToolsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmopenchat-tools-http-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repository = self.root / "репозиторий"
        self.repository.mkdir()
        self.requests = []
        self.source = 'print("Привет, мир")\n'
        self.target = "main.py"
        self.run_script = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(payload)
                if payload["messages"][-1]["role"] == "user":
                    arguments = json.dumps({"path": owner.target, "content": owner.source}, ensure_ascii=False)
                    chunks = [
                        {"choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": 0, "id": "call-http-1", "type": "function", "function": {
                                "name": "write_file", "arguments": arguments[:17]}}
                        ]}}]},
                        {"choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": 0, "function": {"arguments": arguments[17:]}}
                        ]}}]},
                        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                    ]
                elif owner.run_script and len(owner.requests) == 2:
                    chunks = [{"choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": 0, "id": "call-http-2", "type": "function", "function": {
                            "name": "powershell_run", "arguments": json.dumps({"path": owner.target})}
                    }]}, "finish_reason": "tool_calls"}]}]
                else:
                    chunks = [{"choices": [{"index": 0, "delta": {"content": "Действие обработано."},
                                             "finish_reason": "stop"}]}]
                response = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
                response += "data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                self.wfile.write(response.encode("utf-8"))

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def run_chat(self, decision, run_decision=None):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(base_url=f"http://127.0.0.1:{self.server.server_port}/v1", request_timeout=5)
        output = io.StringIO()
        choices = ["/tools", "2", str(self.repository)]
        if self.run_script:
            choices.append("5")
        choices.extend(["0", f"Запиши {self.target}", decision])
        if self.run_script:
            choices.append(run_decision or "")
        choices.append("/quit")
        with (patch.object(cli, "console", Console(file=output, width=160, color_system=None)),
              patch.object(cli.sys.stdin, "isatty", return_value=False),
              patch("builtins.input", side_effect=choices),
              patch.dict(os.environ, {"LLMOPENCHAT_API_KEY": ""})):
            self.assertEqual(cli.chat(config, self.root), 0)
        return output.getvalue()

    def assert_round(self, status):
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0]["tool_choice"], "required")
        names = {tool["function"]["name"] for tool in self.requests[0]["tools"]}
        self.assertIn("write_file", names)
        self.assertNotIn("web_fetch", names)
        self.assertFalse(self.requests[0]["parallel_tool_calls"])
        result = self.requests[1]["messages"][-1]
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "call-http-1")
        self.assertEqual(json.loads(result["content"])["status"], status)
        saved = list((self.root / ".local" / "sessions").glob("*.json"))
        self.assertEqual(len(saved), 1)
        messages = load_session(saved[0])
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool", "assistant"])

    def test_approved_code_write_has_full_preview_and_complete_saved_round(self):
        output = self.run_chat("разрешить")
        self.assertEqual((self.repository / "main.py").read_text(encoding="utf-8"), self.source)
        self.assertIn("Привет, мир", output)
        self.assertIn(str(self.repository), output)
        self.assert_round("ok")

    def test_denied_code_write_never_creates_file(self):
        output = self.run_chat("")
        self.assertEqual(list(self.repository.iterdir()), [])
        self.assertIn("отклонён", output)
        self.assert_round("denied")

    def test_written_powershell_script_needs_second_approval_and_runs_restricted(self):
        self.run_script = True
        self.target = "project/hello.ps1"
        self.source = "Write-Output 'Hello, World!'\n"
        output = self.run_chat("разрешить", "разрешить")
        self.assertEqual(len(self.requests), 3)
        write = json.loads(self.requests[1]["messages"][-1]["content"])
        run = json.loads(self.requests[2]["messages"][-1]["content"])
        self.assertEqual(write["status"], "ok")
        self.assertEqual(write["directories_created"], ["project"])
        self.assertEqual(run["status"], "ok", run)
        self.assertTrue(run["restricted"])
        self.assertIn("Hello, World!", run["stdout"])
        self.assertIn("Вывод PowerShell:", output)
        self.assertIn("успешно выполнено скриптов: 1", output)
        self.assertEqual(output.count("Харнес · ожидание вашего подтверждения"), 2)

    def test_file_write_approval_does_not_authorize_powershell_run(self):
        self.run_script = True
        self.target = "hello.ps1"
        self.source = "Write-Output 'Hello, World!'\n"
        output = self.run_chat("разрешить", "")
        self.assertEqual(len(self.requests), 3)
        run = json.loads(self.requests[2]["messages"][-1]["content"])
        self.assertEqual(run["status"], "denied")
        self.assertIn("успешно выполнено скриптов: 0", output)


if __name__ == "__main__":
    unittest.main()
