"""Real CLI pipelines with a local fake model and backend receiver."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from llmopenchat import cli, menu, scenario_cli
from llmopenchat.config import DEFAULT_CONFIG, ROOT, write_config


class ScenarioCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmopenchat-scenario-cli-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scenario = self.root / "news" / "scenario.json"
        scenario_cli.create_template(self.scenario)
        self.payloads = []
        self.deliveries = []
        self.model_checks = 0
        self.answer = {"main_idea": "Компания открыла завод.", "sentiment": "positive"}
        self.finish = "stop"
        self.receiver_status = 200
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.model_checks += 1
                body = json.dumps({"data": [{"id": "scenario-test"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/receiver":
                    owner.deliveries.append((json.loads(body), self.headers.get("Authorization")))
                    self.send_response(owner.receiver_status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                owner.payloads.append(json.loads(body))
                records = [
                    {"choices": [{"delta": {"content": json.dumps(owner.answer, ensure_ascii=False)}}]},
                    {"choices": [{"delta": {}, "finish_reason": owner.finish}]},
                ]
                response = ("".join("data: " + json.dumps(record, ensure_ascii=False) + "\n\n" for record in records)
                            + "data: [DONE]\n\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.config = self.root / "config.json"
        config = copy.deepcopy(DEFAULT_CONFIG)
        config.update(backend="external", base_url=self.origin + "/v1", model="scenario-test")
        write_config(self.config, config)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def invoke(self, *args, text="Новость: компания открыла завод.", extra_env=None):
        env = os.environ.copy()
        env["LLMOPENCHAT_API_KEY"] = ""
        env.update(extra_env or {})
        return subprocess.run(
            [sys.executable, "-m", "llmopenchat", "--config", str(self.config), "scenario", *map(str, args)],
            cwd=ROOT, input=text.encode("utf-8"), capture_output=True, timeout=15, env=env,
        )

    def edit(self, **updates):
        value = json.loads(self.scenario.read_text(encoding="utf-8"))
        value.update(updates)
        self.scenario.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_stdin_returns_one_json_document_and_sets_schema_request(self):
        result = self.invoke("run", self.scenario, "--no-autostart")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        self.assertEqual(json.loads(result.stdout), self.answer)
        request = self.payloads[0]
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 512)
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["messages"][-1], {"role": "user", "content": "Новость: компания открыла завод."})
        self.assertNotIn("tools", request)
        self.assertFalse((self.root / ".local" / "sessions").exists())

    def test_bad_output_and_truncated_output_have_nonzero_exit_without_stdout(self):
        self.answer = {"main_idea": "секрет-новости", "sentiment": "invalid"}
        result = self.invoke("run", self.scenario)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertNotIn("секрет-новости", result.stderr.decode("utf-8"))
        self.answer = {"main_idea": "Новость", "sentiment": "neutral"}
        self.finish = "length"
        result = self.invoke("run", self.scenario)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")

    def test_input_is_checked_before_model_startup(self):
        invalid_file = self.root / "invalid.txt"
        invalid_file.write_bytes(b"\xff")
        result = self.invoke("run", self.scenario, "--input-file", invalid_file)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.edit(input={"max_bytes": 5})
        result = self.invoke("run", self.scenario, text="Очень длинный текст")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.model_checks, 0)
        self.assertEqual(self.payloads, [])

    def test_fixed_post_delivery_and_failure_exit(self):
        (self.scenario.parent / "hosts.txt").write_text(self.origin + "\n", encoding="utf-8")
        self.edit(output={"type": "post", "url": self.origin + "/receiver", "headers_env": {"Authorization": "SCENARIO_TEST_AUTH"}})
        result = self.invoke("run", self.scenario, extra_env={"SCENARIO_TEST_AUTH": "Bearer backend-secret"})
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        self.assertEqual(result.stdout, b"")
        self.assertEqual(self.deliveries, [(self.answer, "Bearer backend-secret")])
        self.receiver_status = 503
        result = self.invoke("run", self.scenario, extra_env={"SCENARIO_TEST_AUTH": "Bearer backend-secret"})
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(self.deliveries), 2)  # No retry on failure.
        self.assertNotIn(b"backend-secret", result.stderr)

    def test_init_validate_never_contacts_model_and_preserves_existing_definition(self):
        destination = self.root / "new" / "scenario.json"
        result = self.invoke("init", destination)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        original = destination.read_bytes()
        result = self.invoke("init", destination)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(destination.read_bytes(), original)
        result = self.invoke("validate", destination)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["auto_approve"], [])
        hosts = destination.parent / "hosts.txt"
        original_hosts = hosts.read_bytes()
        result = self.invoke("init", destination.parent / "second.json")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(hosts.read_bytes(), original_hosts)
        self.assertEqual(self.model_checks, 0)

    def test_post_outside_hosts_fails_before_model_or_delivery(self):
        self.edit(output={"type": "post", "url": self.origin + "/receiver"})
        result = self.invoke("run", self.scenario)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"hosts.txt", result.stderr)
        self.assertEqual(self.model_checks, 0)
        self.assertEqual(self.deliveries, [])

    def test_missing_webhook_token_fails_before_model_startup(self):
        result = self.invoke("serve", self.scenario, "--port", "0", extra_env={"LLMOPENCHAT_WEBHOOK_TOKEN": ""})
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(self.model_checks, 0)


class ScenarioMenuTests(unittest.TestCase):
    def test_scenario_menu_receives_current_config_without_starting_chat(self):
        from rich.console import Console
        import io

        callback, chat = Mock(), Mock()
        console = Console(file=io.StringIO())
        with patch.object(menu, "read_choice", side_effect=["7", "0"]):
            self.assertEqual(menu.main_menu(DEFAULT_CONFIG, ROOT / "config.json", ROOT, console,
                                            chat, run_scenarios=callback), 0)
        callback.assert_called_once_with(DEFAULT_CONFIG)
        chat.assert_not_called()
