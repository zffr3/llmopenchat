from __future__ import annotations

import contextlib
import http.client
import io
import json
import socket
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llmopenchat.api import StreamEvent
from llmopenchat.scenario_runner import ScenarioRunner
from llmopenchat.scenario_webhook import build_webhook_server
from llmopenchat.scenarios import Scenario, ScenarioOutput, load_hosts


TOKEN = "test-webhook-secret"


class ScenarioWebhookTests(unittest.TestCase):
    def setUp(self):
        self.scenario = SimpleNamespace(input_max_bytes=64)
        self.runner = Mock()
        self.runner.run.return_value = {"summary": "Новость", "sentiment": "neutral"}
        self.server = build_webhook_server(self.scenario, self.runner, port=0, token=TOKEN)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, body=b'{"text":"news"}', *, path="/run", method="POST", headers=None,
                content_length=True, authorization=True, content_type=True):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        try:
            connection.putrequest(method, path)
            if authorization:
                connection.putheader("Authorization", "Bearer " + TOKEN)
            if content_type:
                connection.putheader("Content-Type", "application/json; charset=utf-8")
            if content_length:
                connection.putheader("Content-Length", str(len(body)))
            for name, value in headers or []:
                connection.putheader(name, value)
            connection.endheaders(body)
            response = connection.getresponse()
            raw = response.read()
            return response.status, dict(response.getheaders()), json.loads(raw) if raw else None
        finally:
            connection.close()

    def test_authentication_is_mandatory_at_startup(self):
        for token in (None, "", " ", "secret with spaces", "secret\r\nInjected: value", "секрет", True):
            with self.subTest(token=token), self.assertRaises(ValueError) as raised:
                build_webhook_server(self.scenario, self.runner, port=0, token=token)
            if isinstance(token, str) and len(token) > 2:
                self.assertNotIn(token, str(raised.exception))

    def test_runs_fixed_scenario_with_utf8_text_and_returns_json(self):
        status, headers, result = self.request(json.dumps({"text": "Текст новости"}, ensure_ascii=False).encode())
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Connection"], "close")
        self.assertEqual(result, self.runner.run.return_value)
        self.runner.run.assert_called_once_with("Текст новости")

    def test_rejects_invalid_listen_settings_before_binding(self):
        for port in (-1, 65536, True, "8765", None):
            with self.subTest(port=port), self.assertRaises(ValueError):
                build_webhook_server(self.scenario, self.runner, port=port, token=TOKEN)
        for host in ("", " ", None, 127):
            with self.subTest(host=host), self.assertRaises(ValueError):
                build_webhook_server(self.scenario, self.runner, host=host, port=0, token=TOKEN)

    def test_authentication_failures_never_call_model(self):
        cases = [
            {"authorization": False},
            {"authorization": False, "headers": [("Authorization", "Bearer wrong")]},
            {"headers": [("Authorization", "Bearer " + TOKEN)]},
            {"authorization": False, "headers": [("Authorization", "Basic " + TOKEN)]},
        ]
        for options in cases:
            with self.subTest(options=options):
                status, _, result = self.request(**options)
                self.assertEqual(status, 401)
                self.assertNotIn(TOKEN, str(result))
        self.runner.run.assert_not_called()

    def test_only_post_run_is_available_without_redirects(self):
        for options in ({"path": "/"}, {"path": "/run?scenario=other"}, {"path": "/run/"}, {"method": "GET"}):
            with self.subTest(options=options):
                status, headers, _ = self.request(**options)
                self.assertEqual(status, 404)
                self.assertNotIn("Location", headers)
        self.runner.run.assert_not_called()

    def test_requires_unambiguous_length_and_json_content_type(self):
        cases = [
            ({"content_length": False}, 411),
            ({"headers": [("Content-Length", "15")]}, 400),
            ({"content_length": False, "headers": [("Content-Length", "-1")]}, 400),
            ({"headers": [("Transfer-Encoding", "chunked")]}, 400),
            ({"content_type": False}, 415),
            ({"content_type": False, "headers": [("Content-Type", "text/plain")]}, 415),
            ({"headers": [("Content-Type", "application/json")]}, 415),
            ({"content_type": False, "headers": [("Content-Type", "application/json; charset=latin-1")]}, 415),
        ]
        for options, expected in cases:
            with self.subTest(options=options):
                self.assertEqual(self.request(**options)[0], expected)
        self.runner.run.assert_not_called()

    def test_rejects_malformed_and_nonfinite_json_and_configuration_overrides(self):
        for body in (
            b"invalid", b"[]", b"null", b"{}", b'{"text":1}', b'{"text":null}',
            b'{"text":""}', b'{"text":"   \\n\\t"}',
            b'{"text":"first","text":"second"}', b'{"text":NaN}', b'{"text":Infinity}',
            b'{"text":1e999}', b'{"text":"\\ud800"}', b'{"text":"\xff"}',
            b'{"text":"news","scenario":"other"}',
            b'{"text":"news","config":{"auto_approve":true}}',
            b'{"text":"news","output":{"url":"https://other.test"}}',
        ):
            with self.subTest(body=body):
                status, _, result = self.request(body)
                self.assertEqual(status, 400)
                self.assertNotIn("other.test", str(result))
        self.runner.run.assert_not_called()

    def test_enforces_decoded_utf8_and_envelope_limits_before_generation(self):
        self.assertEqual(self.request(json.dumps({"text": "я" * 33}).encode())[0], 413)
        status, _, _ = self.request(
            body=b"", content_length=False,
            headers=[("Content-Length", str(self.server.body_max_bytes + 1))],
        )
        self.assertEqual(status, 413)
        self.runner.run.assert_not_called()
        # Escaped Unicode remains valid when its decoded UTF-8 length is in bounds.
        self.assertEqual(self.request(json.dumps({"text": "я" * 32}).encode())[0], 200)
        self.runner.run.assert_called_once_with("я" * 32)

    def test_upstream_errors_and_invalid_results_have_safe_502_responses(self):
        private = "private model text " + TOKEN + " https://user:password@host.test"
        for error in (RuntimeError(private), ValueError(private)):
            self.runner.run.side_effect = error
            status, _, result = self.request()
            self.assertEqual(status, 502)
            self.assertNotIn(private, str(result))
            self.assertNotIn(TOKEN, str(result))
            self.assertNotIn("password", str(result))
        self.runner.run.side_effect = None
        for result in ({"value": float("nan")}, {"value": "\ud800"}, object()):
            self.runner.run.return_value = result
            self.assertEqual(self.request()[0], 502)

    def test_request_and_failure_logs_do_not_expose_inputs_or_secrets(self):
        self.runner.run.side_effect = RuntimeError(TOKEN)
        output = io.StringIO()
        with contextlib.redirect_stderr(output), contextlib.redirect_stdout(output):
            self.assertEqual(self.request(b'{"text":"private input"}')[0], 502)
            self.assertEqual(self.request(path="/" + TOKEN)[0], 404)
        self.assertEqual(output.getvalue(), "")

    def test_slow_body_has_a_socket_timeout(self):
        with patch("llmopenchat.scenario_webhook.REQUEST_IO_TIMEOUT", 0.05):
            connection = socket.create_connection(self.server.server_address, timeout=2)
            try:
                connection.sendall(
                    b"POST /run HTTP/1.0\r\nAuthorization: Bearer " + TOKEN.encode()
                    + b"\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n"
                )
                response = http.client.HTTPResponse(connection)
                response.begin()
                self.assertEqual(response.status, 400)
                self.assertIn("error", json.loads(response.read()))
            finally:
                connection.close()
        self.runner.run.assert_not_called()

    def test_model_executions_are_serialized(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        calls = []

        def run(text):
            calls.append(text)
            if len(calls) == 1:
                first_started.set()
                if not release_first.wait(timeout=2):
                    raise RuntimeError("test did not release the first request")
            else:
                second_started.set()
            return {"text": text}

        self.runner.run.side_effect = run
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(self.request, b'{"text":"one"}')
            self.assertTrue(first_started.wait(timeout=1))
            second = executor.submit(self.request, b'{"text":"two"}')
            try:
                self.assertFalse(second_started.wait(timeout=0.05))
                self.assertEqual(calls, ["one"])
            finally:
                release_first.set()
            self.assertEqual(first.result(timeout=2)[0], 200)
            self.assertEqual(second.result(timeout=2)[0], 200)
        self.assertEqual(calls, ["one", "two"])

    def make_delivery_runner(self, model_result):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        hosts_path = Path(directory.name) / "hosts.txt"
        hosts_path.write_text("http://127.0.0.1:8000\n", encoding="utf-8")
        scenario = Scenario(
            name="news", instruction="Extract the key thought and sentiment.",
            response_schema={
                "type": "object", "required": ["summary", "sentiment"], "additionalProperties": False,
                "properties": {"summary": {"type": "string"}, "sentiment": {"enum": ["neutral"]}},
            },
            output=ScenarioOutput(type="post", url="http://127.0.0.1:8000/result", timeout_seconds=7),
            hosts=load_hosts(hosts_path),
        )
        client = Mock()
        client.stream_chat.return_value = iter([
            StreamEvent("content", json.dumps(model_result, ensure_ascii=False)),
            StreamEvent("finish", "stop"), StreamEvent("done", {}),
        ])
        opener = Mock()
        response = io.BytesIO()
        response.status = 204
        opener.open.return_value = response
        with patch("llmopenchat.scenario_runner.build_opener", return_value=opener):
            runner = ScenarioRunner(scenario, client)
        self.server.runner = runner
        return client, opener

    def test_validated_result_is_posted_to_fixed_destination_by_real_runner(self):
        expected = {"summary": "Ключевая мысль", "sentiment": "neutral"}
        client, opener = self.make_delivery_runner(expected)
        status, _, result = self.request(b'{"text":"news"}')
        self.assertEqual(status, 200)
        self.assertEqual(result, expected)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/result")
        self.assertEqual(json.loads(request.data), expected)
        self.assertEqual(opener.open.call_args.kwargs, {"timeout": 7})
        self.assertEqual(client.stream_chat.call_args.args[0][1], {"role": "user", "content": "news"})

    def test_invalid_model_result_is_not_posted(self):
        _, opener = self.make_delivery_runner({"summary": "missing sentiment"})
        self.assertEqual(self.request()[0], 502)
        opener.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
