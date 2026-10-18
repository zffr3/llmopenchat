"""Model HTTP requests retain approval, protocol and public-network boundaries."""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import unittest
from unittest.mock import Mock, patch

from llmopenchat import harness
from llmopenchat.harness import HarnessError, HarnessSettings, ToolHarness


URL = "https://api.example.com/items"


def call(tools, **arguments):
    return json.loads(tools.execute("http_request", json.dumps(arguments, ensure_ascii=False)))


def response(**changes):
    return {
        "url": URL, "method": "GET", "status_code": 200,
        "headers": {"content-type": "application/json"}, "body": '{"ok":true}',
        "truncated": False, "bytes_received": 11, **changes,
    }


class HttpHarnessTests(unittest.TestCase):
    def test_http_permission_is_independent_from_read_only_web(self):
        for settings in (HarnessSettings(), HarnessSettings(web_enabled=True)):
            with self.subTest(settings=settings):
                approve = Mock(return_value=True)
                tools = ToolHarness(settings, approve)
                self.assertNotIn("http_request", [item["function"]["name"] for item in tools.schemas()])
                with patch.object(harness, "_request_http_public") as request:
                    self.assertEqual(call(tools, url=URL)["status"], "error")
                    request.assert_not_called()
                approve.assert_not_called()
        tools = ToolHarness(HarnessSettings(http_enabled=True), lambda request: False)
        self.assertEqual([item["function"]["name"] for item in tools.schemas()], ["http_request"])
        with self.assertRaises(HarnessError):
            ToolHarness(HarnessSettings(http_enabled=1), lambda request: True)

    def test_schema_exposes_supported_methods_and_detached_header_defaults(self):
        tools = ToolHarness(HarnessSettings(http_enabled=True), lambda request: False)
        schema = tools.schemas()[0]["function"]["parameters"]
        self.assertEqual(schema["required"], ["url"])
        properties = schema["properties"]
        self.assertEqual(set(properties["method"]["enum"]), {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
        self.assertEqual(properties["method"]["default"], "GET")
        self.assertEqual(properties["headers"]["type"], "object")
        self.assertEqual(properties["body"]["type"], "string")
        properties["headers"]["default"]["X-Mutated"] = "bad"
        self.assertEqual(tools.schemas()[0]["function"]["parameters"]["properties"]["headers"]["default"], {})

    def test_denial_never_resolves_or_connects_and_each_call_reprompts(self):
        approve = Mock(return_value=False)
        tools = ToolHarness(HarnessSettings(http_enabled=True), approve)
        with patch.object(harness, "_resolve_public") as resolve, patch.object(harness, "_request_http_public") as request:
            for _ in range(2):
                self.assertEqual(call(tools, url=URL, method="POST", body="one")["status"], "denied")
            resolve.assert_not_called()
            request.assert_not_called()
        self.assertEqual(approve.call_count, 2)

    def test_approval_exposes_full_headers_and_body_and_cannot_mutate_request(self):
        approvals = []
        body = '{"message":"Привет\nмир"}'
        headers = {"Authorization": "Bearer explicit-token", "Content-Type": "application/json"}

        def approve(request):
            approvals.append(request)
            self.assertIn("PATCH", request.summary)
            self.assertIn(URL, request.summary)
            self.assertIn("Bearer explicit-token", request.summary + request.preview)
            self.assertIn("application/json", request.summary + request.preview)
            self.assertIn(body, request.preview)
            request.arguments["url"] = "http://127.0.0.1/"
            request.arguments["method"] = "DELETE"
            request.arguments["headers"]["Authorization"] = "mutated"
            request.arguments["body"] = "mutated"
            return True

        tools = ToolHarness(HarnessSettings(http_enabled=True), approve)
        with patch.object(harness, "_request_http_public", return_value=response(method="PATCH")) as request:
            result = call(tools, url=URL, method="PATCH", headers=headers, body=body)
        self.assertEqual(result["status"], "ok")
        request.assert_called_once_with(URL, "PATCH", headers, body)
        self.assertEqual(len(approvals), 1)

    def test_revoke_during_approval_prevents_request(self):
        def revoke(request):
            tools.settings = dataclasses.replace(tools.settings, http_enabled=False)
            return True

        tools = ToolHarness(HarnessSettings(http_enabled=True), revoke)
        with patch.object(harness, "_request_http_public") as request:
            self.assertEqual(call(tools, url=URL)["status"], "denied")
            request.assert_not_called()

    def test_explicit_true_required_and_broken_approval_fails_closed(self):
        decisions = [False, None, 1, "yes", EOFError(), KeyboardInterrupt(), RuntimeError("broken prompt")]
        for decision in decisions:
            with self.subTest(decision=decision):
                approve = Mock(side_effect=decision) if isinstance(decision, BaseException) else Mock(return_value=decision)
                tools = ToolHarness(HarnessSettings(http_enabled=True), approve)
                with patch.object(harness, "_request_http_public") as request:
                    self.assertEqual(call(tools, url=URL)["status"], "denied")
                    request.assert_not_called()

    def test_missing_optional_arguments_get_defaults_and_canonical_url(self):
        tools = ToolHarness(HarnessSettings(http_enabled=True), lambda request: True)
        with patch.object(harness, "_request_http_public", return_value=response()) as request:
            self.assertEqual(call(tools, url="https://api.example.com")["status"], "ok")
        request.assert_called_once_with("https://api.example.com/", "GET", {}, "")

    def test_malformed_types_methods_and_duplicate_arguments_reject_before_approval(self):
        approve = Mock(return_value=True)
        tools = ToolHarness(HarnessSettings(http_enabled=True), approve)
        invalid = [
            "[]", "null", "{", "{}", '{"url":42}',
            '{"url":"https://example.com/","extra":true}',
            '{"url":"https://example.com/","url":"https://other.example/"}',
            '{"url":"https://example.com/","headers":{"X":"1","X":"2"}}',
            '{"url":"https://example.com/","headers":{"X":NaN}}',
        ]
        for arguments in (
            {"method": "CONNECT"}, {"method": "TRACE"}, {"method": "POST\r\nX: 1"},
            {"method": 1}, {"method": None}, {"headers": []}, {"headers": None},
            {"headers": {"X-Value": 2}}, {"headers": {"X-Value": False}},
            {"body": {}}, {"body": None}, {"body": 1},
        ):
            invalid.append(json.dumps({"url": URL, **arguments}))
        with patch.object(harness, "_request_http_public") as request:
            for raw in invalid:
                with self.subTest(arguments=raw):
                    self.assertEqual(json.loads(tools.execute("http_request", raw))["status"], "error")
            request.assert_not_called()
        approve.assert_not_called()

    def test_header_injection_framing_and_encoding_overrides_reject_before_approval(self):
        invalid = [
            {"X-Test\r\nHost": "evil"}, {"X Test": "bad"}, {"": "empty"},
            {"X-Test": "x\r\nHost: evil"}, {"X-Test": "a\x00b"},
            {"X-Test": "a\x7fb"}, {"X-Test": "a\tb"}, {"X-Test": "Привет"},
            {"X-Key": "one", "x-key": "two"},
        ]
        invalid.extend({name: "override"} for name in (
            "Host", "Content-Length", "Transfer-Encoding", "Connection", "Proxy-Authorization",
            "Proxy-Connection", "Accept-Encoding", "Upgrade", "TE", "Trailer",
        ))
        approve = Mock(return_value=True)
        tools = ToolHarness(HarnessSettings(http_enabled=True), approve)
        with patch.object(harness, "_request_http_public") as request:
            for headers in invalid:
                with self.subTest(headers=headers):
                    self.assertEqual(call(tools, url=URL, headers=headers)["status"], "error")
            request.assert_not_called()
        approve.assert_not_called()

    def test_public_standard_ports_only_and_forbidden_targets_never_ask_approval(self):
        urls = [
            "file:///tmp/result", "ftp://example.com/", "http://localhost/",
            "http://127.0.0.1/", "http://10.0.0.1/", "http://169.254.169.254/",
            "http://[::1]/", "http://service.internal/", "https://example.com:8443/",
            "https://user:pass@example.com/", "https://example.com/#fragment",
            "https://example.com/\nHost:evil", "http://example.com\\@127.0.0.1/",
        ]
        approve = Mock(return_value=True)
        tools = ToolHarness(HarnessSettings(http_enabled=True), approve)
        with patch.object(harness, "_request_http_public") as request:
            for url in urls:
                with self.subTest(url=url):
                    self.assertEqual(call(tools, url=url)["status"], "error")
            request.assert_not_called()
        approve.assert_not_called()

    def test_get_and_head_bodies_are_rejected_before_approval(self):
        approve = Mock(return_value=True)
        tools = ToolHarness(HarnessSettings(http_enabled=True), approve)
        with patch.object(harness, "_resolve_public") as resolve, patch.object(harness, "_request_http_public") as request:
            for method in ("GET", "HEAD"):
                with self.subTest(method=method):
                    self.assertEqual(call(tools, url=URL, method=method, body="must not be sent")["status"], "error")
            resolve.assert_not_called()
            request.assert_not_called()
        approve.assert_not_called()

    def test_headers_and_utf8_body_have_byte_limits(self):
        approve = Mock(return_value=True)
        tools = ToolHarness(HarnessSettings(http_enabled=True), approve)
        invalid = [
            {"headers": {f"X-{index}": "x" for index in range(33)}},
            {"headers": {"X-Large": "x" * (16 * 1024)}},
            {"method": "POST", "body": "я" * (128 * 1024 + 1)},
        ]
        with patch.object(harness, "_request_http_public") as request:
            for arguments in invalid:
                with self.subTest(arguments={key: type(value).__name__ for key, value in arguments.items()}):
                    self.assertEqual(call(tools, url=URL, **arguments)["status"], "error")
            request.assert_not_called()
        approve.assert_not_called()

    def test_http_error_responses_remain_available_to_model(self):
        tools = ToolHarness(HarnessSettings(http_enabled=True), lambda request: True)
        for status_code in (204, 302, 400, 404, 429, 500):
            result = response(status_code=status_code, body='{"detail":"rejected"}')
            with self.subTest(status_code=status_code), patch.object(harness, "_request_http_public", return_value=result):
                actual = call(tools, url=URL)
                self.assertEqual(actual["status"], "ok")
                self.assertEqual(actual["status_code"], status_code)
                self.assertEqual(actual["body"], result["body"])


class FakeSocket:
    def __init__(self):
        self.connections = []
        self.closed = False

    def connect(self, address):
        self.connections.append(address)

    def settimeout(self, timeout):
        pass

    def shutdown(self, direction):
        pass

    def close(self):
        self.closed = True

    def do_handshake(self):
        pass


class FakeResponse:
    def __init__(self, status=200, headers=None, content=b"ok"):
        self.status = status
        self.reason = "Test response"
        self.headers = headers if headers is not None else [
            ("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(content))),
        ]
        self.content = content
        self.reads = []

    def getheaders(self):
        return self.headers

    def getheader(self, name, default=None):
        values = [value for key, value in self.headers if key.lower() == name.lower()]
        return ", ".join(values) if values else default

    def read(self, count):
        self.reads.append(count)
        return self.content[:count]


class HttpTransportTests(unittest.TestCase):
    def request(self, response=None, *, url="http://example.com/items?q=1", method="GET", headers=None, body="", connection_error=None):
        sock = FakeSocket()
        connection = Mock()
        connection.getresponse.return_value = response or FakeResponse()
        if connection_error is not None:
            connection.getresponse.side_effect = connection_error
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))]
        context = Mock()
        context.wrap_socket.return_value = sock
        with (patch.object(socket, "getaddrinfo", return_value=addresses) as dns,
              patch.object(socket, "socket", return_value=sock),
              patch.object(harness, "HTTPConnection", return_value=connection),
              patch.object(harness.ssl, "create_default_context", return_value=context)):
            result = harness._request_http_public(url, method, headers or {}, body)
        return result, sock, connection, dns, context

    def test_one_pinned_post_transmits_utf8_and_explicit_auth_without_inherited_secrets(self):
        headers = {"Authorization": "Bearer explicit", "Content-Type": "application/json"}
        with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "LLMOPENCHAT_API_KEY": "private-client-key"}):
            result, sock, connection, dns, _ = self.request(method="POST", headers=headers, body='{"text":"Привет"}')
        self.assertEqual(result["status_code"], 200)
        self.assertEqual(sock.connections, [("8.8.8.8", 80)])
        dns.assert_called_once_with("example.com", 80, type=socket.SOCK_STREAM)
        connection.request.assert_called_once()
        args, kwargs = connection.request.call_args
        self.assertEqual(args[:2], ("POST", "/items?q=1"))
        sent = kwargs.get("body", args[2] if len(args) > 2 else None)
        self.assertEqual(sent, '{"text":"Привет"}'.encode("utf-8"))
        actual_headers = {name.lower(): value for name, value in kwargs["headers"].items()}
        self.assertEqual(actual_headers["authorization"], "Bearer explicit")
        self.assertEqual(actual_headers["accept-encoding"], "identity")
        self.assertNotIn("cookie", actual_headers)
        self.assertNotIn("proxy-authorization", actual_headers)
        self.assertNotIn("private-client-key", str(kwargs))
        self.assertTrue(sock.closed)

    def test_http_statuses_duplicate_response_headers_and_raw_html_are_preserved(self):
        for status in (201, 302, 400, 404, 500):
            original = FakeResponse(status, [("Content-Type", "text/html"), ("X-Result", "one"), ("x-result", "two"), ("Location", "http://127.0.0.1/private")], b"<p>Raw API body</p>")
            with self.subTest(status=status):
                result, sock, connection, dns, _ = self.request(original)
                self.assertEqual(result["status_code"], status)
                self.assertEqual(result["body"], "<p>Raw API body</p>")
                self.assertEqual(result["headers"]["x-result"], "one, two")
                self.assertEqual(result["headers"]["location"], "http://127.0.0.1/private")
                self.assertEqual(result["bytes_received"], len(original.content))
                connection.request.assert_called_once()
                dns.assert_called_once()
                self.assertTrue(sock.closed)

    def test_json_suffix_xml_and_declared_charset_are_text(self):
        for mime, content, expected in (
            ("application/problem+json", b'{"problem":"detail"}', '{"problem":"detail"}'),
            ("application/xml", b"<ok/>", "<ok/>"),
            ("text/plain; charset=windows-1251", "Привет".encode("cp1251"), "Привет"),
            ("text/plain; charset=unknown-encoding", "Привет".encode("utf-8"), "Привет"),
        ):
            with self.subTest(mime=mime):
                result, *_ = self.request(FakeResponse(headers=[("Content-Type", mime)], content=content))
                self.assertEqual(result["body"], expected)

    def test_body_characters_are_bounded_separately_from_received_bytes(self):
        content = ("я" * (24000 + 1)).encode("utf-8")
        result, *_ = self.request(FakeResponse(content=content))
        self.assertEqual(result["body"], "я" * 24000)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["bytes_received"], len(content))

    def test_binary_compressed_and_oversized_responses_raise(self):
        responses = [
            FakeResponse(headers=[("Content-Type", "application/octet-stream")]),
            FakeResponse(headers=[("Content-Type", "text/plain"), ("Content-Encoding", "gzip")]),
            FakeResponse(headers=[("Content-Type", "text/plain"), ("Content-Length", str(512 * 1024 + 1))]),
            FakeResponse(headers=[("Content-Type", "text/plain")], content=b"x" * (512 * 1024 + 1)),
        ]
        for original in responses:
            with self.subTest(headers=original.headers):
                with self.assertRaises(HarnessError):
                    self.request(original)

    def test_head_does_not_read_or_reject_entity_metadata(self):
        original = FakeResponse(headers=[
            ("Content-Type", "application/octet-stream"), ("Content-Encoding", "gzip"),
            ("Content-Length", str(1024 * 1024 * 1024)),
        ])
        result, *_ = self.request(original, method="HEAD")
        self.assertEqual(result["body"], "")
        self.assertEqual(result["bytes_received"], 0)
        self.assertFalse(result["truncated"])
        self.assertEqual(original.reads, [])

    def test_incomplete_content_length_is_not_returned_as_complete_response(self):
        original = FakeResponse(headers=[
            ("Content-Type", "application/json"), ("Content-Length", "200"),
        ], content=b'{"ok":true}')
        with self.assertRaises(HarnessError) as raised:
            self.request(original, method="POST", body="change")
        self.assertTrue(raised.exception.request_started)

    def test_malformed_and_conflicting_content_lengths_are_rejected(self):
        for length in ("-1", "not-a-number", "2, 3", "1.5"):
            with self.subTest(length=length):
                original = FakeResponse(headers=[("Content-Type", "text/plain"), ("Content-Length", length)])
                with self.assertRaises(HarnessError):
                    self.request(original)
        original = FakeResponse(headers=[("Content-Type", "text/plain"), ("Content-Length", "2"), ("Content-Length", "3")])
        with self.assertRaises(HarnessError):
            self.request(original)

    def test_oversized_response_headers_are_rejected_before_reading_body(self):
        original = FakeResponse(headers=[("Content-Type", "text/plain"), ("X-Large", "x" * (16 * 1024))])
        with self.assertRaises(HarnessError) as raised:
            self.request(original)
        self.assertTrue(raised.exception.request_started)
        self.assertEqual(original.reads, [])

    def test_https_uses_verified_context_and_original_server_hostname(self):
        _, sock, _, _, context = self.request(url="https://example.com/items")
        context.wrap_socket.assert_called_once_with(sock, server_hostname="example.com", do_handshake_on_connect=False)
        self.assertTrue(sock.closed)

    def test_post_connection_failure_after_send_is_not_retried_and_marks_uncertainty(self):
        tools = ToolHarness(HarnessSettings(http_enabled=True), lambda request: True)
        sock = FakeSocket()
        connection = Mock()
        connection.getresponse.side_effect = OSError("connection lost after upload")
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))]
        with (patch.object(socket, "getaddrinfo", return_value=addresses),
              patch.object(socket, "socket", return_value=sock),
              patch.object(harness, "HTTPConnection", return_value=connection)):
            result = call(tools, url="http://example.com/items", method="POST", body="change")
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["request_started"])
        connection.request.assert_called_once()
        connection.getresponse.assert_called_once()
        self.assertTrue(sock.closed)


if __name__ == "__main__":
    unittest.main()
