from __future__ import annotations

import io
import json
import unittest
from email.parser import BytesParser
from email.policy import default
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from llmopenchat.telegram_api import TelegramAPI, TelegramError, _NoRedirect


TOKEN = "123456:dummy-secret_for_tests"


def response(result=None, *, body=None):
    return io.BytesIO(
        body if body is not None else json.dumps({"ok": True, "result": result}, ensure_ascii=False).encode("utf-8")
    )


class TelegramAPITests(unittest.TestCase):
    def setUp(self):
        self.opener = Mock()
        with patch("llmopenchat.telegram_api.build_opener", return_value=self.opener):
            self.api = TelegramAPI(TOKEN)
        self.opener.open.return_value = response({"id": 123456, "username": "test_bot"})

    def payload(self, index=0):
        return json.loads(self.opener.open.call_args_list[index].args[0].data)

    def test_posts_utf8_json_to_official_endpoint_only(self):
        self.assertEqual(self.api.call("sendMessage", {"text": "Привет"}), {"id": 123456, "username": "test_bot"})
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, f"https://api.telegram.org/bot{TOKEN}/sendMessage")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json; charset=utf-8")
        self.assertIn("Привет".encode("utf-8"), request.data)
        self.assertEqual(self.opener.open.call_args.kwargs, {"timeout": 30})
        self.assertIsNone(_NoRedirect().redirect_request(request, None, 302, "", {}, "https://other.test/"))

    def test_rejects_invalid_token_method_payload_and_timeout_without_disclosing_secret(self):
        for token in ("", "secret", "123:secret/path", "123:secret?query", None):
            with self.subTest(token=token), self.assertRaises(TelegramError) as raised:
                TelegramAPI(token)
            if token:
                self.assertNotIn(str(token), str(raised.exception))
        for method in ("../getMe", "getMe?x", "https://other.test/", None):
            with self.subTest(method=method), self.assertRaises(TelegramError):
                self.api.call(method)
        for timeout in (0, -1, True, float("nan"), float("inf"), "30"):
            with self.subTest(timeout=timeout), self.assertRaises(TelegramError):
                self.api.call("getMe", timeout=timeout)
        for payload in ([], {"value": float("nan")}, {"value": "\ud800"}):
            with self.subTest(payload=payload), self.assertRaises(TelegramError):
                self.api.call("getMe", payload)
        self.opener.open.assert_not_called()

    def test_polling_uses_explicit_allowed_updates_and_timeout_margin(self):
        self.opener.open.side_effect = [response([{"update_id": 10}]), response([])]
        self.assertEqual(self.api.get_updates(offset=11), [{"update_id": 10}])
        self.assertEqual(self.payload(), {"offset": 11, "timeout": 25, "allowed_updates": ["message", "callback_query"]})
        self.assertEqual(self.opener.open.call_args_list[0].kwargs["timeout"], 35)
        self.assertEqual(self.api.get_updates(timeout=0), [])
        self.assertNotIn("offset", self.payload(1))
        self.assertEqual(self.opener.open.call_args.kwargs["timeout"], 10)

    def test_splits_unicode_text_by_utf16_units_and_places_buttons_on_last_message(self):
        text = "я" * 4095 + "🦊" + "<tag> `code` **bold**\n" + "🦊" * 3000
        markup = {"inline_keyboard": [[{"text": "Разрешить", "callback_data": "approval:1"}]]}
        self.opener.open.side_effect = lambda *args, **kwargs: response({"message_id": self.opener.open.call_count})
        result = self.api.send_message(10, text, markup)
        payloads = [self.payload(index) for index in range(self.opener.open.call_count)]
        self.assertEqual("".join(payload["text"] for payload in payloads), text)
        self.assertGreater(len(payloads), 2)
        self.assertEqual(payloads[0]["text"], "я" * 4095)
        self.assertTrue(all(0 < len(payload["text"].encode("utf-16-le")) // 2 <= 4096 for payload in payloads))
        self.assertTrue(all("reply_markup" not in payload for payload in payloads[:-1]))
        self.assertEqual(payloads[-1]["reply_markup"], markup)
        self.assertTrue(all("parse_mode" not in payload for payload in payloads))
        self.assertEqual(result["message_id"], len(payloads))

    def test_exact_limit_is_one_message_and_invalid_unicode_never_sends_partial_text(self):
        self.api.send_message(10, "🦊" * 2048)
        self.assertEqual(self.opener.open.call_count, 1)
        self.opener.open.reset_mock()
        for text in ("", "x" * 5000 + "\ud800", None):
            with self.subTest(text_length=len(text) if text else 0), self.assertRaises(TelegramError):
                self.api.send_message(10, text)
        self.opener.open.assert_not_called()

    def test_callback_acknowledgement_markup_removal_and_commands(self):
        self.opener.open.side_effect = [response(True), response({"message_id": 9}), response(True), response({"id": 123})]
        self.api.answer_callback_query("callback", "🦊" * 110, True)
        self.assertEqual(self.payload(), {"callback_query_id": "callback", "text": "🦊" * 100, "show_alert": True})
        self.api.edit_message_reply_markup(10, 9)
        self.assertEqual(self.payload(1), {"chat_id": 10, "message_id": 9, "reply_markup": {"inline_keyboard": []}})
        commands = [{"command": "harness", "description": "Подключить харнес"}]
        self.api.set_commands(commands)
        self.assertEqual(self.payload(2), {"commands": commands})
        self.assertEqual(self.api.get_me(), {"id": 123})
        self.assertEqual(self.opener.open.call_args.args[0].full_url.rsplit("/", 1)[1], "getMe")

    def test_network_failure_suppresses_secret_and_chained_exception(self):
        self.opener.open.side_effect = URLError(f"https://api.telegram.org/bot{TOKEN}/getMe")
        with self.assertRaises(TelegramError) as raised:
            self.api.get_me()
        self.assertNotIn(TOKEN, str(raised.exception))
        self.assertNotIn("https://", str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)

    def test_document_upload_preserves_exact_bytes_and_uses_safe_multipart_filename(self):
        content = b"\x00\xffbinary\r\n" + "Исходный текст <tag>".encode("utf-8")
        self.opener.open.return_value = response({"message_id": 9, "document": {"file_id": "file"}})
        result = self.api.send_document(10, 'C:\\private\\Сессия"\r\n.txt', content, "🦊" * 600)
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, f"https://api.telegram.org/bot{TOKEN}/sendDocument")
        self.assertEqual(request.get_method(), "POST")
        multipart = BytesParser(policy=default).parsebytes(
            f"Content-Type: {request.get_header('Content-type')}\r\nMIME-Version: 1.0\r\n\r\n".encode() + request.data
        )
        parts = {part.get_param("name", header="content-disposition"): part for part in multipart.iter_parts()}
        self.assertEqual(parts["chat_id"].get_payload(decode=True), b"10")
        self.assertEqual(parts["caption"].get_payload(decode=True).decode(), "🦊" * 512)
        self.assertEqual(parts["document"].get_payload(decode=True), content)
        self.assertEqual(parts["document"].get_filename(), "Сессия___.txt")
        self.assertNotIn(b"private", request.data)
        self.assertNotIn("parse_mode", parts)
        self.assertEqual(result["message_id"], 9)

    def test_document_network_failure_is_sanitized_and_oversize_never_uploads(self):
        self.opener.open.side_effect = URLError(f"https://api.telegram.org/bot{TOKEN}/sendDocument")
        with self.assertRaises(TelegramError) as raised:
            self.api.send_document(10, "session.json", b"{}")
        self.assertNotIn(TOKEN, str(raised.exception))
        self.opener.open.reset_mock()
        for filename, content in (("", b"x"), ("file", "text"), ("file\ud800", b"x")):
            with self.subTest(filename=filename), self.assertRaises(TelegramError):
                self.api.send_document(10, filename, content)
        with patch("llmopenchat.telegram_api.MAX_DOCUMENT_BYTES", 2), self.assertRaises(TelegramError):
            self.api.send_document(10, "file", b"123")
        self.opener.open.assert_not_called()

    def test_api_failure_redacts_token_and_url_but_keeps_error_code(self):
        body = json.dumps({"ok": False, "error_code": 401, "description": f"Unauthorized {TOKEN} https://api.telegram.org/bot{TOKEN}/getMe"}).encode()
        self.opener.open.side_effect = HTTPError(f"https://api.telegram.org/bot{TOKEN}/getMe", 401, TOKEN, {}, io.BytesIO(body))
        with self.assertRaises(TelegramError) as raised:
            self.api.get_me()
        self.assertEqual(raised.exception.error_code, 401)
        self.assertIn("Unauthorized", str(raised.exception))
        self.assertNotIn(TOKEN, str(raised.exception))
        self.assertNotIn("https://", str(raised.exception))

    def test_non_json_http_failure_does_not_expose_body_or_url(self):
        self.opener.open.side_effect = HTTPError(TOKEN, 502, TOKEN, {}, io.BytesIO(TOKEN.encode()))
        with self.assertRaises(TelegramError) as raised:
            self.api.get_me()
        self.assertEqual(raised.exception.error_code, 502)
        self.assertIn("502", str(raised.exception))
        self.assertNotIn(TOKEN, str(raised.exception))

    def test_retries_short_rate_limit_with_server_delay_and_never_waits_long(self):
        limited = {"ok": False, "error_code": 429, "parameters": {"retry_after": 2}, "description": "Too Many Requests"}
        self.opener.open.side_effect = [response(body=json.dumps(limited).encode()), response({"id": 123})]
        with patch("llmopenchat.telegram_api.time.sleep") as sleep:
            self.assertEqual(self.api.get_me(), {"id": 123})
        sleep.assert_called_once_with(2)
        limited["parameters"]["retry_after"] = 60
        self.opener.open.side_effect = [response(body=json.dumps(limited).encode())]
        with patch("llmopenchat.telegram_api.time.sleep") as sleep, self.assertRaises(TelegramError) as raised:
            self.api.get_me()
        sleep.assert_not_called()
        self.assertEqual(raised.exception.retry_after, 60)

    def test_rate_limit_retries_and_total_wait_are_bounded(self):
        limited = {"ok": False, "error_code": 429, "parameters": {"retry_after": 2}}
        self.opener.open.side_effect = lambda *args, **kwargs: response(body=json.dumps(limited).encode())
        with patch("llmopenchat.telegram_api.time.sleep") as sleep, self.assertRaises(TelegramError):
            self.api.get_me()
        self.assertEqual(self.opener.open.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_rejects_malformed_json_envelope_results_and_oversize_response(self):
        for body in (b"{", b"\xff", b"[]", b'{"ok": 1}', b'{"ok": true}'):
            self.opener.open.side_effect = None
            self.opener.open.return_value = response(body=body)
            with self.subTest(body=body), self.assertRaises(TelegramError):
                self.api.get_me()
        for method, result in ((self.api.get_updates, {}), (self.api.get_updates, [None]), (self.api.get_me, [])):
            self.opener.open.return_value = response(result)
            with self.subTest(result=result), self.assertRaises(TelegramError):
                method()
        self.opener.open.return_value = response(body=b"x" * 21)
        with patch("llmopenchat.telegram_api.MAX_RESPONSE_BYTES", 20), self.assertRaises(TelegramError):
            self.api.get_me()


if __name__ == "__main__":
    unittest.main()
