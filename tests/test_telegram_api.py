from __future__ import annotations

import io
import json
import unittest
from email.parser import BytesParser
from email.policy import default
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from llmopenchat.telegram_api import MAX_DOWNLOAD_BYTES, TelegramAPI, TelegramError, _NoRedirect


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
        self.assertEqual(self.payload(), {
            "offset": 11, "timeout": 25,
            "allowed_updates": ["message", "callback_query", "business_connection", "business_message"],
        })
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

    def test_business_messages_keep_connection_on_every_chunk(self):
        self.opener.open.side_effect = lambda *args, **kwargs: response({"message_id": self.opener.open.call_count})
        markup = {"inline_keyboard": [[{"text": "Готово", "callback_data": "done"}]]}

        result = self.api.send_message(42, "я" * 4097, markup, business_connection_id="connection-1")

        self.assertEqual(self.opener.open.call_count, 2)
        self.assertEqual(self.payload(), {
            "chat_id": 42, "text": "я" * 4096, "business_connection_id": "connection-1",
        })
        self.assertEqual(self.payload(1), {
            "chat_id": 42, "text": "я", "business_connection_id": "connection-1", "reply_markup": markup,
        })
        self.assertEqual(result, {"message_id": 2})

    def test_get_business_connection_uses_official_method_and_rejects_malformed_result(self):
        connection = {"id": "connection-1", "user": {"id": 123}, "is_enabled": True, "rights": {"can_reply": True}}
        self.opener.open.return_value = response(connection)

        self.assertEqual(self.api.get_business_connection("connection-1"), connection)
        self.assertEqual(self.payload(), {"business_connection_id": "connection-1"})
        self.assertEqual(self.opener.open.call_args.args[0].full_url, f"https://api.telegram.org/bot{TOKEN}/getBusinessConnection")

        for result in (None, [], True, "connection-1"):
            with self.subTest(result=result), self.assertRaises(TelegramError):
                self.opener.open.return_value = response(result)
                self.api.get_business_connection("connection-1")

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

    def test_business_reply_markup_edit_includes_connection(self):
        self.opener.open.return_value = response(True)

        self.api.edit_message_reply_markup(42, 9, business_connection_id="connection-1")

        self.assertEqual(self.payload(), {
            "chat_id": 42, "message_id": 9,
            "reply_markup": {"inline_keyboard": []}, "business_connection_id": "connection-1",
        })
        self.assertEqual(self.opener.open.call_args.args[0].full_url.rsplit("/", 1)[1], "editMessageReplyMarkup")

    def test_commands_can_target_a_specific_chat_scope(self):
        commands = [{"command": "tools", "description": "Подключить харнесы"}]
        scope = {"type": "chat", "chat_id": 101}
        self.opener.open.return_value = response(True)

        self.assertTrue(self.api.set_commands(commands, scope=scope))

        self.assertEqual(self.payload(), {"commands": commands, "scope": scope})
        self.assertEqual(self.opener.open.call_args.args[0].full_url.rsplit("/", 1)[1], "setMyCommands")

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
        self.assertNotIn("business_connection_id", parts)
        self.assertEqual(result["message_id"], 9)

    def test_business_document_forwards_connection_to_media_upload(self):
        with patch.object(self.api, "send_media", return_value={"message_id": 9}) as send_media:
            self.assertEqual(
                self.api.send_document(42, "file.txt", b"file", "Описание", business_connection_id="connection-1"),
                {"message_id": 9},
            )
        send_media.assert_called_once_with(
            42, "file.txt", b"file", media_type="document", caption="Описание", business_connection_id="connection-1",
        )

    def test_business_media_uploads_include_connection_in_multipart_fields(self):
        for media_type in ("document", "photo", "video", "audio", "voice", "animation", "video_note", "sticker"):
            with self.subTest(media_type=media_type):
                self.opener.open.return_value = response({"message_id": 9})

                self.api.send_media(42, "file", b"content", media_type=media_type, business_connection_id="connection-1")

                request = self.opener.open.call_args.args[0]
                multipart = BytesParser(policy=default).parsebytes(
                    f"Content-Type: {request.get_header('Content-type')}\r\n\r\n".encode() + request.data
                )
                parts = {part.get_param("name", header="content-disposition"): part for part in multipart.iter_parts()}
                self.assertEqual(parts["business_connection_id"].get_payload(decode=True), b"connection-1")
                self.assertEqual(parts[media_type].get_payload(decode=True), b"content")

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

    def test_native_media_uploads_choose_method_field_and_content_type(self):
        cases = (
            ("document", "data.pdf", "sendDocument", "application/octet-stream"),
            ("photo", "photo.JPG", "sendPhoto", "image/jpeg"),
            ("video", "movie.mp4", "sendVideo", "video/mp4"),
            ("audio", "song.mp3", "sendAudio", "audio/mpeg"),
            ("voice", "voice.ogg", "sendVoice", "audio/ogg"),
            ("animation", "image.gif", "sendAnimation", "image/gif"),
            ("video_note", "note.mp4", "sendVideoNote", "video/mp4"),
            ("sticker", "sticker.webp", "sendSticker", "image/webp"),
        )
        content = b"\x00\xffnative-media\r\n"
        for media_type, filename, method, mime_type in cases:
            with self.subTest(media_type=media_type):
                self.opener.open.return_value = response({"message_id": 10})
                caption = "Описание" if media_type not in {"sticker", "video_note"} else ""
                result = self.api.send_media(42, filename, content, media_type=media_type, caption=caption)
                request = self.opener.open.call_args.args[0]
                self.assertEqual(request.full_url, f"https://api.telegram.org/bot{TOKEN}/{method}")
                multipart = BytesParser(policy=default).parsebytes(
                    f"Content-Type: {request.get_header('Content-type')}\r\n\r\n".encode() + request.data
                )
                parts = {part.get_param("name", header="content-disposition"): part for part in multipart.iter_parts()}
                self.assertEqual(parts[media_type].get_content_type(), mime_type)
                self.assertEqual(parts[media_type].get_payload(decode=True), content)
                self.assertEqual(parts[media_type].get_filename(), filename)
                self.assertEqual(parts["chat_id"].get_payload(decode=True), b"42")
                if caption:
                    self.assertEqual(parts["caption"].get_payload(decode=True).decode(), caption)
                else:
                    self.assertNotIn("caption", parts)
                self.assertEqual(result, {"message_id": 10})

    def test_native_media_validation_happens_before_upload(self):
        invalid = (
            {"media_type": "unknown"}, {"media_type": None}, {"media_type": []},
            {"content": b""}, {"content": bytearray(b"x")},
            {"filename": "folder/"}, {"filename": "."}, {"filename": ".."},
            {"caption": None}, {"caption": "bad\ud800"},
            {"media_type": "sticker", "caption": "unsupported"},
            {"media_type": "video_note", "caption": "unsupported"},
        )
        for overrides in invalid:
            args = {"filename": "file", "content": b"x", **overrides}
            with self.subTest(overrides=overrides), self.assertRaises(TelegramError):
                self.api.send_media(42, **args)
        with patch("llmopenchat.telegram_api.MAX_PHOTO_BYTES", 2), self.assertRaises(TelegramError):
            self.api.send_media(42, "photo.jpg", b"123", media_type="photo")
        with patch("llmopenchat.telegram_api.MAX_DOCUMENT_BYTES", 2), self.assertRaises(TelegramError):
            self.api.send_media(42, "video.mp4", b"123", media_type="video")
        self.opener.open.assert_not_called()

    def test_media_regenerates_boundary_if_content_collides(self):
        self.opener.open.return_value = response({"message_id": 1})
        content = b"llmopenchat-first\r\nraw"
        with patch("llmopenchat.telegram_api.secrets.token_hex", side_effect=["first", "second"]):
            self.api.send_media(42, "file", content)
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Content-type"), "multipart/form-data; boundary=llmopenchat-second")
        self.assertIn(content, request.data)

    def test_download_file_uses_official_getfile_endpoint_and_exact_binary_bytes(self):
        content = b"\x00\xffbinary\r\n"
        self.opener.open.side_effect = [
            response({"file_path": "photos/file_123.jpg", "file_size": len(content)}),
            io.BytesIO(content),
        ]
        self.assertEqual(self.api.download_file("telegram-file-id", max_bytes=len(content)), content)
        self.assertEqual(self.payload(), {"file_id": "telegram-file-id"})
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, f"https://api.telegram.org/file/bot{TOKEN}/photos/file_123.jpg")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(self.opener.open.call_args.kwargs, {"timeout": 60})

    def test_download_document_preserves_compatibility_and_optional_size(self):
        self.opener.open.side_effect = [response({"file_path": "documents/file.bin"}), io.BytesIO(b"file")]
        self.assertEqual(self.api.download_document("id", max_bytes=4), b"file")
        with patch.object(self.api, "download_file", return_value=b"data") as download:
            self.assertEqual(self.api.download_document("alias"), b"data")
        download.assert_called_once_with("alias", max_bytes=MAX_DOWNLOAD_BYTES)

    def test_download_rejects_invalid_id_and_limits_before_network(self):
        for file_id in (None, "", "x" * 1025, 123):
            with self.subTest(file_id=file_id), self.assertRaises(TelegramError):
                self.api.download_file(file_id)
        for max_bytes in (None, 0, -1, True, 1.5, MAX_DOWNLOAD_BYTES + 1):
            with self.subTest(max_bytes=max_bytes), self.assertRaises(TelegramError):
                self.api.download_file("id", max_bytes=max_bytes)
        self.opener.open.assert_not_called()

    def test_download_rejects_unsafe_paths_and_malformed_file_information(self):
        cases = [None, [], {}, {"file_path": 123}]
        cases.extend({"file_path": path} for path in (
            "", "../secret", "/etc/passwd", "a//b", "a/./b", "a/../b",
            "https://other.test/file", "a\\b", "a?b", "a#b", "a%2fb", "x" * 1025,
        ))
        cases.extend({"file_path": "file.bin", "file_size": size} for size in (-1, True, "3", 4))
        for result in cases:
            with self.subTest(result=result), self.assertRaises(TelegramError):
                self.opener.open.reset_mock()
                self.opener.open.return_value = response(result)
                self.api.download_file("id", max_bytes=3)
            self.assertEqual(self.opener.open.call_count, 1)

    def test_download_enforces_actual_read_bound_and_detects_truncated_file(self):
        for metadata, content in (
            ({"file_path": "file.bin"}, b"1234"),
            ({"file_path": "file.bin", "file_size": 3}, b"12"),
        ):
            with self.subTest(metadata=metadata), self.assertRaises(TelegramError):
                self.opener.open.side_effect = [response(metadata), io.BytesIO(content)]
                self.api.download_file("id", max_bytes=3)
        bounded = Mock()
        bounded.read.return_value = b"1234"
        bounded.__enter__ = Mock(return_value=bounded)
        bounded.__exit__ = Mock(return_value=None)
        self.opener.open.side_effect = [response({"file_path": "file.bin"}), bounded]
        with self.assertRaises(TelegramError):
            self.api.download_file("id", max_bytes=3)
        bounded.read.assert_called_once_with(4)

    def test_download_transport_failure_redacts_secret_and_does_not_follow_redirects(self):
        for error in (
            URLError(f"https://api.telegram.org/file/bot{TOKEN}/file.bin"),
            HTTPError(f"https://api.telegram.org/file/bot{TOKEN}/file.bin", 302, TOKEN, {}, None),
        ):
            self.opener.open.side_effect = [response({"file_path": "file.bin"}), error]
            with self.subTest(error=type(error).__name__), self.assertRaises(TelegramError) as raised:
                self.api.download_file("id")
            self.assertNotIn(TOKEN, str(raised.exception))
            self.assertNotIn("https://", str(raised.exception))
            self.assertTrue(raised.exception.__suppress_context__)

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
