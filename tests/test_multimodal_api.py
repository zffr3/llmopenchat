"""Real serialized API payloads and safe diagnostics for image requests."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from llmopenchat.api import ApiClient, ApiError
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session, save_session

from tests.test_multimodal_history import image_part


class MultimodalApiTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["vision"] = {"enabled": True}
        self.image = image_part()
        self.messages = [{"role": "system", "content": "Отвечай точно"},
                         {"role": "user", "content": [{"type": "text", "text": "Опиши изображение"}, self.image]}]
        self.opener = Mock()
        self.environment = patch.dict("os.environ", {"LLMOPENCHAT_API_KEY": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        with patch("llmopenchat.api.build_opener", return_value=self.opener):
            self.client = ApiClient(self.config)

    def request(self, messages=None, **options):
        self.opener.open.return_value = io.BytesIO(b"data: [DONE]\n\n")
        list(self.client.stream_chat(self.messages if messages is None else messages, **options))
        return json.loads(self.opener.open.call_args.args[0].data)

    def test_enabled_payload_contains_exact_inline_image_and_preserves_input(self):
        original = copy.deepcopy(self.messages)
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
        payload = self.request(tools=tools)
        self.assertEqual(payload["messages"], original)
        self.assertEqual(payload["messages"][1]["content"][1]["image_url"]["url"], self.image["image_url"]["url"])
        self.assertEqual(self.messages, original)
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertFalse(payload["parallel_tool_calls"])

    def test_disabled_or_missing_vision_degrades_to_text_and_explicit_unavailable_instruction(self):
        original = copy.deepcopy(self.messages)
        for vision in ({"enabled": False}, None):
            with self.subTest(vision=vision):
                if vision is None:
                    self.config.pop("vision", None)
                else:
                    self.config["vision"] = vision
                payload = self.request()
                serialized = json.dumps(payload, ensure_ascii=False)
                self.assertNotIn("base64", serialized)
                self.assertNotIn("image_url", serialized)
                self.assertNotIn(self.image["image_url"]["url"].split(",")[1], serialized)
                self.assertIn("Опиши изображение", serialized)
                self.assertIn("Изображение", serialized)
                system = "\n".join(message["content"] for message in payload["messages"] if message["role"] == "system")
                self.assertIn("недоступны", system)
                self.assertIn("Не утверждай", system)
                self.assertEqual(self.messages, original)

    def test_invalid_vision_switch_cannot_enable_or_send_images(self):
        for vision in ({"enabled": "true"}, {"enabled": 1}, [], None):
            with self.subTest(vision=vision), self.assertRaises(ApiError):
                self.config["vision"] = vision
                self.request()
        self.opener.open.assert_not_called()

    def test_transport_prunes_a_detached_copy_to_the_configured_latest_image_count(self):
        self.config["vision"] = {"enabled": True, "max_history_images": 2}
        messages = [{"role": "user", "content": [{"type": "text", "text": str(index)}, image_part()]}
                    for index in range(6)]
        original = copy.deepcopy(messages)
        payload = self.request(messages)
        self.assertEqual(messages, original)
        image_messages = [message for message in payload["messages"]
                          if any(part.get("type") == "image_url" for part in message["content"])]
        self.assertEqual([message["content"][0]["text"] for message in image_messages], ["4", "5"])
        self.assertEqual(len(image_messages), 2)
        self.assertIn("удалено", payload["messages"][0]["content"][1]["text"])

    def test_loaded_images_are_retained_on_disk_when_disabled_transport_sends_text_only(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Первое фото"}, image_part()]},
                    {"role": "assistant", "content": "Ответ"},
                    {"role": "user", "content": [{"type": "text", "text": "Сравни со вторым"}, image_part()]}]
        with tempfile.TemporaryDirectory(prefix="llmopenchat-multimodal-api-") as directory:
            path = Path(directory) / "session.json"
            save_session(path, messages, "vision-local")
            loaded = load_session(path)
            self.config["vision"] = {"enabled": False}
            payload = self.request(loaded)
            self.assertNotIn("data:image", json.dumps(payload))
            self.assertIn("недоступны", json.dumps(payload, ensure_ascii=False))
            self.assertEqual(loaded, messages)
            self.assertEqual(load_session(path), messages)

    def test_http_diagnostic_read_and_display_are_bounded(self):
        class TrackingBody(io.BytesIO):
            def __init__(self, body):
                super().__init__(body)
                self.read_sizes = []

            def read(self, size=-1):
                self.read_sizes.append(size)
                return super().read(size)

        body = TrackingBody(("Описание: " + "я" * 10000).encode("utf-8"))
        self.opener.open.side_effect = HTTPError("https://model.example/api", 503, "error", {}, body)
        with self.assertRaises(ApiError) as raised:
            list(self.client.stream_chat(self.messages))
        self.assertEqual(body.read_sizes, [8192])
        self.assertTrue(body.closed)
        self.assertLess(len(str(raised.exception)), 1600)
        self.assertIn("Описание", str(raised.exception))

    def test_text_and_tool_rounds_keep_existing_request_shape(self):
        messages = [{"role": "system", "content": "Инструкция"},
                    {"role": "user", "content": "Вопрос"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "read_file", "arguments": '{"path":"file.txt"}'}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"ok"}'}]
        self.config["vision"] = {"enabled": False}
        self.assertEqual(self.request(messages)["messages"], messages)

    def test_request_extra_cannot_replace_messages_and_bypass_disabled_vision(self):
        self.config["vision"] = {"enabled": False}
        self.config["request_extra"] = {"messages": self.messages, "top_p": 0.7}
        messages = [{"role": "user", "content": "Only this text"}]
        payload = self.request(messages)
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["top_p"], 0.7)
        self.assertNotIn("base64", json.dumps(payload))

    def test_invalid_remote_image_parts_and_non_user_arrays_never_reach_network(self):
        invalid = [[{"type": "image_url", "image_url": {"url": url}}] for url in (
            "https://example.com/image.png", "file:///tmp/image.png",
            "https://api.telegram.org/file/bot123456:private-token-secret/photo.png",
            "data:image/png;base64,not-valid",
        )]
        invalid.extend(([self.image] * 5, [{"type": "video_url", "url": "remote"}]))
        for enabled in (True, False):
            self.config["vision"] = {"enabled": enabled}
            for parts in invalid:
                with self.subTest(enabled=enabled, shape=str(parts)[:70]), self.assertRaises(ApiError):
                    self.request([{"role": "user", "content": parts}])
            for role in ("system", "assistant", "tool"):
                with self.subTest(role=role), self.assertRaises(ApiError):
                    self.request([{"role": role, "content": [self.image]}])
        self.opener.open.assert_not_called()

    def test_http_image_rejection_is_actionable_and_redacts_image_urls_tokens_and_api_key(self):
        token = "123456:private-telegram-token-secret"
        key = "private-model-api-key"
        encoded = self.image["image_url"]["url"].split(",")[1]
        detail = ("Модель не поддерживает изображения " + self.image["image_url"]["url"] + "\n" + encoded
                  + f" https://api.telegram.org/file/bot{token}/photo.png {token} {key}")
        for code in (400, 422):
            error = HTTPError("https://model.example/v1/chat/completions", code, "Bad image", {},
                              io.BytesIO(detail.encode("utf-8")))
            self.opener.open.side_effect = error
            with (self.subTest(code=code), patch.dict("os.environ", {"LLMOPENCHAT_API_KEY": key}),
                  self.assertRaises(ApiError) as raised):
                list(self.client.stream_chat(self.messages))
            message = str(raised.exception)
            self.assertIn(f"HTTP {code}", message)
            self.assertIn("Модель не приняла изображение", message)
            self.assertIn("--mmproj", message)
            self.assertIn("Модель не поддерживает изображения", message)
            for secret in (token, key, encoded, "data:image", "https://api.telegram.org"):
                self.assertNotIn(secret, message)
            self.assertLess(len(message), 2000)
            self.assertTrue(raised.exception.__suppress_context__)

    def test_plain_http_error_retains_status_and_detail_without_vision_warning(self):
        self.opener.open.side_effect = HTTPError("https://model.example/v1/chat/completions", 400, "error", {},
                                                io.BytesIO("Неверный параметр temperature".encode("utf-8")))
        with self.assertRaises(ApiError) as raised:
            list(self.client.stream_chat([{"role": "user", "content": "Text only"}]))
        self.assertIn("HTTP 400", str(raised.exception))
        self.assertIn("Неверный параметр temperature", str(raised.exception))
        self.assertNotIn("--mmproj", str(raised.exception))

    def test_stream_error_and_network_exception_never_echo_encoded_image_or_secret_url(self):
        token = "123456:private-telegram-token-secret"
        detail = f"Ошибка сервера https://api.telegram.org/bot{token}/getFile " + self.image["image_url"]["url"]
        self.opener.open.return_value = io.BytesIO(
            ("data: " + json.dumps({"error": {"message": detail}}, ensure_ascii=False) + "\n\n").encode("utf-8"))
        with self.assertRaises(ApiError) as raised:
            list(self.client.stream_chat(self.messages))
        self.assertIn("Ошибка сервера", str(raised.exception))
        for secret in (token, "data:image", self.image["image_url"]["url"].split(",")[1]):
            self.assertNotIn(secret, str(raised.exception))
        self.opener.open.side_effect = URLError(detail)
        with self.assertRaises(ApiError) as raised:
            list(self.client.stream_chat(self.messages))
        self.assertNotIn(token, str(raised.exception))
        self.assertNotIn("data:image", str(raised.exception))
        self.assertTrue(raised.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
