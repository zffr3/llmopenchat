"""Real Telegram -> conversation -> HTTP JSON vision requests, without paid APIs."""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from PIL import Image

from llmopenchat.history import load_session
from llmopenchat.telegram_api import MAX_DOWNLOAD_BYTES
from llmopenchat.vision import content_text, has_images
from tests.test_api import sse_record
from tests.test_telegram import USER, OTHER
from tests.test_telegram_media_integration import MediaTestCase


def picture(color="red", format="PNG", size=(120, 80)):
    result = io.BytesIO()
    Image.new("RGB", size, color).save(result, format=format)
    return result.getvalue()


class VisionTelegramTests(MediaTestCase):
    def setUp(self):
        super().setUp()
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(sse_record({"choices": [{"index": 0, "delta": {"content": "Изображение получено."}, "finish_reason": "stop"}]}))
                self.wfile.write(sse_record("[DONE]"))

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(target=self.server.serve_forever,
                                              kwargs={"poll_interval": 0.01}, daemon=True)
        self.server_thread.start()
        def close_server():
            self.server.shutdown()
            self.server.server_close()
            self.server_thread.join(3)
        self.addCleanup(close_server)
        self.bot.runtime.config.update(base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
                                       backend="external", request_extra={})
        self.bot.runtime.config["vision"]["enabled"] = True
        self.api.content = picture()

    def photo(self, caption="Оцени блюдо", user_id=USER):
        return self.upload("photo", [{"file_id": "telegram-photo-secret", "width": 120, "height": 80,
                                       "file_size": len(self.api.content)}], caption=caption, user_id=user_id)

    def image_content(self, request=-1):
        return next(message["content"] for message in reversed(self.requests[request]["messages"])
                    if message["role"] == "user" and has_images(message["content"]))

    def test_photo_pixels_and_caption_reach_actual_http_payload_without_telegram_credentials(self):
        user = self.photo()
        self.assertEqual(len(self.requests), 1)
        content = self.image_content()
        self.assertIn("Оцени блюдо", content_text(content))
        part = next(part for part in content if part["type"] == "image_url")
        header, payload = part["image_url"]["url"].split(",", 1)
        self.assertEqual(header, "data:image/jpeg;base64")
        with Image.open(io.BytesIO(base64.b64decode(payload, validate=True))) as image:
            self.assertEqual(image.size, (120, 80))
            self.assertGreater(image.getpixel((10, 10))[0], 200)
        serialized = json.dumps(self.requests)
        self.assertNotIn("api.telegram.org", serialized)
        self.assertNotIn("telegram-photo-secret", serialized)
        self.assertNotIn("tools", self.requests[0])
        self.assertNotIn("Telegram file delivery is available", serialized)
        self.assertEqual(load_session(user.directory / "current.json"), user.messages)

    def test_file_transfer_intent_keeps_tool_schemas_with_image_input(self):
        self.photo(caption="Отправь этот файл обратно")
        self.assertTrue(has_images(self.image_content()))
        self.assertIn("telegram_send_file", json.dumps(self.requests[-1]["tools"]))

    def test_photo_without_caption_generates_visual_answer_when_vision_enabled(self):
        user = self.photo(caption=None)
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(has_images(user.messages[-2]["content"]))
        self.assertIn("Рассмотри изображение", content_text(self.image_content()))

    def test_supported_image_documents_use_actual_bytes_even_if_mime_is_generic(self):
        for format, extension in (("JPEG", "jpg"), ("PNG", "png"), ("WEBP", "webp")):
            with self.subTest(format=format):
                self.api.content = picture(format=format)
                document = self.document("meal." + extension)
                document["mime_type"] = "application/octet-stream"
                self.upload("document", document, caption="Посмотри изображение")
                self.assertTrue(has_images(self.image_content()))
        self.assertEqual(len(self.requests), 3)

    def test_largest_photo_is_selected_and_personal_histories_do_not_leak(self):
        first = self.photo(user_id=USER)
        first_uri = self.image_content()[1]["image_url"]["url"]
        self.api.content = picture("blue")
        second = self.upload("photo", [{"file_id": "small", "width": 20, "height": 10},
                                       {"file_id": "largest", "width": 120, "height": 80}],
                             caption="Посмотри моё фото", user_id=OTHER)
        self.assertEqual(self.api.downloads[-1][0], "largest")
        self.assertNotIn(first_uri, json.dumps(self.requests[-1]))
        self.assertNotEqual(first.directory, second.directory)
        self.assertNotEqual(self.image_content()[1]["image_url"]["url"], first_uri)

    def test_disabled_vision_keeps_file_delivery_and_explicit_metadata_warning(self):
        self.bot.runtime.config["vision"]["enabled"] = False
        user = self.photo()
        self.assertEqual(len(self.requests), 1)
        serialized = json.dumps(self.requests[-1], ensure_ascii=False)
        self.assertNotIn("data:image", serialized)
        self.assertIn("содержимое изображения модели не передано", serialized)
        self.assertFalse(has_images(user.messages[-2]["content"]))
        self.assertEqual(self.api.downloads[-1][1], MAX_DOWNLOAD_BYTES)

    def test_invalid_bytes_and_input_limit_do_not_reach_model(self):
        self.api.content = b"not a JPEG, PNG or WebP"
        self.photo()
        self.assertFalse(self.requests)
        self.assertIn("Ошибка", self.api.messages[-1]["text"])
        self.api.downloads.clear()
        too_big = {"file_id": "oversize", "width": 1, "height": 1,
                   "file_size": self.bot.runtime.config["vision"]["max_input_bytes"] + 1}
        self.upload("photo", [too_big], caption="Что на фото?")
        self.assertFalse(self.api.downloads)
        self.assertFalse(self.requests)

    def test_rag_without_matches_does_not_block_pixels(self):
        user = self.bot.user_chat(USER)
        user.rag.enabled = True
        self.photo()
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(has_images(self.image_content()))

    def test_retention_save_load_and_text_mode_resume_are_bounded(self):
        self.bot.runtime.config["vision"]["max_history_images"] = 2
        for color in ("red", "green", "blue"):
            self.api.content = picture(color)
            user = self.photo()
        self.assertEqual(sum(has_images(m["content"]) for m in user.messages), 2)
        self.send("/save visual.json")
        self.send("/clear")
        self.send("/load visual.json")
        self.assertEqual(sum(has_images(m["content"]) for m in user.messages), 2)
        self.bot.runtime.config["vision"]["enabled"] = False
        self.send("Что помнишь?")
        payload = json.dumps(self.requests[-1], ensure_ascii=False)
        self.assertNotIn("data:image", payload)
        self.assertIn("Vision выключен", payload)
        self.assertLess((user.directory / "current.json").stat().st_size, 200_000)

    def test_revoked_access_after_download_prevents_image_storage_and_inference(self):
        actor = OTHER if self.shared else USER
        def revoke(*args, **kwargs):
            self.bot.usernames[actor] = None
            return self.api.content
        with patch.object(self.api, "download_document", side_effect=revoke):
            user = self.photo(user_id=actor)
        self.assertFalse(self.requests)
        self.assertIsNone(user.media_store)


class SharedVisionTelegramTests(VisionTelegramTests):
    shared = True

    def test_client_photo_uses_shared_vision_but_cannot_change_rag(self):
        user = self.bot.user_chat(OTHER)
        with patch.object(user.rag, "command") as rag:
            self.photo(user_id=OTHER)
        rag.assert_not_called()
        self.assertTrue(has_images(self.image_content()))
        self.assertTrue(any(item["chat_id"] == OTHER and "Изображение получено" in item["text"]
                            for item in self.api.messages))
        self.assertFalse(self.bot.approvals)

    def test_largest_photo_is_selected_and_personal_histories_do_not_leak(self):
        # Shared participants intentionally share images; private isolation is tested above.
        user = self.photo(user_id=OTHER)
        self.assertIs(user, self.bot.user_chat(USER))
        self.assertTrue(has_images(self.image_content()))
