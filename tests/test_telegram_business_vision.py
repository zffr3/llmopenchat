"""Authorized Business image inputs, bounded queues and private vision history."""

from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from llmopenchat.api import StreamEvent
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session
from llmopenchat.telegram import BotRuntime, TelegramBot
from llmopenchat.telegram_api import TelegramError
from llmopenchat.telegram_business import BusinessResponder
from llmopenchat.vision import VisionSettings, content_text, has_images

from tests.test_telegram_business import CONNECTION, CONTACT, OWNER, FakeAPI, connection_update, message_update


def picture(color="red"):
    output = io.BytesIO()
    Image.new("RGB", (80, 40), color).save(output, format="PNG")
    return output.getvalue()


def photo_update(caption="Что на фото?", *, identifier=CONNECTION, contact=CONTACT, message_id=1):
    update = message_update(identifier=identifier, contact=contact, message_id=message_id)
    message = update["business_message"]
    message.pop("text")
    message["photo"] = [
        {"file_id": "medium-file", "width": 640, "height": 480, "file_size": 100},
        {"file_id": "largest-file", "width": 1280, "height": 960, "file_size": 200},
        {"file_id": "small-file", "width": 100, "height": 80, "file_size": 50},
    ]
    if caption is not None:
        message["caption"] = caption
    return update


class VisionClient:
    def __init__(self, requests):
        self.requests = requests
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def stream_chat(self, messages, tools=None, tool_choice=None):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools), tool_choice))
        yield StreamEvent("reasoning", "PRIVATE REASONING")
        yield StreamEvent("content", "На изображении красная область.")
        yield StreamEvent("finish", "stop")


class BusinessVisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-business-vision-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["backend"] = "external"
        self.config["telegram"].update(admin_user_id=OWNER, business_enabled=True)
        self.config["vision"] = {"enabled": True}
        self.api = FakeAPI()
        self.api.download_file = Mock(return_value=picture())
        self.requests = []
        self.emitted = []
        client_patch = patch("llmopenchat.telegram_business.ApiClient",
                             side_effect=lambda config: VisionClient(self.requests))
        self.make_client = client_patch.start()
        self.addCleanup(client_patch.stop)
        self.responder = self.new_responder()

    def new_responder(self, config=None):
        config = self.config if config is None else config
        runtime = BotRuntime(config, self.root, self.emitted.append)
        responder = BusinessResponder(self.api, config, self.root, runtime, self.emitted.append)
        self.addCleanup(responder.close)
        return responder

    def join(self, responder=None):
        responder = responder or self.responder
        with responder.lock:
            worker = responder._worker
        if worker is not None:
            worker.join(4)
            self.assertFalse(worker.is_alive(), "Business image worker did not finish")

    def send(self, update, responder=None):
        responder = responder or self.responder
        responder.handle_update(update)
        self.join(responder)

    def test_caption_and_largest_photo_reach_model_as_image_parts_without_telegram_secrets(self):
        self.send(photo_update())
        self.api.download_file.assert_called_once_with(
            "largest-file", max_bytes=VisionSettings.from_config(self.config).max_input_bytes,
        )
        messages, tools, choice = self.requests[0]
        content = messages[-1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "Что на фото?"})
        self.assertTrue(has_images(content))
        image_url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        self.assertTrue(image_url.startswith("data:image/"))
        with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as image:
            self.assertEqual(image.size, (80, 40))
        self.assertNotIn("api.telegram.org", json.dumps(messages))
        self.assertNotIn("largest-file", json.dumps(messages))
        self.assertIsNone(tools)
        self.assertIsNone(choice)
        self.assertEqual(len(self.api.messages), 1)
        self.assertEqual(self.api.messages[0]["business_connection_id"], CONNECTION)
        self.assertNotIn("PRIVATE REASONING", self.api.messages[0]["text"])
        chat = self.responder.chats[(CONNECTION, CONTACT)]
        self.assertEqual(load_session(chat.directory / "current.json"), chat.messages)

    def test_photo_without_caption_uses_default_visual_prompt(self):
        self.send(photo_update(None))
        content = self.requests[0][0][-1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Опиши изображение."})
        self.assertTrue(has_images(content))

    def test_image_sent_as_document_is_a_vision_input(self):
        update = photo_update("Прочитай изображение")
        message = update["business_message"]
        message.pop("photo")
        message["document"] = {"file_id": "document-image", "file_name": "picture.PNG",
                               "mime_type": "image/png", "file_size": len(picture())}
        self.send(update)
        self.assertEqual(self.api.download_file.call_args.args, ("document-image",))
        self.assertTrue(has_images(self.requests[0][0][-1]["content"]))

    def test_disabled_vision_keeps_text_warning_and_never_downloads(self):
        config = copy.deepcopy(self.config)
        config["vision"]["enabled"] = False
        responder = self.new_responder(config)
        self.send(photo_update(), responder)
        self.api.download_file.assert_not_called()
        content = self.requests[0][0][-1]["content"]
        self.assertIsInstance(content, str)
        self.assertIn("Что на фото?", content)
        self.assertIn("недоступно", content)
        self.assertFalse(has_images(content))

    def test_bad_bytes_oversize_metadata_and_download_failure_reply_safely_without_model(self):
        limit = VisionSettings.from_config(self.config).max_input_bytes
        cases = [(b"not an image", None), (picture(), limit + 1),
                 (TelegramError("https://api.telegram.org/file/botSECRET_TOKEN/private.png"), None)]
        for number, (result, size) in enumerate(cases, 1):
            with self.subTest(number=number):
                self.api.download_file.reset_mock()
                self.api.download_file.side_effect = result if isinstance(result, Exception) else None
                self.api.download_file.return_value = result
                update = photo_update(message_id=number)
                if size is not None:
                    for photo in update["business_message"]["photo"]:
                        photo["file_size"] = size
                self.send(update)
                self.assertEqual(self.requests, [])
                self.assertEqual(self.api.messages[-1]["business_connection_id"], CONNECTION)
                self.assertEqual(self.api.messages[-1]["chat_id"], CONTACT)
                self.assertIn("Не удалось обработать изображение", self.api.messages[-1]["text"])
                if size is not None:
                    self.api.download_file.assert_not_called()
        self.assertNotIn("SECRET_TOKEN", str(self.api.messages) + str(self.emitted))
        self.make_client.assert_not_called()

    def test_unauthorized_connections_do_not_download_any_image(self):
        for number, options in enumerate(({"owner": OWNER + 1}, {"can_reply": False}, {"enabled": False})):
            identifier = f"unauthorized-{number}"
            self.responder.handle_update(connection_update(identifier, **options))
            self.send(photo_update(identifier=identifier))
        self.api.download_file.assert_not_called()
        self.make_client.assert_not_called()
        self.assertEqual(self.api.messages, [])

    def test_rights_revoked_during_download_suppress_model_and_reply_and_clear_pending_images(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def download(*args, **kwargs):
            started.set()
            if not release.wait(3):
                raise AssertionError("Download was not released")
            return picture()

        self.api.download_file.side_effect = download
        self.responder.handle_update(photo_update())
        self.assertTrue(started.wait(2), "Image download did not start")
        self.responder.handle_update(photo_update(message_id=2))
        self.responder.handle_update(connection_update(can_reply=False))
        release.set()
        self.join()
        self.make_client.assert_not_called()
        self.assertEqual(self.requests, [])
        self.assertEqual(self.api.messages, [])
        self.assertFalse(self.responder.chats[(CONNECTION, CONTACT)].pending)
        self.api.download_file.assert_called_once()

    def test_pending_images_are_descriptors_and_revocation_before_execution_prevents_download(self):
        self.responder.handle_update(connection_update())
        with self.responder.runtime.lock, self.responder.lock:
            self.responder.handle_update(photo_update())
            chat = self.responder.chats[(CONNECTION, CONTACT)]
            queued = chat.pending[0]
            self.assertEqual(queued[2]["file_id"], "largest-file")
            self.assertNotIn("data:image", json.dumps(queued))
            self.responder.handle_update(connection_update(enabled=False))
        self.join()
        self.api.download_file.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_business_image_is_not_shared_with_another_contact_connection_or_private_chat(self):
        whitelist = self.root / "whitelist.txt"
        whitelist.write_text("", encoding="utf-8")
        bot = TelegramBot(self.api, self.config, self.root, whitelist, emit=self.emitted.append)
        self.addCleanup(bot.close)
        private = bot.user_chat(OWNER)
        private.messages.append({"role": "user", "content": "PRIVATE BOT HISTORY"})
        bot.handle_update(photo_update("SECRET BUSINESS IMAGE"))
        self.join(bot.business)
        bot.handle_update(message_update("Другой контакт", contact=303))
        self.join(bot.business)
        bot.handle_update(message_update("Другое подключение", identifier="another-connection"))
        self.join(bot.business)
        self.assertNotIn("PRIVATE BOT HISTORY", str(self.requests))
        self.assertNotIn("SECRET BUSINESS IMAGE", str(private.messages))
        self.assertTrue(has_images(self.requests[0][0][-1]["content"]))
        for messages, tools, choice in self.requests[1:]:
            self.assertFalse(any(has_images(message["content"]) for message in messages))
            self.assertNotIn("SECRET BUSINESS IMAGE", str(messages))
        self.assertNotIn(CONTACT, bot.users)

    def test_history_keeps_only_configured_recent_images_and_reloads_multimodal_turns(self):
        self.responder.runtime.config["vision"]["max_history_images"] = 1
        self.send(photo_update("Первое изображение", message_id=1))
        self.send(photo_update("Второе изображение", message_id=2))
        chat = self.responder.chats[(CONNECTION, CONTACT)]
        self.assertEqual(sum(has_images(message["content"]) for message in chat.messages), 1)
        self.assertIn("Первое изображение", content_text(chat.messages[1]["content"]))
        saved = load_session(chat.directory / "current.json")
        self.assertEqual(saved, chat.messages)
        self.responder.close()
        restarted = self.new_responder()
        self.send(message_update("Продолжи", message_id=3), restarted)
        self.assertEqual(sum(has_images(message["content"]) for message in self.requests[-1][0]), 1)

    def test_disabling_vision_redacts_saved_images_from_next_request_without_destroying_history(self):
        self.send(photo_update("Сохрани контекст", message_id=1))
        chat = self.responder.chats[(CONNECTION, CONTACT)]
        self.assertTrue(any(has_images(message["content"]) for message in chat.messages))
        self.responder.runtime.config["vision"]["enabled"] = False
        self.send(message_update("Продолжим текстом", message_id=2))
        outbound = self.requests[-1][0]
        self.assertFalse(any(has_images(message["content"]) for message in outbound))
        self.assertNotIn("data:image/", json.dumps(outbound))
        self.assertIn("Сохрани контекст", str(outbound))
        self.assertTrue(any(has_images(message["content"]) for message in chat.messages))
        self.api.download_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
