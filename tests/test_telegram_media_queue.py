"""Private Telegram media bursts are serialized without losing album items."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from llmopenchat import telegram
from llmopenchat._files import supported
from llmopenchat.api import StreamEvent
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.telegram_api import MAX_DOWNLOAD_BYTES, TelegramError

from tests.test_telegram import FakeAPI, USER, message_update


class BlockingMediaAPI(FakeAPI):
    def __init__(self):
        super().__init__()
        self.downloads = []
        self.download_started = threading.Event()
        self.release_download = threading.Event()

    @staticmethod
    def content(file_id):
        return file_id.encode("ascii") + b"\x00\xffbinary-photo"

    def download_document(self, file_id, *, max_bytes):
        with self.condition:
            self.downloads.append((file_id, max_bytes))
            self.condition.notify_all()
        if file_id == "photo-0":
            self.download_started.set()
            if not self.release_download.wait(5):
                raise TelegramError("Test download was not released")
        return self.content(file_id)


@unittest.skipUnless(supported(), "Secure attachment storage requires Windows or Linux")
class TelegramMediaQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-telegram-media-queue-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.whitelist = self.root / "whitelist.txt"
        self.whitelist.write_text("alice_user,bob_user", encoding="utf-8")
        self.api = BlockingMediaAPI()
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["backend"] = "external"
        self.bot = telegram.TelegramBot(self.api, config, self.root, self.whitelist,
                                        emit=lambda _: None)
        self.addCleanup(self.cleanup_workers)
        self.bot.handle_update(message_update("/id"))
        self.api.messages.clear()

    def cleanup_workers(self):
        self.api.release_download.set()
        self.bot.close()
        for user in self.bot.users.values():
            if user.worker is not None:
                user.worker.join(3)

    def upload_photo(self, index):
        update = message_update("")
        message = update["message"]
        message.pop("text")
        message["media_group_id"] = "album-123"
        message["photo"] = [{"file_id": f"photo-{index}", "width": 640, "height": 480}]
        self.bot.handle_update(update)

    def begin_album(self):
        self.upload_photo(0)
        self.assertTrue(self.api.download_started.wait(3), "First photo download did not start")
        user = self.bot.users[USER]
        self.assertTrue(user.lock.locked())
        self.upload_photo(1)
        self.upload_photo(2)
        return user

    def join_worker(self):
        user = self.bot.users[USER]
        user.worker.join(4)
        self.assertFalse(user.worker.is_alive(), "Private media queue did not finish")
        self.assertFalse(user.lock.locked(), "Private media queue left the chat locked")
        return user

    def attachment_metadata(self, user):
        return [json.loads(message["content"].split("[Вложение Telegram]\n", 1)[1])
                for message in user.messages
                if message["role"] == "user" and "[Вложение Telegram]\n" in message["content"]]

    def test_album_photos_are_saved_serially_in_order_and_text_remains_busy(self):
        with patch.object(telegram, "ApiClient") as model:
            self.begin_album()
            self.assertEqual(self.api.downloads, [("photo-0", MAX_DOWNLOAD_BYTES)])
            self.bot.handle_update(message_update("Этот текст не должен попасть в модель"))
            self.assertTrue(any("Предыдущий запрос ещё выполняется" in message["text"]
                                for message in self.api.messages))
            self.api.release_download.set()
            user = self.join_worker()
            model.assert_not_called()
        self.assertEqual(self.api.downloads, [(f"photo-{index}", MAX_DOWNLOAD_BYTES) for index in range(3)])
        metadata = self.attachment_metadata(user)
        self.assertEqual(len(metadata), 3)
        self.assertEqual(len({item["file_id"] for item in metadata}), 3)
        store = self.bot._media_store(user)
        for index, item in enumerate(metadata):
            self.assertEqual(item["media_type"], "photo")
            self.assertEqual(store.read(item["file_id"])[1], self.api.content(f"photo-{index}"))
        self.assertNotIn("Этот текст", json.dumps(user.messages, ensure_ascii=False))

    def assert_pending_album_cancelled(self, command):
        self.begin_album()
        self.bot.handle_update(message_update(command))
        self.api.release_download.set()
        user = self.join_worker()
        self.assertEqual(self.api.downloads, [("photo-0", MAX_DOWNLOAD_BYTES)])
        self.assertEqual(self.attachment_metadata(user), [])
        self.assertEqual(list(user.media_queue), [])
        # A later request must succeed without reviving the discarded album.
        self.upload_photo(9)
        user = self.join_worker()
        self.assertEqual([item[0] for item in self.api.downloads], ["photo-0", "photo-9"])
        metadata = self.attachment_metadata(user)
        self.assertEqual(len(metadata), 1)
        self.assertEqual(self.bot._media_store(user).read(metadata[0]["file_id"])[1], self.api.content("photo-9"))

    def test_cancel_discards_queued_photos_without_replaying_them_on_next_request(self):
        self.assert_pending_album_cancelled("/cancel")

    def test_quit_discards_queued_photos_and_releases_the_private_chat(self):
        self.assert_pending_album_cancelled("/quit")

    def test_access_revocation_stops_queue_before_any_later_download(self):
        self.begin_album()
        self.whitelist.write_text("bob_user", encoding="utf-8")
        self.api.release_download.set()
        user = self.join_worker()
        self.assertEqual(self.api.downloads, [("photo-0", MAX_DOWNLOAD_BYTES)])
        self.assertEqual(self.attachment_metadata(user), [])
        self.assertEqual(list(user.media_queue), [])
        self.assertIsNone(user.media_store)
        self.whitelist.write_text("alice_user,bob_user", encoding="utf-8")
        self.upload_photo(9)
        user = self.join_worker()
        self.assertEqual([item[0] for item in self.api.downloads], ["photo-0", "photo-9"])
        self.assertEqual(len(self.attachment_metadata(user)), 1)

    def test_shutdown_stops_pending_album_and_rejects_new_uploads(self):
        self.begin_album()
        close_thread = threading.Thread(target=self.bot.close, daemon=True)
        close_thread.start()
        self.assertTrue(self.bot.stopped.wait(3))
        self.api.release_download.set()
        close_thread.join(4)
        self.assertFalse(close_thread.is_alive(), "Bot shutdown did not finish")
        user = self.join_worker()
        self.assertEqual(self.api.downloads, [("photo-0", MAX_DOWNLOAD_BYTES)])
        self.assertEqual(self.attachment_metadata(user), [])
        self.assertEqual(list(user.media_queue), [])
        self.upload_photo(9)
        self.assertEqual(self.api.downloads, [("photo-0", MAX_DOWNLOAD_BYTES)])

    def test_media_queued_while_model_busy_is_processed_after_original_reply(self):
        started, released = threading.Event(), threading.Event()

        class BlockingClient:
            def cancel(self):
                released.set()

            def stream_chat(self, messages, tools=None, tool_choice=None):
                started.set()
                if not released.wait(5):
                    raise AssertionError("Test model was not released")
                yield StreamEvent("content", "Обычный ответ")
                yield StreamEvent("finish", "stop")

        self.addCleanup(released.set)
        with patch.object(telegram, "ApiClient", return_value=BlockingClient()) as model:
            self.bot.handle_update(message_update("Обычный вопрос"))
            self.assertTrue(started.wait(3), "Original model request did not start")
            self.upload_photo(1)
            self.upload_photo(2)
            self.assertEqual(self.api.downloads, [])
            released.set()
            user = self.join_worker()
            self.assertEqual(model.call_count, 1)
        self.assertEqual([item[0] for item in self.api.downloads], ["photo-1", "photo-2"])
        self.assertEqual(len(self.attachment_metadata(user)), 2)
        self.assertEqual(user.messages[0 if user.messages[0]["role"] != "system" else 1]["content"],
                         "Обычный вопрос")
        texts = [message["text"] for message in self.api.messages]
        answer_index = next(index for index, text in enumerate(texts) if "Обычный ответ" in text)
        first_saved = next(index for index, text in enumerate(texts) if "Файл сохранён" in text)
        self.assertLess(answer_index, first_saved)


if __name__ == "__main__":
    unittest.main()
