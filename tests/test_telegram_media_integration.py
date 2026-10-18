"""Telegram media uploads and approved binary-download/send model rounds."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from llmopenchat import telegram
from llmopenchat._files import supported
from llmopenchat.api import StreamEvent
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session
from llmopenchat.telegram_api import MAX_DOWNLOAD_BYTES

from tests.test_telegram import FakeAPI, OTHER, USER, callback_update, message_update, tool_call


URL = "https://files.example.com/report.pdf"
PDF = b"%PDF-1.7\n\x00\xffbinary report\n%%EOF"


class MediaAPI(FakeAPI):
    def __init__(self):
        super().__init__()
        self.downloads = []
        self.media = []
        self.content = b"Uploaded content is cobalt-71."

    def download_document(self, file_id, *, max_bytes):
        self.downloads.append((file_id, max_bytes))
        return self.content

    def send_media(self, chat_id, filename, content, *, media_type="document", caption=""):
        with self.condition:
            result = {"chat_id": chat_id, "filename": filename, "content": content,
                      "media_type": media_type, "caption": caption, "message_id": 900 + len(self.media)}
            self.media.append(result)
            self.condition.notify_all()
            return {"message_id": result["message_id"]}


class ModelClient:
    """Real conversation loop with deterministic model events and no model network."""

    def __init__(self, action=None, *, final_answer="Готово, файл обработан."):
        self.action = action
        self.final_answer = final_answer
        self.requests = []
        self.tool_choices = []
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def stream_chat(self, messages, tools=None, tool_choice=None):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        self.tool_choices.append(tool_choice)
        number = len(self.requests)
        action = self.action(number, messages) if self.action is not None else None
        if action is not None:
            name, arguments = action
            yield StreamEvent("finish", "tool_calls")
            yield StreamEvent("tool_call", tool_call(f"media-call-{number}", name,
                                                     json.dumps(arguments, ensure_ascii=False)))
        else:
            if self.final_answer:
                yield StreamEvent("content", self.final_answer)
            yield StreamEvent("finish", "stop")
        yield StreamEvent("done", {})


def download_then_send(number, messages):
    if number == 1:
        return "http_download", {"url": URL, "filename": "report.pdf",
                                 "headers": {"Authorization": "Bearer explicit-download"}}
    if number == 2:
        result = json.loads(messages[-1]["content"])
        if result["status"] == "ok":
            return "telegram_send_file", {"file_id": result["file_id"], "caption": "Ваш отчёт"}
    return None


class MediaTestCase(unittest.TestCase):
    shared = False

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-telegram-media-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.whitelist = self.root / "whitelist.txt"
        self.whitelist.write_text("alice_user,bob_user", encoding="utf-8")
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["backend"] = "external"
        if self.shared:
            self.config["telegram"] = {"admin_user_id": USER, "repository": None}
        self.api = MediaAPI()
        self.bot = telegram.TelegramBot(self.api, self.config, self.root, self.whitelist,
                                        emit=lambda _: None)
        self.addCleanup(self.bot.close)
        for user_id in (USER, OTHER):
            self.send("/id", user_id)
        self.api.messages.clear()

    def join(self, user_id=USER):
        user = self.bot.users.get(user_id)
        if user is not None and user.worker is not None:
            user.worker.join(timeout=4)
            self.assertFalse(user.worker.is_alive(), "Media request worker did not finish")
        return user

    def send(self, text, user_id=USER, *, wait=True):
        self.bot.handle_update(message_update(text, user_id))
        return self.join(user_id) if wait else None

    def upload(self, kind, media, *, caption=None, user_id=USER, wait=True):
        update = message_update("", user_id)
        update["message"].pop("text")
        update["message"][kind] = media
        if caption is not None:
            update["message"]["caption"] = caption
        self.bot.handle_update(update)
        return self.join(user_id) if wait else None

    def document(self, filename="archive.zip", file_id="telegram-upload"):
        return {"file_id": file_id, "file_name": filename, "file_size": len(self.api.content),
                "mime_type": "application/zip" if filename.endswith(".zip") else "text/plain"}

    def metadata(self, user):
        text = next(item["content"] for item in reversed(user.messages) if item["role"] == "user")
        return json.loads(text.split("[Вложение Telegram]\n", 1)[1])

    def approval(self, chat_id=USER):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            with self.bot.control_lock:
                ready = [pending for pending in self.bot.approvals.values()
                         if pending.chat_id == chat_id and pending.message_id is not None
                         and not pending.event.is_set()]
                if ready:
                    message_id = ready[-1].message_id
                    return next(item for item in self.api.messages if item["message_id"] == message_id)
            with self.api.condition:
                self.api.condition.wait(timeout=0.01)
        self.fail("No media approval was registered")

    def decide(self, message, *, user_id=USER, allow=True):
        data = message["reply_markup"]["inline_keyboard"][0][0 if allow else 1]["callback_data"]
        self.bot.handle_update(callback_update(data, message["message_id"], user_id))

    def download_response(self):
        return {"url": URL, "status_code": 200, "content": PDF, "content_type": "application/pdf",
                "headers": {"content-type": "application/pdf"}, "filename": "report.pdf"}

    def results(self, user):
        return [json.loads(item["content"]) for item in load_session(user.directory / "current.json")
                if item["role"] == "tool"]


@unittest.skipUnless(supported(), "Secure media storage requires Windows or Linux")
class TelegramMediaIntegrationTests(MediaTestCase):
    def test_caption_reaches_model_with_opaque_metadata_and_bytes_stay_out_of_history(self):
        self.api.content = b"\x00\xffprivate-binary-payload"
        client = ModelClient()
        with patch.object(telegram, "ApiClient", return_value=client):
            user = self.upload("document", self.document(), caption="Верни этот файл")
        self.assertEqual(len(client.requests), 1)
        prompt = client.requests[0][0][-1]["content"]
        self.assertIn("Верни этот файл", prompt)
        metadata = self.metadata(user)
        self.assertEqual(metadata["filename"], "archive.zip")
        self.assertEqual(metadata["media_type"], "document")
        self.assertEqual(metadata["size"], len(self.api.content))
        self.assertNotEqual(metadata["file_id"], "telegram-upload")
        self.assertNotIn("private-binary-payload", json.dumps(user.messages))
        stored, content = self.bot._media_store(user).read(metadata["file_id"])
        self.assertEqual(content, self.api.content)
        self.assertEqual(stored["file_id"], metadata["file_id"])
        self.assertEqual(self.api.downloads, [("telegram-upload", MAX_DOWNLOAD_BYTES)])
        names = {tool["function"]["name"] for tool in client.requests[0][1]}
        self.assertIn("telegram_send_file", names)

    def test_uncaptioned_media_types_are_saved_without_starting_inference(self):
        with patch.object(telegram, "ApiClient") as model:
            for kind, extension in (("document", ".zip"), ("video", ".mp4"), ("audio", ".mp3"),
                                    ("voice", ".ogg"), ("animation", ".gif"),
                                    ("video_note", ".mp4"), ("sticker", ".webp")):
                with self.subTest(kind=kind):
                    media = self.document("upload" + extension, "tg-" + kind)
                    user = self.upload(kind, media)
                    metadata = self.metadata(user)
                    self.assertEqual(metadata["media_type"], kind)
                    self.assertEqual(self.bot._media_store(user).read(metadata["file_id"])[1], self.api.content)
                    self.assertIn(metadata["file_id"], self.api.messages[-1]["text"])
            model.assert_not_called()

    def test_largest_photo_variant_is_downloaded_and_available_to_followup_prompt(self):
        photos = [{"file_id": "middle", "width": 640, "height": 480, "file_size": 20},
                  {"file_id": "largest", "width": 1280, "height": 960, "file_size": 40},
                  {"file_id": "small", "width": 160, "height": 120, "file_size": 10}]
        with patch.object(telegram, "ApiClient") as model:
            user = self.upload("photo", photos)
            model.assert_not_called()
        metadata = self.metadata(user)
        self.assertEqual(metadata["media_type"], "photo")
        self.assertEqual(self.api.downloads, [("largest", MAX_DOWNLOAD_BYTES)])
        client = ModelClient()
        with patch.object(telegram, "ApiClient", return_value=client):
            self.send("Отправь предыдущее фото")
        self.assertIn(metadata["file_id"], json.dumps(client.requests[0][0]))

    def test_private_supported_document_keeps_automatic_rag_import(self):
        user = self.upload("document", self.document("notes.txt"))
        self.assertIn("notes", user.rag.command("list"))
        self.assertFalse(user.rag.enabled)
        self.assertIn("cobalt", user.rag.command("search cobalt"))
        self.assertNotIn("notes", self.bot.user_chat(OTHER).rag.command("list"))

    def test_uploaded_photo_can_be_returned_without_enabling_repository_or_http(self):
        user = self.upload("photo", [{"file_id": "photo", "width": 100, "height": 100}])
        identifier = self.metadata(user)["file_id"]
        arguments = {"file_id": identifier, "media_type": "photo", "caption": "Ваше фото"}
        client = ModelClient(lambda number, _: ("telegram_send_file", arguments) if number == 1 else None)
        with patch.object(telegram, "ApiClient", return_value=client):
            self.send("Пришли предыдущее фото", wait=False)
            self.assertEqual(self.api.media, [])
            self.decide(self.approval())
            self.join()
        self.assertFalse(user.settings.http_enabled)
        self.assertFalse(user.settings.code_enabled)
        self.assertEqual(len(self.api.media), 1)
        self.assertEqual(self.api.media[0]["media_type"], "photo")
        self.assertEqual(self.api.media[0]["caption"], "Ваше фото")
        self.assertEqual(self.api.media[0]["content"], self.api.content)
        self.assertEqual(self.results(user)[-1]["status"], "ok")

    def test_get_download_then_send_are_separately_approved_and_saved_as_tool_rounds(self):
        self.send("/tools http on")
        client = ModelClient(download_then_send)
        with (patch.object(telegram, "ApiClient", return_value=client),
              patch("llmopenchat.telegram_media._download_public", return_value=self.download_response()) as download):
            self.send("Скачай PDF и пришли сюда", wait=False)
            first = self.approval()
            self.assertIn("HTTP GET", first["text"])
            self.assertIn(URL, first["text"])
            download.assert_not_called()
            self.decide(first)
            second = self.approval()
            self.assertNotEqual(second["message_id"], first["message_id"])
            self.assertIn("report.pdf", second["text"])
            self.assertIn("Ваш отчёт", second["text"])
            self.assertEqual(self.api.media, [])
            self.decide(second)
            user = self.join()
        self.assertEqual(len(client.requests), 3)
        download.assert_called_once()
        self.assertEqual(self.api.media, [{"chat_id": USER, "filename": "report.pdf", "content": PDF,
                                         "media_type": "document", "caption": "Ваш отчёт", "message_id": 900}])
        self.assertEqual([result["status"] for result in self.results(user)], ["ok", "ok"])
        history = load_session(user.directory / "current.json")
        self.assertEqual([item["role"] for item in history],
                         ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])
        self.assertNotIn("binary report", json.dumps(history))

    def test_denied_download_does_not_fetch_or_send_and_model_receives_denial(self):
        self.send("/tools http on")
        client = ModelClient(download_then_send)
        with (patch.object(telegram, "ApiClient", return_value=client),
              patch("llmopenchat.telegram_media._download_public") as download):
            self.send("Скачай и отправь PDF", wait=False)
            self.decide(self.approval(), allow=False)
            user = self.join()
        download.assert_not_called()
        self.assertEqual(self.api.media, [])
        self.assertEqual(self.results(user)[0]["status"], "denied")
        self.assertEqual(len(client.requests), 2)

    def test_explicit_filename_transfer_runs_even_when_rag_has_no_matching_documents(self):
        self.send("/tools http on")
        user = self.send("/rag on")
        self.assertTrue(user.rag.enabled)
        self.assertEqual(user.rag.search("report.pdf"), [])
        client = ModelClient(download_then_send)
        with (patch.object(telegram, "ApiClient", return_value=client),
              patch.object(user.rag, "prepare", wraps=user.rag.prepare) as retrieve,
              patch("llmopenchat.telegram_media._download_public", return_value=self.download_response())):
            self.send("Пришли report.pdf", wait=False)
            self.decide(self.approval())
            self.decide(self.approval())
            self.join()
        retrieve.assert_not_called()
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(self.api.media[0]["content"], PDF)
        self.assertEqual([result["status"] for result in self.results(user)], ["ok", "ok"])

    def test_download_and_save_with_code_and_rag_uses_media_tools_and_empty_reply_gets_delivery_ack(self):
        self.send("/tools code on")
        self.send("/tools http on")
        user = self.send("/rag on")
        client = ModelClient(download_then_send, final_answer="")
        with (patch.object(telegram, "ApiClient", return_value=client),
              patch("llmopenchat.telegram_media._download_public", return_value=self.download_response())):
            self.send("Скачай и сохрани файл " + URL, wait=False)
            self.decide(self.approval())
            self.decide(self.approval())
            self.join()
        self.assertTrue(user.settings.code_enabled)
        self.assertTrue(user.rag.enabled)
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(client.tool_choices, [None, None, None])
        self.assertEqual(user.messages[-1], {"role": "assistant", "content": "Файл отправлен."})
        self.assertEqual(self.api.media[0]["content"], PDF)
        self.assertTrue(any("Файл отправлен." in item["text"] for item in self.api.messages))
        self.assertFalse(any("Ошибка" in item["text"] or "Файлы не созданы" in item["text"]
                             for item in self.api.messages))

    def test_cancel_pending_send_preserves_complete_denied_round_without_delivery(self):
        user = self.upload("document", self.document())
        identifier = self.metadata(user)["file_id"]
        client = ModelClient(lambda number, _: ("telegram_send_file", {"file_id": identifier}) if number == 1 else None)
        with patch.object(telegram, "ApiClient", return_value=client):
            self.send("Пришли вложение", wait=False)
            approval = self.approval()
            self.send("/cancel", wait=False)
            self.join()
            self.decide(approval)
        self.assertEqual(self.api.media, [])
        self.assertEqual(self.results(user)[-1]["status"], "denied")
        self.assertFalse(self.bot.approvals)
        self.assertEqual(len(client.requests), 1)

    def test_approved_download_does_not_authorize_subsequent_send(self):
        self.send("/tools http on")
        client = ModelClient(download_then_send)
        with (patch.object(telegram, "ApiClient", return_value=client),
              patch("llmopenchat.telegram_media._download_public", return_value=self.download_response()) as download):
            self.send("Скачай и перешли отчёт", wait=False)
            self.decide(self.approval())
            send_approval = self.approval()
            download.assert_called_once()
            self.assertEqual(self.api.media, [])
            self.decide(send_approval, allow=False)
            user = self.join()
        self.assertEqual(self.api.media, [])
        self.assertEqual([result["status"] for result in self.results(user)], ["ok", "denied"])
        self.assertEqual(len(client.requests), 3)

    def test_private_attachment_id_cannot_be_used_by_another_user(self):
        alice = self.upload("document", self.document())
        identifier = self.metadata(alice)["file_id"]
        client = ModelClient(lambda number, _: ("telegram_send_file", {"file_id": identifier}) if number == 1 else None)
        with patch.object(telegram, "ApiClient", return_value=client):
            bob = self.send("Пришли файл", OTHER)
        self.assertEqual(self.api.media, [])
        self.assertEqual(self.results(bob)[-1]["status"], "error")
        self.assertNotIn(identifier, json.dumps(bob.messages[0]))
        self.assertFalse(self.bot.approvals)

    def test_non_whitelisted_sender_cannot_upload_or_create_media_storage(self):
        update = message_update("", USER, username="outsider")
        update["message"].pop("text")
        update["message"]["document"] = self.document()
        self.bot.handle_update(update)
        self.assertEqual(self.api.downloads, [])
        self.assertNotIn(USER, self.bot.users)


@unittest.skipUnless(supported(), "Secure media storage requires Windows or Linux")
class SharedTelegramMediaIntegrationTests(MediaTestCase):
    shared = True

    def test_client_upload_caption_uses_shared_history_without_mutating_rag(self):
        client = ModelClient()
        with patch.object(telegram, "ApiClient", return_value=client):
            user = self.upload("document", self.document("client-notes.txt"),
                               caption="Используй этот файл", user_id=OTHER)
        self.assertIs(user, self.bot.user_chat(USER))
        self.assertEqual(self.metadata(user)["filename"], "client-notes.txt")
        self.assertNotIn("client-notes", user.rag.command("list"))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(self.api.messages[-1]["chat_id"], OTHER)
        self.upload("document", self.document("admin-notes.txt"), user_id=USER)
        self.assertIn("admin-notes", user.rag.command("list"))

    def test_admin_approves_each_client_download_and_send_but_client_receives_file(self):
        self.send("/tools http on", USER)
        client = ModelClient(download_then_send)
        with (patch.object(telegram, "ApiClient", return_value=client),
              patch("llmopenchat.telegram_media._download_public", return_value=self.download_response()) as download):
            self.send("Скачай PDF и отправь мне", OTHER, wait=False)
            first = self.approval(USER)
            self.assertIn(str(OTHER), first["text"])
            self.decide(first, user_id=OTHER)
            download.assert_not_called()
            self.decide(first, user_id=USER)
            second = self.approval(USER)
            self.assertEqual(self.api.media, [])
            self.decide(second, user_id=OTHER)
            self.assertEqual(self.api.media, [])
            self.decide(second, user_id=USER)
            user = self.join(OTHER)
        self.assertEqual(len(self.api.media), 1)
        self.assertEqual(self.api.media[0]["chat_id"], OTHER)
        self.assertEqual(self.api.media[0]["content"], PDF)
        self.assertEqual([result["status"] for result in self.results(user)], ["ok", "ok"])


if __name__ == "__main__":
    unittest.main()
