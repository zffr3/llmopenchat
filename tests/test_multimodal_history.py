"""Multimodal snapshots, safe terminal previews and current-turn RAG queries."""

from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from llmopenchat import cli, history
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import list_sessions, load_session, save_session
from llmopenchat.rag import RagSession


def image_part():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (21, 87, 145)).save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


class MultimodalHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="llmopenchat-multimodal-history-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.image = image_part()
        self.parts = [{"type": "text", "text": "Что изображено на фотографии?"}, self.image]
        self.messages = [{"role": "system", "content": "Будь точен."},
                         {"role": "user", "content": self.parts},
                         {"role": "assistant", "content": "Ответ."}]

    def write(self, messages):
        path = self.root / "snapshot.json"
        path.write_text(json.dumps({"messages": messages}, ensure_ascii=False), encoding="utf-8")
        return path

    def test_multimodal_save_load_round_trip_retains_exact_image_and_only_protocol_fields(self):
        messages = copy.deepcopy(self.messages)
        messages[1]["approved"] = True
        path = self.root / "session.json"
        original = copy.deepcopy(messages)
        save_session(path, messages, "vision-local", target_model="Visual Model")
        loaded = load_session(path)
        self.assertEqual(loaded, self.messages)
        self.assertEqual(messages, original)
        self.assertEqual(loaded[1]["content"][1]["image_url"]["url"], self.image["image_url"]["url"])
        loaded[1]["content"][0]["text"] = "Изменённая копия"
        loaded[1]["content"][1]["image_url"]["url"] = "modified"
        self.assertEqual(load_session(path), self.messages)
        self.assertEqual(messages, original)

    def test_validation_copies_nested_multimodal_parts_without_sharing_caller_state(self):
        validated = history._validated_messages(self.messages)
        self.assertIsNot(validated[1]["content"], self.parts)
        self.assertIsNot(validated[1]["content"][0], self.parts[0])
        self.assertIsNot(validated[1]["content"][1]["image_url"], self.image["image_url"])
        validated[1]["content"][0]["text"] = "changed"
        self.assertEqual(self.parts[0]["text"], "Что изображено на фотографии?")

    def test_history_title_and_image_only_preview_never_contain_encoded_bytes(self):
        path = self.write(self.messages)
        title = list_sessions(self.root)[0].title
        self.assertIn("Что изображено", title)
        self.assertNotIn("data:image", title)
        self.assertNotIn(self.image["image_url"]["url"].split(",")[1], title)
        self.assertNotIn("image_url", title)
        save_session(path, [{"role": "user", "content": [self.image]}], "vision-local")
        title = list_sessions(self.root)[0].title
        self.assertIn("Изображение", title)
        self.assertNotIn("base64", title)

    def test_only_user_role_accepts_multimodal_parts(self):
        for role in ("system", "assistant", "tool"):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "Неверный формат диалога"):
                load_session(self.write([{"role": role, "content": self.parts}]))

    def test_malformed_remote_and_oversized_content_is_rejected_without_echoing_it(self):
        invalid = [
            [{"type": "image_url", "image_url": {"url": url}}]
            for url in ("https://example.com/private-photo.jpg", "file:///C:/private.jpg",
                        "https://api.telegram.org/file/bot123456:private-token-secret/photos/file.jpg",
                        "data:image/png;base64,%%%invalid%%%", "data:image/png;base64,YWJj")
        ]
        invalid.extend((
            [{"type": "image_url", "image_url": {"url": self.image["image_url"]["url"], "detail": "high"}}],
            [{"type": "text", "text": "hello", "permission": "always"}],
            [{"type": "audio", "data": "secret"}],
            [self.image] * 5,
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * (1400 * 1024)}}],
        ))
        for parts in invalid:
            with self.subTest(shape=str(parts)[:70]):
                path = self.write([{"role": "user", "content": parts}])
                with self.assertRaisesRegex(ValueError, "Неверный формат диалога") as raised:
                    load_session(path)
                self.assertNotIn("data:image", str(raised.exception))
                self.assertNotIn("private-token", str(raised.exception))
                notices = []
                self.assertEqual(list_sessions(self.root, emit=notices.append), [])
                self.assertEqual(len(notices), 1)
                self.assertNotIn("data:image", notices[0])
                self.assertNotIn("private-token", notices[0])

    def test_invalid_or_oversized_multimodal_save_keeps_prior_snapshot(self):
        path = self.write(self.messages)
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            save_session(path, [{"role": "user", "content": [self.image] * 5}], "vision-local")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(history.MAX_SESSION_BYTES, 16 * 1024 * 1024)
        with patch.object(history, "MAX_SESSION_BYTES", 256), self.assertRaisesRegex(ValueError, "16 МБ"):
            save_session(path, self.messages, "vision-local")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])
        with patch.object(history, "MAX_SESSION_BYTES", 256), self.assertRaisesRegex(ValueError, "16 МБ"):
            load_session(path)

    def test_resume_terminal_preview_renders_text_and_placeholder_only(self):
        rendered = []
        with (patch.object(cli, "console", Mock()), patch.object(cli, "say", side_effect=rendered.append),
              patch.object(cli, "ApiClient"), patch("llmopenchat.cli.sys.stdin.isatty", return_value=False),
              patch("builtins.input", side_effect=EOFError)):
            self.assertEqual(cli.chat(copy.deepcopy(DEFAULT_CONFIG), self.root, self.messages), 0)
        output = "\n".join(rendered)
        self.assertIn("Вы > Что изображено", output)
        self.assertIn("Изображение", output)
        self.assertNotIn("image_url", output)
        self.assertNotIn("base64", output)
        self.assertNotIn(self.image["image_url"]["url"].split(",")[1], output)

    def test_rag_uses_latest_multimodal_text_instead_of_previous_user_prompt(self):
        rag = RagSession(self.root, copy.deepcopy(DEFAULT_CONFIG))
        rag.enabled = True
        messages = [{"role": "user", "content": "Старый вопрос про звёзды"},
                    {"role": "assistant", "content": "Ответ"},
                    {"role": "user", "content": [{"type": "text", "text": "Новая проверка чертежа"}, self.image]}]
        with patch.object(rag, "search", return_value=[]) as search:
            rag.prepare(messages)
        query = search.call_args.args[0]
        self.assertIn("Новая проверка чертежа", query)
        self.assertNotIn("Старый вопрос", query)
        self.assertNotIn("base64", query)
        self.assertNotIn("image_url", query)
        messages[-1]["content"] = [self.image]
        with patch.object(rag, "search", return_value=[]) as search:
            rag.prepare(messages)
        self.assertNotIn("Старый вопрос", search.call_args.args[0])
        self.assertIn("Изображение", search.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
