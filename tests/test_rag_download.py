"""Telegram RAG uploads use bounded, official downloads without leaking tokens."""

import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from llmopenchat.telegram_api import TelegramAPI, TelegramError


class RagDownloadTests(unittest.TestCase):
    def setUp(self):
        self.opener = Mock()
        with patch("llmopenchat.telegram_api.build_opener", return_value=self.opener):
            self.api = TelegramAPI("123456:test-secret")

    def metadata(self, **values):
        return io.BytesIO(json.dumps({"ok": True, "result": values}).encode())

    def test_download_uses_official_endpoint_and_checks_size(self):
        self.opener.open.side_effect = [self.metadata(file_path="documents/file_1.txt", file_size=3), io.BytesIO(b"abc")]
        self.assertEqual(self.api.download_document("id", max_bytes=4), b"abc")
        first, second = self.opener.open.call_args_list
        self.assertEqual(json.loads(first.args[0].data), {"file_id": "id"})
        self.assertEqual(second.args[0].full_url, "https://api.telegram.org/file/bot123456:test-secret/documents/file_1.txt")
        self.assertEqual(second.args[0].get_method(), "GET")

    def test_untrusted_metadata_cannot_change_host_or_escape_path(self):
        for path in ("../secrets", "/document.txt", "documents/../file", "https://evil.test/x", "docs//file", "docs/file?x=1", "docs/file\\x"):
            with self.subTest(path=path):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = [self.metadata(file_path=path)]
                with self.assertRaises(TelegramError):
                    self.api.download_document("id")
                self.assertEqual(self.opener.open.call_count, 1)

    def test_size_is_bounded_even_if_metadata_omits_it(self):
        self.opener.open.side_effect = [self.metadata(file_path="docs/file.txt"), io.BytesIO(b"abcde")]
        with self.assertRaisesRegex(TelegramError, "размер"):
            self.api.download_document("id", max_bytes=4)

    def test_oversize_metadata_does_not_download(self):
        self.opener.open.side_effect = [self.metadata(file_path="docs/file.txt", file_size=5)]
        with self.assertRaises(TelegramError):
            self.api.download_document("id", max_bytes=4)
        self.assertEqual(self.opener.open.call_count, 1)

    def test_truncated_download_and_transport_errors_are_safe(self):
        self.opener.open.side_effect = [self.metadata(file_path="docs/file.txt", file_size=4), io.BytesIO(b"ab")]
        with self.assertRaisesRegex(TelegramError, "не полностью"):
            self.api.download_document("id")
        self.opener.open.side_effect = [self.metadata(file_path="docs/file.txt"), URLError("123456:test-secret")]
        with self.assertRaises(TelegramError) as raised:
            self.api.download_document("id")
        self.assertNotIn("test-secret", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
