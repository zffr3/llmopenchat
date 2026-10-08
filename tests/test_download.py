"""Real HTTP range/resume/error tests against an isolated loopback server."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import socket
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from llmopenchat import download


class RangeHandler(BaseHTTPRequestHandler):
    data = bytes(range(256)) * 32
    mode = "normal"
    requests = []
    attempts = {}
    guard = threading.Lock()

    def log_message(self, *args):
        pass

    def do_GET(self):
        selected = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", selected)
        if not match:
            self.send_error(400)
            return
        start, end = map(int, match.groups())
        with self.guard:
            type(self).requests.append((start, end))
            type(self).attempts[start] = type(self).attempts.get(start, 0) + 1
            attempt = type(self).attempts[start]
        if self.mode == "range416" or start >= len(self.data):
            self.send_error(416)
            return
        if self.mode == "fail_after_first" and start:
            self.send_error(503)
            return
        if self.mode == "ignore_range":
            status, body = 200, self.data
        else:
            status, body = 206, self.data[start:end + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            reported_start = start + 1 if self.mode == "wrong_range" else start
            self.send_header("Content-Range", f"bytes {reported_start}-{end}/{len(self.data)}")
        self.end_headers()
        try:
            if self.mode == "drop_once" and attempt == 1:
                self.wfile.write(body[:len(body) // 2])
                self.wfile.flush()
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            if self.mode == "corrupt_content":
                body = b"Z" * len(body)
            self.wfile.write(body)
        except OSError:
            pass


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/model"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="llmopenchat-download-test-")
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "model.gguf"
        self.partial = self.destination.with_name("model.gguf.partial")
        self.checkpoint = self.destination.with_name("model.gguf.partial.json")
        self.digest = hashlib.sha256(RangeHandler.data).hexdigest()
        self.emit = Mock()
        with RangeHandler.guard:
            RangeHandler.mode = "normal"
            RangeHandler.requests = []
            RangeHandler.attempts = {}
        for name, value in [("_CHUNK_SIZE", 1024), ("_MIN_CHUNK_SIZE", 1024), ("_TIMEOUT", 2), ("_RETRIES", 2)]:
            context = patch.object(download, name, value)
            context.start()
            self.addCleanup(context.stop)

    def run_download(self, workers=4):
        return download.download_file(self.url, self.destination, len(RangeHandler.data), self.digest, self.emit, workers)

    def test_parallel_ranges_produce_exact_file_and_remove_checkpoint(self):
        result = self.run_download()
        self.assertEqual(result, self.destination)
        self.assertEqual(result.read_bytes(), RangeHandler.data)
        self.assertEqual(set(RangeHandler.requests), {(i, i + 1023) for i in range(0, 8192, 1024)})
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.checkpoint.exists())

    def test_interrupted_ranges_retry_without_checkpointing_short_bytes(self):
        RangeHandler.mode = "drop_once"
        self.run_download()
        self.assertEqual(self.destination.read_bytes(), RangeHandler.data)
        self.assertTrue(all(count == 2 for count in RangeHandler.attempts.values()))

    def test_completed_ranges_resume_without_redownloading(self):
        RangeHandler.mode = "fail_after_first"
        with self.assertRaises(download.DownloadError):
            self.run_download(workers=1)
        saved = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(saved["completed"], [0])
        self.assertEqual(self.partial.stat().st_size, len(RangeHandler.data))
        with RangeHandler.guard:
            RangeHandler.mode = "normal"
            RangeHandler.requests = []
        self.run_download(workers=1)
        self.assertNotIn((0, 1023), RangeHandler.requests)
        self.assertEqual(self.destination.read_bytes(), RangeHandler.data)

    def test_http_416_is_actionable_and_preserves_partial(self):
        RangeHandler.mode = "range416"
        with self.assertRaisesRegex(download.DownloadError, "HTTP 416"):
            self.run_download()
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.destination.exists())

    def test_ignored_range_is_rejected(self):
        RangeHandler.mode = "ignore_range"
        with self.assertRaisesRegex(download.DownloadError, "HTTP 206"):
            self.run_download()
        self.assertFalse(self.destination.exists())

    def test_wrong_content_range_is_rejected(self):
        RangeHandler.mode = "wrong_range"
        with self.assertRaisesRegex(download.DownloadError, "Content-Range"):
            self.run_download()
        self.assertFalse(self.destination.exists())

    def test_same_size_corrupted_data_fails_final_hash(self):
        RangeHandler.mode = "corrupt_content"
        with self.assertRaisesRegex(download.DownloadError, "Контрольная сумма"):
            self.run_download()
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.partial.exists())
        self.assertTrue(self.checkpoint.exists())

    def test_existing_valid_final_is_verified_without_network(self):
        self.destination.write_bytes(RangeHandler.data)
        self.assertEqual(self.run_download(), self.destination)
        self.assertEqual(RangeHandler.requests, [])

    def test_existing_corrupt_final_is_preserved(self):
        content = b"Z" * len(RangeHandler.data)
        self.destination.write_bytes(content)
        with self.assertRaisesRegex(download.DownloadError, "существующего файла"):
            self.run_download()
        self.assertEqual(self.destination.read_bytes(), content)
        self.assertEqual(RangeHandler.requests, [])

    def test_incompatible_partial_is_preserved_without_network(self):
        self.partial.write_bytes(b"user-owned unfinished file")
        before = self.partial.read_bytes()
        with self.assertRaisesRegex(download.DownloadError, "принадлежит другому файлу"):
            self.run_download()
        self.assertEqual(self.partial.read_bytes(), before)
        self.assertEqual(RangeHandler.requests, [])

    def test_wrong_checkpoint_metadata_is_preserved(self):
        self.partial.write_bytes(b"\0" * len(RangeHandler.data))
        saved = {"url": self.url, "size": len(RangeHandler.data), "sha256": "0" * 64, "chunk_size": 1024, "completed": []}
        original = json.dumps(saved)
        self.checkpoint.write_text(original, encoding="utf-8")
        with self.assertRaises(download.DownloadError):
            self.run_download()
        self.assertEqual(self.checkpoint.read_text(encoding="utf-8"), original)
        self.assertEqual(RangeHandler.requests, [])

    def test_keyboard_interrupt_stops_workers_and_preserves_checkpoint(self):
        with patch.object(download, "wait", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_download()
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.partial.exists())
        self.assertIsInstance(json.loads(self.checkpoint.read_text(encoding="utf-8"))["completed"], list)


if __name__ == "__main__":
    unittest.main()
