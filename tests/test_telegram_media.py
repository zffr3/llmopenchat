from __future__ import annotations

import dataclasses
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llmopenchat import harness, telegram_media
from llmopenchat._files import FileAccessError, LockedHandle, supported
from llmopenchat.harness import HarnessError, HarnessSettings, ToolHarness
from llmopenchat.telegram_media import AttachmentStore, MAX_MEDIA_BYTES, TelegramToolHarness


def call(tools, name, **arguments):
    return json.loads(tools.execute(name, json.dumps(arguments, ensure_ascii=False)))


@unittest.skipUnless(supported(), "Secure attachment access requires Windows or Linux")
class AttachmentStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.store = AttachmentStore(self.workspace)

    def test_binary_round_trip_persists_ids_and_isolates_users(self):
        content = b"\x00\xff\x80arbitrary binary"
        metadata = self.store.add("report.pdf", content, "document", "application/pdf; charset=binary")
        self.assertEqual(metadata["size"], len(content))
        self.assertEqual(metadata["mime_type"], "application/pdf")
        self.assertEqual(AttachmentStore(self.workspace).read(metadata["file_id"]), (metadata, content))
        self.assertEqual(self.store.path(metadata["file_id"]).read_bytes(), content)
        other = self.workspace / "other"
        other.mkdir()
        with self.assertRaises((HarnessError, FileAccessError)):
            AttachmentStore(other).read(metadata["file_id"])

    def test_names_are_safe_and_identical_names_never_overwrite(self):
        for filename in ("../../report.pdf", "C:\\outside\\report.pdf", "CON.txt", ".METADATA.JSON", "../.git", "evil\x00name?.bin", "a\u202eexe.pdf"):
            with self.subTest(filename=filename):
                first = self.store.add(filename, b"first")
                second = self.store.add(filename, b"second")
                self.assertNotEqual(first["path"], second["path"])
                self.assertNotIn("..", first["path"].split("/"))
                self.assertEqual(self.store.read(first["file_id"])[1], b"first")
                self.assertEqual(self.store.read(second["file_id"])[1], b"second")
        self.assertFalse((self.workspace.parent / "report.pdf").exists())

    def test_invalid_ids_do_not_read_outside_store(self):
        for file_id in ("../secret", "a" * 31, "A" * 32, "a" * 33, None, 5):
            with self.subTest(file_id=file_id), self.assertRaises(HarnessError):
                self.store.read(file_id)

    def test_metadata_cannot_point_outside_its_unique_folder(self):
        metadata = self.store.add("safe.bin", b"safe")
        record = self.workspace / "attachments" / metadata["file_id"] / ".metadata.json"
        changed = {**metadata, "filename": "../../secret", "path": "secret"}
        record.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(HarnessError):
            self.store.read(metadata["file_id"])

    def test_modified_size_and_hardlinks_are_rejected(self):
        metadata = self.store.add("file.bin", b"safe")
        path = self.store.path(metadata["file_id"])
        path.write_bytes(b"different size")
        with self.assertRaises(HarnessError):
            self.store.read(metadata["file_id"])
        path.write_bytes(b"safe")
        alias = self.workspace / "alias.bin"
        os.link(path, alias)
        with self.assertRaises(FileAccessError):
            self.store.read(metadata["file_id"])

    def test_size_limit_and_wrong_content_type_fail_before_creating(self):
        with patch.object(telegram_media, "MAX_MEDIA_BYTES", 3):
            with self.assertRaises(HarnessError):
                self.store.add("big.bin", b"1234")
        with self.assertRaises(HarnessError):
            self.store.add("text.txt", "not bytes")
        self.assertEqual(list(self.workspace.iterdir()), [])


@unittest.skipUnless(supported(), "Secure attachment access requires Windows or Linux")
class TelegramMediaToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.store = AttachmentStore(self.workspace)
        self.approve = Mock(return_value=True)
        self.send = Mock(return_value={"message_id": 123})
        self.tools = TelegramToolHarness(HarnessSettings(), self.approve, store=self.store, send=self.send)
        self.metadata = self.store.add("file.bin", b"\x00\xffbytes")

    def test_telegram_always_has_send_and_cli_schemas_are_unchanged(self):
        self.assertEqual(ToolHarness(HarnessSettings(), self.approve).schemas(), [])
        self.assertEqual([item["function"]["name"] for item in self.tools.schemas()], ["telegram_send_file"])
        schemas = self.tools.schemas()
        schemas[0]["function"]["name"] = "changed"
        self.assertEqual(self.tools.schemas()[0]["function"]["name"], "telegram_send_file")
        self.assertNotIn("chat_id", json.dumps(self.tools.schemas()))
        self.assertNotIn("recipient", self.tools.schemas()[0]["function"]["parameters"]["properties"])

    def test_each_send_is_approved_and_bound_to_trusted_callback(self):
        for _ in range(2):
            result = call(self.tools, "telegram_send_file", file_id=self.metadata["file_id"], caption="caption")
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["send_started"])
            self.assertEqual(result["telegram"], {"message_id": 123})
        self.assertEqual(self.approve.call_count, 2)
        self.assertEqual(self.send.call_count, 2)
        self.send.assert_called_with("file.bin", b"\x00\xffbytes", "document", "caption")
        self.assertIn("file.bin", self.approve.call_args.args[0].summary)
        self.assertIn("caption", self.approve.call_args.args[0].summary)

    def test_denial_does_not_read_binary_or_send(self):
        self.approve.return_value = False
        original = LockedHandle.read
        reads = []

        def read(handle, limit):
            reads.append(handle.path.name)
            return original(handle, limit)

        with patch.object(LockedHandle, "read", new=read):
            result = call(self.tools, "telegram_send_file", file_id=self.metadata["file_id"])
        self.assertEqual(result["status"], "denied")
        self.assertFalse(result["send_started"])
        self.assertEqual(reads, [".metadata.json"])
        self.send.assert_not_called()

    def test_argument_mutation_does_not_change_content_caption_or_type(self):
        def approve(request):
            request.arguments.update(file_id="a" * 32, caption="changed", media_type="photo")
            return True

        self.tools.approve = approve
        result = call(self.tools, "telegram_send_file", file_id=self.metadata["file_id"], caption="original")
        self.assertEqual(result["status"], "ok")
        self.send.assert_called_once_with("file.bin", b"\x00\xffbytes", "document", "original")

    def test_malformed_or_recipient_arguments_never_request_approval(self):
        cases = ["[]", "{", '{"file_id":"a","file_id":"b"}', '{"file_id":NaN}']
        for raw in cases:
            self.assertEqual(json.loads(self.tools.execute("telegram_send_file", raw))["status"], "error")
        for values in ({}, {"file_id": self.metadata["file_id"], "path": "x"}, {"file_id": self.metadata["file_id"], "chat_id": 7}, {"file_id": self.metadata["file_id"], "media_type": "unknown"}, {"file_id": self.metadata["file_id"], "caption": "x", "media_type": "sticker"}):
            self.assertEqual(call(self.tools, "telegram_send_file", **values)["status"], "error")
        self.approve.assert_not_called()
        self.send.assert_not_called()

    def test_repository_path_requires_code_access(self):
        path = self.workspace / "export.zip"
        path.write_bytes(b"PK\x00\x01")
        self.assertEqual(call(self.tools, "telegram_send_file", path="export.zip")["status"], "error")
        self.approve.assert_not_called()
        tools = TelegramToolHarness(HarnessSettings(code_enabled=True, repository=self.workspace), self.approve, store=self.store, send=self.send)
        self.assertEqual(call(tools, "telegram_send_file", path="export.zip")["status"], "ok")
        self.send.assert_called_once_with("export.zip", b"PK\x00\x01", "document", "")
        self.assertEqual(call(tools, "telegram_send_file", path="../outside.zip")["status"], "error")

    def test_repository_hardlink_is_rejected_before_approval(self):
        path = self.workspace / "original.bin"
        path.write_bytes(b"secret")
        os.link(path, self.workspace / "alias.bin")
        tools = TelegramToolHarness(HarnessSettings(code_enabled=True, repository=self.workspace), self.approve, store=self.store, send=self.send)
        self.assertEqual(call(tools, "telegram_send_file", path="alias.bin")["status"], "error")
        self.approve.assert_not_called()
        self.send.assert_not_called()

    def test_empty_files_and_oversized_photos_fail_before_approval(self):
        empty = self.store.add("empty.bin", b"")
        result = call(self.tools, "telegram_send_file", file_id=empty["file_id"])
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["send_started"])
        with patch.object(telegram_media, "MAX_PHOTO_BYTES", 2):
            result = call(self.tools, "telegram_send_file", file_id=self.metadata["file_id"], media_type="photo")
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["send_started"])
        self.approve.assert_not_called()
        self.send.assert_not_called()

    def test_file_cannot_be_swapped_during_approval(self):
        path = self.store.path(self.metadata["file_id"])

        def approve(request):
            path.unlink()
            path.write_bytes(b"changed")
            return True

        self.tools.approve = approve
        result = call(self.tools, "telegram_send_file", file_id=self.metadata["file_id"])
        self.assertIn(result["status"], {"error", "denied"})
        self.send.assert_not_called()

    def test_revocation_during_approval_and_truthy_values_fail_closed(self):
        def revoke(request):
            self.tools.settings = dataclasses.replace(self.tools.settings, http_enabled=True)
            return True

        self.tools.approve = revoke
        self.assertEqual(call(self.tools, "telegram_send_file", file_id=self.metadata["file_id"])["status"], "denied")
        for answer in (1, "yes", None):
            self.tools.approve = Mock(return_value=answer)
            self.assertEqual(call(self.tools, "telegram_send_file", file_id=self.metadata["file_id"])["status"], "denied")
        self.send.assert_not_called()

    def test_send_failure_is_sanitized_marked_uncertain_and_never_retried(self):
        self.send.side_effect = RuntimeError("token-secret/12345")
        result = call(self.tools, "telegram_send_file", file_id=self.metadata["file_id"])
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["send_started"])
        self.assertNotIn("token-secret", json.dumps(result))
        self.send.assert_called_once()

    def test_cancelled_callback_can_report_send_not_started(self):
        error = HarnessError("cancelled")
        error.send_started = False
        self.send.side_effect = error
        result = call(self.tools, "telegram_send_file", file_id=self.metadata["file_id"])
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["send_started"])

    def test_download_needs_http_permission_and_fresh_approval(self):
        with patch.object(telegram_media, "_download_public") as download:
            self.assertEqual(call(self.tools, "http_download", url="https://example.com/a.pdf")["status"], "error")
            self.tools.settings = dataclasses.replace(self.tools.settings, http_enabled=True)
            self.assertIn("http_download", [item["function"]["name"] for item in self.tools.schemas()])
            self.approve.return_value = False
            self.assertEqual(call(self.tools, "http_download", url="https://example.com/a.pdf")["status"], "denied")
            download.assert_not_called()

    def test_download_stores_binary_metadata_then_can_send_without_text_conversion(self):
        self.tools.settings = dataclasses.replace(self.tools.settings, http_enabled=True)
        response = {"content": b"\x00\xffPDF", "content_type": "application/pdf", "status_code": 200,
                    "headers": {"content-disposition": "attachment; filename*=UTF-8''%D1%84%D0%B0%D0%B9%D0%BB.pdf"}}
        with patch.object(telegram_media, "_download_public", return_value=response) as download:
            result = call(self.tools, "http_download", url="https://example.com/a.pdf", headers={"Authorization": "Bearer explicit"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["filename"], "файл.pdf")
        self.assertEqual(result["size"], 5)
        self.assertNotIn("content", result)
        download.assert_called_once_with("https://example.com/a.pdf", {"Authorization": "Bearer explicit"})
        self.assertIn("Bearer explicit", self.approve.call_args.args[0].summary)
        self.assertEqual(call(self.tools, "telegram_send_file", file_id=result["file_id"])["status"], "ok")
        self.send.assert_called_once_with("файл.pdf", b"\x00\xffPDF", "document", "")

    def test_download_arguments_block_private_urls_headers_and_injected_fields(self):
        self.tools.settings = dataclasses.replace(self.tools.settings, http_enabled=True)
        for values in ({"url": "http://127.0.0.1/a"}, {"url": "https://example.com/a", "headers": {"Host": "localhost"}}, {"url": "https://example.com/a", "headers": {"X-Test": "x\r\ninject"}}, {"url": "https://example.com/a", "method": "POST"}):
            self.assertEqual(call(self.tools, "http_download", **values)["status"], "error")
        self.approve.assert_not_called()


class BinaryDownloadTransportTests(unittest.TestCase):
    def transfer(self, content=b"\x00\xff", *, status=200, headers=None, limit=10, records=None):
        from tests.test_harness import FakeSocket

        response = Mock()
        response.status = status
        values = {"Content-Type": "application/octet-stream", "Content-Length": str(len(content)), **(headers or {})}
        response.getheader.side_effect = lambda name, default=None: values.get(name, default)
        response.getheaders.return_value = list(values.items())
        response.read.side_effect = lambda count: content[:count]
        sock = FakeSocket()
        connection = Mock()
        connection.getresponse.return_value = response
        addresses = records or [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))]
        with patch.object(socket, "getaddrinfo", return_value=addresses) as dns, patch.object(socket, "socket", return_value=sock), patch.object(harness, "HTTPConnection", return_value=connection):
            try:
                result = harness._download_http_public("http://example.com/download", {"Authorization": "Bearer user"}, limit)
            except HarnessError as error:
                result = error
        return result, connection, response, sock, dns

    def test_binary_get_uses_one_pinned_public_request_and_preserves_bytes(self):
        result, connection, response, sock, dns = self.transfer()
        self.assertEqual(result["content"], b"\x00\xff")
        self.assertEqual(result["content_type"], "application/octet-stream")
        self.assertEqual(sock.connections, [("8.8.8.8", 80)])
        connection.request.assert_called_once()
        args, kwargs = connection.request.call_args
        self.assertEqual(args, ("GET", "/download"))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer user")
        self.assertEqual(kwargs["headers"]["Accept-Encoding"], "identity")
        response.read.assert_called_once_with(11)
        dns.assert_called_once()
        self.assertTrue(sock.closed)

    def test_redirect_error_oversize_compression_and_truncation_fail_without_retry(self):
        cases = [dict(status=302, headers={"Location": "http://127.0.0.1/private"}), dict(status=404),
                 dict(headers={"Content-Length": "11"}), dict(content=b"x" * 11, headers={"Content-Length": ""}),
                 dict(headers={"Content-Encoding": "gzip"}), dict(headers={"Content-Length": "5"})]
        for values in cases:
            with self.subTest(values=values):
                result, connection, _response, sock, dns = self.transfer(**values)
                self.assertIsInstance(result, HarnessError)
                self.assertTrue(result.request_started)
                connection.request.assert_called_once()
                dns.assert_called_once()
                self.assertTrue(sock.closed)

    def test_mixed_public_private_dns_is_rejected_before_connection(self):
        records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80)) for ip in ("8.8.8.8", "127.0.0.1")]
        result, connection, _response, sock, _dns = self.transfer(records=records)
        self.assertIsInstance(result, HarnessError)
        self.assertFalse(sock.connections)
        connection.request.assert_not_called()

    def test_public_entrypoint_rejects_unbounded_limits(self):
        for limit in (0, -1, MAX_MEDIA_BYTES + 1, True):
            with self.subTest(limit=limit), self.assertRaises(HarnessError):
                harness._download_http_public("http://example.com/download", {}, limit)


if __name__ == "__main__":
    unittest.main()
