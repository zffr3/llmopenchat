"""Installer integrity and interrupted-download tests, without network access."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import tarfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from llmopenchat import setup


class FakeHubHttpError(Exception):
    pass


def response(body: bytes, status=200, headers=None):
    stream = io.BytesIO(body)
    stream.status = status
    stream.headers = headers or {}
    return stream


class InterruptedResponse(io.BytesIO):
    status = 200
    headers = {}

    def __init__(self, first_block):
        super().__init__(first_block)
        self.was_read = False

    def read(self, size=-1):
        if self.was_read:
            raise OSError("Connection interrupted")
        self.was_read = True
        return super().read(size)


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="llmopenchat-setup-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.emit = Mock()

    def asset(self, payload):
        return {"name": "test.zip", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    def hub(self, metadata=None, error=None, download=None):
        module = ModuleType("huggingface_hub")
        api = Mock()
        api.model_info.side_effect = error
        api.model_info.return_value = metadata
        module.HfApi = Mock(return_value=api)
        module.hf_hub_download = Mock(side_effect=download)
        return module, api

    def metadata(self, filename=setup.DEFAULT_FILENAME, size=18132722048, digest=setup.DEFAULT_SHA256):
        sibling = SimpleNamespace(rfilename=filename, size=size, lfs=SimpleNamespace(sha256=digest))
        return SimpleNamespace(sha=setup.DEFAULT_REVISION, siblings=[sibling])

    def call_install(self, hub, repo=setup.DEFAULT_REPO, filename=setup.DEFAULT_FILENAME):
        with (
            patch.dict(sys.modules, {"huggingface_hub": hub}),
            patch.object(setup, "os", SimpleNamespace(name="nt")),
            patch.object(setup.platform, "machine", return_value="AMD64"),
            patch.object(setup.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)),
        ):
            return setup.install_local(self.root, repo, filename, emit=self.emit)

    def test_archive_rejects_traversal_and_absolute_paths(self):
        for malicious in ("../outside.dll", "..\\outside.dll", "/absolute.dll", "C:/outside.dll"):
            with self.subTest(path=malicious):
                archive = self.root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as compressed:
                    compressed.writestr(malicious, b"unsafe")
                with self.assertRaises(setup.SetupError):
                    setup._extract_archive(archive, self.root / "target")
        self.assertFalse((self.root / "outside.dll").exists())

    def test_cached_archive_requires_matching_hash(self):
        archive = self.root / "test.zip"
        archive.write_bytes(b"corrupt archive")
        with patch.object(setup.urllib.request, "urlopen") as fetch:
            with self.assertRaisesRegex(setup.SetupError, "Поврежден архив"):
                setup._download_asset(self.asset(b"trusted archive"), archive, self.emit)
        fetch.assert_not_called()

    def test_interrupted_download_resumes_with_range(self):
        payload = b"abcdef"
        archive = self.root / "test.zip"
        resumed = response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})
        with patch.object(setup.urllib.request, "urlopen", side_effect=[InterruptedResponse(b"abc"), resumed]) as fetch:
            setup._download_asset(self.asset(payload), archive, self.emit)
        self.assertEqual(archive.read_bytes(), payload)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(fetch.call_args_list[1].args[0].get_header("Range"), "bytes=3-")
        self.assertFalse(archive.with_suffix(".zip.partial").exists())

    def test_server_ignoring_range_restarts_without_appending(self):
        archive = self.root / "test.zip"
        archive.with_suffix(".zip.partial").write_bytes(b"abc")
        with patch.object(setup.urllib.request, "urlopen", return_value=response(b"abcdef")):
            setup._download_asset(self.asset(b"abcdef"), archive, self.emit)
        self.assertEqual(archive.read_bytes(), b"abcdef")

    def test_downloaded_archive_hash_failure_never_becomes_final(self):
        archive = self.root / "test.zip"
        with patch.object(setup.urllib.request, "urlopen", return_value=response(b"bad")):
            with self.assertRaisesRegex(setup.SetupError, "SHA256"):
                setup._download_asset(self.asset(b"good"[:3]), archive, self.emit)
        self.assertFalse(archive.exists())

    def test_pinned_metadata_mismatch_stops_before_downloads(self):
        hub, _ = self.hub(metadata=self.metadata(digest="0" * 64))
        with patch.object(setup, "_install_runtime") as runtime:
            with self.assertRaisesRegex(setup.SetupError, "Метаданные"):
                self.call_install(hub)
        runtime.assert_not_called()
        hub.hf_hub_download.assert_not_called()

    def test_metadata_http_error_is_actionable_setup_error(self):
        original = FakeHubHttpError("HTTP 503")
        hub, _ = self.hub(error=original)
        with self.assertRaisesRegex(setup.SetupError, "метаданные Hugging Face") as raised:
            self.call_install(hub)
        self.assertIs(raised.exception.__cause__, original)

    def test_model_download_error_preserves_resume_instruction(self):
        original = FakeHubHttpError("HTTP 502")
        hub, _ = self.hub(metadata=self.metadata(), download=original)
        with patch.object(setup, "_install_runtime", return_value=self.root / "llama-server.exe"):
            with self.assertRaisesRegex(setup.SetupError, "Повторный запуск продолжит загрузку") as raised:
                self.call_install(hub)
        self.assertIs(raised.exception.__cause__, original)
        self.assertFalse((self.root / ".local/install.json").exists())

    def test_keyboard_interrupt_is_not_wrapped(self):
        hub, _ = self.hub(error=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.call_install(hub)

    def test_model_integrity_and_exact_file_recorded(self):
        payload = b"GGUF local model test"
        digest = hashlib.sha256(payload).hexdigest()
        filename = "model.gguf"
        metadata = self.metadata(filename=filename, size=len(payload), digest=digest)

        def download(**kwargs):
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.write_bytes(payload)
            return str(target)

        hub, api = self.hub(metadata=metadata, download=download)
        executable = self.root / ".local/runtime/llama-server.exe"
        with patch.object(setup, "_install_runtime", return_value=executable):
            result = self.call_install(hub, repo="example/model", filename=filename)
        self.assertEqual(result["executable"], ".local/runtime/llama-server.exe")
        self.assertEqual(result["model_path"], "models/example--model/model.gguf")
        lock = json.loads((self.root / ".local/install.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["model"]["sha256"], digest)
        self.assertEqual(lock["runtime_release"], setup.LLAMA_RELEASE)
        api.model_info.assert_called_once_with("example/model", revision="main", files_metadata=True)
        hub.hf_hub_download.assert_called_once_with(
            repo_id="example/model", filename=filename, revision=metadata.sha,
            local_dir=self.root / "models/example--model",
        )

    def test_model_hash_mismatch_does_not_write_lock(self):
        filename = "model.gguf"

        def download(**kwargs):
            target = Path(kwargs["local_dir"]) / filename
            target.write_bytes(b"bad")
            return str(target)

        hub, _ = self.hub(metadata=self.metadata(filename=filename, size=3, digest="0" * 64), download=download)
        with patch.object(setup, "_install_runtime", return_value=self.root / "llama-server.exe"):
            with self.assertRaisesRegex(setup.SetupError, "SHA256 модели"):
                self.call_install(hub, repo="example/model", filename=filename)
        self.assertFalse((self.root / ".local/install.json").exists())

    def test_runtime_extracts_verified_vulkan_bundle_and_reuses_marker(self):
        archive = self.root / ".local/downloads/test.zip"
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, "w") as compressed:
            compressed.writestr("bin/llama-server.exe", b"verified server")
            compressed.writestr("lib/ggml-vulkan.dll", b"verified Vulkan backend")
        asset = self.asset(archive.read_bytes())
        with (
            patch.object(setup, "_ASSETS", (asset,)),
            patch.object(setup, "os", SimpleNamespace(name="nt")),
            patch.object(setup.platform, "machine", return_value="AMD64"),
            patch.object(setup.urllib.request, "urlopen") as network,
        ):
            executable = setup._install_runtime(self.root, self.emit)
            self.assertEqual(executable.read_bytes(), b"verified server")
            self.assertEqual((executable.parent / "ggml-vulkan.dll").read_bytes(), b"verified Vulkan backend")
            expected = self.root / f".local/runtime/llama-{setup.LLAMA_RELEASE}-vulkan/bin/llama-server.exe"
            self.assertEqual(executable, expected)
            self.assertEqual(setup._install_runtime(self.root, self.emit), executable)
        network.assert_not_called()

    def test_runtime_without_vulkan_backend_is_rejected(self):
        archive = self.root / ".local/downloads/test.zip"
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, "w") as compressed:
            compressed.writestr("llama-server.exe", b"server without GPU backend")
        asset = self.asset(archive.read_bytes())
        with patch.object(setup, "_ASSETS", (asset,)), patch.object(setup, "os", SimpleNamespace(name="nt")), patch.object(setup.platform, "machine", return_value="AMD64"):
            with self.assertRaisesRegex(setup.SetupError, "ggml-vulkan.dll"):
                setup._install_runtime(self.root, self.emit)
        self.assertFalse((self.root / f".local/runtime/llama-{setup.LLAMA_RELEASE}-vulkan").exists())


class LinuxRuntimeSetupTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def spec(self, machine="x86_64", backend=None):
        environment = {} if backend is None else {"LLMOPENCHAT_RUNTIME_BACKEND": backend}
        with patch.object(setup, "os", SimpleNamespace(name="posix", environ=environment)), patch.object(setup.platform, "system", return_value="Linux"), patch.object(setup.platform, "machine", return_value=machine):
            return setup._runtime_spec()

    def archive(self, entries):
        path = self.root / "test.tar.gz"
        with tarfile.open(path, "w:gz") as compressed:
            for name, kind, value in entries:
                member = tarfile.TarInfo(name)
                member.mode = 0o755 if name.endswith("llama-server") else 0o644
                if kind == "file":
                    member.size = len(value)
                    compressed.addfile(member, io.BytesIO(value))
                else:
                    member.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
                    member.linkname = value
                    compressed.addfile(member)
        return path

    def test_verified_bundles_cover_cpu_vulkan_and_cuda_on_both_architectures(self):
        for machine in ("x86_64", "aarch64"):
            for backend in ("cpu", "vulkan", "cuda", "cuda13"):
                with self.subTest(machine=machine, backend=backend):
                    spec = self.spec(machine, backend)
                    self.assertEqual(spec["platform"], "linux")
                    self.assertEqual(spec["executable"], "llama-server")
                    self.assertTrue(all(asset["size"] > 0 and len(asset["sha256"]) == 64 for asset in spec["assets"]))
                    self.assertEqual(len(spec["assets"]), 2 if backend.startswith("cuda") else 1)
        self.assertEqual(self.spec()["backend"], "cpu")
        self.assertIn("12.8", self.spec(backend="cuda")["assets"][0]["name"])
        self.assertIn("13.4", self.spec("aarch64", "cuda")["assets"][0]["name"])
        with self.assertRaises(setup.SetupError):
            self.spec("x86_64", "typo")
        with self.assertRaises(setup.SetupError):
            self.spec("i686")

    def test_tar_rejects_traversal_links_outside_archive_and_special_files(self):
        for entries in (
            [("../escape", "file", b"bad")],
            [("/absolute", "file", b"bad")],
            [("C:/drive", "file", b"bad")],
            [("link", "symlink", "../escape")],
            [("link", "hardlink", "/etc/passwd")],
            [("link", "symlink", "missing")],
            [("a", "symlink", "b"), ("b", "symlink", "a")],
            [("fifo", "fifo", "")],
        ):
            with self.subTest(entries=entries):
                with self.assertRaises(setup.SetupError):
                    setup._extract_archive(self.archive(entries), self.root / "target")
        self.assertFalse((self.root / "escape").exists())

    def test_linux_runtime_materializes_library_links_and_reuses_installation(self):
        archive = self.archive([
            ("llama-b11445/llama-server", "file", b"server"),
            ("llama-b11445/libggml.so.0", "file", b"backend"),
            ("llama-b11445/libggml.so", "symlink", "libggml.so.0"),
        ])
        spec = self.spec()
        asset = {"name": archive.name, "size": archive.stat().st_size, "sha256": setup._sha256(archive)}
        spec["assets"] = (asset,)
        downloads = self.root / ".local/downloads"
        downloads.mkdir(parents=True)
        archive.replace(downloads / asset["name"])
        with patch.object(setup, "_runtime_spec", return_value=spec), patch.object(setup.urllib.request, "urlopen") as fetch:
            executable = setup._install_runtime(self.root, lambda message: None)
            self.assertEqual(executable, self.root / ".local/runtime" / spec["directory"] / "llama-server")
            self.assertEqual(executable.read_bytes(), b"server")
            self.assertEqual((executable.parent / "libggml.so").read_bytes(), b"backend")
            self.assertFalse((executable.parent / "libggml.so").is_symlink())
            self.assertEqual(setup._install_runtime(self.root, lambda message: None), executable)
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
