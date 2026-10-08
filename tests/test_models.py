"""Package transactions and safety tested with tiny mocked GGUF transfers."""

import copy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llmopenchat import models
from llmopenchat.config import DEFAULT_CONFIG


class ModelManagerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="llmopenchat-model-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.payload = b"GGUF tiny verified fixture"
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.spec = models.ModelSpec(
            "test-flash", "Test Flash", "example/test-flash-GGUF", 30,
            "Test description", "flash", "Original", "код / программирование",
            "https://huggingface.co/example/test-flash", "a" * 40,
            "Test-Q4_K_M.gguf", len(self.payload), self.digest,
        )
        self.dense = replace(self.spec, id="test-32b", name="Test Dense", repo_id="example/test-32b-GGUF", family="glm4", architecture="glm4", parameters_b=32, context_length=32768)
        self.emit = Mock()
        self.manager = models.ModelManager(self.root, emit=self.emit)
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.original_config = self.config_path.read_bytes()
        self.template = self.root / "llmopenchat/templates/GLM-4.7-Flash.jinja"
        self.template.parent.mkdir(parents=True)
        shutil.copyfile(Path(models.__file__).parent / "templates/GLM-4.7-Flash.jinja", self.template)
        self.hub = ModuleType("huggingface_hub")
        self.api = Mock()
        self.hub.HfApi = Mock(return_value=self.api)
        self.api.model_info.side_effect = self.metadata
        self.downloader = Mock(side_effect=self.transfer)
        self.runtime = Mock(side_effect=self.install_runtime)
        patches = [
            patch.object(models, "CATALOG", (self.spec, self.dense)),
            patch.dict(sys.modules, {"huggingface_hub": self.hub}),
            patch.object(models, "download_file", self.downloader),
            patch.object(models, "_install_runtime", self.runtime),
            patch.object(models, "os", SimpleNamespace(name="nt", fsync=os.fsync, getpid=os.getpid)),
            patch.object(models.platform, "machine", return_value="AMD64"),
            patch.object(models.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * models._GIB)),
        ]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    def sibling(self, filename=None, data=None, digest=None):
        data = self.payload if data is None else data
        return SimpleNamespace(rfilename=filename or self.spec.default_filename, size=len(data), lfs=SimpleNamespace(sha256=digest or hashlib.sha256(data).hexdigest()))

    def metadata(self, repo, revision, files_metadata):
        spec = self.dense if repo == self.dense.repo_id else self.spec
        return SimpleNamespace(sha=spec.revision, siblings=[self.sibling()], gguf={"total": int(spec.parameters_b * 1e9), "architecture": spec.architecture})

    def transfer(self, url, destination, size, sha256, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != self.payload:
                raise models.DownloadError("Контрольная сумма существующего файла не совпала; переименуйте его.")
        else:
            destination.write_bytes(self.payload)
        return destination

    def install_runtime(self, root, emit):
        executable = root / ".local/runtime/test/llama-server.exe"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b"test runtime")
        return executable

    def install(self, dense=False):
        return self.manager.install(self.dense.id if dense else self.spec.id)

    def manifest(self):
        return json.loads(self.manager.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, data):
        self.manager.manifest_path.write_text(json.dumps(data), encoding="utf-8")

    def legacy(self, path="models/legacy/Legacy-Q4_K_M.gguf"):
        model = self.root / path
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(self.payload)
        executable = self.install_runtime(self.root, self.emit)
        lock = {"model_path": path, "executable": executable.relative_to(self.root).as_posix(), "repo_id": "huihui_ai/glm-4.7-flash-abliterated",
                "model": {"size": len(self.payload), "sha256": self.digest, "quantization": "Q4_K_M", "source": "https://ollama.com/huihui_ai/glm-4.7-flash-abliterated"}}
        (self.root / ".local/install.json").write_text(json.dumps(lock), encoding="utf-8")
        config = copy.deepcopy(self.config)
        config["server"].update(model_path=path, executable=lock["executable"])
        return config

    def test_listing_is_read_only_and_does_not_require_network(self):
        self.assertEqual(len(self.manager.catalog()), 2)
        records = self.manager.installed(self.config)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["managed"])
        self.assertEqual(records[0]["status"], "missing")
        self.assertFalse(self.manager.manifest_path.exists())
        self.assertFalse((self.root / ".local").exists())
        self.api.model_info.assert_not_called()

    def test_quantizations_select_only_exact_selected_gguf(self):
        metadata = self.metadata(self.spec.repo_id, self.spec.revision, True)
        metadata.siblings.extend([self.sibling("Test-Q8_0.gguf"), self.sibling("README.md"), self.sibling("Test-F16.gguf")])
        self.api.model_info.side_effect = None
        self.api.model_info.return_value = metadata
        variants = self.manager.available_quantizations(self.spec.id)
        self.assertEqual({item["quantization"] for item in variants}, {"Q4_K_M", "Q8_0", "F16"})
        record = self.install()
        self.assertEqual(len(record["files"]), 1)
        self.assertEqual(self.downloader.call_count, 1)
        self.assertIn(f"/resolve/{self.spec.revision}/Test-Q4_K_M.gguf", self.downloader.call_args.args[0])
        self.assertEqual(self.downloader.call_args.args[2:4], (len(self.payload), self.digest))
        self.api.model_info.assert_called_with(self.spec.repo_id, revision=self.spec.revision, files_metadata=True)

    def test_split_shards_complete_sorted_and_downloaded_as_one_package(self):
        info = self.metadata(self.spec.repo_id, self.spec.revision, True)
        info.siblings.extend([self.sibling("Q5_K_M/Test-Q5_K_M-00002-of-00002.gguf"), self.sibling("Q5_K_M/Test-Q5_K_M-00001-of-00002.gguf")])
        self.api.model_info.side_effect = None
        self.api.model_info.return_value = info
        record = self.manager.install(self.spec.id, "Q5_K_M")
        self.assertEqual(len(record["files"]), 2)
        self.assertTrue(record["model_path"].endswith("00001-of-00002.gguf"))
        self.assertEqual(record["total_size"], 2 * len(self.payload))
        self.assertEqual(self.downloader.call_count, 2)
        self.assertEqual(self.manager.installed(verify=True)[0]["status"], "ready")

    def test_incomplete_or_ambiguous_variants_not_installable(self):
        info = self.metadata(self.spec.repo_id, self.spec.revision, True)
        info.siblings.extend([self.sibling("Test-Q5_K_M-00001-of-00002.gguf"), self.sibling("Test-Q8_0.gguf"), self.sibling("Different-Q8_0.gguf")])
        self.api.model_info.side_effect = None
        self.api.model_info.return_value = info
        self.assertEqual([item["quantization"] for item in self.manager.available_quantizations(self.spec.id)], ["Q4_K_M"])
        for quant in ("Q5_K_M", "Q8_0"):
            with self.subTest(quant=quant), self.assertRaisesRegex(models.ModelError, "набор"):
                self.manager.install(self.spec.id, quant)
        self.downloader.assert_not_called()

    def test_missing_size_or_lfs_digest_refuses_download(self):
        for change in ({"size": None}, {"size": -1}, {"lfs": None}, {"lfs": {"sha256": "bad"}}):
            with self.subTest(change=change):
                info = self.metadata(self.spec.repo_id, self.spec.revision, True)
                for key, value in change.items():
                    setattr(info.siblings[0], key, value)
                self.api.model_info.side_effect = None
                self.api.model_info.return_value = info
                with self.assertRaises(models.ModelError):
                    self.install()
        self.downloader.assert_not_called()
        self.runtime.assert_not_called()

    def test_bad_pinned_metadata_rejected_before_mutations(self):
        for changes in ({"sha": "b" * 40}, {"gguf": {"total": 61_000_000_000}}, {"gguf": {"architecture": "unknown"}}):
            with self.subTest(changes=changes):
                info = self.metadata(self.spec.repo_id, self.spec.revision, True)
                for key, value in changes.items():
                    setattr(info, key, value)
                self.api.model_info.side_effect = None
                self.api.model_info.return_value = info
                with self.assertRaises(models.ModelError):
                    self.install()
        self.downloader.assert_not_called()
        self.runtime.assert_not_called()
        self.assertFalse(self.manager.manifest_path.exists())
        self.assertFalse((self.root / "models").exists())

    def test_pinned_default_sha_mismatch_rejected(self):
        info = self.metadata(self.spec.repo_id, self.spec.revision, True)
        info.siblings[0].lfs.sha256 = "0" * 64
        self.api.model_info.side_effect = None
        self.api.model_info.return_value = info
        with self.assertRaisesRegex(models.ModelError, "закрепленной"):
            self.install()
        self.downloader.assert_not_called()

    def test_remote_path_traversal_and_windows_device_names_rejected(self):
        for filename in ("../Test-Q8_0.gguf", "/Test-Q8_0.gguf", "C:/Test-Q8_0.gguf", "bad\\Test-Q8_0.gguf", "CON/Test-Q8_0.gguf"):
            with self.subTest(filename=filename):
                info = self.metadata(self.spec.repo_id, self.spec.revision, True)
                info.siblings.append(self.sibling(filename))
                self.api.model_info.side_effect = None
                self.api.model_info.return_value = info
                with self.assertRaises(models.ModelError):
                    self.install()
        self.downloader.assert_not_called()

    def test_successful_install_manifest_atomic_and_config_untouched(self):
        first = self.install()
        second = self.install(dense=True)
        data = self.manifest()
        self.assertEqual(set(data["packages"]), {first["id"], second["id"]})
        self.assertEqual(data["version"], 1)
        self.assertEqual(self.config_path.read_bytes(), self.original_config)
        self.assertEqual(self.config, DEFAULT_CONFIG)
        self.assertFalse(list((self.root / ".local").glob("models.*.tmp")))
        with self.manager._mutation():
            pass  # The OS lock is released, even though its file remains.

    def test_download_failure_and_keyboard_interrupt_keep_old_manifest_and_resume_files(self):
        first = self.install()
        before = self.manager.manifest_path.read_bytes()

        def interrupted(url, destination, size, sha256, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.with_name(destination.name + ".partial").write_bytes(self.payload)
            raise models.DownloadError("HTTP 503")

        self.downloader.side_effect = interrupted
        with self.assertRaisesRegex(models.ModelError, "продолжит"):
            self.install(dense=True)
        self.assertEqual(self.manager.manifest_path.read_bytes(), before)
        plan = self.manager.inspect(self.dense.id)
        self.assertTrue((self.root / (plan["model_path"] + ".partial")).is_file())
        self.downloader.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.install(dense=True)
        self.assertEqual(set(self.manifest()["packages"]), {first["id"]})
        with self.manager._mutation():
            pass

    def test_runtime_failure_keeps_verified_download_but_never_commits(self):
        self.runtime.side_effect = models.SetupError("runtime unavailable")
        with self.assertRaisesRegex(models.ModelError, "runtime unavailable"):
            self.install()
        self.assertFalse(self.manager.manifest_path.exists())
        plan = self.manager.inspect(self.spec.id)
        self.assertEqual((self.root / plan["model_path"]).read_bytes(), self.payload)
        self.assertEqual(self.config_path.read_bytes(), self.original_config)

    def test_existing_verified_download_reused_by_real_downloader_without_network(self):
        record = self.install()
        from llmopenchat.download import download_file
        with patch.object(models, "download_file", download_file), patch("llmopenchat.download.urllib.request.build_opener") as fetch:
            self.install()
        fetch.assert_not_called()
        self.assertEqual((self.root / record["model_path"]).read_bytes(), self.payload)

    def test_corrupt_existing_download_preserved_and_manifest_not_changed(self):
        record = self.install()
        before = self.manager.manifest_path.read_bytes()
        model = self.root / record["model_path"]
        model.write_bytes(b"X" * len(self.payload))
        from llmopenchat.download import download_file
        with patch.object(models, "download_file", download_file), self.assertRaisesRegex(models.ModelError, "переименуйте"):
            self.install()
        self.assertEqual(model.read_bytes(), b"X" * len(self.payload))
        self.assertEqual(self.manager.manifest_path.read_bytes(), before)
        self.assertEqual(self.manager.installed(verify=True)[0]["status"], "corrupt")

    def test_disk_guard_stops_before_runtime_or_download(self):
        with patch.object(models.shutil, "disk_usage", return_value=SimpleNamespace(free=1024)), self.assertRaisesRegex(models.ModelError, "Недостаточно места"):
            self.install()
        self.downloader.assert_not_called()
        self.runtime.assert_not_called()
        self.assertFalse((self.root / "models").exists())

    def test_resume_checkpoint_credits_only_matching_reserved_space(self):
        plan = self.manager.inspect(self.spec.id)
        self.manager._prepare(plan)
        path = self.root / plan["model_path"]
        partial = path.with_name(path.name + ".partial")
        partial.write_bytes(self.payload)
        checkpoint = path.with_name(path.name + ".partial.json")
        file = plan["files"][0]
        saved = {"url": self.manager._url(plan, file), "size": file["size"], "sha256": file["sha256"]}
        checkpoint.write_text(json.dumps(saved), encoding="utf-8")
        self.assertEqual(self.manager._needed(plan), 2 * models._GIB)
        saved["sha256"] = "0" * 64
        checkpoint.write_text(json.dumps(saved), encoding="utf-8")
        self.assertEqual(self.manager._needed(plan), 2 * models._GIB + len(self.payload))

    def test_activation_unique_alias_preserves_tuning_clears_old_template(self):
        flash = self.install()
        dense = self.install(dense=True)
        config = copy.deepcopy(self.config)
        config.update(base_url="http://localhost:9876/v1", temperature=0.33, system_prompt="Custom system", request_extra={"top_p": 0.91, "chat_template_kwargs": {"enable_thinking": False}})
        config["server"]["extra_args"].extend(["--chat-template=old", "--chat-template-kwargs", "{}"])
        config["server"]["context_size"] = 64000
        original = copy.deepcopy(config)
        activated = self.manager.activate(dense["id"], config, self.config_path)
        self.assertEqual(config, original)
        self.assertEqual(activated["request_extra"], {"top_p": 0.91})
        self.assertEqual(activated["server"]["extra_args"], ["--fit", "on", "--fit-target", "1024"])
        self.assertEqual(activated["server"]["context_size"], 32768)
        self.assertEqual(activated["base_url"], "http://localhost:9876/v1")
        self.assertEqual(activated["temperature"], 0.33)
        self.assertEqual(activated["system_prompt"], "Custom system")
        self.assertEqual(activated["model_package"], dense["id"])
        self.assertNotEqual(activated["model"], "glm-local")
        another = self.manager.activate(flash["id"], activated)
        self.assertNotEqual(another["model"], activated["model"])
        self.assertEqual(another["request_extra"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(sum(argument == "--chat-template-file" for argument in another["server"]["extra_args"]), 1)
        again = self.manager.activate(flash["id"], another)
        self.assertEqual(again["model"], another["model"])

    def test_switch_from_remote_backend_selects_local_endpoint(self):
        record = self.install(dense=True)
        remote = copy.deepcopy(self.config)
        remote.update(backend="external", base_url="https://example.com/v1")
        activated = self.manager.activate(record["id"], remote)
        self.assertEqual(activated["base_url"], DEFAULT_CONFIG["base_url"])
        self.assertEqual(activated["backend"], "llama_cpp")

    def test_generic_gguf_install_and_activation_round_trip_uses_embedded_template(self):
        generic = replace(self.spec, id="test-small", name="Test Small", repo_id="example/test-small-GGUF",
                          parameters_b=1.7, family="gguf", architecture="llama", context_length=8192)
        info = SimpleNamespace(sha=generic.revision, siblings=[self.sibling()],
                               gguf={"total": 1_700_000_000, "architecture": generic.architecture})
        self.api.model_info.side_effect = None
        self.api.model_info.return_value = info
        config = copy.deepcopy(self.config)
        config.update(temperature=0.25, max_tokens=2048, system_prompt="Extract facts",
                      request_extra={"top_p": 0.9, "chat_template_kwargs": {"enable_thinking": False}})
        config["server"]["context_size"] = 16384
        config["server"]["extra_args"].extend([
            "--chat-template=old", "--chat-template-kwargs", "{}",
            "--chat_template", "old", "--chat_template_file", "old.jinja", "--chat_template_kwargs={}",
            "--chat_template=old", "--chat_template_file=old.jinja", "--chat_template_kwargs", "{}",
        ])
        original = copy.deepcopy(config)
        with patch.object(models, "CATALOG", (*self.manager.catalog(), generic)):
            record = self.manager.install(generic.id)
            self.assertEqual(self.config_path.read_bytes(), self.original_config)
            reloaded = models.ModelManager(self.root, emit=self.emit)
            installed = reloaded.installed(verify=True)
            self.assertEqual(len(installed), 1)
            self.assertEqual(installed[0]["family"], "gguf")
            self.assertEqual(installed[0]["status"], "ready")
            activated = reloaded.activate(record["id"], config, self.config_path)
            self.assertEqual(config, original)
            self.assertEqual(activated["request_extra"], {"top_p": 0.9})
            self.assertEqual(activated["server"]["extra_args"], ["--fit", "on", "--fit-target", "1024"])
            self.assertEqual(activated["server"]["context_size"], 8192)
            self.assertEqual(activated["server"]["model_path"], record["model_path"])
            self.assertEqual(activated["target_model"], generic.repo_id)
            self.assertEqual(activated["model_package"], record["id"])
            self.assertEqual(activated["temperature"], 0.25)
            self.assertEqual(activated["max_tokens"], 2048)
            self.assertEqual(activated["system_prompt"], "Extract facts")
            self.assertEqual(json.loads(self.config_path.read_text(encoding="utf-8")), activated)
            self.assertTrue(reloaded.installed(activated)[0]["active"])
            self.assertEqual(reloaded.activate(record["id"], activated), activated)

    def test_missing_or_same_size_corrupt_model_never_activated(self):
        record = self.install(dense=True)
        model = self.root / record["model_path"]
        model.write_bytes(b"X" * len(self.payload))
        with self.assertRaisesRegex(models.ModelError, "повреждена"):
            self.manager.activate(record["id"], self.config, self.config_path)
        self.assertEqual(self.config_path.read_bytes(), self.original_config)
        model.unlink()
        self.assertEqual(self.manager.installed()[0]["status"], "missing")
        with self.assertRaises(models.ModelError):
            self.manager.activate(record["id"], self.config)

    def test_legacy_read_only_import_survives_switch_and_cannot_be_deleted(self):
        config = self.legacy()
        record = self.manager.installed(config)[0]
        self.assertEqual(record["status"], "ready")
        self.assertTrue(record["active"])
        self.assertFalse(record["managed"])
        self.assertFalse(self.manager.manifest_path.exists())
        dense = self.install(dense=True)
        activated = self.manager.activate(dense["id"], config)
        records = self.manager.installed(activated)
        self.assertEqual(len(records), 2)
        self.assertIn(record["id"], self.manifest()["packages"])
        restored = self.manager.activate(record["id"], activated)
        self.assertEqual(restored["server"]["model_path"], record["model_path"])
        with self.assertRaisesRegex(models.ModelError, "не удаляются"):
            self.manager.remove(record["id"], activated)
        self.assertEqual((self.root / record["model_path"]).read_bytes(), self.payload)

    def test_distinct_legacy_installs_stay_listed_and_stale_package_not_active(self):
        first_config = self.legacy()
        first = self.manager.installed(first_config)[0]
        self.manager.activate(first["id"], first_config)
        new_config = self.legacy("models/other/Other-Q4_K_M.gguf")
        new_config["model_package"] = first["id"]
        records = self.manager.installed(new_config)
        self.assertEqual(len(records), 2)
        self.assertFalse(next(record for record in records if record["id"] == first["id"])["active"])
        self.assertEqual(sum(record["active"] for record in records), 1)

    def test_removal_preserves_unknown_and_shared_files_and_blocks_active(self):
        record = self.install(dense=True)
        directory = self.root / record["owned_dir"]
        unknown = directory / "user-notes.txt"
        unknown.write_text("keep me", encoding="utf-8")
        active = self.manager.activate(record["id"], self.config)
        with self.assertRaisesRegex(models.ModelError, "активную"):
            self.manager.remove(record["id"], active)
        self.manager.remove(record["id"], self.config)
        self.assertFalse((self.root / record["model_path"]).exists())
        self.assertEqual(unknown.read_text(encoding="utf-8"), "keep me")
        self.assertTrue((self.root / record["executable"]).is_file())
        self.assertNotIn(record["id"], self.manifest()["packages"])

    def test_remove_checks_persisted_active_config_when_no_config_argument(self):
        record = self.install(dense=True)
        self.manager.activate(record["id"], self.config, self.config_path)
        with self.assertRaisesRegex(models.ModelError, "активную"):
            self.manager.remove(record["id"])

    def test_failed_registry_commit_retains_ownership_for_delete_retry(self):
        record = self.install(dense=True)
        with patch.object(self.manager, "_save", side_effect=OSError("disk full")), self.assertRaises(models.ModelError):
            self.manager.remove(record["id"], self.config)
        self.assertTrue((self.root / record["owned_dir"] / models._MARKER).is_file())
        self.assertIn(record["id"], self.manifest()["packages"])
        self.manager.remove(record["id"], self.config)
        self.assertEqual(self.manifest()["packages"], {})

    def test_missing_install_directory_can_be_unregistered_safely(self):
        record = self.install(dense=True)
        directory = (self.root / record["owned_dir"]).resolve()
        self.assertTrue(directory.is_relative_to(self.root.resolve()))
        shutil.rmtree(directory)
        self.manager.remove(record["id"], self.config)
        self.assertEqual(self.manifest()["packages"], {})

    def test_unowned_preexisting_directory_or_modified_marker_not_overwritten(self):
        plan = self.manager.inspect(self.spec.id)
        directory = self.root / plan["owned_dir"]
        directory.mkdir(parents=True)
        unknown = directory / "my-file.txt"
        unknown.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(models.ModelError, "не принадлежит"):
            self.install()
        self.assertEqual(unknown.read_text(encoding="utf-8"), "keep")
        self.assertFalse(self.manager.manifest_path.exists())

    def test_linked_partial_file_not_written_or_deleted(self):
        record = self.install(dense=True)
        partial = self.root / (record["model_path"] + ".partial")
        partial.write_text("untouched", encoding="utf-8")
        original_check = Path.is_symlink

        def is_symlink(path):
            return path == partial or original_check(path)

        self.downloader.reset_mock()
        with patch.object(Path, "is_symlink", is_symlink):
            with self.assertRaisesRegex(models.ModelError, "Ссылки"):
                self.install(dense=True)
            with self.assertRaisesRegex(models.ModelError, "Ссылки"):
                self.manager.remove(record["id"], self.config)
        self.downloader.assert_not_called()
        self.assertEqual(partial.read_text(encoding="utf-8"), "untouched")
        self.assertTrue((self.root / record["model_path"]).exists())

    def test_registry_schema_prevents_unverified_launch_and_runtime_deletion(self):
        record = self.install(dense=True)
        original = self.manifest()
        for changes in ({"model_path": record["executable"]}, {"owned_dir": ".local/runtime/test"}, {"repo_id": None}, {"files": [{}]}):
            with self.subTest(changes=changes):
                data = copy.deepcopy(original)
                data["packages"][record["id"]].update(changes)
                self.save_manifest(data)
                with self.assertRaises(models.ModelError):
                    self.manager.activate(record["id"], self.config)
                with self.assertRaises(models.ModelError):
                    self.manager.remove(record["id"], self.config)
        self.assertTrue((self.root / record["executable"]).is_file())
        self.assertTrue((self.root / record["model_path"]).is_file())

    def test_modified_ownership_marker_prevents_activation_and_deletion(self):
        record = self.install(dense=True)
        marker = self.root / record["owned_dir"] / models._MARKER
        marker.write_text("{}", encoding="utf-8")
        self.assertEqual(self.manager.installed()[0]["status"], "corrupt")
        with self.assertRaises(models.ModelError):
            self.manager.activate(record["id"], self.config)
        with self.assertRaises(models.ModelError):
            self.manager.remove(record["id"], self.config)
        self.assertTrue((self.root / record["model_path"]).exists())

    def test_stale_lock_file_allows_resume_but_live_operation_rejected(self):
        lock = self.root / ".local/models.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("other process", encoding="utf-8")
        with self.manager._mutation():
            with self.assertRaisesRegex(models.ModelError, "уже выполняет"):
                self.install()
        self.assertEqual(lock.read_text(encoding="utf-8"), "other process")
        self.api.model_info.assert_not_called()
        self.install()

    def test_process_exit_releases_operation_lock_for_resume(self):
        script = "import sys; from pathlib import Path; from llmopenchat.models import ModelManager; manager=ModelManager(Path(sys.argv[1])); context=manager._mutation(); context.__enter__(); print('locked',flush=True); sys.stdin.read()"
        child = subprocess.Popen([sys.executable, "-c", script, str(self.root)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        try:
            self.assertEqual(child.stdout.readline().strip(), "locked")
            with self.assertRaisesRegex(models.ModelError, "уже выполняет"):
                self.install()
            child.terminate()
            child.communicate(timeout=5)
            self.install()
        finally:
            if child.poll() is None:
                child.terminate()
                child.communicate(timeout=5)

    def test_real_range_downloader_receives_pinned_url_and_verifies_bytes(self):
        from llmopenchat.download import download_file

        class Response(io.BytesIO):
            status = 206

            def geturl(response):
                return self.manager._url(self.manager.inspect(self.spec.id), {"filename": self.spec.default_filename})

        response = Response(self.payload)
        response.headers = {"Content-Range": f"bytes 0-{len(self.payload) - 1}/{len(self.payload)}", "Content-Length": str(len(self.payload))}
        opener = Mock()
        opener.open.return_value = response
        with patch.object(models, "download_file", download_file), patch("llmopenchat.download.urllib.request.build_opener", return_value=opener):
            record = self.install()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Range"), f"bytes=0-{len(self.payload) - 1}")
        self.assertIn(self.spec.revision, request.full_url)
        self.assertEqual((self.root / record["model_path"]).read_bytes(), self.payload)


class CuratedCatalogTests(unittest.TestCase):
    def test_curated_models_under_limit_with_pinned_hashes(self):
        self.assertGreaterEqual(len(models.CATALOG), 7)
        self.assertEqual(len({spec.id for spec in models.CATALOG}), len(models.CATALOG))
        for spec in models.CATALOG:
            with self.subTest(model=spec.id):
                self.assertGreater(spec.parameters_b, 0)
                self.assertLessEqual(spec.parameters_b, 60)
                self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
                self.assertRegex(spec.default_sha256, r"^[0-9a-f]{64}$")
                self.assertGreater(spec.default_size, 0)
                self.assertTrue(spec.description)
                self.assertTrue(spec.censorship)
                self.assertIn(spec.family, {"flash", "glm4", "gguf"})
                self.assertIn(spec.architecture, {"glm4", "deepseek2", "qwen3", "qwen3moe", "qwen2", "llama", "phi3", "granite", "lfm2", "smollm3", "gemma4", "gpt-oss", "deci"})
        self.assertNotIn("glm-z1-rumination-32b", {spec.id for spec in models.CATALOG})
        coder = next(spec for spec in models.CATALOG if spec.id == "glm-4.7-flash-coder")
        self.assertIn("дообучена", coder.description.lower())
        z1 = next(spec for spec in models.CATALOG if spec.id == "glm-z1-32b")
        self.assertEqual(z1.repo_id, "unsloth/GLM-Z1-32B-0414-GGUF")


if __name__ == "__main__":
    unittest.main()
