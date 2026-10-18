"""Real Linux descriptor access, consent boundaries, and concurrent swaps."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llmopenchat import _posixfiles
from llmopenchat.harness import HarnessError, HarnessSettings, ToolHarness


@unittest.skipUnless(_posixfiles.supported(), "Linux no-follow descriptor tests")
class LinuxRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmopenchat-linux-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "репозиторий"
        self.root.mkdir()
        self.approve = Mock(return_value=True)
        self.tools = ToolHarness(HarnessSettings(code_enabled=True, powershell_enabled=True,
                                                  repository=self.root), self.approve)

    def call(self, name, **arguments):
        return json.loads(self.tools.execute(name, json.dumps(arguments, ensure_ascii=False)))

    def test_linux_filenames_unicode_and_case_are_preserved(self):
        for name in ("readme", "README", "NUL", "notes:version~1.txt", "имя с пробелами.txt", "trailing."):
            self.assertEqual(self.call("write_file", path=name, content=name)["status"], "ok")
            self.assertEqual(self.call("read_file", path=name)["content"], name)
        listed = self.call("list_files")
        self.assertEqual(len(listed["entries"]), 6)

    def test_subdirectories_with_posix_link_counts_are_scanned(self):
        (self.root / "a" / "b").mkdir(parents=True)
        (self.root / "a" / "b" / "hello.txt").write_text("needle", encoding="utf-8")
        result = self.call("list_files", recursive=True)
        self.assertEqual({entry["path"] for entry in result["entries"]}, {"a", "a/b", "a/b/hello.txt"})
        self.assertEqual(self.call("search_files", query="needle")["matches"][0]["path"], "a/b/hello.txt")

    def test_repeated_and_partial_scans_rewind_shared_directory_descriptor(self):
        for name in ("one.txt", "two.txt", "three.txt"):
            (self.root / name).write_text(name, encoding="utf-8")
        with _posixfiles.LockedRepository(self.root) as locked:
            with locked.scandir(()) as entries:
                next(entries)
            for _ in range(2):
                with locked.scandir(()) as entries:
                    self.assertEqual({entry.name for entry in entries}, {"one.txt", "two.txt", "three.txt"})

    def test_root_symlink_and_symlink_ancestor_are_rejected(self):
        (self.root / "nested").mkdir()
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        for path in (alias, alias / "nested"):
            with self.subTest(path=path), self.assertRaises(HarnessError):
                ToolHarness(HarnessSettings(code_enabled=True, repository=path), self.approve)
        self.approve.assert_not_called()

    def test_links_and_fifo_are_not_read_or_scanned(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside secret", encoding="utf-8")
        (self.root / "link.txt").symlink_to(outside)
        (self.root / "dangling.txt").symlink_to(self.base / "missing")
        os.link(outside, self.root / "hard.txt")
        os.mkfifo(self.root / "pipe")
        for name in ("link.txt", "dangling.txt", "hard.txt", "pipe"):
            for tool, arguments in (("read_file", {}), ("write_file", {"content": "bad"})):
                self.assertEqual(self.call(tool, path=name, **arguments)["status"], "error")
        self.approve.assert_not_called()
        self.assertEqual(self.call("list_files", recursive=True)["entries"], [])
        self.assertEqual(self.call("search_files", query="secret")["matches"], [])
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside secret")

    def test_target_replacement_during_approval_fails_closed(self):
        target = self.root / "target.txt"
        replacement = self.base / "replacement.txt"
        for tool, arguments in (("read_file", {}), ("write_file", {"content": "new"})):
            target.write_text("original", encoding="utf-8")
            replacement.write_text("replacement", encoding="utf-8")
            self.tools.approve = lambda request: os.replace(replacement, target) or True
            self.assertEqual(self.call(tool, path="target.txt", **arguments)["status"], "error")
            self.assertEqual(target.read_text(encoding="utf-8"), "replacement")

    def test_root_replacement_during_approval_prevents_all_effects(self):
        moved = self.base / "moved"

        def approve(request):
            self.root.rename(moved)
            self.root.mkdir()
            return True

        self.tools.approve = approve
        self.assertEqual(self.call("write_file", path="new/target.txt", content="new")["status"], "error")
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(list(moved.iterdir()), [])

    def test_existing_parent_renamed_during_approval_is_detected(self):
        parent = self.root / "parent"
        parent.mkdir()
        moved = self.base / "moved"
        self.tools.approve = lambda request: parent.rename(moved) or True
        self.assertEqual(self.call("create_directory", path="parent/new")["status"], "error")
        self.assertEqual(list(moved.iterdir()), [])

    def test_missing_parent_replaced_by_symlink_after_approval_cannot_escape(self):
        outside = self.base / "outside"
        outside.mkdir()
        alias = self.root / "racing"
        self.tools.approve = lambda request: alias.symlink_to(outside, target_is_directory=True) or True
        self.assertEqual(self.call("write_file", path="racing/new/target.txt", content="bad")["status"], "error")
        self.assertEqual(list(outside.iterdir()), [])

    def test_created_parent_swapped_before_open_cannot_escape(self):
        outside = self.base / "outside"
        outside.mkdir()
        original = os.mkdir

        def mkdir(name, *args, **kwargs):
            original(name, *args, **kwargs)
            if name == "fresh":
                (self.root / "fresh").rmdir()
                (self.root / "fresh").symlink_to(outside, target_is_directory=True)

        with patch.object(os, "mkdir", side_effect=mkdir):
            self.assertEqual(self.call("write_file", path="fresh/new/target.txt", content="bad")["status"], "error")
        self.assertEqual(list(outside.iterdir()), [])

    def test_content_change_during_approval_requires_fresh_request(self):
        target = self.root / "target.txt"
        target.write_text("original", encoding="utf-8")

        def approve(request):
            target.write_text("modified", encoding="utf-8")
            return True

        self.tools.approve = approve
        self.assertEqual(self.call("read_file", path="target.txt")["status"], "error")

    def test_hardlink_added_during_approval_cannot_receive_replacement(self):
        target = self.root / "target.txt"
        target.write_text("original", encoding="utf-8")
        alias = self.base / "alias.txt"
        self.tools.approve = lambda request: os.link(target, alias) or True
        self.assertEqual(self.call("write_file", path="target.txt", content="bad")["status"], "error")
        self.assertEqual(alias.read_text(encoding="utf-8"), "original")

    def test_atomic_replacement_preserves_file_permissions_and_cleans_temporaries(self):
        target = self.root / "target.txt"
        target.write_text("original", encoding="utf-8")
        target.chmod(0o640)
        old_inode = target.stat().st_ino
        self.assertEqual(self.call("write_file", path="target.txt", content="replacement")["status"], "ok")
        self.assertNotEqual(target.stat().st_ino, old_inode)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assertEqual(list(self.root.iterdir()), [target])
        self.assertEqual(self.call("write_file", path="new.txt", content="")["status"], "ok")
        self.assertEqual((self.root / "new.txt").read_bytes(), b"")

    def test_failed_atomic_write_preserves_original_and_cleans_temporaries(self):
        target = self.root / "target.txt"
        target.write_text("original", encoding="utf-8")
        with patch.object(os, "fsync", side_effect=OSError("simulated disk failure")):
            self.assertEqual(self.call("write_file", path="target.txt", content="replacement")["status"], "error")
        self.assertEqual(target.read_text(encoding="utf-8"), "original")
        self.assertEqual(list(self.root.iterdir()), [target])

    def test_denied_powershell_source_is_never_read(self):
        (self.root / "hello.ps1").write_text("Write-Output 'hello'", encoding="utf-8")
        self.tools.approve = lambda request: False
        with patch.object(_posixfiles.LockedHandle, "read") as read:
            result = self.call("powershell_run", path="hello.ps1")
        self.assertEqual(result["status"], "denied")
        self.assertFalse(result["execution_started"])
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
