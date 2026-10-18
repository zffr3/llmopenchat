"""POSIX entry points preserve argv, working directory, and exit status."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
SHELL = shutil.which("sh")
if not SHELL and os.name == "nt":
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    if candidate.is_file():
        SHELL = str(candidate)


@unittest.skipUnless(SHELL, "A POSIX shell is required")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="llmopenchat launchers ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in ("Install.sh", "Start-Chat.sh", "Start-Telegram.sh"):
            shutil.copyfile(ROOT / name, self.root / name)
        binary = self.root / ".venv/bin/python"
        binary.parent.mkdir(parents=True)
        binary.write_text('#!/bin/sh\nprintf "[%s]\\n" "$@"\nexit "${TEST_EXIT_CODE:-0}"\n', encoding="utf-8")
        binary.chmod(0o755)

    def run_script(self, name, *arguments, code=0):
        environment = dict(os.environ, TEST_EXIT_CODE=str(code))
        return subprocess.run([SHELL, str(self.root / name), *arguments],
                              cwd=ROOT, env=environment, text=True, encoding="utf-8",
                              capture_output=True, timeout=15)

    def test_chat_forwards_arguments_and_child_failure(self):
        result = self.run_script("Start-Chat.sh", "ask", "текст с пробелами", code=7)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("[-m]\n[llmopenchat]\n[ask]\n[текст с пробелами]", result.stdout)

    def test_telegram_places_global_config_before_command(self):
        result = self.run_script("Start-Telegram.sh", "--config", "profile with spaces.json", "--no-autostart")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[--config]\n[profile with spaces.json]\n[telegram]\n[--no-autostart]", result.stdout)

    def test_install_propagates_options_and_can_install_only_client(self):
        result = self.run_script("Install.sh", "--model", "qwen3-0.6b")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[install]\n[--model]\n[qwen3-0.6b]", result.stdout)
        client = self.run_script("Install.sh", "--client-only")
        self.assertEqual(client.returncode, 0, client.stderr)
        self.assertNotIn("[llmopenchat]", client.stdout)

    def test_missing_environment_reports_installation_command(self):
        (self.root / ".venv/bin/python").unlink()
        result = self.run_script("Start-Telegram.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Install.sh", result.stderr)
