from __future__ import annotations

import dataclasses
import importlib
import json
import os
import socket
import subprocess
import tempfile
import threading
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llmopenchat import harness
from llmopenchat._winfiles import LockedHandle, supported
from llmopenchat import _winfiles
from llmopenchat.harness import HarnessError, HarnessSettings, ToolHarness


def call(tools, name, **arguments):
    return json.loads(tools.execute(name, json.dumps(arguments, ensure_ascii=False)))


class HarnessControlTests(unittest.TestCase):
    def test_default_has_no_tools_and_model_cannot_change_settings(self):
        approve = Mock(return_value=True)
        tools = ToolHarness(HarnessSettings(), approve)
        self.assertEqual(tools.schemas(), [])
        for name in ["web_fetch", "read_file", "powershell_run", "configure_tools", "run_shell", "enable_web"]:
            self.assertEqual(call(tools, name, url="https://example.com/")["status"], "error")
        approve.assert_not_called()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            tools.settings.web_enabled = True

    def test_schemas_are_detached_and_do_not_expose_settings_or_identity(self):
        tools = ToolHarness(HarnessSettings(web_enabled=True), lambda request: False)
        schemas = tools.schemas()
        self.assertEqual([item["function"]["name"] for item in schemas], ["web_fetch", "web_search"])
        self.assertNotIn("repository_identity", json.dumps(schemas))
        schemas[0]["function"]["name"] = "enable_web"
        self.assertEqual(tools.schemas()[0]["function"]["name"], "web_fetch")

    def test_malformed_web_arguments_never_ask_approval(self):
        approve = Mock(return_value=True)
        tools = ToolHarness(HarnessSettings(web_enabled=True), approve)
        for arguments in ["[]", "null", "{", '{"url":42}', '{"url":"https://example.com/","extra":true}', '{"url":"https://example.com/","url":"https://example.org/"}', '{"query":NaN}']:
            with self.subTest(arguments=arguments):
                self.assertEqual(json.loads(tools.execute("web_fetch", arguments))["status"], "error")
        approve.assert_not_called()

    def test_denied_web_does_not_resolve_or_connect_and_always_reprompts(self):
        approve = Mock(return_value=False)
        tools = ToolHarness(HarnessSettings(web_enabled=True), approve)
        with patch.object(harness, "_resolve_public") as resolve, patch.object(harness, "_request_public") as request:
            for _ in range(2):
                self.assertEqual(call(tools, "web_fetch", url="https://example.com/")["status"], "denied")
            self.assertEqual(call(tools, "web_search", query="test")["status"], "denied")
            resolve.assert_not_called()
            request.assert_not_called()
        self.assertEqual(approve.call_count, 3)

    def test_truthy_values_and_broken_approval_fail_closed(self):
        for answer in [1, "yes", None, False]:
            with self.subTest(answer=answer):
                tools = ToolHarness(HarnessSettings(web_enabled=True), lambda request: answer)
                self.assertEqual(call(tools, "web_fetch", url="https://example.com")["status"], "denied")
        for error in [EOFError(), KeyboardInterrupt(), RuntimeError("prompt failed")]:
            tools = ToolHarness(HarnessSettings(web_enabled=True), Mock(side_effect=error))
            self.assertEqual(call(tools, "web_fetch", url="https://example.com")["status"], "denied")

    def test_approval_cannot_change_canonical_arguments_or_grant_a_revoked_tool(self):
        def approve(request):
            request.arguments["url"] = "http://127.0.0.1/"
            return True

        tools = ToolHarness(HarnessSettings(web_enabled=True), approve)
        with patch.object(harness, "_request_public", return_value=(b"ok", "text/plain")) as request:
            self.assertEqual(call(tools, "web_fetch", url="https://example.com")["status"], "ok")
            request.assert_called_once_with("https://example.com/")

        def revoke(request):
            tools.settings = dataclasses.replace(tools.settings, web_enabled=False)
            return True

        tools = ToolHarness(HarnessSettings(web_enabled=True), revoke)
        with patch.object(harness, "_request_public") as request:
            self.assertEqual(call(tools, "web_fetch", url="https://example.com")["status"], "denied")
            request.assert_not_called()

    def test_search_approval_names_query_and_actual_public_target(self):
        approvals = []
        tools = ToolHarness(HarnessSettings(web_enabled=True), lambda request: approvals.append(request) or True)
        with patch.object(harness, "_request_public", return_value=(b"results", "text/plain")) as request:
            self.assertEqual(call(tools, "web_search", query="a & b")["status"], "ok")
            request.assert_called_once_with("https://html.duckduckgo.com/html/?q=a+%26+b")
        self.assertIn("a & b", approvals[0].summary)
        self.assertIn("https://html.duckduckgo.com/html/?q=a+%26+b", approvals[0].summary)

    def test_unsupported_code_platform_cannot_claim_code_is_enabled(self):
        with patch.object(harness, "supported", return_value=False):
            with self.assertRaises(HarnessError):
                ToolHarness(HarnessSettings(code_enabled=True, repository=Path.cwd()), lambda request: True)

    def test_powershell_settings_require_exact_boolean_and_selected_secure_repository(self):
        with self.assertRaises(HarnessError):
            ToolHarness(HarnessSettings(powershell_enabled="true"), lambda request: True)
        with self.assertRaises(HarnessError):
            ToolHarness(HarnessSettings(powershell_enabled=True), lambda request: True)
        with patch.object(harness, "supported", return_value=False):
            with self.assertRaises(HarnessError):
                ToolHarness(HarnessSettings(powershell_enabled=True, repository=Path.cwd()), lambda request: True)


@unittest.skipUnless(supported(), "Secure PowerShell file loading requires Windows native handles")
class PowerShellHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repository = self.base / "repo"
        self.repository.mkdir()
        self.script = self.repository / "hello.ps1"
        self.source = 'Write-Output "Hello, World!"\n'
        self.script.write_bytes(self.source.encode("utf-8"))
        self.approve = Mock(return_value=True)
        self.tools = ToolHarness(HarnessSettings(powershell_enabled=True, repository=self.repository), self.approve)
        self.response = {"stdout": "Hello, World!\n", "stderr": "", "exit_code": 0,
                         "restricted": True, "available_cmdlets": list(harness._POWERSHELL_CMDLETS)}

    def test_powershell_is_independent_of_code_permissions_and_preserves_identity(self):
        self.assertFalse(self.tools.settings.code_enabled)
        self.assertTrue(self.tools.settings.powershell_enabled)
        self.assertEqual([item["function"]["name"] for item in self.tools.schemas()], ["powershell_run"])
        self.assertIsNotNone(self.tools.settings.repository_identity)
        restored = ToolHarness(self.tools.settings, lambda request: False)
        self.assertEqual(restored.settings.repository_identity, self.tools.settings.repository_identity)
        self.assertEqual(call(self.tools, "read_file", path="hello.ps1")["status"], "error")
        self.assertEqual(call(self.tools, "write_file", path="hello.ps1", content="bad")["status"], "error")
        self.approve.assert_not_called()
        disabled = ToolHarness(HarnessSettings(code_enabled=True, repository=self.repository), self.approve)
        self.assertNotIn("powershell_run", [item["function"]["name"] for item in disabled.schemas()])
        self.assertEqual(call(disabled, "powershell_run", path="hello.ps1")["status"], "error")
        self.approve.assert_not_called()

    def test_denied_script_is_not_read_and_never_reaches_runner(self):
        self.approve.return_value = False
        with patch.object(LockedHandle, "read") as read, patch.object(harness, "_run_restricted_powershell") as runner:
            for _ in range(2):
                result = call(self.tools, "powershell_run", path="hello.ps1")
                self.assertEqual(result["status"], "denied")
                self.assertFalse(result["execution_started"])
            read.assert_not_called()
            runner.assert_not_called()
        self.assertEqual(self.approve.call_count, 2)

    def test_fresh_approval_precedes_content_read_and_cannot_change_path_or_timeout(self):
        reads = []
        original_read = LockedHandle.read

        def read(handle, limit):
            reads.append(handle.path)
            return original_read(handle, limit)

        def approve(request):
            self.assertEqual(reads, [])
            self.assertEqual(request.preview, "")
            self.assertIn(str(self.script), request.summary)
            self.assertIn("7 с", request.summary)
            self.assertIn("Write-Host", request.summary)
            request.arguments["path"] = "../outside.ps1"
            request.arguments["timeout_seconds"] = 1000
            return True

        self.tools.approve = approve
        with patch.object(LockedHandle, "read", new=read), patch.object(harness, "_run_restricted_powershell", return_value=self.response) as runner:
            result = call(self.tools, "powershell_run", path="hello.ps1", timeout_seconds=7)
        runner.assert_called_once_with(self.source, self.repository, 7)
        self.assertEqual(reads, [self.script])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["path"], "hello.ps1")
        self.assertTrue(result["execution_started"])

    def test_every_script_run_requires_approval_and_keeps_file_and_root_pinned(self):
        def run(source, repository, timeout):
            self.assertEqual(source, self.source)
            self.assertEqual(repository, self.repository)
            with self.assertRaises(OSError):
                self.script.rename(self.repository / "swapped.ps1")
            with self.assertRaises(OSError):
                self.repository.rename(self.base / "swapped-repo")
            with self.assertRaises(OSError):
                os.link(self.script, self.base / "outside-alias.ps1")
            return self.response

        with patch.object(harness, "run_powershell", side_effect=run) as runner:
            for _ in range(2):
                result = call(self.tools, "powershell_run", path="hello.ps1")
                self.assertEqual(result["status"], "ok")
                self.assertTrue(result["restricted"])
                self.assertEqual(result["stdout"], "Hello, World!\n")
        self.assertEqual(self.approve.call_count, 2)
        self.assertEqual(runner.call_count, 2)

    def test_malformed_paths_extensions_and_timeouts_fail_before_approval(self):
        invalid = [{"path": "../hello.ps1"}, {"path": r"C:\hello.ps1"}, {"path": "hello.ps1:ads"},
                   {"path": "script.exe"}, {"path": "script.txt"}, {"path": ".ps1"},
                   {"path": "hello.ps1", "command": "Write-Output bad"}]
        invalid += [{"path": "hello.ps1", "timeout_seconds": value} for value in (0, 16, True, "5", None)]
        with patch.object(LockedHandle, "read") as read, patch.object(harness, "_run_restricted_powershell") as runner:
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    result = call(self.tools, "powershell_run", **arguments)
                    self.assertEqual(result["status"], "error")
                    self.assertFalse(result["execution_started"])
            read.assert_not_called()
            runner.assert_not_called()
        self.approve.assert_not_called()

    def test_oversized_script_is_rejected_using_metadata_without_content_read(self):
        self.script.write_bytes(b"x" * (harness.MAX_FILE_BYTES + 1))
        with patch.object(LockedHandle, "read") as read, patch.object(harness, "_run_restricted_powershell") as runner:
            self.assertEqual(call(self.tools, "powershell_run", path="hello.ps1")["status"], "error")
            read.assert_not_called()
            runner.assert_not_called()
        self.approve.assert_not_called()

    def test_revoked_permission_stops_before_script_read_or_execution(self):
        def approve(request):
            self.tools.settings = dataclasses.replace(self.tools.settings, powershell_enabled=False)
            return True

        self.tools.approve = approve
        with patch.object(LockedHandle, "read") as read, patch.object(harness, "_run_restricted_powershell") as runner:
            result = call(self.tools, "powershell_run", path="hello.ps1")
            self.assertEqual(result["status"], "denied")
            self.assertFalse(result["execution_started"])
            read.assert_not_called()
            runner.assert_not_called()

    def test_hardlinked_script_is_rejected_before_approval_or_execution(self):
        os.link(self.script, self.base / "outside-alias.ps1")
        with patch.object(LockedHandle, "read") as read, patch.object(harness, "_run_restricted_powershell") as runner:
            self.assertEqual(call(self.tools, "powershell_run", path="hello.ps1")["status"], "error")
            read.assert_not_called()
            runner.assert_not_called()
        self.approve.assert_not_called()

    def test_junction_script_parent_is_rejected_before_approval_or_execution(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "hello.ps1").write_text(self.source, encoding="utf-8")
        link = self.repository / "junction"
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True)
        if completed.returncode:
            self.skipTest("Junction creation unavailable: " + completed.stderr)
        try:
            with patch.object(LockedHandle, "read") as read, patch.object(harness, "_run_restricted_powershell") as runner:
                self.assertEqual(call(self.tools, "powershell_run", path="junction/hello.ps1")["status"], "error")
                read.assert_not_called()
                runner.assert_not_called()
            self.approve.assert_not_called()
        finally:
            link.rmdir()

    def test_non_utf8_and_binary_scripts_are_not_executed_after_approval(self):
        with patch.object(harness, "_run_restricted_powershell") as runner:
            for content in (b"\xff\xfeW\x00", b"Write-Output\x00 hi"):
                self.script.write_bytes(content)
                result = call(self.tools, "powershell_run", path="hello.ps1")
                self.assertEqual(result["status"], "error")
                self.assertFalse(result["execution_started"])
            runner.assert_not_called()
        self.assertEqual(self.approve.call_count, 2)

    def test_runner_rejection_and_timeout_report_whether_execution_started(self):
        with patch.object(harness, "run_powershell") as runner:
            for started in (False, True):
                runner.side_effect = harness.PowerShellError("rejected or timed out", started=started)
                result = call(self.tools, "powershell_run", path="hello.ps1")
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["execution_started"], started)
                self.assertTrue(result["restricted"])
                self.assertIn("Write-Output", result["available_cmdlets"])

    def test_nonzero_exit_and_missing_restriction_evidence_fail_closed(self):
        with patch.object(harness, "run_powershell", return_value={**self.response, "exit_code": 1, "stderr": "script error"}) as runner:
            result = call(self.tools, "powershell_run", path="hello.ps1")
            self.assertEqual(result["status"], "error")
            self.assertTrue(result["execution_started"])
            self.assertEqual(result["stderr"], "script error")
            runner.return_value = {**self.response, "restricted": False}
            result = call(self.tools, "powershell_run", path="hello.ps1")
            self.assertEqual(result["status"], "error")
            self.assertTrue(result["execution_started"])
        self.assertEqual(runner.call_count, 2)

    def test_trusted_runner_is_loaded_before_approved_application_source_write(self):
        # Isolate a fake application package in the temporary repository. Never
        # modify the real application's module or invoke actual PowerShell here.
        package_name = "llmopenchat_capture_regression"
        package = self.base / package_name
        package.mkdir()
        (package / "__init__.py").write_bytes(b"")
        (package / "harness.py").write_bytes(Path(harness.__file__).read_bytes())
        (package / "_winfiles.py").write_bytes(Path(_winfiles.__file__).read_bytes())
        trusted_source = (
            f"AVAILABLE_CMDLETS = {harness._POWERSHELL_CMDLETS!r}\n"
            "class PowerShellError(RuntimeError):\n    started = False\n"
            "def run_powershell(source, repository, timeout_seconds):\n"
            "    return {'stdout': 'trusted', 'stderr': '', 'exit_code': 0, 'restricted': True, 'available_cmdlets': list(AVAILABLE_CMDLETS)}\n"
        )
        (package / "powershell.py").write_bytes(trusted_source.encode("utf-8"))
        (package / "hello.ps1").write_bytes(self.source.encode("utf-8"))
        try:
            with patch.object(sys, "path", [str(self.base), *sys.path]):
                isolated = importlib.import_module(package_name + ".harness")
            self.assertIn(package_name + ".powershell", sys.modules)
            runner_module = sys.modules[package_name + ".powershell"]
            captured = isolated.run_powershell
            self.assertIs(captured, runner_module.run_powershell)
            tools = isolated.ToolHarness(isolated.HarnessSettings(code_enabled=True, powershell_enabled=True, repository=package), lambda request: True)
            overwritten = call(tools, "write_file", path="powershell.py", content="raise AssertionError('untrusted module loaded after consent')\n")
            self.assertEqual(overwritten["status"], "ok")
            runner_module.run_powershell = Mock(side_effect=AssertionError("late module rebinding used"))
            self.assertIs(isolated.run_powershell, captured)
            result = call(tools, "powershell_run", path="hello.ps1")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["stdout"], "trusted")
            runner_module.run_powershell.assert_not_called()
        finally:
            for name in [name for name in sys.modules if name == package_name or name.startswith(package_name + ".")]:
                del sys.modules[name]


@unittest.skipUnless(supported(), "Secure file tooling requires Windows native handles")
class RepositoryHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.repository = self.base / "repo"
        self.repository.mkdir()
        self.approvals = []
        self.tools = ToolHarness(HarnessSettings(code_enabled=True, repository=self.repository), lambda request: self.approvals.append(request) or True)

    def tearDown(self):
        self.temporary.cleanup()

    def deny(self):
        self.tools.approve = lambda request: self.approvals.append(request) or False

    def test_requires_existing_absolute_local_root(self):
        for path in [Path("relative"), self.base / "missing", self.repository / "..", Path(r"\\server\share"), Path(r"\\?\C:\repo")]:
            with self.subTest(path=path), self.assertRaises(HarnessError):
                ToolHarness(HarnessSettings(code_enabled=True, repository=path), lambda request: True)
        with self.assertRaises(HarnessError):
            ToolHarness(HarnessSettings(code_enabled=True), lambda request: True)

    def test_unsafe_paths_are_rejected_before_approval(self):
        forbidden = ["../outside", "a/../../outside", "./a", "/outside", r"C:\outside", "C:relative", r"\\server\share\file", r"\\?\C:\outside", "file:ads", ".git/config", "a/.GiT/config", "NUL", "nul.txt", "CONIN$", "CONOUT$", "CLOCK$", "COM¹.txt", "LPT²", "a.", "a ", "a//b", "a~1", "a\x00b", "a\x1bb", "a\u202eb"]
        for path in forbidden:
            with self.subTest(path=path):
                self.assertEqual(call(self.tools, "write_file", path=path, content="bad")["status"], "error")
                self.assertEqual(call(self.tools, "create_directory", path=path)["status"], "error")
        self.assertEqual(self.approvals, [])
        self.assertEqual(list(self.repository.iterdir()), [])

    def test_each_read_write_listing_search_requires_new_approval(self):
        for _ in range(2):
            self.assertEqual(call(self.tools, "write_file", path="code.py", content="print('hello')\n")["status"], "ok")
            self.assertEqual(call(self.tools, "read_file", path="code.py")["content"], "print('hello')\n")
            self.assertEqual(call(self.tools, "list_files")["entries"][0]["path"], "code.py")
            self.assertEqual(call(self.tools, "search_files", query="HELLO")["matches"][0]["line"], 1)
        self.assertEqual(len(self.approvals), 8)

    def test_denied_invocations_never_read_write_enumerate_or_create(self):
        target = self.repository / "existing.txt"
        target.write_text("secret", encoding="utf-8")
        self.deny()
        with patch.object(LockedHandle, "read") as read, patch.object(LockedHandle, "write") as write, patch.object(harness.os, "scandir") as scan, patch.object(_winfiles._kernel, "CreateDirectoryW") as mkdir:
            for name, arguments in [("read_file", {"path": "existing.txt"}), ("write_file", {"path": "existing.txt", "content": "replacement"}), ("write_file", {"path": "new.txt", "content": "new"}), ("write_file", {"path": "new-project/src/Program.cs", "content": "new"}), ("create_directory", {"path": "new-directory/nested"}), ("list_files", {}), ("search_files", {"query": "secret"})]:
                self.assertEqual(call(self.tools, name, **arguments)["status"], "denied")
            read.assert_not_called()
            write.assert_not_called()
            scan.assert_not_called()
            mkdir.assert_not_called()
        self.assertEqual(target.read_text(encoding="utf-8"), "secret")
        self.assertFalse((self.repository / "new.txt").exists())
        self.assertFalse((self.repository / "new-project").exists())
        self.assertFalse((self.repository / "new-directory").exists())
        self.assertEqual(len(self.approvals), 7)

    def test_write_preview_is_exact_full_new_content_and_cannot_redirect_write(self):
        def approve(request):
            self.approvals.append(request)
            self.assertEqual(request.preview, "print('привет')\n")
            self.assertIn(str(self.repository / "safe.py"), request.summary)
            request.arguments["path"] = "../outside.py"
            request.arguments["content"] = "malicious"
            return True

        self.tools.approve = approve
        result = call(self.tools, "write_file", path="safe.py", content="print('привет')\n")
        self.assertEqual(result["status"], "ok")
        self.assertEqual((self.repository / "safe.py").read_text(encoding="utf-8"), "print('привет')\n")
        self.assertFalse((self.base / "outside.py").exists())

    def test_empty_write_truncates_and_nested_files_stay_in_repo(self):
        (self.repository / "nested").mkdir()
        self.assertEqual(call(self.tools, "write_file", path=r"nested\code.py", content="x = 1\n")["status"], "ok")
        self.assertEqual(call(self.tools, "write_file", path="nested/code.py", content="")["bytes_written"], 0)
        self.assertEqual((self.repository / "nested" / "code.py").read_bytes(), b"")

    def test_create_directory_approves_before_mkdir_and_is_idempotent(self):
        def approve(request):
            self.approvals.append(request)
            if len(self.approvals) == 1:
                self.assertFalse((self.repository / "app").exists())
                self.assertIn(str(self.repository / "app"), request.summary)
                self.assertIn(str(self.repository / "app" / "src"), request.summary)
            return True

        self.tools.approve = approve
        created = call(self.tools, "create_directory", path="app/src")
        self.assertEqual(created, {"status": "ok", "path": "app/src", "created": True, "directories_created": ["app", "app/src"]})
        self.assertTrue((self.repository / "app" / "src").is_dir())
        existing = call(self.tools, "create_directory", path="app/src")
        self.assertEqual(existing, {"status": "ok", "path": "app/src", "created": False, "directories_created": []})
        self.assertEqual(len(self.approvals), 2)

    def test_write_file_creates_missing_parent_directories_after_full_approval(self):
        program = 'System.Console.WriteLine("Hello, World!");\n'

        def approve(request):
            self.approvals.append(request)
            self.assertFalse((self.repository / "projects").exists())
            self.assertEqual(request.preview, program)
            self.assertIn(str(self.repository / "projects"), request.summary)
            self.assertIn(str(self.repository / "projects" / "HelloWorld"), request.summary)
            return True

        self.tools.approve = approve
        result = call(self.tools, "write_file", path="projects/HelloWorld/Program.cs", content=program)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["created"])
        self.assertEqual(result["directories_created"], ["projects", "projects/HelloWorld"])
        self.assertEqual((self.repository / "projects" / "HelloWorld" / "Program.cs").read_text(encoding="utf-8"), program)
        self.tools.approve = lambda request: self.approvals.append(request) or True
        project = '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework></PropertyGroup></Project>\n'
        self.assertEqual(call(self.tools, "write_file", path="projects/HelloWorld/HelloWorld.csproj", content=project)["directories_created"], [])
        self.assertTrue((self.repository / "projects" / "HelloWorld" / "HelloWorld.csproj").is_file())
        self.assertEqual(len(self.approvals), 2)

    def test_existing_parent_prefix_stays_pinned_when_missing_children_are_approved(self):
        (self.repository / "existing").mkdir()

        def approve(request):
            with self.assertRaises(OSError):
                (self.repository / "existing").rename(self.base / "moved")
            return True

        self.tools.approve = approve
        result = call(self.tools, "write_file", path="existing/new/deep/Program.cs", content="class Program {}")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["directories_created"], ["existing/new", "existing/new/deep"])
        self.assertFalse((self.base / "moved").exists())

    def test_create_directory_request_arguments_cannot_redirect_creation(self):
        def approve(request):
            request.arguments["path"] = "../outside"
            return True

        self.tools.approve = approve
        self.assertEqual(call(self.tools, "create_directory", path="safe/nested")["status"], "ok")
        self.assertTrue((self.repository / "safe" / "nested").is_dir())
        self.assertFalse((self.base / "outside").exists())

    def test_racing_ordinary_parent_is_checked_and_target_still_uses_create_new(self):
        def approve(request):
            (self.repository / "racing").mkdir()
            (self.repository / "racing" / "Program.cs").write_text("racing contents", encoding="utf-8")
            return True

        self.tools.approve = approve
        self.assertEqual(call(self.tools, "write_file", path="racing/Program.cs", content="replacement")["status"], "error")
        self.assertEqual((self.repository / "racing" / "Program.cs").read_text(encoding="utf-8"), "racing contents")

    def test_racing_ordinary_directory_is_pinned_before_new_children(self):
        def approve(request):
            (self.repository / "racing").mkdir()
            return True

        self.tools.approve = approve
        result = call(self.tools, "write_file", path="racing/src/Program.cs", content="class Program {}")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["directories_created"], ["racing/src"])
        self.assertTrue((self.repository / "racing" / "src" / "Program.cs").is_file())

    def test_racing_hardlink_in_missing_parent_never_writes_outside(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside", encoding="utf-8")

        def approve(request):
            (self.repository / "racing").mkdir()
            try:
                os.link(outside, self.repository / "racing" / "Program.cs")
            except OSError:
                (self.repository / "racing" / "Program.cs").write_text("racing", encoding="utf-8")
            return True

        self.tools.approve = approve
        self.assertEqual(call(self.tools, "write_file", path="racing/Program.cs", content="replacement")["status"], "error")
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside")

    def test_files_and_hardlinks_cannot_be_used_as_directory_parents(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        os.link(outside, self.repository / "linked")
        for name, arguments in [("write_file", {"path": "linked/new/Program.cs", "content": "replacement"}), ("create_directory", {"path": "linked/new"})]:
            self.assertEqual(call(self.tools, name, **arguments)["status"], "error")
        self.assertEqual(self.approvals, [])
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside")

    def test_limits_and_argument_types_rejected_before_approval(self):
        for name, arguments in [("list_files", {"max_results": True}), ("list_files", {"max_results": 101}), ("search_files", {"query": ""}), ("search_files", {"query": "x", "case_sensitive": 1}), ("write_file", {"path": "too-big", "content": "x" * (harness.MAX_FILE_BYTES + 1)})]:
            self.assertEqual(call(self.tools, name, **arguments)["status"], "error")
        self.assertEqual(self.approvals, [])

    def test_read_limits_apply_after_approval_without_partial_contents(self):
        (self.repository / "big.txt").write_bytes(b"x" * (harness.MAX_FILE_BYTES + 1))
        (self.repository / "binary.bin").write_bytes(b"\xff\xfe")
        self.assertEqual(call(self.tools, "read_file", path="big.txt")["status"], "error")
        self.assertEqual(call(self.tools, "read_file", path="binary.bin")["status"], "error")
        self.assertEqual(len(self.approvals), 2)

    def test_git_and_hardlinks_are_never_scanned_or_read(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside secret", encoding="utf-8")
        os.link(outside, self.repository / "linked.txt")
        (self.repository / ".git").mkdir()
        (self.repository / ".git" / "config").write_text("outside secret", encoding="utf-8")
        (self.repository / "ordinary.txt").write_text("inside", encoding="utf-8")
        for name in ["read_file", "write_file"]:
            arguments = {"path": "linked.txt"}
            if name == "write_file":
                arguments["content"] = "bad"
            self.assertEqual(call(self.tools, name, **arguments)["status"], "error")
        self.assertEqual(self.approvals, [])
        self.assertEqual(call(self.tools, "list_files", recursive=True)["entries"], [{"path": "ordinary.txt", "type": "file", "bytes": 6}])
        self.assertEqual(call(self.tools, "search_files", query="outside")["matches"], [])
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside secret")

    def test_scan_is_bounded_and_reports_truncation(self):
        for name in ["one.txt", "two.txt", "three.txt"]:
            (self.repository / name).write_text("match\nmatch", encoding="utf-8")
        self.assertTrue(call(self.tools, "list_files", max_results=1)["truncated"])
        result = call(self.tools, "search_files", query="match", max_results=2)
        self.assertEqual(len(result["matches"]), 2)
        self.assertTrue(result["truncated"])

    def test_repository_identity_survives_new_harness_across_turns(self):
        settings = self.tools.settings
        self.assertIsNotNone(settings.repository_identity)
        self.repository.rename(self.base / "old-repo")
        self.repository.mkdir()
        with self.assertRaises(HarnessError):
            ToolHarness(settings, lambda request: True)
        self.assertEqual(call(self.tools, "list_files")["status"], "error")
        self.assertEqual(self.approvals, [])

    def test_repo_ancestors_and_existing_target_cannot_be_swapped_during_approval(self):
        target = self.repository / "existing.txt"
        target.write_text("original", encoding="utf-8")
        substitute = self.repository / "substitute.txt"
        substitute.write_text("substitute", encoding="utf-8")

        def approve(request):
            for source, destination in [(self.repository, self.base / "moved"), (target, self.repository / "moved.txt"), (substitute, target)]:
                with self.assertRaises(OSError):
                    os.replace(source, destination)
            # A write-capable directory handle is needed to set a junction.
            from llmopenchat import _winfiles
            kernel = _winfiles._kernel
            handle = kernel.CreateFileW(str(self.repository), 0x40000000, 7, None, 3, 0x02200000, None)
            self.assertEqual(handle, _winfiles.ctypes.c_void_p(-1).value)
            return True

        self.tools.approve = approve
        self.assertEqual(call(self.tools, "write_file", path="existing.txt", content="new")["status"], "ok")
        self.assertEqual(target.read_text(encoding="utf-8"), "new")

    def test_new_target_creation_race_fails_without_touching_outside_hardlink(self):
        outside = self.base / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        def approve(request):
            try:
                os.link(outside, self.repository / "new.txt")
            except OSError:
                # The pinned directory can itself block hardlink creation;
                # an ordinary racing file must still never be overwritten.
                (self.repository / "new.txt").write_text("racing", encoding="utf-8")
            return True

        self.tools.approve = approve
        self.assertEqual(call(self.tools, "write_file", path="new.txt", content="bad")["status"], "error")
        self.assertEqual(outside.read_text(encoding="utf-8"), "secret")

    def test_locked_target_cannot_gain_outside_hardlink_during_approval(self):
        target = self.repository / "existing.txt"
        target.write_text("original", encoding="utf-8")
        outside_alias = self.base / "outside-alias.txt"

        def approve(request):
            with self.assertRaises(OSError):
                os.link(target, outside_alias)
            return True

        self.tools.approve = approve
        self.assertEqual(call(self.tools, "read_file", path="existing.txt")["content"], "original")
        self.assertEqual(call(self.tools, "write_file", path="existing.txt", content="new")["status"], "ok")
        self.assertFalse(outside_alias.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "new")

    def test_preexisting_write_capable_directory_handle_rejects_root_lock(self):
        from llmopenchat import _winfiles
        kernel = _winfiles._kernel
        handle = kernel.CreateFileW(str(self.repository), 0x40000000, 7, None, 3, 0x02200000, None)
        self.assertNotEqual(handle, _winfiles.ctypes.c_void_p(-1).value)
        try:
            with self.assertRaises(HarnessError):
                ToolHarness(self.tools.settings, lambda request: True)
            self.assertEqual(call(self.tools, "list_files")["status"], "error")
            self.assertEqual(self.approvals, [])
        finally:
            kernel.CloseHandle(handle)

    def test_settings_revocation_during_approval_prevents_write(self):
        def approve(request):
            self.tools.settings = dataclasses.replace(self.tools.settings, code_enabled=False)
            return True

        self.tools.approve = approve
        self.assertEqual(call(self.tools, "write_file", path="new.txt", content="bad")["status"], "denied")
        self.assertFalse((self.repository / "new.txt").exists())

    def test_settings_revocation_during_approval_prevents_directory_creation(self):
        for name, arguments in [("write_file", {"path": "new-project/src/Program.cs", "content": "class Program {}"}), ("create_directory", {"path": "new-project/src"})]:
            def approve(request):
                active.settings = dataclasses.replace(active.settings, code_enabled=False)
                return True

            active = ToolHarness(HarnessSettings(code_enabled=True, repository=self.repository), approve)
            self.assertEqual(call(active, name, **arguments)["status"], "denied")
            self.assertFalse((self.repository / "new-project").exists())

    def make_junction(self, link, target):
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if completed.returncode:
            self.skipTest("Junction creation unavailable: " + completed.stderr)

    def test_junctions_in_target_root_and_root_ancestors_are_blocked(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("outside", encoding="utf-8")
        link = self.repository / "junction"
        self.make_junction(link, outside)
        try:
            self.assertEqual(call(self.tools, "read_file", path="junction/secret.txt")["status"], "error")
            self.assertEqual(call(self.tools, "write_file", path="junction/new/Program.cs", content="class Program {}")["status"], "error")
            self.assertEqual(call(self.tools, "create_directory", path="junction/new")["status"], "error")
            self.assertEqual(self.approvals, [])
            self.assertFalse((outside / "new").exists())
            self.assertEqual(call(self.tools, "list_files", recursive=True)["entries"], [])
            self.assertEqual(call(self.tools, "search_files", query="outside")["matches"], [])
            with self.assertRaises(HarnessError):
                ToolHarness(HarnessSettings(code_enabled=True, repository=link), lambda request: True)
            with self.assertRaises(HarnessError):
                ToolHarness(HarnessSettings(code_enabled=True, repository=link / "nested"), lambda request: True)
        finally:
            link.rmdir()

    def test_junction_racing_in_missing_parent_cannot_receive_new_children(self):
        outside = self.base / "outside"
        outside.mkdir()
        link = self.repository / "racing"

        def approve(request):
            self.make_junction(link, outside)
            return True

        self.tools.approve = approve
        for name, arguments in [("write_file", {"path": "racing/new/Program.cs", "content": "class Program {}"}), ("create_directory", {"path": "racing/new"})]:
            try:
                self.assertEqual(call(self.tools, name, **arguments)["status"], "error")
                self.assertFalse((outside / "new").exists())
            finally:
                if link.exists():
                    link.rmdir()

    def test_created_directory_replaced_by_junction_before_pin_fails_closed(self):
        outside = self.base / "outside"
        outside.mkdir()
        link = self.repository / "fresh"
        create = _winfiles._kernel.CreateDirectoryW

        def race(path, security):
            result = create(path, security)
            if result and Path(path.removeprefix("\\\\?\\")) == link:
                link.rmdir()
                self.make_junction(link, outside)
            return result

        try:
            with patch.object(_winfiles._kernel, "CreateDirectoryW", side_effect=race):
                self.assertEqual(call(self.tools, "write_file", path="fresh/new/Program.cs", content="class Program {}")["status"], "error")
            self.assertFalse((outside / "new").exists())
        finally:
            if link.exists():
                link.rmdir()

    def test_symlink_file_and_dangling_symlink_block_before_approval(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        try:
            os.symlink(outside, self.repository / "link.txt")
            os.symlink(self.base / "missing.txt", self.repository / "dangling.txt")
        except OSError as error:
            self.skipTest("Symlink privilege unavailable: " + str(error))
        for path in ["link.txt", "dangling.txt"]:
            self.assertEqual(call(self.tools, "write_file", path=path, content="bad")["status"], "error")
        self.assertEqual(self.approvals, [])
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside")


class WebBoundaryTests(unittest.TestCase):
    def test_forbidden_urls_are_rejected_before_approval(self):
        approve = Mock(return_value=True)
        tools = ToolHarness(HarnessSettings(web_enabled=True), approve)
        urls = ["file:///etc/passwd", "ftp://example.com/", "http://127.0.0.1/", "http://10.0.0.1/", "http://169.254.169.254/", "http://100.64.0.1/", "http://[::1]/", "http://[::ffff:8.8.8.8]/", "http://localhost/", "http://service.local/", "http://user:pass@example.com/", "https://example.com:8080/", "https://example.com/#secret", "https://example.com/\nHeader:bad", r"http://example.com\@127.0.0.1/"]
        with patch.object(harness, "_request_public") as request:
            for url in urls:
                with self.subTest(url=url):
                    self.assertEqual(call(tools, "web_fetch", url=url)["status"], "error")
            request.assert_not_called()
        approve.assert_not_called()

    def test_dns_requires_every_address_to_be_public(self):
        public = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))
        for address in ["127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1", "192.0.0.8", "100.64.0.1", "224.0.0.1"]:
            private = (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
            with self.subTest(address=address), patch.object(socket, "getaddrinfo", return_value=[public, private]):
                with self.assertRaises(HarnessError):
                    harness._resolve_public("example.com", 443, harness.time.monotonic() + 1)
        with patch.object(socket, "getaddrinfo", return_value=[public]):
            self.assertEqual(harness._resolve_public("example.com", 443, harness.time.monotonic() + 1), [public])

    def test_ipv6_special_ranges_and_mapped_addresses_are_blocked(self):
        for address in ["::1", "::ffff:8.8.8.8", "64:ff9b::a00:1", "2002:0a00:0001::", "2001::1", "fec0::1", "ff02::1"]:
            self.assertFalse(harness._public_address(address), address)
        self.assertTrue(harness._public_address("2606:4700:4700::1111"))

    def test_html_is_bounded_and_links_require_their_own_future_approval(self):
        content = b'<html><script>secret-script</script><p>Hello world</p><a href="https://example.org/read">Read</a><a href="http://127.0.0.1/">Private</a></html>'
        with patch.object(harness, "_request_public", return_value=(content, "text/html; charset=utf-8")) as request:
            result = harness._web_text("https://example.com/")
            self.assertIn("Hello world", result["text"])
            self.assertNotIn("secret-script", result["text"])
            self.assertEqual(result["links"], [{"title": "Read", "url": "https://example.org/read"}])
            request.assert_called_once()


class FakeSocket:
    def __init__(self):
        self.connections = []
        self.timeouts = []
        self.closed = False
        self.shut_down = False

    def connect(self, address):
        self.connections.append(address)

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def shutdown(self, direction):
        self.shut_down = True

    def close(self):
        self.closed = True

    def do_handshake(self):
        pass


class FakeResponse:
    def __init__(self, status=200, headers=None, content=b"ok"):
        self.status = status
        self.headers = headers or {"Content-Type": "text/plain", "Content-Length": str(len(content))}
        self.content = content

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read(self, count):
        return self.content[:count]


class PublicRequestTests(unittest.TestCase):
    def request(self, response, *, url="http://example.com/", tls_context=None):
        sock = FakeSocket()
        connection = Mock()
        connection.getresponse.return_value = response
        records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))]
        with patch.object(socket, "getaddrinfo", return_value=records) as dns, patch.object(socket, "socket", return_value=sock), patch.object(harness, "HTTPConnection", return_value=connection), patch.object(harness.ssl, "create_default_context", return_value=tls_context):
            try:
                result = harness._request_public(url)
            except HarnessError as error:
                result = error
        return result, sock, connection, dns

    def test_only_one_pinned_direct_get_no_proxy_cookie_or_credentials(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "LLMOPENCHAT_API_KEY": "secret"}):
            result, sock, connection, dns = self.request(FakeResponse())
        self.assertEqual(result, (b"ok", "text/plain"))
        self.assertEqual(sock.connections, [("8.8.8.8", 80)])
        dns.assert_called_once_with("example.com", 80, type=socket.SOCK_STREAM)
        connection.request.assert_called_once()
        args, kwargs = connection.request.call_args
        self.assertEqual(args[:2], ("GET", "/"))
        self.assertFalse({"Authorization", "Proxy-Authorization", "Cookie"} & set(kwargs["headers"]))
        self.assertTrue(sock.closed)

    def test_redirect_reports_canonical_target_without_following(self):
        for location, expected in [("/next", "http://example.com/next"), ("https://example.org/next", "https://example.org/next"), ("http://127.0.0.1/private", "небезопасен")]:
            with self.subTest(location=location):
                result, sock, connection, dns = self.request(FakeResponse(302, {"Location": location}))
                self.assertIsInstance(result, HarnessError)
                self.assertIn(expected, str(result))
                self.assertEqual(len(sock.connections), 1)
                connection.request.assert_called_once()
                dns.assert_called_once()

    def test_oversized_compressed_and_binary_responses_are_rejected(self):
        cases = [FakeResponse(headers={"Content-Type": "text/plain", "Content-Length": str(harness.MAX_WEB_BYTES + 1)}), FakeResponse(headers={"Content-Type": "text/plain", "Content-Encoding": "gzip"}), FakeResponse(headers={"Content-Type": "application/octet-stream"}), FakeResponse(headers={"Content-Type": "text/plain"}, content=b"x" * (harness.MAX_WEB_BYTES + 1))]
        for response in cases:
            with self.subTest(headers=response.headers):
                result, sock, _, _ = self.request(response)
                self.assertIsInstance(result, HarnessError)
                self.assertTrue(sock.closed)

    def test_tls_handoff_is_registered_before_handshake(self):
        tls_socket = FakeSocket()
        context = Mock()
        context.wrap_socket.return_value = tls_socket
        result, sock, _, _ = self.request(FakeResponse(), url="https://example.com/", tls_context=context)
        self.assertEqual(result, (b"ok", "text/plain"))
        context.wrap_socket.assert_called_once_with(sock, server_hostname="example.com", do_handshake_on_connect=False)
        self.assertTrue(tls_socket.closed)

    def test_timer_can_interrupt_tls_handshake_without_losing_socket_ownership(self):
        tls_socket = FakeSocket()
        context = Mock()
        timer_fired = threading.Event()

        def handshake():
            self.assertTrue(timer_fired.wait(timeout=1))
            self.assertTrue(tls_socket.closed)
            raise OSError("deadline interrupted handshake")

        tls_socket.do_handshake = handshake

        def wrap(sock, **kwargs):
            self.assertFalse(kwargs["do_handshake_on_connect"])
            return tls_socket

        context.wrap_socket.side_effect = wrap
        actual_timer = harness.threading.Timer

        def timer(interval, interrupt):
            def fire():
                interrupt()
                timer_fired.set()
            return actual_timer(0.025, fire)

        with patch.object(harness.threading, "Timer", side_effect=timer):
            with self.assertRaises(OSError):
                self.request(FakeResponse(), url="https://example.com/", tls_context=context)


@unittest.skipUnless(supported(), "Repository policy protection requires Windows native handles")
class HostsPolicyProtectionTests(unittest.TestCase):
    def test_policy_writes_are_rejected_before_approval_and_cannot_disable_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "hosts.txt"
            original = b"https://backend.example.com\n"
            policy.write_bytes(original)
            approve = Mock(return_value=True)
            tools = ToolHarness(HarnessSettings(code_enabled=True, repository=root), approve,
                                protected_names=frozenset())
            for path in ("hosts.txt", "HOSTS.TXT", "nested/Hosts.Txt", "hosts.txt/child.txt"):
                with self.subTest(path=path):
                    self.assertEqual(call(tools, "write_file", path=path, content="https://attacker.example/")["status"], "error")
                    self.assertEqual(call(tools, "create_directory", path=path)["status"], "error")
            approve.assert_not_called()
            self.assertEqual(policy.read_bytes(), original)
            self.assertEqual(list(root.iterdir()), [policy])
            self.assertEqual(call(tools, "write_file", path="result.json", content="{}")["status"], "ok")

    def test_policy_hardlink_alias_cannot_be_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, alias = root / "hosts.txt", root / "other.txt"
            original = b"https://backend.example.com\n"
            policy.write_bytes(original)
            os.link(policy, alias)
            approve = Mock(return_value=True)
            tools = ToolHarness(HarnessSettings(code_enabled=True, repository=root), approve)
            self.assertEqual(call(tools, "write_file", path="other.txt", content="changed")["status"], "error")
            approve.assert_not_called()
            self.assertEqual(policy.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
