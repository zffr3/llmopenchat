"""Restricted Windows/Linux runspaces: output, blocked escapes and bounds."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from llmopenchat import powershell
from llmopenchat.powershell import AVAILABLE_CMDLETS, PowerShellError, availability_error, run_powershell, run_script


class PowerShellValidationTests(unittest.TestCase):
    def test_unsupported_platform_never_falls_back(self):
        repository = Path.cwd()
        with patch("llmopenchat.powershell.sys.platform", "darwin"), patch("llmopenchat.powershell.subprocess.Popen") as launch:
            with self.assertRaisesRegex(PowerShellError, "Windows.*Linux"):
                run_script("Write-Output hello", repository)
            launch.assert_not_called()

    def test_availability_reports_prerequisite_error_without_launch(self):
        with patch.object(powershell.sys, "platform", "linux"), patch.object(powershell, "_linux_configuration", side_effect=PowerShellError("Install pwsh")), patch.object(powershell.subprocess, "Popen") as launch:
            self.assertEqual(availability_error(), "Install pwsh")
            launch.assert_not_called()

    def test_linux_requires_non_root_user_for_kernel_task_limit(self):
        with patch.object(powershell.os, "geteuid", return_value=0, create=True), patch.object(powershell, "_trusted_linux_executable") as executable:
            with self.assertRaisesRegex(PowerShellError, "root"):
                powershell._linux_configuration(5)
            executable.assert_not_called()

    def test_linux_applies_hard_limits_and_clean_environment_before_pwsh(self):
        paths = (Path.cwd() / "trusted-pwsh", Path.cwd() / "trusted-prlimit")
        with patch.object(powershell.os, "geteuid", return_value=1000, create=True), patch.object(powershell, "_trusted_linux_executable", side_effect=paths), patch.object(powershell, "_linux_task_limit", return_value=700), patch.dict(os.environ, {"LD_PRELOAD": "evil", "PSModulePath": "evil", "DOTNET_GCHeapHardLimit": "0"}):
            command, directory, environment = powershell._linux_configuration(3)
        self.assertEqual(command[0], str(paths[1]))
        self.assertEqual(command[-2:], ["--", str(paths[0])])
        self.assertIn(f"--as={powershell.MAX_LINUX_ADDRESS_BYTES}", command)
        self.assertIn("--cpu=8", command)
        self.assertIn("--nproc=700", command)
        self.assertIn("--fsize=0", command)
        self.assertIn("--core=0", command)
        self.assertEqual(environment["DOTNET_GCHeapHardLimit"], f"{powershell.MAX_PROCESS_BYTES:x}")
        self.assertEqual(environment["PSModulePath"], str(paths[0].parent / "Modules"))
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertEqual(environment["HOME"], "/nonexistent")

    def test_linux_uses_same_driver_new_process_group_and_bounded_reader(self):
        repository = Path.cwd()
        response = b'{"stdout":"ok","stderr":"","exit_code":0,"execution_started":true}'
        with patch.object(powershell.sys, "platform", "linux"), patch.object(powershell, "_linux_configuration", return_value=(["trusted-prlimit", "--", "trusted-pwsh"], repository, {})), patch.object(powershell.subprocess, "Popen") as launch, patch.object(powershell, "_RestrictedJob") as job, patch.object(powershell, "_communicate_linux", return_value=(response, b"")) as reader, patch.object(powershell.os, "killpg", create=True) as kill, patch.object(signal, "SIGKILL", 9, create=True):
            launch.return_value.pid = 42
            launch.return_value.returncode = 0
            result = run_script("Write-Output ok", repository, 3)
        self.assertEqual(result["stdout"], "ok")
        self.assertTrue(result["restricted"])
        self.assertEqual(launch.call_args.args[0][:3], ["trusted-prlimit", "--", "trusted-pwsh"])
        self.assertIn("-EncodedCommand", launch.call_args.args[0])
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertNotIn("creationflags", launch.call_args.kwargs)
        self.assertEqual(reader.call_args.args[2], 8)
        self.assertEqual(json.loads(reader.call_args.args[1])["timeout"], 3)
        kill.assert_called_once_with(42, 9)
        job.assert_not_called()

    def test_linux_does_not_launch_without_prerequisites(self):
        repository = Path.cwd()
        with patch.object(powershell.sys, "platform", "linux"), patch.object(powershell, "_linux_configuration", side_effect=PowerShellError("No trusted pwsh")), patch.object(powershell.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(PowerShellError, "trusted pwsh"):
                run_script("Write-Output ok", repository)
            launch.assert_not_called()

    def test_linux_io_failure_after_dispatch_preserves_started_flag(self):
        repository = Path.cwd()
        with patch.object(powershell.sys, "platform", "linux"), patch.object(powershell, "_linux_configuration", return_value=(["trusted-prlimit", "--", "trusted-pwsh"], repository, {})), patch.object(powershell.subprocess, "Popen") as launch, patch.object(powershell, "_communicate_linux", side_effect=OSError("Pipe failure")), patch.object(powershell.os, "killpg", create=True), patch.object(signal, "SIGKILL", 9, create=True):
            launch.return_value.wait.side_effect = subprocess.TimeoutExpired("pwsh", 5)
            with self.assertRaisesRegex(PowerShellError, "Pipe failure") as raised:
                run_script("Write-Output ok", repository)
            self.assertTrue(raised.exception.started)


@unittest.skipUnless(sys.platform == "win32" or (sys.platform == "linux" and availability_error() is None),
                     "Requires trusted PowerShell; Linux also requires util-linux and a non-root user")
class RestrictedPowerShellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmopenchat-restricted-ps-")
        self.addCleanup(self.temp.cleanup)
        self.repository = Path(self.temp.name)

    def run_source(self, source, timeout=3):
        return run_powershell(source, self.repository, timeout)

    def test_output_arithmetic_variables_and_control_flow(self):
        result = self.run_source('$total=0; for($i=0;$i -lt 3;$i++){ $total += $i }; if($total -eq 3){Write-Output "Hello, World!"}; Write-Output $total')
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].splitlines(), ["Hello, World!", "3"])
        self.assertTrue(result["restricted"])
        self.assertTrue(result["execution_started"])
        self.assertEqual(result["available_cmdlets"], list(AVAILABLE_CMDLETS))
        self.assertEqual(list(self.repository.iterdir()), [])

    def test_write_host_unicode_and_warning_error_streams(self):
        result = self.run_source('Write-Host "Привет, мир!"; Write-Warning "warning"; Write-Error "failure"')
        self.assertIn("Привет, мир!", result["stdout"])
        self.assertIn("warning", result["stderr"])
        self.assertIn("failure", result["stderr"])
        self.assertEqual(result["exit_code"], 1)

    def test_all_escape_routes_are_rejected_before_execution(self):
        sources = [
            'cmd.exe /c echo hello', '/bin/sh -c "echo hello"', 'bash -c "echo hello"',
            'pwsh -Command "echo hello"', '& "Write-Output" hello', '. ./script.ps1',
            'Get-Content ../secret.txt', 'Invoke-WebRequest https://example.com',
            'powershell.exe -EncodedCommand aABpAA==', 'Write-Output hello > ../escape.txt',
            'Write-Output $env:LLMOPENCHAT_TEST_SECRET', 'Write-Output "$env:LLMOPENCHAT_TEST_SECRET"',
            '[System.IO.File]::ReadAllText("../secret.txt")', '("a").ToString()',
            '$x = { Write-Output hi }; & $x', 'function Run { cmd.exe }; Run',
            'Write-Output $(cmd.exe /c echo hello)', 'Write-Output $ExecutionContext',
            'Write-Output $PWD', 'Write-Output $PSHOME', 'Write-Output -OutVariable local hi',
            '$x=[string]"a"; Write-Output $x', 'New-Object System.Net.WebClient',
            '#requires -Modules C:\\outside\\evil.psm1\nWrite-Output hello',
            '#requires -Assembly C:\\outside\\evil.dll\nWrite-Output hello',
            'using assembly "C:\\outside\\evil.dll"\nWrite-Output hello',
            'configuration Setup { Node localhost { } }',
            'using module C:\\outside\\evil.psm1\nWrite-Output hello',
            'if($false){$HOME="x"}; Write-Output $HOME',
            'if($false){$PSHOME="x"}; Write-Output $PSHOME',
        ]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(PowerShellError) as raised:
                self.run_source(source)
            self.assertFalse(raised.exception.started)
        self.assertEqual(list(self.repository.iterdir()), [])

    def test_timeout_stops_infinite_loop_and_reports_execution_started(self):
        with self.assertRaisesRegex(PowerShellError, "timeout|таймаут") as raised:
            self.run_source('while($true) { $i=1 }', timeout=1)
        self.assertTrue(raised.exception.started)

    def test_output_is_bounded_for_strings_and_streams(self):
        for source in ('Write-Output ("x" * 70000)', 'Write-Host ("x" * 70000)', 'foreach($i in 1..3000){Write-Output $i}',
                       'while($true){Write-Warning "x"}'):
            with self.subTest(source=source), self.assertRaisesRegex(PowerShellError, "Output limit") as raised:
                self.run_source(source)
            self.assertTrue(raised.exception.started)

    def test_system_executable_ignores_environment_search_and_profile(self):
        with patch.dict(os.environ, {"SystemRoot": str(self.repository), "PATH": str(self.repository),
                                     "PSModulePath": str(self.repository), "LLMOPENCHAT_TEST_SECRET": "secret-value"}):
            result = self.run_source('Write-Output "trusted"')
        self.assertEqual(result["stdout"].strip(), "trusted")
        self.assertNotIn("secret-value", str(result))
        self.assertEqual(list(self.repository.iterdir()), [])

    def test_invalid_timeout_source_and_repository_fail_closed(self):
        for source, repository, timeout in (("x", self.repository, 0), ("x", self.repository, 16),
                                             ("x", self.repository, True), (7, self.repository, 3),
                                             ("x" * (256 * 1024 + 1), self.repository, 3),
                                             ("Write-Output hi", Path("relative"), 3)):
            with self.subTest(timeout=timeout), patch("llmopenchat.powershell.subprocess.Popen") as launch:
                with self.assertRaises(PowerShellError):
                    run_powershell(source, repository, timeout)
                launch.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows Job Object integration")
    def test_failed_job_assignment_never_sends_model_source(self):
        with patch("llmopenchat.powershell._RestrictedJob") as job_type, patch("llmopenchat.powershell.subprocess.Popen") as launch:
            process = launch.return_value
            process.pid = 42
            process.poll.return_value = 0
            process.communicate.return_value = (b"", b"")
            job_type.return_value.assign.side_effect = PowerShellError("Job assignment failed")
            with self.assertRaisesRegex(PowerShellError, "Job assignment failed") as raised:
                self.run_source('Write-Output "MODEL SOURCE"')
            self.assertFalse(raised.exception.started)
            process.communicate.assert_called_once_with()
            job_type.return_value.close.assert_called_once_with()

    def test_very_large_allocation_fails_with_bounded_output(self):
        try:
            result = self.run_source('Write-Output ("x" * 1073741824)', timeout=2)
        except PowerShellError as error:
            self.assertTrue(error.started)
        else:
            self.assertNotEqual(result["exit_code"], 0)
            self.assertLessEqual(len(result["stdout"].encode("utf-8")), 65536)


@unittest.skipUnless(sys.platform == "linux", "Requires Linux pipe selectors and process groups")
class LinuxPipeLimitTests(unittest.TestCase):
    def test_task_budget_allows_existing_model_threads_and_respects_parent_hard_limit(self):
        import resource

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            own_pid = str(os.getpid())
            for pid, uid, threads in ((own_pid, 1000, 4), (own_pid + "1", 1000, 300), (own_pid + "2", 2000, 500)):
                entry = directory / pid
                entry.mkdir()
                (entry / "status").write_text(f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\nThreads:\t{threads}\n", encoding="ascii")
            with patch.object(powershell, "Path", return_value=directory), patch.object(powershell.os, "getuid", return_value=1000):
                with patch.object(resource, "getrlimit", return_value=(500, 1000)):
                    self.assertEqual(powershell._linux_task_limit(), 304 + powershell.MAX_LINUX_TASKS)
                with patch.object(resource, "getrlimit", return_value=(350, 350)):
                    self.assertEqual(powershell._linux_task_limit(), 350)
                with patch.object(resource, "getrlimit", return_value=(320, 320)):
                    with self.assertRaisesRegex(PowerShellError, "задач"):
                        powershell._linux_task_limit()

    def process(self, source):
        process = subprocess.Popen([sys.executable, "-c", source], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        def cleanup():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            for pipe in (process.stdin, process.stdout, process.stderr):
                pipe.close()
        self.addCleanup(cleanup)
        return process

    def test_reader_handles_large_request_and_both_streams(self):
        process = self.process("import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data); sys.stderr.write('warning')")
        request = b"x" * (256 * 1024)
        output, error = powershell._communicate_linux(process, request, 5)
        self.assertEqual(output, request)
        self.assertEqual(error, b"warning")

    def test_reader_rejects_excessive_output_before_buffering_it_all(self):
        process = self.process("import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(b'x'*1048576)")
        with self.assertRaisesRegex(PowerShellError, "лимит вывода"):
            powershell._communicate_linux(process, b"{}", 5)

    def test_reader_times_out_a_silent_process(self):
        process = self.process("import sys,time; sys.stdin.buffer.read(); time.sleep(60)")
        with self.assertRaises(subprocess.TimeoutExpired):
            powershell._communicate_linux(process, b"{}", 0.1)


if __name__ == "__main__":
    unittest.main()
