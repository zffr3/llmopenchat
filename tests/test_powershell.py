"""Real restricted Windows runspaces: safe output, blocked escapes and bounds."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llmopenchat.powershell import AVAILABLE_CMDLETS, PowerShellError, run_powershell, run_script


class PowerShellValidationTests(unittest.TestCase):
    def test_unsupported_platform_never_falls_back(self):
        repository = Path.cwd()
        with patch("llmopenchat.powershell.os.name", "posix"), patch("llmopenchat.powershell.subprocess.Popen") as launch:
            with self.assertRaisesRegex(PowerShellError, "Windows"):
                run_script("Write-Output hello", repository)
            launch.assert_not_called()


@unittest.skipUnless(os.name == "nt", "Requires the trusted Windows PowerShell executable")
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
            'cmd.exe /c echo hello', '& "Write-Output" hello', '. ./script.ps1',
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


if __name__ == "__main__":
    unittest.main()
