"""Exercise the Windows bootstrap without downloading or installing software."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = (
    Path(os.environ.get("SystemRoot", r"C:\Windows"))
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)
EXPECTED_HASH = "5da012101ba2e17113a9a1cd901e9ce1441086f2b7ecb5de06ec94163899a190"


@unittest.skipUnless(os.name == "nt" and POWERSHELL.is_file(), "Windows PowerShell 5.1 is required")
class WindowsPythonBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "проверка ' & (bootstrap)"
        self.root.mkdir()
        self.local_app_data = self.root / "local app data"
        self.local_app_data.mkdir()

    def powershell(self, body):
        wrapper = self.root / "test-bootstrap.ps1"
        wrapper.write_text(
            "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)\n"
            ". $env:TEST_BOOTSTRAP_SCRIPT\n"
            "function Complete-Test {\n"
            "    param($Value)\n"
            "    [Console]::Out.WriteLine('BOOTSTRAP_RESULT:' + "
            "(ConvertTo-Json -InputObject $Value -Compress -Depth 8))\n"
            "}\n" + body,
            encoding="utf-8-sig",
        )
        environment = os.environ.copy()
        environment.update({
            "TEST_BOOTSTRAP_SCRIPT": str(ROOT / "scripts" / "bootstrap-python.ps1"),
            "TEST_PROJECT_ROOT": str(self.root),
            "TEST_PYTHON": sys.executable,
            "LOCALAPPDATA": str(self.local_app_data),
            "PROCESSOR_ARCHITECTURE": "AMD64",
        })
        environment.pop("PROCESSOR_ARCHITEW6432", None)
        result = subprocess.run(
            [str(POWERSHELL), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(wrapper)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
        )
        output = result.stdout.decode("utf-8")
        error = result.stderr.decode("utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, f"{output}\n{error}")
        records = [line.partition("BOOTSTRAP_RESULT:")[2] for line in output.splitlines()
                   if line.startswith("BOOTSTRAP_RESULT:")]
        self.assertEqual(len(records), 1, f"{output}\n{error}")
        return json.loads(records[0])

    def installer_mocks(self, *, installer_exit=0, create_runtime=True, hash_value=EXPECTED_HASH, download_error=False):
        # Each wrapper runs in a new PowerShell process. These functions shadow
        # cmdlets only there, so no registry, network or system installation changes occur.
        return r"""
$script:events = New-Object 'System.Collections.Generic.List[string]'
$script:downloadPath = $null
$script:installerArguments = @()
$script:installerOptions = $null
$script:installedExecutable = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
function Test-PythonRuntime {
    param([string]$Executable, [string[]]$Arguments = @())
    return (Test-Path -LiteralPath $Executable -PathType Leaf)
}
function Invoke-WebRequest {
    [CmdletBinding()]
    param([string]$Uri, [string]$OutFile, [switch]$UseBasicParsing, [int]$TimeoutSec)
    $script:events.Add('download')
    $script:downloadUri = $Uri
    $script:downloadPath = $OutFile
    [IO.File]::WriteAllText($OutFile, 'fake installer, never executed')
    if (DOWNLOAD_ERROR) { throw 'simulated connection failure' }
}
function Get-FileHash {
    [CmdletBinding()]
    param([string]$LiteralPath, [string]$Algorithm)
    $script:events.Add('hash')
    if ($LiteralPath -ne $script:downloadPath -or $Algorithm -ne 'SHA256') { throw 'Unexpected hash input' }
    return [pscustomobject]@{ Hash = 'HASH_VALUE' }
}
function Start-Process {
    [CmdletBinding()]
    param([string]$FilePath, [string[]]$ArgumentList, [switch]$Wait, [switch]$PassThru, [string]$WindowStyle)
    $script:events.Add('install')
    if ($FilePath -ne $script:downloadPath) { throw 'Unexpected installer path' }
    $script:installerArguments = $ArgumentList
    $script:installerOptions = @{ Wait = [bool]$Wait; PassThru = [bool]$PassThru; WindowStyle = $WindowStyle }
    $log = Join-Path $env:TEST_PROJECT_ROOT '.local\logs\python-install.log'
    [IO.File]::WriteAllText($log, 'retained installer diagnostics')
    if (CREATE_RUNTIME) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $script:installedExecutable) -Force | Out-Null
        [IO.File]::WriteAllText($script:installedExecutable, 'fake Python, never executed')
    }
    return [pscustomobject]@{ ExitCode = INSTALLER_EXIT }
}
""".replace("DOWNLOAD_ERROR", "$true" if download_error else "$false").replace(
            "CREATE_RUNTIME", "$true" if create_runtime else "$false"
        ).replace("INSTALLER_EXIT", str(installer_exit)).replace("HASH_VALUE", hash_value)

    @staticmethod
    def install_and_report(*, initialize=False):
        action = "Initialize-PythonEnvironment" if initialize else "Install-PythonRuntime"
        return r"""
$caught = $null
$runtime = $null
try { $runtime = ACTION -ProjectRoot $env:TEST_PROJECT_ROOT }
catch { $caught = $_.Exception.Message }
Complete-Test @{
    Error = $caught
    Runtime = $runtime
    Events = @($script:events.ToArray())
    InstallerArguments = @($script:installerArguments)
    InstallerOptions = $script:installerOptions
    DownloadUri = $script:downloadUri
    DownloadExists = ($script:downloadPath -and (Test-Path -LiteralPath $script:downloadPath))
    LogExists = (Test-Path -LiteralPath (Join-Path $env:TEST_PROJECT_ROOT '.local\logs\python-install.log'))
    VenvExists = (Test-Path -LiteralPath (Join-Path $env:TEST_PROJECT_ROOT '.venv'))
}
""".replace("ACTION", action)

    def test_runtime_probe_accepts_python_with_venv_and_rejects_missing_file(self):
        result = self.powershell(r"""
Complete-Test @{
    Existing = (Test-PythonRuntime -Executable $env:TEST_PYTHON)
    WithArguments = (Test-PythonRuntime -Executable $env:TEST_PYTHON -Arguments @('-I'))
    Missing = (Test-PythonRuntime -Executable (Join-Path $env:TEST_PROJECT_ROOT 'missing.exe'))
}
""")
        self.assertEqual(result, {"Existing": True, "WithArguments": True, "Missing": False})

    def test_downloaded_installer_is_verified_and_installed_for_current_user(self):
        result = self.powershell(self.installer_mocks() + self.install_and_report())
        self.assertIsNone(result["Error"])
        self.assertEqual(result["Events"], ["download", "hash", "install"])
        self.assertEqual(result["DownloadUri"], "https://www.python.org/ftp/python/3.13.17/python-3.13.17-amd64.exe")
        self.assertEqual(result["InstallerOptions"], {"Wait": True, "PassThru": True, "WindowStyle": "Hidden"})
        for argument in ("/quiet", "/norestart", "InstallAllUsers=0", "Include_launcher=0", "Include_pip=1"):
            self.assertIn(argument, result["InstallerArguments"])
        expected_python = self.local_app_data / "Programs" / "Python" / "Python313" / "python.exe"
        self.assertEqual(Path(result["Runtime"]["Executable"]), expected_python)
        self.assertEqual(result["Runtime"]["Arguments"], [])
        self.assertIn(f'TargetDir="{expected_python.parent}"', result["InstallerArguments"])
        self.assertIn(f'"{self.root / ".local" / "logs" / "python-install.log"}"', result["InstallerArguments"])
        self.assertTrue(result["LogExists"])
        self.assertFalse(result["DownloadExists"])

    def test_checksum_mismatch_never_launches_installer_and_cleans_download(self):
        result = self.powershell(self.installer_mocks(hash_value="0" * 64) + self.install_and_report())
        self.assertIn("checksum", result["Error"].lower())
        self.assertEqual(result["Events"], ["download", "hash"])
        self.assertFalse(result["DownloadExists"])
        self.assertFalse(result["VenvExists"])

    def test_interrupted_download_is_cleaned_without_launching_installer(self):
        result = self.powershell(self.installer_mocks(download_error=True) + self.install_and_report())
        self.assertIn("simulated connection failure", result["Error"])
        self.assertEqual(result["Events"], ["download"])
        self.assertFalse(result["DownloadExists"])
        self.assertFalse(result["VenvExists"])

    def test_failed_installer_preserves_diagnostics_and_stops_environment_creation(self):
        result = self.powershell(self.installer_mocks(installer_exit=1603, create_runtime=False) + r"""
function Find-PythonRuntime { return $null }
""" + self.install_and_report(initialize=True))
        self.assertIn("1603", result["Error"])
        self.assertIn("python-install.log", result["Error"])
        self.assertEqual(result["Events"], ["download", "hash", "install"])
        self.assertTrue(result["LogExists"])
        self.assertFalse(result["DownloadExists"])
        self.assertFalse(result["VenvExists"])

    def test_installer_success_requires_a_usable_python(self):
        for exit_code in (0, 3010):
            with self.subTest(exit_code=exit_code):
                result = self.powershell(self.installer_mocks(installer_exit=exit_code, create_runtime=False)
                                         + self.install_and_report())
                self.assertIsNotNone(result["Error"])
                self.assertIn("Python", result["Error"])
                if exit_code == 3010:
                    self.assertIn("restart", result["Error"].lower())
                self.assertFalse(result["DownloadExists"])
                self.assertFalse(result["VenvExists"])

    def test_restart_code_can_continue_when_python_is_already_usable(self):
        result = self.powershell(self.installer_mocks(installer_exit=3010) + self.install_and_report())
        self.assertIsNone(result["Error"])
        self.assertIsNotNone(result["Runtime"])
        self.assertFalse(result["DownloadExists"])

    def test_unsupported_architecture_does_not_download(self):
        result = self.powershell(self.installer_mocks() + r"""
$env:PROCESSOR_ARCHITECTURE = 'ARM64'
""" + self.install_and_report())
        self.assertIn("Windows x64", result["Error"])
        self.assertEqual(result["Events"], [])

    def test_unregistered_runtime_at_install_destination_is_reused(self):
        result = self.powershell(self.installer_mocks() + r"""
New-Item -ItemType Directory -Path (Split-Path -Parent $script:installedExecutable) -Force | Out-Null
[IO.File]::WriteAllText($script:installedExecutable, 'existing runtime')
""" + self.install_and_report())
        self.assertIsNone(result["Error"])
        self.assertEqual(result["Events"], [])
        self.assertIsNotNone(result["Runtime"])

    def test_store_aliases_are_skipped_without_executing_them(self):
        result = self.powershell(r"""
$script:probed = New-Object 'System.Collections.Generic.List[string]'
function Get-Command {
    [CmdletBinding()]
    param([string]$Name, [string]$CommandType)
    if ($Name -ne 'py') {
        return [pscustomobject]@{ Source = (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\$Name.exe") }
    }
}
function Get-ChildItem { [CmdletBinding()] param([string]$LiteralPath) }
function Test-PythonRuntime {
    param([string]$Executable, [string[]]$Arguments = @())
    $script:probed.Add($Executable)
    return $false
}
$runtime = Find-PythonRuntime
Complete-Test @{ Runtime = $runtime; Probed = @($script:probed.ToArray()) }
""")
        self.assertIsNone(result["Runtime"])
        self.assertEqual(result["Probed"], [])

    def test_command_discovery_uses_first_compatible_runtime_and_launcher_arguments(self):
        for accepted_command in ("py", "python", "python3"):
            with self.subTest(accepted_command=accepted_command):
                result = self.powershell(r"""
$script:probed = New-Object 'System.Collections.Generic.List[object]'
function Get-Command {
    [CmdletBinding()]
    param([string]$Name, [string]$CommandType)
    return [pscustomobject]@{ Source = "$Name.exe" }
}
function Get-ChildItem { [CmdletBinding()] param([string]$LiteralPath) throw 'A compatible PATH runtime must win over registry lookup' }
function Test-PythonRuntime {
    param([string]$Executable, [string[]]$Arguments = @())
    $script:probed.Add(@{ Executable = $Executable; Arguments = @($Arguments) })
    return ($Executable -eq 'ACCEPTED_COMMAND.exe')
}
Complete-Test @{ Runtime = (Find-PythonRuntime); Probed = @($script:probed.ToArray()) }
""".replace("ACCEPTED_COMMAND", accepted_command))
                self.assertEqual(result["Runtime"]["Executable"], f"{accepted_command}.exe")
                self.assertEqual(result["Runtime"]["Arguments"], ["-3"] if accepted_command == "py" else [])
                expected_commands = ["py", "python", "python3"][:["py", "python", "python3"].index(accepted_command) + 1]
                self.assertEqual([probe["Executable"] for probe in result["Probed"]],
                                 [f"{command}.exe" for command in expected_commands])
                self.assertEqual(result["Probed"][0]["Arguments"], ["-3"])

    def test_registered_python_is_found_without_path_or_launcher(self):
        for use_executable_value in (True, False):
            with self.subTest(use_executable_value=use_executable_value):
                result = self.powershell(r"""
$script:registeredPython = Join-Path $env:TEST_PROJECT_ROOT 'registered Python\python.exe'
$script:checkedRoots = New-Object 'System.Collections.Generic.List[string]'
function Get-Command { [CmdletBinding()] param([string]$Name, [string]$CommandType) }
function Get-ChildItem {
    [CmdletBinding()]
    param([string]$LiteralPath)
    $script:checkedRoots.Add($LiteralPath)
    if ($LiteralPath -eq 'HKLM:\Software\WOW6432Node\Python\PythonCore') {
        return [pscustomobject]@{ PSPath = 'Registry::MockPython\3.12' }
    }
}
function Get-Item {
    [CmdletBinding()]
    param([string]$LiteralPath)
    if ($LiteralPath -ne 'Registry::MockPython\3.12\InstallPath') { throw 'Unexpected registry lookup' }
    $key = [pscustomobject]@{}
    $key | Add-Member -MemberType ScriptMethod -Name GetValue -Value {
        param([string]$Name)
        if ($Name -eq 'ExecutablePath' -and USE_EXECUTABLE_VALUE) { return $script:registeredPython }
        if ($Name -eq '') { return (Split-Path -Parent $script:registeredPython) }
        return $null
    }
    return $key
}
function Test-PythonRuntime {
    param([string]$Executable, [string[]]$Arguments = @())
    return ($Executable -eq $script:registeredPython)
}
Complete-Test @{ Runtime = (Find-PythonRuntime); CheckedRoots = @($script:checkedRoots.ToArray()) }
""".replace("USE_EXECUTABLE_VALUE", "$true" if use_executable_value else "$false"))
                self.assertEqual(Path(result["Runtime"]["Executable"]), self.root / "registered Python" / "python.exe")
                self.assertEqual(result["Runtime"]["Arguments"], [])
                self.assertIn(r"HKCU:\Software\Python\PythonCore", result["CheckedRoots"])

    def test_existing_runtime_creates_environment_without_installer_or_path_refresh(self):
        result = self.powershell(r"""
$script:venvArguments = @()
function Find-PythonRuntime { return [pscustomobject]@{ Executable = 'Invoke-FakePython'; Arguments = @('-3') } }
function Install-PythonRuntime { throw 'An existing runtime must not trigger installation' }
function Invoke-FakePython {
    $script:venvArguments = @($args)
    $directory = Join-Path $env:TEST_PROJECT_ROOT '.venv\Scripts'
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $directory 'python.exe'), 'fake environment')
    $global:LASTEXITCODE = 0
}
function Test-PythonRuntime {
    param([string]$Executable, [string[]]$Arguments = @())
    return (Test-Path -LiteralPath $Executable -PathType Leaf)
}
Initialize-PythonEnvironment -ProjectRoot $env:TEST_PROJECT_ROOT
Complete-Test @{ Arguments = @($script:venvArguments); Exists = (Test-Path -LiteralPath (Join-Path $env:TEST_PROJECT_ROOT '.venv\Scripts\python.exe')) }
""")
        self.assertEqual(result["Arguments"], ["-3", "-m", "venv", str(self.root / ".venv")])
        self.assertTrue(result["Exists"])

    def test_environment_creation_error_and_missing_executable_are_reported(self):
        for exit_code in (1, 0):
            with self.subTest(exit_code=exit_code):
                result = self.powershell(r"""
function Find-PythonRuntime { return [pscustomobject]@{ Executable = 'Invoke-FakePython'; Arguments = @() } }
function Install-PythonRuntime { throw 'Existing runtime must be reused' }
function Invoke-FakePython { $global:LASTEXITCODE = EXIT_CODE }
function Test-PythonRuntime { param([string]$Executable, [string[]]$Arguments = @()) return $false }
$caught = $null
try { Initialize-PythonEnvironment -ProjectRoot $env:TEST_PROJECT_ROOT }
catch { $caught = $_.Exception.Message }
Complete-Test @{ Error = $caught }
""".replace("EXIT_CODE", str(exit_code)))
                self.assertIsNotNone(result["Error"])
                if exit_code:
                    self.assertIn("create .venv", result["Error"])
                else:
                    self.assertIn("verified", result["Error"])


if __name__ == "__main__":
    unittest.main()
