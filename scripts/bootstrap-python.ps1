# Bootstrap for Install.cmd; compatible with the Windows PowerShell 5.1 bundled with Windows.
param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = 'Stop'

function Test-PythonRuntime {
    param([string]$Executable, [string[]]$Arguments = @())

    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        return $false
    }
    try {
        & $Executable @Arguments -I -c 'import sys, venv, ensurepip; sys.exit(sys.version_info < (3, 10))' > $null 2>&1
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Find-PythonRuntime {
    foreach ($name in @('py', 'python', 'python3')) {
        $command = Get-Command -Name $name -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $command) { continue }
        # These Windows aliases can open Microsoft Store instead of running Python.
        if ($name -ne 'py' -and $command.Source -like '*\Microsoft\WindowsApps\*') { continue }
        $arguments = @()
        if ($name -eq 'py') { $arguments = @('-3') }
        if (Test-PythonRuntime -Executable $command.Source -Arguments $arguments) {
            return [pscustomobject]@{ Executable = $command.Source; Arguments = $arguments }
        }
    }

    # An installed Python need not be on PATH. Check the standard PEP 514 registrations.
    foreach ($registryRoot in @(
        'HKCU:\Software\Python\PythonCore',
        'HKLM:\Software\Python\PythonCore',
        'HKCU:\Software\WOW6432Node\Python\PythonCore',
        'HKLM:\Software\WOW6432Node\Python\PythonCore'
    )) {
        foreach ($version in (Get-ChildItem -LiteralPath $registryRoot -ErrorAction SilentlyContinue)) {
            $key = Get-Item -LiteralPath ($version.PSPath + '\InstallPath') -ErrorAction SilentlyContinue
            if (-not $key) { continue }
            $executable = $key.GetValue('ExecutablePath')
            if (-not $executable) {
                $directory = $key.GetValue('')
                if (-not $directory) { continue }
                $executable = Join-Path $directory 'python.exe'
            }
            if (Test-PythonRuntime -Executable $executable) {
                return [pscustomobject]@{ Executable = $executable; Arguments = @() }
            }
        }
    }
    return $null
}

function Install-PythonRuntime {
    param([string]$ProjectRoot)

    $architecture = $env:PROCESSOR_ARCHITEW6432
    if (-not $architecture) { $architecture = $env:PROCESSOR_ARCHITECTURE }
    if ($architecture -ne 'AMD64') {
        throw 'Automatic Python installation requires Windows x64. Install Python 3.10+ manually for this platform.'
    }
    if (-not $env:LOCALAPPDATA) {
        throw 'LOCALAPPDATA is not set. Cannot install Python for the current user.'
    }

    $version = '3.13.17'
    # SHA256 published at https://www.python.org/downloads/release/python-31317/
    $expectedHash = '5da012101ba2e17113a9a1cd901e9ce1441086f2b7ecb5de06ec94163899a190'
    $url = "https://www.python.org/ftp/python/$version/python-$version-amd64.exe"
    $targetDirectory = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313'
    $executable = Join-Path $targetDirectory 'python.exe'
    $logDirectory = Join-Path $ProjectRoot '.local\logs'
    $logPath = Join-Path $logDirectory 'python-install.log'
    $installerPath = Join-Path ([System.IO.Path]::GetTempPath()) ('llmopenchat-python-' + [guid]::NewGuid().ToString('N') + '.exe')

    # Also handle an unregistered installation left at the standard destination.
    if (Test-PythonRuntime -Executable $executable) {
        return [pscustomobject]@{ Executable = $executable; Arguments = @() }
    }

    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    try {
        Write-Host "Python 3.10+ was not found. Downloading Python $version from python.org..."
        $ProgressPreference = 'SilentlyContinue'
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $installerPath -UseBasicParsing -TimeoutSec 300
        $actualHash = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash
        if ($actualHash -ne $expectedHash) {
            throw 'Python installer checksum mismatch. The downloaded file will not be run. Run Install.cmd again to retry.'
        }

        Write-Host "Installing Python $version for the current user..."
        $installerArguments = @(
            '/quiet', '/norestart', 'InstallAllUsers=0', 'Include_launcher=0',
            'InstallLauncherAllUsers=0', 'Include_test=0', 'Include_pip=1',
            'Include_doc=0', 'Include_tcltk=0', 'Shortcuts=0', 'PrependPath=0',
            "TargetDir=`"$targetDirectory`"", '/log', "`"$logPath`""
        )
        $process = Start-Process -FilePath $installerPath -ArgumentList $installerArguments -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -notin @(0, 3010)) {
            throw "Python installer exited with code $($process.ExitCode). See $logPath"
        }
        if (-not (Test-PythonRuntime -Executable $executable)) {
            if ($process.ExitCode -eq 3010) {
                throw "Python installation requires a Windows restart. Restart and run Install.cmd again. See $logPath"
            }
            throw "Python installation did not produce a working Python 3.10+ with venv and pip. See $logPath"
        }
        return [pscustomobject]@{ Executable = $executable; Arguments = @() }
    }
    finally {
        # Only this randomly named installer file is removed; keep the installation log.
        if (Test-Path -LiteralPath $installerPath) {
            Remove-Item -LiteralPath $installerPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Initialize-PythonEnvironment {
    param([string]$ProjectRoot)

    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
    $runtime = Find-PythonRuntime
    if (-not $runtime) { $runtime = Install-PythonRuntime -ProjectRoot $ProjectRoot }
    $arguments = @($runtime.Arguments)
    $environmentPath = Join-Path $ProjectRoot '.venv'
    Write-Host 'Creating the Python environment...'
    & $runtime.Executable @arguments -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create .venv. See the Python error above and run Install.cmd again.'
    }
    if (-not (Test-PythonRuntime -Executable (Join-Path $environmentPath 'Scripts\python.exe'))) {
        throw 'The Python environment could not be verified. Run Install.cmd again.'
    }
}

# Dot-sourcing exposes the functions for isolated tests without downloading or installing.
if ($MyInvocation.InvocationName -ne '.') {
    try {
        Initialize-PythonEnvironment -ProjectRoot $ProjectRoot
    }
    catch {
        [Console]::Error.WriteLine("Python setup failed: $($_.Exception.Message)")
        exit 1
    }
}
