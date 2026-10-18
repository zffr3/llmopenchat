"""A small PowerShell language subset with no filesystem or network capabilities.

The harness owns file locking and human approval. This adapter accepts already
approved source, never a command line, and never falls back to an unrestricted
PowerShell session. This is intentionally unsuitable for builds or C# execution.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time

from .runtime import _WindowsJob, RuntimeErrorDetail

AVAILABLE_CMDLETS = ("Write-Output", "Write-Host", "Write-Error", "Write-Warning", "Write-Verbose", "Write-Debug")
MAX_SOURCE_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_PROCESS_BYTES = 256 * 1024 * 1024
# CoreCLR reserves virtual memory in addition to its managed heap. Bound both;
# applying the Windows physical-memory limit as RLIMIT_AS prevents it starting.
MAX_LINUX_ADDRESS_BYTES = 8 * 1024 * 1024 * 1024
MAX_LINUX_TASKS = 128


class PowerShellError(RuntimeError):
    def __init__(self, message: str, *, started: bool = False):
        super().__init__(message)
        self.started = started


# This entire driver is trusted constant code. User source reaches it only as
# stdin JSON data and is parsed before being added to a separate runspace.
_DRIVER = r'''
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$started = $false
$ps = $null
$space = $null
try {
    $request = [Console]::In.ReadToEnd() | ConvertFrom-Json
    $source = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($request.source))
    # Some grammar directives can load modules/types during parsing itself.
    # Reject them before the full-language host calls Parser.ParseInput.
    if ($source -match '(?i)\b(using|configuration)\b|#\s*requires\b|[\[\]]') { throw 'Parser directives and type syntax are blocked.' }
    $tokens = $null
    $parseErrors = $null
    $ast = [Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -gt 0) { throw 'PowerShell syntax rejected.' }
    if ($ast.ScriptRequirements -ne $null) { throw 'Script requirements and automatic imports are blocked.' }
    $allowedNodes = @(
        'ScriptBlockAst','NamedBlockAst','StatementBlockAst','PipelineAst',
        'CommandAst','CommandExpressionAst','CommandParameterAst',
        'StringConstantExpressionAst','ExpandableStringExpressionAst','ConstantExpressionAst',
        'VariableExpressionAst','AssignmentStatementAst','BinaryExpressionAst','UnaryExpressionAst',
        'ArrayLiteralAst','ArrayExpressionAst','ParenExpressionAst','SubExpressionAst',
        'IfStatementAst','ForStatementAst','ForEachStatementAst','WhileStatementAst',
        'DoWhileStatementAst','DoUntilStatementAst','BreakStatementAst','ContinueStatementAst'
    )
    $allowedCommands = @('Write-Output','Write-Host','Write-Error','Write-Warning','Write-Verbose','Write-Debug')
    $binary = @('Plus','Minus','Multiply','Divide','Rem','Ieq','Ine','Igt','Ige','Ilt','Ile','And','Or','Xor','DotDot')
    $unary = @('Plus','Minus','Not','Exclaim','PostfixPlusPlus','PrefixPlusPlus','PostfixMinusMinus','PrefixMinusMinus')
    $assign = @('Equals','PlusEquals','MinusEquals','MultiplyEquals','DivideEquals','RemainderEquals')
    $locals = @('true','false','null')
    foreach ($item in $ast.FindAll({ param($item) $true }, $true)) {
        if ($item.GetType().Name -eq 'AssignmentStatementAst' -and $item.Left.GetType().Name -eq 'VariableExpressionAst') { $locals += $item.Left.VariablePath.UserPath }
        if ($item.GetType().Name -eq 'ForEachStatementAst') { $locals += $item.Variable.VariablePath.UserPath }
    }
    $parameters = @{
        'Write-Output'=@('InputObject','NoEnumerate'); 'Write-Host'=@('Object','NoNewline','Separator','ForegroundColor','BackgroundColor');
        'Write-Error'=@('Message'); 'Write-Warning'=@('Message'); 'Write-Verbose'=@('Message','Verbose'); 'Write-Debug'=@('Message','Debug')
    }
    foreach ($node in $ast.FindAll({ param($item) $true }, $true)) {
        $kind = $node.GetType().Name
        if ($allowedNodes -notcontains $kind) { throw ('AST construct blocked: ' + $kind) }
        if ($kind -eq 'ScriptBlockAst' -and $node -ne $ast) { throw 'Nested scripts are blocked.' }
        if ($kind -eq 'CommandAst') {
            $name = $node.GetCommandName()
            if ($node.InvocationOperator.ToString() -ne 'Unknown' -or $allowedCommands -notcontains $name) {
                throw 'Only the documented output cmdlets may be invoked literally.'
            }
            if ($node.Redirections.Count -ne 0) { throw 'Redirection is blocked.' }
            foreach ($element in $node.CommandElements) {
                if ($element.GetType().Name -eq 'CommandParameterAst' -and $parameters[$name] -notcontains $element.ParameterName) { throw 'Command parameter blocked.' }
            }
        }
        if ($kind -eq 'VariableExpressionAst') {
            $name = $node.VariablePath.UserPath
            if ($node.Splatted -or $name -notmatch '^[A-Za-z_][A-Za-z_0-9]*$' -or $locals -notcontains $name -or
                $name -match '^(?i:PS|ExecutionContext|Host|Home|PID|PWD|LastExitCode|OutputEncoding|Error|Args|Input|Matches|This|MyInvocation|ShellId|Profile|NestedPromptLevel|StackTrace|Event|Sender)') {
                throw 'Scoped, automatic, environment and dynamic variables are blocked.'
            }
        }
        if ($kind -eq 'BinaryExpressionAst' -and $binary -notcontains $node.Operator.ToString()) { throw 'Binary operator blocked.' }
        if ($kind -eq 'UnaryExpressionAst' -and $unary -notcontains $node.TokenKind.ToString()) { throw 'Unary operator blocked.' }
        if ($kind -eq 'AssignmentStatementAst' -and $assign -notcontains $node.Operator.ToString()) { throw 'Assignment operator blocked.' }
    }
    $state = [Management.Automation.Runspaces.InitialSessionState]::CreateDefault2()
    $state.Commands.Clear()
    $state.Providers.Clear()
    $state.Variables.Clear()
    $state.LanguageMode = [Management.Automation.PSLanguageMode]::ConstrainedLanguage
    $state.Variables.Add((New-Object Management.Automation.Runspaces.SessionStateVariableEntry('PSModuleAutoLoadingPreference','None','Disabled')))
    foreach ($name in $allowedCommands) {
        $command = Get-Command -Name $name -CommandType Cmdlet
        $state.Commands.Add((New-Object Management.Automation.Runspaces.SessionStateCmdletEntry($name,$command.ImplementingType,$null)))
    }
    $space = [Management.Automation.Runspaces.RunspaceFactory]::CreateRunspace($state)
    $space.Open()
    $ps = [Management.Automation.PowerShell]::Create()
    $ps.Runspace = $space
    [void]$ps.AddScript($source, $true)
    $output = New-Object 'Management.Automation.PSDataCollection[psobject]'
    $method = $ps.GetType().GetMethods() | Where-Object { $_.Name -eq 'BeginInvoke' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 2 } | Select-Object -First 1
    $method = $method.MakeGenericMethod([psobject],[psobject])
    $started = $true
    $invokeArgs = New-Object 'object[]' 2
    $invokeArgs[0] = $null
    $invokeArgs[1] = $output.psobject.BaseObject
    $async = $method.Invoke($ps, $invokeArgs)
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $stdout = New-Object Text.StringBuilder
    $stderr = New-Object Text.StringBuilder
    $positions = @(0,0,0,0,0,0)
    $records = 0
    do {
        $streams = @($output,$ps.Streams.Error,$ps.Streams.Warning,$ps.Streams.Verbose,$ps.Streams.Debug,$ps.Streams.Information)
        for ($s = 0; $s -lt $streams.Count; $s++) {
            while ($positions[$s] -lt $streams[$s].Count) {
                $text = $streams[$s][$positions[$s]].ToString()
                $positions[$s]++
                $records++
                if ($records -gt 2048 -or $text.Length -gt 65536) { throw 'Output limit exceeded.' }
                if ($s -eq 0 -or $s -eq 5) { [void]$stdout.AppendLine($text) } else { [void]$stderr.AppendLine($text) }
                if ([Text.Encoding]::UTF8.GetByteCount($stdout.ToString()) + [Text.Encoding]::UTF8.GetByteCount($stderr.ToString()) -gt 65536) { throw 'Output limit exceeded.' }
            }
        }
        if ($watch.Elapsed.TotalSeconds -gt $request.timeout) { throw 'Execution timeout exceeded.' }
        if (-not $async.IsCompleted) { Start-Sleep -Milliseconds 10 }
    } while (-not $async.IsCompleted)
    # Drain once after completion in case the producer completed between polls.
    $streams = @($output,$ps.Streams.Error,$ps.Streams.Warning,$ps.Streams.Verbose,$ps.Streams.Debug,$ps.Streams.Information)
    for ($s = 0; $s -lt $streams.Count; $s++) {
        while ($positions[$s] -lt $streams[$s].Count) {
            $text = $streams[$s][$positions[$s]].ToString(); $positions[$s]++; $records++
            if ($records -gt 2048 -or $text.Length -gt 65536) { throw 'Output limit exceeded.' }
            if ($s -eq 0 -or $s -eq 5) { [void]$stdout.AppendLine($text) } else { [void]$stderr.AppendLine($text) }
        }
    }
    if ([Text.Encoding]::UTF8.GetByteCount($stdout.ToString()) + [Text.Encoding]::UTF8.GetByteCount($stderr.ToString()) -gt 65536) { throw 'Output limit exceeded.' }
    try { [void]$ps.EndInvoke($async) } catch { [void]$stderr.AppendLine('Restricted script execution failed.') }
    $exitCode = 0
    if ($ps.HadErrors) { $exitCode = 1 }
    $response = @{stdout=$stdout.ToString();stderr=$stderr.ToString();exit_code=$exitCode;execution_started=$true}
} catch {
    $response = @{error=$_.Exception.Message;execution_started=$started}
} finally {
    if ($ps -ne $null) { $ps.Stop(); $ps.Dispose() }
    if ($space -ne $null) { $space.Dispose() }
}
[Console]::Out.Write(($response | ConvertTo-Json -Compress -Depth 4))
'''


class _RestrictedJob(_WindowsJob):
    def __init__(self):
        super().__init__()
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64), ("flags", ctypes.c_uint32),
                        ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t), ("processes", ctypes.c_uint32),
                        ("affinity", ctypes.c_size_t), ("priority", ctypes.c_uint32), ("scheduling", ctypes.c_uint32)]
        class Io(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("read", "write", "other", "read_bytes", "write_bytes", "other_bytes")]
        class Limits(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", Io), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        limits = Limits()
        limits.basic.flags = 0x2000 | 0x8 | 0x100  # kill on close, one process, process memory
        limits.basic.processes = 1
        limits.process_memory = MAX_PROCESS_BYTES
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise PowerShellError("Windows не ограничила процесс PowerShell; выполнение запрещено.")


def _windows_configuration() -> tuple[list[str], Path, dict[str, str]]:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetSystemDirectoryW.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32)
    kernel.GetSystemDirectoryW.restype = ctypes.c_uint32
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel.GetSystemDirectoryW(buffer, len(buffer))
    if not length or length >= len(buffer):
        raise PowerShellError("Windows не указала доверенную системную папку.")
    system_directory = Path(buffer.value)
    system = system_directory.parent
    executable = system_directory / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if not executable.is_file():
        raise PowerShellError("Системный Windows PowerShell не найден; небезопасного запасного запуска нет.")
    environment = {"SystemRoot": str(system), "WINDIR": str(system),
                   "PSModulePath": str(executable.parent / "Modules")}
    return [str(executable)], system_directory, environment


def _trusted_linux_executable(candidates: tuple[str, ...], label: str) -> Path:
    """Never discover executors from PATH, the repository, or user settings."""
    for candidate in candidates:
        path = Path(candidate)
        try:
            resolved = path.resolve(strict=True)
            # Check both the link's directories and its resolved installation.
            # The final symlink's mode is immaterial; its directory owns it.
            directories = set(path.parents) | set(resolved.parents)
            for directory in directories:
                info = directory.stat()
                if info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                    raise PowerShellError(f"Недоверенная системная папка {label}: {directory}.")
            info = resolved.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise PowerShellError(f"Недоверенный системный исполняемый файл {label}.")
            if not os.access(resolved, os.X_OK):
                continue
            return resolved
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PowerShellError(f"Не удалось проверить системный {label}: {error}") from error
    raise PowerShellError(f"Не найден системный {label}; установите его через администратора Linux.")


def _linux_task_limit() -> int:
    """Reserve bounded new tasks without counting the model against that budget."""
    import resource

    uid, own_pid = os.getuid(), str(os.getpid())
    tasks = 0
    own_seen = False
    try:
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal():
                continue
            try:
                with (path / "status").open("r", encoding="ascii", errors="replace") as stream:
                    status = dict(line.split(":", 1) for line in stream.read(16384).splitlines() if ":" in line)
                if int(status["Uid"].split()[0]) == uid:
                    tasks += int(status["Threads"].strip())
                    own_seen = own_seen or path.name == own_pid
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                # A different process may exit while /proc is being scanned.
                # hidepid mounts can also hide other users' status files.
                continue
        if not own_seen:
            raise PowerShellError("Для ограничения задач PowerShell нужен доступ к /proc своего пользователя.")
        _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        limit = tasks + MAX_LINUX_TASKS
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        if limit - tasks < 32:
            raise PowerShellError("Недостаточно свободных задач пользователя для ограниченного PowerShell.")
        return limit
    except (OSError, ValueError, KeyError) as error:
        raise PowerShellError(f"Не удалось проверить лимит задач PowerShell через /proc: {error}") from error


def _linux_configuration(timeout_seconds: int) -> tuple[list[str], Path, dict[str, str]]:
    # RLIMIT_NPROC is not enforced for root. Require an ordinary service account
    # so the same fail-closed task bound also applies on a headless server.
    if os.geteuid() == 0:
        raise PowerShellError("Ограниченный PowerShell на Linux запускайте от обычного пользователя, не root.")
    executable = _trusted_linux_executable(("/usr/bin/pwsh", "/usr/local/bin/pwsh"), "PowerShell 7 (pwsh)")
    limiter = _trusted_linux_executable(("/usr/bin/prlimit", "/bin/prlimit"), "prlimit (пакет util-linux)")
    command = [str(limiter), f"--as={MAX_LINUX_ADDRESS_BYTES}", f"--cpu={timeout_seconds + 5}",
               f"--nproc={_linux_task_limit()}", "--nofile=64", "--fsize=0", "--core=0", "--", str(executable)]
    environment = {
        "PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "XDG_CONFIG_HOME": "/nonexistent",
        "LC_ALL": "C.UTF-8", "TERM": "dumb", "PSModulePath": str(executable.parent / "Modules"),
        "PSModuleAnalysisCachePath": "/dev/null", "POWERSHELL_TELEMETRY_OPTOUT": "1",
        "POWERSHELL_UPDATECHECK": "Off", "DOTNET_GCHeapHardLimit": f"{MAX_PROCESS_BYTES:x}",
        "DOTNET_GCHeapCount": "1", "DOTNET_gcServer": "0", "DOTNET_PROCESSOR_COUNT": "1",
        "DOTNET_EnableDiagnostics": "0",
    }
    return command, Path("/"), environment


def availability_error() -> str | None:
    """Return an actionable prerequisite diagnostic without starting a process."""
    try:
        if sys.platform == "win32":
            _windows_configuration()
        elif sys.platform == "linux":
            _linux_configuration(5)
        else:
            return "Ограниченный PowerShell поддерживается в Windows и Linux."
    except (OSError, PowerShellError) as error:
        return str(error)
    return None


def _communicate_linux(process: subprocess.Popen, request: bytes, timeout: float) -> tuple[bytes, bytes]:
    """Drain pipes while bounding allocations; communicate() buffers forever."""
    deadline = time.monotonic() + timeout
    output = {"stdout": bytearray(), "stderr": bytearray()}
    position = 0
    with selectors.DefaultSelector() as selector:
        for pipe, name, event in ((process.stdin, "stdin", selectors.EVENT_WRITE),
                                  (process.stdout, "stdout", selectors.EVENT_READ),
                                  (process.stderr, "stderr", selectors.EVENT_READ)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, event, name)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            for key, event in selector.select(min(remaining, 0.1)):
                pipe, name = key.fileobj, key.data
                if name == "stdin":
                    try:
                        position += os.write(pipe.fileno(), request[position:position + 65536])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        position = len(request)
                    if position == len(request):
                        selector.unregister(pipe)
                        pipe.close()
                else:
                    try:
                        chunk = os.read(pipe.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(pipe)
                        continue
                    output[name].extend(chunk)
                    limit = 512 * 1024 if name == "stdout" else MAX_OUTPUT_BYTES
                    if len(output[name]) > limit:
                        raise PowerShellError("PowerShell превысил лимит вывода.", started=True)
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
    return bytes(output["stdout"]), bytes(output["stderr"])


def run_powershell(source: str, repository: Path, timeout_seconds: int = 5) -> dict:
    """Run already-approved source in a capability-free restricted runspace."""
    if sys.platform not in {"win32", "linux"}:
        raise PowerShellError("Ограниченный PowerShell поддерживается в Windows и Linux.")
    if not isinstance(source, str) or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 15:
        raise PowerShellError("Неверный исходный текст или таймаут PowerShell (1–15 секунд).")
    try:
        encoded = source.encode("utf-8")
    except UnicodeError as error:
        raise PowerShellError("Скрипт должен быть корректным UTF-8.") from error
    if len(encoded) > MAX_SOURCE_BYTES:
        raise PowerShellError("Скрипт PowerShell превышает 256 КБ.")
    repository = Path(repository)
    if not repository.is_absolute() or not repository.is_dir():
        raise PowerShellError("Для PowerShell нужна выбранная существующая папка репозитория.")
    windows = sys.platform == "win32"
    command, directory, environment = _windows_configuration() if windows else _linux_configuration(timeout_seconds)
    driver = base64.b64encode(_DRIVER.encode("utf-16-le")).decode("ascii")
    request = json.dumps({"source": base64.b64encode(encoded).decode("ascii"), "timeout": timeout_seconds}).encode("utf-8")
    process = None
    job = None
    execution_possible = False
    try:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if windows else {"start_new_session": True}
        if windows:
            job = _RestrictedJob()
        process = subprocess.Popen([*command, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", driver],
                                   cwd=directory, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, **options)
        if job is not None:
            job.assign(process.pid)
        # Windows withholds source until job assignment. On Linux prlimit applies
        # hard limits before exec, so a failed limit setup cannot evaluate source.
        execution_possible = True
        stdout, stderr = (process.communicate(request, timeout=timeout_seconds + 5) if windows else
                          _communicate_linux(process, request, timeout_seconds + 5))
        if len(stdout) > 512 * 1024 or len(stderr) > MAX_OUTPUT_BYTES:
            raise PowerShellError("PowerShell превысил лимит вывода.", started=True)
        if process.returncode != 0:
            raise PowerShellError("Ограниченный процесс PowerShell завершился с ошибкой.", started=True)
        try:
            result = json.loads(stdout.decode("utf-8-sig"))
        except (ValueError, UnicodeError) as error:
            raise PowerShellError("PowerShell вернул неверный результат.", started=True) from error
        if not isinstance(result, dict):
            raise PowerShellError("PowerShell вернул неверный результат.", started=True)
        if "error" in result:
            raise PowerShellError(str(result["error"]), started=result.get("execution_started") is True)
        return {**result, "restricted": True, "available_cmdlets": list(AVAILABLE_CMDLETS)}
    except subprocess.TimeoutExpired as error:
        raise PowerShellError("Превышен таймаут PowerShell.", started=True) from error
    except (OSError, RuntimeErrorDetail) as error:
        raise PowerShellError("Безопасный запуск PowerShell не выполнен: " + str(error),
                              started=execution_possible) from error
    finally:
        if job is not None:
            job.close()
        if process is not None:
            if not windows:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                # Do not let communicate() collect unbounded residual output.
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
                try:
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            else:
                if process.poll() is None:
                    process.kill()
                process.communicate()


run_script = run_powershell
