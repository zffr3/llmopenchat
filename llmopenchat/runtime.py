"""Own the local inference server for the lifetime of a console session."""

from __future__ import annotations

import ipaddress
import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
from typing import BinaryIO, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


# These are execution controls, not inference tuning. The console owns its
# approval-gated tools and must never start a second, unrestricted executor.
_SERVER_TOOL_ARGUMENTS = {
    "--tools", "--tools-runtime", "--mcp-servers", "--mcp-servers-config",
    "--mcp-servers-json", "--agent", "-ag", "--ui-mcp-proxy",
    "--webui-mcp-proxy", "--models-preset",
}
_SERVER_TOOL_ENVIRONMENT = {
    "LLAMA_ARG_TOOLS", "LLAMA_ARG_TOOLS_RUNTIME", "LLAMA_ARG_MCP_SERVERS",
    "LLAMA_ARG_MCP_SERVERS_CONFIG", "LLAMA_ARG_MCP_SERVERS_JSON",
    "LLAMA_ARG_AGENT", "LLAMA_ARG_NO_AGENT", "LLAMA_ARG_UI_MCP_PROXY",
    "LLAMA_ARG_NO_UI_MCP_PROXY", "LLAMA_ARG_MODELS_PRESET",
}


class RuntimeErrorDetail(Exception):
    """An actionable startup or configuration error for the console client."""


class _WindowsJob:
    """Keep children in a job that Windows destroys when its owner disappears."""

    def __init__(self) -> None:
        # Explicit fixed-width integers avoid the Windows/Linux c_long difference.
        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
        self.kernel.CreateJobObjectW.restype = ctypes.c_void_p
        self.kernel.SetInformationJobObject.argtypes = (
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
        )
        self.kernel.SetInformationJobObject.restype = ctypes.c_int
        self.kernel.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self.kernel.AssignProcessToJobObject.restype = ctypes.c_int
        self.kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        self.kernel.OpenProcess.restype = ctypes.c_void_p
        self.kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        self.kernel.CloseHandle.restype = ctypes.c_int
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise RuntimeErrorDetail(f"Windows не создала Job Object: {ctypes.WinError(ctypes.get_last_error())}")
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise RuntimeErrorDetail(f"Windows не настроила завершение модели вместе с клиентом: {error}")

    def assign(self, pid: int) -> None:
        # AssignProcessToJobObject needs PROCESS_SET_QUOTA | PROCESS_TERMINATE.
        process_handle = self.kernel.OpenProcess(0x0100 | 0x0001, False, pid)
        if not process_handle:
            raise RuntimeErrorDetail(f"Windows не открыла процесс модели: {ctypes.WinError(ctypes.get_last_error())}")
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process_handle):
                raise RuntimeErrorDetail(
                    f"Windows не связала модель с клиентом: {ctypes.WinError(ctypes.get_last_error())}"
                )
        finally:
            self.kernel.CloseHandle(process_handle)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class ManagedServer:
    """Start, verify and stop a server, without taking ownership of existing ones.

    Entering an external backend only verifies its ``/models`` endpoint. Managed
    backends reuse an existing endpoint only when it advertises the requested
    model. Subprocesses use argument lists, never shell command strings.
    """

    def __init__(
        self,
        config: dict,
        root: Path,
        emit: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.root = Path(root).resolve()
        self.emit = emit
        self.backend = config.get("backend", "llama_cpp")
        self.model = config.get("model", "glm-local")
        self.base_url = str(config.get("base_url", "http://127.0.0.1:8081/v1")).rstrip("/")
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: BinaryIO | None = None
        self._job: _WindowsJob | None = None
        self._entered = False
        self.log_path = self.root / ".local" / "logs" / "server.log"
        self._opener = build_opener(ProxyHandler({}))

    def __enter__(self) -> ManagedServer:
        if self._entered:
            raise RuntimeErrorDetail("Этот экземпляр ManagedServer уже используется.")
        self._validate_endpoint()
        self._entered = True
        try:
            if self.backend == "external":
                self._require_model(self._read_models())
                return self
            if self._port_is_open():
                try:
                    self._require_model(self._read_models())
                except RuntimeErrorDetail as exc:
                    raise RuntimeErrorDetail(
                        f"Адрес {self.base_url} уже занят. {exc} "
                        "Существующий процесс не был изменён. Выберите другой порт "
                        "или подключитесь к нужной модели."
                    ) from exc
                self.emit(f"Используется уже запущенный сервер: {self.base_url}")
                return self
            command = self._build_command()
            timeout = self._startup_timeout()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("ab", buffering=0)
            options: dict = {
                "cwd": str(self.root),
                "stdin": subprocess.DEVNULL,
                "stdout": self._log_handle,
                "stderr": subprocess.STDOUT,
                "shell": False,
                "env": self._child_environment(),
            }
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._job = _WindowsJob()
            else:
                options["start_new_session"] = True
            self.emit(f"Запуск модели; журнал: {self.log_path}")
            try:
                self._process = subprocess.Popen(command, **options)
                if self._job is not None:
                    self._job.assign(self._process.pid)
            except OSError as exc:
                raise RuntimeErrorDetail(f"Не удалось запустить сервер: {exc}") from exc
            self._wait_until_ready(timeout)
            self.emit(f"Модель готова: {self.model}")
            return self
        except BaseException:
            # __exit__ is not called by Python if __enter__ fails, including Ctrl+C.
            self.stop()
            self._entered = False
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop()
        self._entered = False
        return False

    def stop(self) -> None:
        """Stop only the subprocess created by this object; safe to call twice."""
        process, self._process = self._process, None
        try:
            if process is not None and process.poll() is None:
                try:
                    if os.name == "nt":
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=8)
                except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
                    try:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
                        pass
        finally:
            if self._job is not None:
                self._job.close()
                self._job = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def _validate_endpoint(self) -> None:
        if self.backend not in {"llama_cpp", "vllm", "external"}:
            raise RuntimeErrorDetail(f"Неизвестный backend: {self.backend!r}")
        if not isinstance(self.model, str) or not self.model.strip():
            raise RuntimeErrorDetail("Поле model должно содержать API-имя модели.")
        try:
            parsed = urlsplit(self.base_url)
            self._host = parsed.hostname
            self._port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise RuntimeErrorDetail(f"Некорректный base_url: {exc}") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not self._host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or self._port < 1
        ):
            raise RuntimeErrorDetail("base_url должен быть HTTP(S)-адресом API без учётных данных, query и fragment.")
        if self.backend != "external":
            if parsed.scheme != "http" or self._host not in {"127.0.0.1", "localhost"}:
                raise RuntimeErrorDetail("Автозапуск поддерживает только http://127.0.0.1:<порт>/v1 или localhost.")
            if parsed.path != "/v1":
                raise RuntimeErrorDetail("Для автозапуска base_url должен оканчиваться на /v1.")
        # Bypass proxies only for local servers; honor proxy settings for external APIs.
        try:
            local = self._host == "localhost" or ipaddress.ip_address(self._host).is_loopback
        except ValueError:
            local = False
        self._opener = build_opener(ProxyHandler({})) if local else build_opener()

    def _port_is_open(self) -> bool:
        try:
            with socket.create_connection((self._host, self._port), timeout=0.75):
                return True
        except OSError:
            return False

    def _read_models(self) -> list[str]:
        headers = {"Accept": "application/json"}
        api_key = os.environ.get("LLMOPENCHAT_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(f"{self.base_url}/models", headers=headers)
        try:
            with self._opener.open(request, timeout=2) as response:
                body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024:
                raise RuntimeErrorDetail("Ответ /models слишком велик.")
            payload = json.loads(body)
        except HTTPError as exc:
            raise RuntimeErrorDetail(f"Проверка API вернула HTTP {exc.code}: {self.base_url}/models") from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise RuntimeErrorDetail(f"API недоступен: {self.base_url}. {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeErrorDetail("Сервер вернул некорректный JSON для /models.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise RuntimeErrorDetail("Сервер не вернул OpenAI-совместимый список моделей (data).")
        return [item["id"] for item in payload["data"] if isinstance(item, dict) and isinstance(item.get("id"), str)]

    def _require_model(self, model_ids: list[str]) -> None:
        if self.model not in model_ids:
            available = ", ".join(model_ids) or "список пуст"
            raise RuntimeErrorDetail(f"Ожидалась модель {self.model!r}; сервер предлагает: {available}.")

    def _server_config(self) -> dict:
        server = self.config.get("server", {})
        if not isinstance(server, dict):
            raise RuntimeErrorDetail("Поле server должно быть объектом JSON.")
        return server

    def _positive_integer(self, value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeErrorDetail(f"{name} должно быть положительным целым числом.")
        return value

    def _startup_timeout(self) -> float:
        value = self._server_config().get("startup_timeout", 300)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeErrorDetail("server.startup_timeout должно быть положительным числом секунд.")
        return float(value)

    def _resolve_file(self, value, name: str) -> Path:
        if not isinstance(value, str) or not value:
            raise RuntimeErrorDetail(f"Укажите {name} в конфигурации.")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else self.root / path
        path = path.resolve()
        if not path.is_file():
            raise RuntimeErrorDetail(f"Файл {name} не найден: {path}. Выполните установку или исправьте путь.")
        return path

    def _extra_args(self) -> list[str]:
        arguments = self._server_config().get("extra_args", [])
        if not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments):
            raise RuntimeErrorDetail("server.extra_args должен быть массивом строк.")
        protected = {
            "--host", "--port", "--api-prefix", "--model", "-m", "--alias", "-a",
            "--served-model-name", "--api-key", "--config", "-c",
        }
        for arg in arguments:
            # llama.cpp accepts underscores in long option names as hyphens.
            option = arg.split("=", 1)[0]
            if option.startswith("--"):
                option = option.replace("_", "-")
            if option in _SERVER_TOOL_ARGUMENTS:
                raise RuntimeErrorDetail(
                    f"Параметр {arg!r} включает инструменты сервера в обход подтверждений "
                    "клиента и недопустим в extra_args."
                )
            if option in protected:
                raise RuntimeErrorDetail(f"Параметр {arg!r} задаётся конфигурацией клиента и недопустим в extra_args.")
        return arguments

    def _child_environment(self) -> dict[str, str]:
        """Retain normal model/GPU settings while removing native executors."""
        return {
            key: value for key, value in os.environ.items()
            if key.upper() not in _SERVER_TOOL_ENVIRONMENT
        }

    def _build_command(self) -> list[str]:
        server = self._server_config()
        context = self._positive_integer(server.get("context_size", 8192), "server.context_size")
        extra = self._extra_args()
        if self.backend == "llama_cpp":
            executable = self._resolve_file(server.get("executable"), "server.executable")
            model_path = self._resolve_file(server.get("model_path"), "server.model_path")
            threads = self._positive_integer(server.get("threads", 16), "server.threads")
            gpu_layers = server.get("gpu_layers", "auto")
            if isinstance(gpu_layers, bool) or not (
                isinstance(gpu_layers, int) and gpu_layers >= 0
                or isinstance(gpu_layers, str) and (gpu_layers in {"auto", "all"} or gpu_layers.isdecimal())
            ):
                raise RuntimeErrorDetail("server.gpu_layers: используйте auto, all или неотрицательное число.")
            # Supported by current upstream tools/server/README.md, including 'auto'.
            return [
                str(executable), "--model", str(model_path), "--alias", self.model,
                "--host", "127.0.0.1", "--port", str(self._port),
                "--ctx-size", str(context), "--n-gpu-layers", str(gpu_layers),
                "--threads", str(threads), "--parallel", "1", "--jinja", *extra,
                # b11445 also reads user/system llama.cpp config.ini before
                # env and CLI. Explicit final overrides close that bypass.
                # --no-agent must follow --tools: the CSV parser represents an
                # empty string as one empty entry, which --no-agent clears.
                "--tools", "", "--no-agent", "--no-ui-mcp-proxy",
                "--tools-runtime", "", "--mcp-servers-config", "",
                "--mcp-servers-json", "",
            ]
        if os.name == "nt":
            raise RuntimeErrorDetail(
                "Локальный vLLM требует Linux и подходящего GPU-сервера. "
                "GLM-5.3-UNCENSORED-FP8 содержит 753 млрд параметров, около 756 ГБ весов; "
                "64 ГБ RAM и 16 ГБ VRAM недостаточно. Автор модели использует 8 × H200. "
                "Выберите компактную GGUF-модель с backend=llama_cpp или готовый API с backend=external."
            )
        target = self.config.get("target_model", "dealignai/GLM-5.3-UNCENSORED-FP8")
        if not isinstance(target, str) or not target or target.startswith("-"):
            raise RuntimeErrorDetail("target_model должен содержать Hugging Face ID или путь к модели.")
        tp = self._positive_integer(self.config.get("tensor_parallel_size", 8), "tensor_parallel_size")
        configured_executable = server.get("executable", "vllm")
        if not isinstance(configured_executable, str) or not configured_executable:
            raise RuntimeErrorDetail("server.executable должен содержать путь к vllm.")
        candidate = self.root / configured_executable
        if Path(configured_executable).is_absolute() or candidate.is_file() or any(separator in configured_executable for separator in ("/", "\\")):
            executable = str(self._resolve_file(configured_executable, "server.executable"))
        else:
            executable = shutil.which(configured_executable)
            if executable is None:
                raise RuntimeErrorDetail("vllm не найден в PATH. Установите совместимый vLLM на Linux-сервере.")
        # Model-card serving recipe; a single chat slot keeps KV memory bounded.
        return [
            executable, "serve", target, "--served-model-name", self.model,
            "--host", "127.0.0.1", "--port", str(self._port),
            "--tensor-parallel-size", str(tp), "--gpu-memory-utilization", "0.90",
            "--enforce-eager", "--disable-custom-all-reduce", "--enable-prefix-caching",
            "--max-num-seqs", "1", "--max-model-len", str(context),
            "--reasoning-parser", "glm45", "--tool-call-parser", "glm47",
            "--enable-auto-tool-choice", *extra,
        ]

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        next_progress = time.monotonic() + 15
        last_error = "API ещё не ответил"
        while time.monotonic() < deadline:
            assert self._process is not None
            exit_code = self._process.poll()
            if exit_code is not None:
                raise RuntimeErrorDetail(
                    f"Сервер завершился с кодом {exit_code}. Журнал: {self.log_path}\n{self._log_tail()}"
                )
            try:
                models = self._read_models()
            except RuntimeErrorDetail as exc:
                last_error = str(exc)
            else:
                self._require_model(models)
                return
            now = time.monotonic()
            if now >= next_progress:
                self.emit(f"Модель загружается… журнал: {self.log_path}")
                next_progress = now + 15
            time.sleep(min(0.25, max(0, deadline - now)))
        raise RuntimeErrorDetail(
            f"Модель не запустилась за {timeout:g} с. {last_error}. "
            f"Журнал: {self.log_path}. При долгой загрузке увеличьте server.startup_timeout."
        )

    def _log_tail(self) -> str:
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell() - 8192))
                return "\n".join(handle.read().decode("utf-8", errors="replace").splitlines()[-12:])
        except OSError:
            return ""
