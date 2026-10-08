"""Opt-in, human-approved tools for one chat's repository and public web.

Settings are owned by the CLI, never exposed as a model tool. Every invocation
gets a fresh approval, including listing, searching, and reading. Code tools use
Windows native handles; unsupported platforms fail closed rather than weakening
the repository boundary. Web requests use direct, pinned public-IP sockets.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import queue
import re
import socket
import ssl
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit, urlunsplit

from ._winfiles import FileAccessError, LockedHandle, LockedRepository, supported
from .powershell import AVAILABLE_CMDLETS as _POWERSHELL_CMDLETS, PowerShellError, run_powershell


MAX_FILE_BYTES = 256 * 1024
MAX_ARGUMENT_BYTES = 8 * MAX_FILE_BYTES
MAX_FILES = 1000
MAX_RESULTS = 100
MAX_DEPTH = 24
MAX_SEARCH_BYTES = 4 * 1024 * 1024
MAX_WEB_BYTES = 512 * 1024
MAX_WEB_TEXT = 24000
NETWORK_TIMEOUT = 15.0


class HarnessError(RuntimeError):
    pass


class _PowerShellFailure(HarnessError):
    def __init__(self, message: str, *, execution_started: bool):
        super().__init__(message)
        self.execution_started = execution_started


@dataclass(frozen=True)
class HarnessSettings:
    web_enabled: bool = False
    code_enabled: bool = False
    repository: Path | None = None
    # UI-owned identity survives constructing a new harness on the next turn.
    # It is never exposed in tool schemas or accepted from model arguments.
    repository_identity: tuple[int, int, int] | None = None
    powershell_enabled: bool = False


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: dict
    summary: str
    preview: str = ""


_DOS_NAMES = {"con", "prn", "aux", "nul", "conin$", "conout$", "clock$"} | {
    prefix + suffix for prefix in ("com", "lpt") for suffix in "123456789¹²³"
}
_FORBIDDEN = re.compile(r'[<>:"|?*~\x00-\x1f\x7f-\x9f]')
_BIDI = re.compile(r"[\u202a-\u202e\u2066-\u2069]")


def _component(part: str):
    if (
        not part or part in (".", "..") or len(part) > 255
        or part.endswith((".", " ")) or _FORBIDDEN.search(part) or _BIDI.search(part)
        or part.casefold() == ".git" or part.split(".", 1)[0].casefold() in _DOS_NAMES
    ):
        raise HarnessError("Недопустимый путь: ссылки, .git, traversal, ADS и имена устройств запрещены.")


def _relative(value: str, *, directory: bool = False) -> tuple[str, ...]:
    if not isinstance(value, str) or len(value) > 2048:
        raise HarnessError("Путь должен быть короткой строкой относительно выбранного репозитория.")
    if directory and value in ("", "."):
        return ()
    normalized = value.replace("\\", "/")
    if normalized.startswith("/"):
        raise HarnessError("Абсолютные, UNC и device пути запрещены.")
    parts = tuple(normalized.split("/"))
    for part in parts:
        _component(part)
    if len(parts) > MAX_DEPTH:
        raise HarnessError(f"Глубина пути ограничена {MAX_DEPTH} папками.")
    return parts


def _repository_path(value: Path | None) -> Path:
    if not supported():
        raise HarnessError("Строгий файловый доступ поддерживается только в Windows; инструменты репозитория отключены.")
    if value is None:
        raise HarnessError("Сначала выберите существующую папку репозитория в меню инструментов.")
    path = Path(value)
    raw = str(path)
    if not path.is_absolute() or raw.startswith(("\\\\", "//")) or len(path.anchor) != 3:
        raise HarnessError("Репозиторий должен быть абсолютной папкой на локальном диске Windows.")
    for part in path.parts[1:]:
        _component(part)
    # No resolve(): it would follow precisely the links this boundary prohibits.
    return path


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description + " Each call requires human approval.",
            "parameters": {
                "type": "object", "properties": properties,
                "required": required, "additionalProperties": False,
            },
        },
    }


_PATH = {"type": "string", "description": "Relative path inside this chat's selected repository. No .git or links."}
_SCHEMAS = {
    "list_files": _function("list_files", "List repository entries, at most 100 results (scan bounded to 1000 entries).", {
        "path": {**_PATH, "default": "."},
        "recursive": {"type": "boolean", "default": False},
        "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "default": 100},
    }, []),
    "read_file": _function("read_file", "Read one UTF-8 file, up to 256 KiB.", {"path": _PATH}, ["path"]),
    "write_file": _function("write_file", "Create or replace one UTF-8 file, up to 256 KiB. Missing parent directories are created inside the repository after approval. Approval shows all new directories and the complete proposed contents.", {
        "path": _PATH, "content": {"type": "string", "description": "Complete new file contents, not a patch."},
    }, ["path", "content"]),
    "create_directory": _function("create_directory", "Create a repository directory and its missing parents after approval. Existing ordinary directories are accepted; links are forbidden.", {
        "path": _PATH,
    }, ["path"]),
    "powershell_run": _function("powershell_run", "Read and run one existing repository .ps1 script in restricted PowerShell. Only pure computation/control flow and " + ", ".join(_POWERSHELL_CMDLETS) + " are allowed. No filesystem/network commands, external programs, .NET types or member access. Script contents are read only after approval; use a separate approved read_file to inspect them first.", {
        "path": _PATH,
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 15, "default": 5},
    }, ["path"]),
    "search_files": _function("search_files", "Search literal text in UTF-8 files inside the repository; bounded scan. No regular expressions.", {
        "query": {"type": "string", "minLength": 1, "maxLength": 500},
        "path": {**_PATH, "default": "."},
        "case_sensitive": {"type": "boolean", "default": False},
        "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "default": 30},
    }, ["query"]),
    "web_fetch": _function("web_fetch", "Read a public HTTP(S) page using GET, at most 512 KiB. No private networks, redirects, cookies or credentials.", {
        "url": {"type": "string", "description": "Public HTTP(S) URL on its standard port."},
    }, ["url"]),
    "web_search": _function("web_search", "Search the public web using a DuckDuckGo HTML GET and return bounded text and links.", {
        "query": {"type": "string", "minLength": 1, "maxLength": 500},
    }, ["query"]),
}
_CODE_TOOLS = frozenset({"list_files", "read_file", "write_file", "search_files", "create_directory"})
_REPOSITORY_TOOLS = _CODE_TOOLS | {"powershell_run"}
_WEB_TOOLS = frozenset({"web_fetch", "web_search"})


def _run_restricted_powershell(source: str, repository: Path, timeout_seconds: int) -> dict:
    # Capture trusted executable code at application import, before a selected
    # coding repository can modify application source. Importing a runner only
    # after consent would permit an edited .py file to execute inside the host.
    # Actual script text reads and runner activity still happen after consent.
    try:
        result = run_powershell(source, repository, timeout_seconds)
    except PowerShellError as error:
        raise _PowerShellFailure(str(error), execution_started=getattr(error, "started", False) is True) from error
    if not isinstance(result, dict) or result.get("restricted") is not True:
        raise _PowerShellFailure("Исполнитель не подтвердил ограничения PowerShell; результат отклонён.", execution_started=True)
    return {**result, "execution_started": True}


def _public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if not address.is_global or address.is_multicast or address.is_reserved:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return False
        blocked = ("::/96", "64:ff9b::/96", "64:ff9b:1::/48", "2001::/23", "2002::/16", "fec0::/10")
    else:
        # Explicit ranges also cover older Python ipaddress classifications.
        blocked = ("0.0.0.0/8", "100.64.0.0/10", "192.0.0.0/24", "192.0.2.0/24", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24")
    return not any(address in ipaddress.ip_network(network) for network in blocked)


def _url(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise HarnessError("URL должен быть строкой длиной до 4096 символов.")
    if any(char.isspace() or ord(char) < 32 or 127 <= ord(char) <= 159 for char in value) or _BIDI.search(value) or "\\" in value:
        raise HarnessError("URL содержит недопустимые символы.")
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.hostname:
            raise HarnessError("Разрешены только публичные URL HTTP и HTTPS.")
        if parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise HarnessError("URL с учётными данными или фрагментом запрещён.")
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        if "%" in host or not re.fullmatch(r"[a-z0-9.:-]+", host):
            raise HarnessError("Недопустимое имя сервера.")
        port = parsed.port
        if port is not None and port != (443 if scheme == "https" else 80):
            raise HarnessError("Разрешены только стандартные порты HTTP (80) и HTTPS (443).")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if host.rstrip(".") in {"localhost", "local", "internal"} or host.rstrip(".").endswith((".localhost", ".local", ".internal", ".onion")):
                raise HarnessError("Локальные и служебные серверы запрещены.")
        else:
            if not _public_address(str(address)):
                raise HarnessError("Локальные, приватные и служебные IP-адреса запрещены.")
        authority = f"[{host}]" if ":" in host else host
        path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
        query = quote(parsed.query, safe="%/?@!$&'()*+,;=:-._~")
        return urlunsplit((scheme, authority, path, query, ""))
    except (ValueError, UnicodeError) as error:
        raise HarnessError("Некорректный URL.") from error


def _resolve_public(host: str, port: int, deadline: float) -> list[tuple]:
    """Resolve once, under a deadline. The connect never resolves the name again."""
    resolved: queue.Queue = queue.Queue(maxsize=1)

    def lookup():
        try:
            resolved.put((True, socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
        except Exception as error:
            resolved.put((False, error))

    worker = threading.Thread(target=lookup, daemon=True)
    worker.start()
    try:
        success, entries = resolved.get(timeout=max(0.001, deadline - time.monotonic()))
    except queue.Empty as error:
        raise HarnessError("Истекло время DNS-запроса.") from error
    if not success:
        raise HarnessError("Не удалось разрешить публичное имя сервера.") from entries
    if not entries or len(entries) > 32:
        raise HarnessError("DNS вернул недопустимое число адресов.")
    for family, socktype, protocol, _, address in entries:
        if family not in (socket.AF_INET, socket.AF_INET6) or socktype != socket.SOCK_STREAM or not _public_address(address[0]):
            raise HarnessError("DNS вернул локальный, приватный или служебный адрес; запрос заблокирован.")
    return entries


def _request_public(url: str) -> tuple[bytes, str]:
    parsed = urlsplit(url)
    host = parsed.hostname
    port = 443 if parsed.scheme == "https" else 80
    deadline = time.monotonic() + NETWORK_TIMEOUT
    addresses = _resolve_public(host, port, deadline)
    family, socktype, protocol, _, address = addresses[0]
    connection = HTTPConnection(host, port, timeout=NETWORK_TIMEOUT)
    sockets: list[socket.socket] = []
    socket_lock = threading.Lock()
    cancelled = threading.Event()

    def interrupt():
        # A hard wall-clock deadline also interrupts slow trickle responses; a
        # per-recv timeout by itself would not bound their total duration.
        cancelled.set()
        with socket_lock:
            for current in sockets:
                try:
                    current.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                current.close()

    timer = threading.Timer(max(0.001, deadline - time.monotonic()), interrupt)
    timer.daemon = True
    timer.start()
    try:
        with socket_lock:
            if cancelled.is_set() or time.monotonic() >= deadline:
                raise HarnessError("Истекло время веб-запроса.")
            sock = socket.socket(family, socktype, protocol)
            sockets.append(sock)
        sock.settimeout(max(0.001, deadline - time.monotonic()))
        sock.connect(address)  # approved, validated IP; no second DNS lookup
        if parsed.scheme == "https":
            context = ssl.create_default_context()
            with socket_lock:
                if cancelled.is_set() or time.monotonic() >= deadline:
                    raise HarnessError("Истекло время веб-запроса.")
                sock = context.wrap_socket(sock, server_hostname=host, do_handshake_on_connect=False)
                sockets.append(sock)
            if time.monotonic() >= deadline:
                raise HarnessError("Истекло время веб-запроса.")
            sock.settimeout(max(0.001, deadline - time.monotonic()))
            sock.do_handshake()
        connection.sock = sock
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        connection.request("GET", target, headers={
            "User-Agent": "llmopenchat/0.1 human-approved-public-web",
            "Accept": "text/html,text/plain,application/json,application/xml;q=0.8",
            "Accept-Encoding": "identity",
            "Connection": "close",
        })
        response = connection.getresponse()
        if 300 <= response.status < 400:
            location = response.getheader("Location")
            message = "Сервер вернул перенаправление. Автоматический переход запрещён; для нового URL нужен отдельный вызов и апрув."
            if location:
                try:
                    message += " Новый URL: " + _url(urljoin(url, location))
                except (HarnessError, ValueError):
                    message += " Адрес перенаправления небезопасен или некорректен."
            raise HarnessError(message)
        if not 200 <= response.status < 300:
            raise HarnessError(f"Сервер вернул HTTP {response.status}.")
        if response.getheader("Content-Encoding", "identity").lower() not in {"", "identity"}:
            raise HarnessError("Сжатые сетевые ответы запрещены.")
        content_type = response.getheader("Content-Type", "text/plain")
        mime = content_type.split(";", 1)[0].strip().lower()
        if not (mime.startswith("text/") or mime in {"application/json", "application/xml", "application/xhtml+xml", "application/javascript"}):
            raise HarnessError("Разрешены только текстовые веб-ответы.")
        length = response.getheader("Content-Length")
        if length is not None and (not length.isdecimal() or int(length) > MAX_WEB_BYTES):
            raise HarnessError(f"Веб-ответ превышает лимит {MAX_WEB_BYTES} байт.")
        content = response.read(MAX_WEB_BYTES + 1)
        if len(content) > MAX_WEB_BYTES:
            raise HarnessError(f"Веб-ответ превышает лимит {MAX_WEB_BYTES} байт.")
        if time.monotonic() >= deadline:
            raise HarnessError("Истекло время веб-запроса.")
        return content, content_type
    finally:
        timer.cancel()
        connection.close()
        interrupt()


class _ReadableHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.text: list[str] = []
        self.links: list[dict] = []
        self.anchor: tuple[str, list[str]] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr", "section"}:
            self.text.append("\n")
        if tag == "a" and len(self.links) < 50:
            href = dict(attrs).get("href")
            if href:
                self.anchor = (href, [])

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1
        if tag == "a" and self.anchor:
            href, labels = self.anchor
            self.links.append({"url": href, "title": " ".join("".join(labels).split())[:300]})
            self.anchor = None

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)
            if self.anchor:
                self.anchor[1].append(data)


def _web_text(url: str) -> dict:
    content, content_type = _request_public(url)
    charset = re.search(r"charset\s*=\s*[\"']?([\w-]+)", content_type, re.I)
    encoding = charset.group(1) if charset else "utf-8"
    try:
        text = content.decode(encoding, errors="replace")
    except LookupError:
        text = content.decode("utf-8", errors="replace")
    links = []
    if "html" in content_type.lower():
        parser = _ReadableHTML()
        parser.feed(text)
        text = "\n".join(line for raw in "".join(parser.text).splitlines() if (line := " ".join(raw.split())))
        for link in parser.links:
            target = urljoin(url, link["url"])
            parsed = urlsplit(target)
            if parsed.hostname in {"duckduckgo.com", "html.duckduckgo.com"}:
                target = parse_qs(parsed.query).get("uddg", [target])[0]
            try:
                links.append({"title": link["title"], "url": _url(target)})
            except (HarnessError, ValueError):
                continue
    return {"url": url, "text": text[:MAX_WEB_TEXT], "links": links[:30], "truncated": len(text) > MAX_WEB_TEXT}


class ToolHarness:
    def __init__(self, settings: HarnessSettings, approve: Callable[[ToolRequest], bool],
                 *, protected_names: frozenset[str] = frozenset()):
        if any(type(value) is not bool for value in (settings.web_enabled, settings.code_enabled, settings.powershell_enabled)):
            raise HarnessError("Переключатели инструментов должны быть логическими значениями.")
        if not callable(approve):
            raise HarnessError("Для инструментов требуется обработчик ручного подтверждения.")
        if not isinstance(protected_names, frozenset) or not all(isinstance(name, str) for name in protected_names):
            raise HarnessError("Защищённые имена файлов должны быть неизменяемым набором строк.")
        # A trusted caller can reserve policy files. Model arguments never alter
        # this set, and native file handles also reject hardlinks and reparse points.
        self._protected_names = frozenset({"hosts.txt"}) | frozenset(name.casefold() for name in protected_names)
        repository = _repository_path(settings.repository) if settings.code_enabled or settings.powershell_enabled else settings.repository
        self.settings = replace(settings, repository=repository)
        self.approve = approve
        self._identity = None
        if settings.code_enabled or settings.powershell_enabled:
            try:
                with LockedRepository(repository, settings.repository_identity) as locked:
                    self._identity = locked.identity
                    self.settings = replace(self.settings, repository_identity=locked.identity)
            except FileAccessError as error:
                raise HarnessError(str(error)) from error

    def schemas(self) -> list[dict]:
        names = []
        if self.settings.code_enabled:
            names.extend(("list_files", "read_file", "write_file", "search_files", "create_directory"))
        if self.settings.powershell_enabled:
            names.append("powershell_run")
        if self.settings.web_enabled:
            names.extend(("web_fetch", "web_search"))
        return [copy.deepcopy(_SCHEMAS[name]) for name in names]

    def _arguments(self, name: str, raw: str) -> dict:
        if name not in _SCHEMAS:
            raise HarnessError("Неизвестный инструмент.")
        if name in _CODE_TOOLS and not self.settings.code_enabled:
            raise HarnessError("Инструменты кода отключены для этого чата.")
        if name == "powershell_run" and not self.settings.powershell_enabled:
            raise HarnessError("PowerShell отключён для этого чата.")
        if name in _WEB_TOOLS and not self.settings.web_enabled:
            raise HarnessError("Веб-доступ отключён для этого чата.")
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_ARGUMENT_BYTES:
            raise HarnessError("Аргументы должны быть JSON-объектом ограниченного размера.")

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise HarnessError("Повторяющиеся JSON-ключи запрещены.")
                result[key] = value
            return result

        try:
            arguments = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(HarnessError("Некорректное JSON-число.")))
        except (ValueError, RecursionError) as error:
            raise HarnessError("Некорректные JSON-аргументы.") from error
        schema = _SCHEMAS[name]["function"]["parameters"]
        if not isinstance(arguments, dict) or set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(arguments):
            raise HarnessError("Аргументы не соответствуют схеме инструмента.")
        for key, definition in schema["properties"].items():
            if key not in arguments:
                if "default" in definition:
                    arguments[key] = definition["default"]
                continue
            value = arguments[key]
            expected = {"string": str, "boolean": bool, "integer": int}[definition["type"]]
            if type(value) is not expected:
                raise HarnessError(f"Недопустимый тип аргумента {key}.")
            if isinstance(value, str) and (len(value) < definition.get("minLength", 0) or len(value) > definition.get("maxLength", MAX_ARGUMENT_BYTES)):
                raise HarnessError(f"Недопустимая длина аргумента {key}.")
            if expected is int and not definition["minimum"] <= value <= definition["maximum"]:
                raise HarnessError(f"Недопустимое значение аргумента {key}.")
        if name in _REPOSITORY_TOOLS:
            parts = _relative(arguments["path"], directory=name in {"list_files", "search_files", "create_directory"})
            if name == "powershell_run" and Path(parts[-1]).suffix.casefold() != ".ps1":
                raise HarnessError("PowerShell запускает только существующие файлы .ps1 внутри выбранного репозитория.")
        if name == "write_file" and len(arguments["content"].encode("utf-8")) > MAX_FILE_BYTES:
            raise HarnessError(f"Содержимое превышает лимит {MAX_FILE_BYTES} байт.")
        if name == "web_fetch":
            arguments["url"] = _url(arguments["url"])
        return arguments

    def _request(self, name: str, arguments: dict, directories_to_create: tuple[str, ...] = (), *, script_bytes: int | None = None) -> ToolRequest:
        if name in _WEB_TOOLS:
            url = arguments["url"] if name == "web_fetch" else "https://html.duckduckgo.com/html/?" + urlencode({"q": arguments["query"]})
            summary = f"Публичный HTTP GET: {url} (один запрос, без перенаправлений, лимит {MAX_WEB_BYTES} байт)."
            if name == "web_search":
                summary = f"Поиск DuckDuckGo: {arguments['query']}\n" + summary
            return ToolRequest(name, copy.deepcopy(arguments), summary)
        path = str(self.settings.repository.joinpath(*_relative(arguments["path"], directory=name in {"list_files", "search_files", "create_directory"})))
        operations = {"list_files": "Список файлов", "read_file": "Чтение UTF-8 файла", "write_file": "Создание или полная перезапись UTF-8 файла", "search_files": "Поиск текста в UTF-8 файлах", "create_directory": "Создание каталога", "powershell_run": "Чтение и запуск .ps1 в ограниченном PowerShell"}
        summary = f"{operations[name]}: {path}\nГраница доступа: {self.settings.repository}"
        if directories_to_create:
            summary += "\nПосле подтверждения будут созданы каталоги:\n" + "\n".join(
                str(self.settings.repository.joinpath(*directory.split("/"))) for directory in directories_to_create
            )
        elif name == "create_directory":
            summary += "\nКаталог уже существует; создавать его повторно не требуется."
        if name == "search_files":
            summary += f"\nТекст поиска: {arguments['query']}"
        if name == "list_files":
            summary += f"\nРекурсивно: {'да' if arguments['recursive'] else 'нет'}"
        preview = arguments["content"] if name == "write_file" else ""
        if name == "write_file":
            summary += f"\nПолное новое содержимое: {len(preview.encode('utf-8'))} байт. Запись начинается только после подтверждения."
        if name == "powershell_run":
            summary += (f"\nФайл: {script_bytes} байт. Лимит выполнения: {arguments['timeout_seconds']} с."
                        "\nСодержимое будет прочитано только после этого подтверждения."
                        "\nРазрешены только вычисления, управление потоком и " + ", ".join(_POWERSHELL_CMDLETS) + "."
                        "\nФайловые и сетевые команды, внешние программы, .NET типы и доступ к членам запрещены.")
        return ToolRequest(name, copy.deepcopy(arguments), summary, preview)

    def execute(self, name: str, arguments: str) -> str:
        try:
            values = self._arguments(name, arguments)
            if name in {"write_file", "create_directory"}:
                parts = _relative(values["path"])
                if any(part.casefold() in self._protected_names for part in parts):
                    raise HarnessError("Изменение файла политики сценария и его подмена запрещены.")
            snapshot = replace(self.settings)
            with ExitStack() as stack:
                locked = None
                file = None
                new_file = False
                directory_parts: tuple[str, ...] = ()
                existing_count = 0
                directories_to_create: tuple[str, ...] = ()
                if name in _REPOSITORY_TOOLS:
                    locked = stack.enter_context(LockedRepository(snapshot.repository, self._identity))
                    parts = _relative(values["path"], directory=name in {"list_files", "search_files", "create_directory"})
                    if name in {"write_file", "create_directory"}:
                        directory_parts = parts[:-1] if name == "write_file" else parts
                        existing_count = locked.plan_directory(directory_parts)
                        directories_to_create = tuple(
                            "/".join(directory_parts[:index + 1]) for index in range(existing_count, len(directory_parts))
                        )
                    if name in {"read_file", "write_file", "powershell_run"}:
                        target = locked.root.joinpath(*parts)
                        if name in {"read_file", "powershell_run"}:
                            locked.directory(parts[:-1])
                        # No target can preexist while an ancestor is missing.
                        # It remains CREATE_NEW after approval even if another
                        # process supplies parents during the approval prompt.
                        exists = False
                        if not directories_to_create:
                            try:
                                os.lstat(target)
                                exists = True
                            except FileNotFoundError:
                                pass
                        if name == "write_file" and not exists:
                            # Missing targets are created with CREATE_NEW only
                            # after consent; a racing new file cannot be replaced.
                            new_file = True
                        else:
                            file = stack.enter_context(LockedHandle(target, write=name == "write_file"))
                            if name == "powershell_run" and file.size > MAX_FILE_BYTES:
                                raise HarnessError(f"PowerShell-скрипт превышает лимит {MAX_FILE_BYTES} байт.")
                    elif name != "create_directory":
                        locked.directory(parts)
                request = self._request(name, values, directories_to_create, script_bytes=file.size if name == "powershell_run" else None)
                try:
                    approved = self.approve(request) is True
                except (Exception, KeyboardInterrupt):
                    approved = False
                if not approved:
                    result = {"status": "denied", "message": "Пользователь не разрешил использование инструмента."}
                    if name == "powershell_run":
                        result.update({"restricted": True, "available_cmdlets": list(_POWERSHELL_CMDLETS), "execution_started": False})
                    return json.dumps(result, ensure_ascii=False)
                if self.settings != snapshot:
                    result = {"status": "denied", "message": "Настройки инструментов изменились во время подтверждения."}
                    if name == "powershell_run":
                        result.update({"restricted": True, "available_cmdlets": list(_POWERSHELL_CMDLETS), "execution_started": False})
                    return json.dumps(result, ensure_ascii=False)
                status = "ok"
                if name == "read_file":
                    result = {"path": values["path"], "content": file.read(MAX_FILE_BYTES).decode("utf-8")}
                elif name == "write_file":
                    created_directories = locked.create_directories(directory_parts, existing_count)
                    if new_file:
                        file = stack.enter_context(LockedHandle(target, write=True, create=True))
                    content = values["content"].encode("utf-8")
                    file.write(content)
                    result = {"path": values["path"], "bytes_written": len(content), "created": new_file, "directories_created": created_directories}
                elif name == "create_directory":
                    created_directories = locked.create_directories(directory_parts, existing_count)
                    result = {"path": values["path"], "created": bool(created_directories), "directories_created": created_directories}
                elif name == "powershell_run":
                    try:
                        source = file.read(MAX_FILE_BYTES).decode("utf-8-sig")
                    except UnicodeDecodeError as error:
                        raise HarnessError("PowerShell-скрипт должен иметь кодировку UTF-8.") from error
                    if "\x00" in source:
                        raise HarnessError("PowerShell-скрипт содержит бинарные данные; требуется текст UTF-8.")
                    result = {**_run_restricted_powershell(source, snapshot.repository, values["timeout_seconds"]), "path": values["path"], "execution_started": True}
                    if type(result.get("exit_code")) is not int or result["exit_code"] != 0:
                        status = "error"
                        result["message"] = "Ограниченный PowerShell завершился с ошибкой."
                elif name in {"list_files", "search_files"}:
                    result = self._scan(locked, values, search=name == "search_files")
                else:
                    url = values["url"] if name == "web_fetch" else "https://html.duckduckgo.com/html/?" + urlencode({"q": values["query"]})
                    result = _web_text(url)
                return json.dumps({**result, "status": status}, ensure_ascii=False)
        except (HarnessError, FileAccessError, OSError, UnicodeError, HTTPException, ValueError, TypeError, OverflowError) as error:
            result = {"status": "error", "message": str(error)}
            if name == "powershell_run":
                result.update({"restricted": True, "available_cmdlets": list(_POWERSHELL_CMDLETS),
                               "execution_started": getattr(error, "execution_started", False) is True})
            return json.dumps(result, ensure_ascii=False)

    def _scan(self, locked: LockedRepository, values: dict, *, search: bool) -> dict:
        start_parts = _relative(values["path"], directory=True)
        root = locked.root
        results: list[dict] = []
        visited = 0
        bytes_read = 0
        skipped = 0
        truncated = False
        query = values.get("query", "")
        match_query = query if values.get("case_sensitive") else query.casefold()

        def walk(parts: tuple[str, ...]):
            nonlocal visited, bytes_read, skipped, truncated
            if len(parts) > MAX_DEPTH:
                truncated = True
                return
            path = root.joinpath(*parts)
            with os.scandir(path) as entries:
                for entry in entries:
                    if visited >= MAX_FILES or len(results) >= values["max_results"] or bytes_read >= MAX_SEARCH_BYTES:
                        truncated = True
                        return
                    visited += 1
                    try:
                        _component(entry.name)
                    except HarnessError:
                        skipped += 1
                        continue
                    target_parts = (*parts, entry.name)
                    relative = "/".join(target_parts)
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                        if getattr(metadata, "st_file_attributes", 0) & 0x400 or metadata.st_nlink > 1:
                            skipped += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            with LockedHandle(Path(entry.path), directory=True):
                                if not search:
                                    results.append({"path": relative, "type": "directory"})
                                if search or values.get("recursive"):
                                    walk(target_parts)
                        elif entry.is_file(follow_symlinks=False):
                            with LockedHandle(Path(entry.path)) as handle:
                                if not search:
                                    results.append({"path": relative, "type": "file", "bytes": handle.size})
                                    continue
                                if handle.size > MAX_FILE_BYTES or bytes_read + handle.size > MAX_SEARCH_BYTES:
                                    skipped += 1
                                    continue
                                content = handle.read(MAX_FILE_BYTES)
                                bytes_read += len(content)
                            if b"\x00" in content:
                                skipped += 1
                                continue
                            text = content.decode("utf-8")
                            for number, line in enumerate(text.splitlines(), 1):
                                haystack = line if values.get("case_sensitive") else line.casefold()
                                if match_query in haystack:
                                    results.append({"path": relative, "line": number, "text": line[:1000]})
                                    if len(results) >= values["max_results"]:
                                        truncated = True
                                        return
                        else:
                            skipped += 1
                    except (FileAccessError, OSError, UnicodeError):
                        skipped += 1

        walk(start_parts)
        return {"matches" if search else "entries": results, "truncated": truncated, "skipped": skipped, "scanned": visited}
