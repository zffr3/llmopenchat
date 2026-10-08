"""Trusted, persisted definitions for unattended text-processing scenarios."""

from __future__ import annotations

import copy
import ipaddress
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import NoSuchResource, Unresolvable

from ._winfiles import FileAccessError, LockedHandle, LockedRepository, supported
from .harness import HarnessSettings, ToolRequest


MAX_JSON_BYTES = 1024 * 1024
MAX_INPUT_BYTES = 16 * MAX_JSON_BYTES
_CODE_TOOLS = frozenset({"list_files", "read_file", "write_file", "search_files", "create_directory"})
_WEB_TOOLS = frozenset({"web_fetch", "web_search"})
_REQUEST_FIELDS = frozenset({
    "model", "messages", "stream", "stream_options", "temperature", "max_tokens",
    "tools", "tool_choice", "parallel_tool_calls", "response_format",
})
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_RESERVED_HEADERS = frozenset({
    "host", "content-type", "content-length", "transfer-encoding", "connection",
    "proxy-authorization", "proxy-connection", "trailer", "upgrade", "expect",
})
_DRAFT = "https://json-schema.org/draft/2020-12/schema"
MAX_HOSTS_BYTES = 64 * 1024
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


class ScenarioError(ValueError):
    pass


def _origin(url: str, *, entry: bool = False) -> str:
    if not isinstance(url, str) or not url:
        raise ScenarioError("Ожидается HTTP(S)-адрес ограниченного размера.")
    try:
        if len(url.encode("utf-8")) > 8192:
            raise ValueError("oversized URL")
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
        if (parsed.scheme not in ("http", "https") or not host
                or parsed.username is not None or parsed.password is not None
                or parsed.fragment or "#" in url or "\\" in url
                or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url)
                or (port is not None and not 1 <= port <= 65535)
                or (entry and (parsed.path not in ("", "/") or parsed.query or "?" in url))):
            raise ValueError("invalid origin")
        if parsed.netloc.endswith(":") or "%" in host:
            raise ValueError("invalid host or port")
        if host.endswith("."):
            host = host[:-1]
        try:
            address = ipaddress.ip_address(host)
            host = address.compressed
            if address.version == 6:
                host = f"[{host}]"
        except ValueError:
            host = host.encode("idna").decode("ascii").lower()
            if len(host) > 253 or not all(_DNS_LABEL.fullmatch(label) for label in host.split(".")):
                raise ValueError("invalid hostname")
        return f"{parsed.scheme}://{host}:{port or (443 if parsed.scheme == 'https' else 80)}"
    except (ValueError, UnicodeError) as error:
        raise ScenarioError("Ожидается HTTP(S)-адрес с корректным хостом, без логина, пароля и фрагмента.") from error


@dataclass(frozen=True)
class HostPolicy:
    path: Path
    origins: frozenset[str]

    def allows(self, url: str) -> bool:
        try:
            return _origin(url) in self.origins
        except ScenarioError:
            return False


def load_hosts(path: Path) -> HostPolicy:
    """Read an immutable allowlist of exact HTTP(S) origins, without links."""
    path = Path(path).absolute()

    def regular(info) -> bool:
        return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and not getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))

    try:
        before = path.lstat()
        if not regular(before):
            raise ScenarioError("hosts.txt должен быть обычным файлом без ссылок и reparse points.")
        if supported():
            # Pin every ancestor and read through an exclusive native handle.
            # No ancestor can be renamed or become a junction during the read.
            with LockedRepository(path.parent), LockedHandle(path) as handle:
                raw = handle.read(MAX_HOSTS_BYTES)
        else:
            with path.open("rb") as handle:
                opened = os.fstat(handle.fileno())
                if not regular(opened) or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    raise ScenarioError("hosts.txt был заменён при открытии.")
                raw = handle.read(MAX_HOSTS_BYTES + 1)
        after = path.lstat()
        if not regular(after) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ScenarioError("hosts.txt был заменён при чтении.")
        if len(raw) > MAX_HOSTS_BYTES:
            raise ScenarioError(f"hosts.txt превышает лимит {MAX_HOSTS_BYTES} байт.")
        source = raw.decode("utf-8-sig")
    except (OSError, UnicodeError, FileAccessError) as error:
        raise ScenarioError(f"Не удалось прочитать обязательный список разрешённых хостов {path}.") from error
    origins = set()
    for number, line in enumerate(source.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            origins.add(_origin(line, entry=True))
        except ScenarioError as error:
            raise ScenarioError(f"Некорректный origin в hosts.txt, строка {number}; укажите scheme://host[:port].") from error
        if len(origins) > 1024:
            raise ScenarioError("hosts.txt допускает не более 1024 origins.")
    return HostPolicy(path=path, origins=frozenset(origins))


def _pairs(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ScenarioError("Повторяющиеся JSON-ключи запрещены.")
        result[key] = value
    return result


def _constant(value: str):
    raise ScenarioError("Неконечные числа запрещены в JSON.")


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ScenarioError("Неконечные числа запрещены в JSON.")
    return number


def parse_json_document(text: str, *, max_bytes: int = MAX_JSON_BYTES) -> object:
    """Parse one bounded UTF-8 JSON document, without lossy repair."""
    if not isinstance(text, str):
        raise ScenarioError("JSON должен быть текстом UTF-8.")
    try:
        if len(text.encode("utf-8")) > max_bytes:
            raise ScenarioError(f"JSON превышает лимит {max_bytes} байт.")
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
        # Escaped lone surrogates and non-finite numbers must not survive parsing.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, ScenarioError):
            raise
        raise ScenarioError("Некорректный JSON или кодировка UTF-8.") from error
    return value


def _object(value: object, label: str, fields: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ScenarioError(f"{label} должен быть JSON-объектом.")
    unknown = value.keys() - fields
    if unknown:
        raise ScenarioError(f"Неизвестные поля {label}: {', '.join(sorted(unknown))}.")
    return value


def _text(value: object, label: str, *, maximum: int = MAX_JSON_BYTES) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > maximum or "\x00" in value:
        raise ScenarioError(f"{label} должен быть непустой строкой (максимум {maximum} байт).")
    return value


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ScenarioError(f"{label} должен быть целым числом от {minimum} до {maximum}.")
    return value


def _number(value: object, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
        raise ScenarioError(f"{label} должен быть числом от {minimum:g} до {maximum:g}.")
    return float(value)


def _no_external_schema(resource: str):
    # Keep local validation offline even if a nested $id changes resolution scope.
    raise NoSuchResource(ref=resource)


def _validate_schema(schema: object) -> dict:
    if not isinstance(schema, dict):
        raise ScenarioError("response_schema должен быть JSON Schema объектом.")
    if schema.get("$schema", _DRAFT) != _DRAFT:
        raise ScenarioError("response_schema поддерживает JSON Schema Draft 2020-12.")
    pending = [schema]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                if key in item and (not isinstance(item[key], str) or not item[key].startswith("#")):
                    raise ScenarioError("response_schema допускает только локальные ссылки #; загрузка внешних схем запрещена.")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    try:
        Draft202012Validator.check_schema(schema)
    except (SchemaError, RecursionError) as error:
        raise ScenarioError("Некорректная response_schema (JSON Schema Draft 2020-12).") from error
    return schema


@dataclass(frozen=True)
class ScenarioHarness:
    web_enabled: bool = False
    code_enabled: bool = False
    powershell_enabled: bool = False
    repository: Path | None = None
    auto_approve: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ScenarioOutput:
    type: str = "stdout"
    url: str | None = None
    headers_env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 15.0


@dataclass(frozen=True)
class Scenario:
    name: str
    instruction: str
    response_schema: dict
    generation: dict = field(default_factory=dict)
    harness: ScenarioHarness = field(default_factory=ScenarioHarness)
    input_max_bytes: int = MAX_JSON_BYTES
    output: ScenarioOutput = field(default_factory=ScenarioOutput)
    source_path: Path | None = None
    schema_version: int = 1
    hosts: HostPolicy | None = None

    @property
    def hosts_path(self) -> Path | None:
        return self.source_path.parent / "hosts.txt" if self.source_path is not None else None

    @property
    def allowed_origins(self) -> frozenset[str]:
        return self.hosts.origins if self.hosts is not None else frozenset()

    def network_allowed(self, url: str) -> bool:
        return self.hosts is not None and self.hosts.allows(url)

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self.harness.auto_approve

    def harness_settings(self) -> HarnessSettings:
        return HarnessSettings(
            web_enabled=self.harness.web_enabled, code_enabled=self.harness.code_enabled,
            powershell_enabled=self.harness.powershell_enabled, repository=self.harness.repository,
        )

    def approval(self, request: ToolRequest) -> bool:
        """Autoapproval is restricted to allowlisted network requests."""
        if request.name not in self.allowed_tools:
            return False
        if request.name == "web_fetch":
            return self.network_allowed(request.arguments.get("url"))
        if request.name == "web_search":
            return self.network_allowed("https://html.duckduckgo.com/")
        return False

    def make_config(self, base: dict) -> dict:
        config = copy.deepcopy(base)
        config["system_prompt"] = self.instruction
        for key in ("temperature", "max_tokens"):
            if key in self.generation:
                config[key] = self.generation[key]
        extra = copy.deepcopy(self.generation.get("request_extra", config.get("request_extra", {})))
        for key in _REQUEST_FIELDS:
            extra.pop(key, None)
        mode = self.generation.get("response_mode", "json_schema")
        if mode == "json_schema":
            schema_name = re.sub(r"[^A-Za-z0-9_-]", "_", self.name)[:64].strip("_") or "scenario"
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": copy.deepcopy(self.response_schema)},
            }
        elif mode == "json_object":
            extra["response_format"] = {"type": "json_object"}
        config["request_extra"] = extra
        return config

    def validate_response(self, text: str) -> object:
        value = parse_json_document(text)
        try:
            Draft202012Validator(
                self.response_schema, registry=Registry(retrieve=_no_external_schema),
                format_checker=Draft202012Validator.FORMAT_CHECKER,
            ).validate(value)
        except ValidationError as error:
            path = ".".join(str(part) for part in error.absolute_path) or "$"
            # Do not echo source text or generated values into operational logs.
            raise ScenarioError(f"Ответ модели не соответствует response_schema: {path} ({error.validator}).") from error
        except (Unresolvable, RecursionError, ValueError) as error:
            raise ScenarioError("Не удалось проверить ответ по response_schema: некорректная локальная ссылка или схема.") from error
        return value


def _load_harness(value: object, parent: Path) -> ScenarioHarness:
    values = _object(value, "harness", {"web_enabled", "code_enabled", "powershell_enabled", "repository", "auto_approve"})
    switches = {}
    for key in ("web_enabled", "code_enabled", "powershell_enabled"):
        switch = values.get(key, False)
        if type(switch) is not bool:
            raise ScenarioError(f"harness.{key} должен быть логическим значением.")
        switches[key] = switch
    repository = values.get("repository")
    if repository is not None:
        repository = Path(_text(repository, "harness.repository", maximum=4096))
        if not repository.is_absolute():
            # Avoid resolve(), which would silently follow repository symlinks.
            repository = Path(os.path.abspath(parent / repository))
    if (switches["code_enabled"] or switches["powershell_enabled"]) and repository is None:
        raise ScenarioError("Для инструментов репозитория требуется harness.repository.")
    names = values.get("auto_approve", [])
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ScenarioError("harness.auto_approve должен быть списком уникальных имён инструментов.")
    enabled = set()
    if switches["code_enabled"]:
        enabled.update(_CODE_TOOLS)
    if switches["web_enabled"]:
        enabled.update(_WEB_TOOLS)
    if switches["powershell_enabled"]:
        enabled.add("powershell_run")
    if set(names) - enabled:
        raise ScenarioError("harness.auto_approve допускает только явно включённые существующие инструменты.")
    if set(names) - _WEB_TOOLS:
        raise ScenarioError("Автоподтверждение разрешено только для сетевых инструментов, ограниченных hosts.txt.")
    return ScenarioHarness(**switches, repository=repository, auto_approve=frozenset(names))


def _load_output(value: object) -> ScenarioOutput:
    values = _object(value, "output", {"type", "url", "headers_env", "timeout_seconds"})
    mode = values.get("type", "stdout")
    if mode not in ("stdout", "post"):
        raise ScenarioError("output.type должен быть stdout или post.")
    if mode == "stdout":
        if values.keys() - {"type"}:
            raise ScenarioError("Для output.type=stdout параметры POST запрещены.")
        return ScenarioOutput()
    url = _text(values.get("url"), "output.url", maximum=8192)
    _origin(url)
    if not url.isascii():
        # urllib sends an ASCII request target and Host header. Normalize IDNs
        # and encode Unicode path/query text before constructing its Request.
        parsed = urlsplit(url)
        host = urlsplit(_origin(url)).hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host += f":{parsed.port}"
        url = urlunsplit((
            parsed.scheme, host,
            quote(parsed.path, safe="/:@!$&'()*+,;=-._~%"),
            quote(parsed.query, safe="/?:@!$&'()*+,;=-._~%[]"), "",
        ))
    headers = values.get("headers_env", {})
    if not isinstance(headers, dict):
        raise ScenarioError("output.headers_env должен быть объектом: имя HTTP-заголовка → переменная окружения.")
    seen = set()
    for name, variable in headers.items():
        if not _HEADER_NAME.fullmatch(name) or name.lower() in _RESERVED_HEADERS or name.lower() in seen:
            raise ScenarioError("Некорректный, повторяющийся или служебный HTTP-заголовок output.headers_env.")
        if not isinstance(variable, str) or not _ENV_NAME.fullmatch(variable):
            raise ScenarioError("output.headers_env должен содержать имена переменных окружения.")
        seen.add(name.lower())
    timeout = _number(values.get("timeout_seconds", 15), "output.timeout_seconds", 1, 300)
    return ScenarioOutput(type="post", url=url, headers_env=copy.deepcopy(headers), timeout_seconds=timeout)


def load_scenario(path: Path) -> Scenario:
    """Load a trusted scenario file; no request data can change this policy."""
    path = Path(path).absolute()
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ScenarioError(f"Файл сценария превышает лимит {MAX_JSON_BYTES} байт.")
        values = parse_json_document(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError) as error:
        raise ScenarioError(f"Не удалось прочитать сценарий {path}.") from error
    values = _object(values, "сценария", {
        "schema_version", "name", "instruction", "response_schema", "generation", "harness", "input", "output",
    })
    if type(values.get("schema_version")) is not int or values["schema_version"] != 1:
        raise ScenarioError("schema_version должен быть целым числом 1.")
    name = _text(values.get("name"), "name", maximum=128)
    instruction = _text(values.get("instruction"), "instruction")
    schema = _validate_schema(values.get("response_schema"))
    generation = _object(values.get("generation", {}), "generation", {"temperature", "max_tokens", "request_extra", "response_mode"})
    if "temperature" in generation:
        _number(generation["temperature"], "generation.temperature", 0, 2)
    if "max_tokens" in generation:
        _integer(generation["max_tokens"], "generation.max_tokens", 1, 1048576)
    extra = generation.get("request_extra", {})
    if not isinstance(extra, dict) or extra.keys() & _REQUEST_FIELDS:
        raise ScenarioError("generation.request_extra должен быть объектом без основных полей запроса и response_format.")
    if generation.get("response_mode", "json_schema") not in ("json_schema", "json_object", "prompt"):
        raise ScenarioError("generation.response_mode должен быть json_schema, json_object или prompt.")
    inputs = _object(values.get("input", {}), "input", {"max_bytes"})
    maximum = _integer(inputs.get("max_bytes", MAX_JSON_BYTES), "input.max_bytes", 1, MAX_INPUT_BYTES)
    harness = _load_harness(values.get("harness", {}), path.parent)
    output = _load_output(values.get("output", {}))
    hosts_path = path.parent / "hosts.txt"
    hosts = load_hosts(hosts_path) if harness.web_enabled or output.type == "post" or os.path.lexists(hosts_path) else None
    if output.type == "post" and not hosts.allows(output.url):
        raise ScenarioError("output.url отсутствует в hosts.txt этого сценария.")
    return Scenario(
        name=name, instruction=instruction, response_schema=copy.deepcopy(schema), generation=copy.deepcopy(generation),
        harness=harness, input_max_bytes=maximum, output=output, source_path=path, hosts=hosts,
    )
