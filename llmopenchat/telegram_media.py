"""Per-user Telegram attachments and approved model file transfers.

Attachment identifiers are local, persistent opaque IDs, never Telegram file
IDs or arbitrary paths. Repository reads and downloads retain the harness's
native file boundary, explicit consent and pinned public-network transport.
"""

from __future__ import annotations

import copy
import json
import re
import secrets
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from email.message import Message
from http.client import HTTPException
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlsplit

from ._files import FileAccessError, LockedRepository
from .harness import (
    MAX_ARGUMENT_BYTES, MAX_HTTP_HEADERS, HarnessError, HarnessSettings,
    ToolHarness, ToolRequest, _BIDI, _DOS_NAMES, _FORBIDDEN,
    _component, _download_http_public, _function, _http_arguments,
    _relative, _repository_path, _url,
)


MAX_MEDIA_BYTES = 50 * 1024 * 1024
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MEDIA_TYPES = ("document", "photo", "video", "audio", "voice", "animation", "video_note", "sticker")
_ID = re.compile(r"[a-f0-9]{32}")
_METADATA_LIMIT = 16 * 1024


def _safe_filename(value: str) -> str:
    if not isinstance(value, str):
        raise HarnessError("Имя файла должно быть строкой.")
    # Remote/user-provided names are labels only, never directory components.
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    name = _FORBIDDEN.sub("_", _BIDI.sub("_", name)).strip(" .")[:180].rstrip(" .")
    if not name or name.casefold() in {".", "..", ".git", ".metadata.json"}:
        name = "attachment.bin"
    if name.split(".", 1)[0].casefold() in _DOS_NAMES or name.casefold() == ".git":
        name = "_" + name
    _component(name)
    return name


def _mime_type(value: str) -> str:
    if not isinstance(value, str):
        return "application/octet-stream"
    mime = value.split(";", 1)[0].strip().lower()
    return mime if len(mime) <= 200 and re.fullmatch(r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+", mime) else "application/octet-stream"


class AttachmentStore:
    """Unique attachments rooted in an existing user's workspace.

    Metadata is stored next to each attachment so IDs survive service restarts.
    All creation and reads use no-follow native handles and CREATE_NEW semantics.
    """

    def __init__(self, workspace: Path):
        self.workspace = _repository_path(Path(workspace))
        with LockedRepository(self.workspace) as locked:
            self._identity = locked.identity

    def add(self, filename: str, content: bytes, media_type: str = "document",
            mime_type: str = "application/octet-stream") -> dict:
        if type(content) is not bytes or len(content) > MAX_MEDIA_BYTES:
            raise HarnessError(f"Файл должен содержать не более {MAX_MEDIA_BYTES} байт.")
        if media_type not in MEDIA_TYPES:
            raise HarnessError("Неподдерживаемый тип медиа.")
        filename = _safe_filename(filename)
        with LockedRepository(self.workspace, self._identity) as locked:
            count = locked.plan_directory(("attachments",))
            locked.create_directories(("attachments",), count)
            for _ in range(16):
                file_id = secrets.token_hex(16)
                folder = ("attachments", file_id)
                if not locked.exists(folder):
                    break
            else:
                raise HarnessError("Не удалось выделить уникальное имя вложения.")
            count = locked.plan_directory(folder)
            locked.create_directories(folder, count)
            parts = (*folder, filename)
            metadata = {
                "file_id": file_id, "filename": filename, "path": "/".join(parts),
                "size": len(content), "media_type": media_type, "mime_type": _mime_type(mime_type),
            }
            with locked.open_file(parts, write=True, create=True) as target:
                target.write(content)
            with locked.open_file((*folder, ".metadata.json"), write=True, create=True) as target:
                target.write(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
            locked.validate()
        return metadata

    @contextmanager
    def open(self, file_id: str):
        """Hold metadata, ancestors and attachment through an approval prompt."""
        if not isinstance(file_id, str) or not _ID.fullmatch(file_id):
            raise HarnessError("Недопустимый идентификатор вложения.")
        with LockedRepository(self.workspace, self._identity) as locked:
            folder = ("attachments", file_id)
            locked.directory(folder)
            with locked.open_file((*folder, ".metadata.json")) as record:
                metadata = json.loads(record.read(_METADATA_LIMIT).decode("utf-8"))
                self._validate_metadata(file_id, metadata)
                with locked.open_file((*folder, metadata["filename"])) as target:
                    if target.size != metadata["size"] or target.size > MAX_MEDIA_BYTES:
                        raise HarnessError("Размер вложения отличается от сохранённых данных.")
                    yield copy.deepcopy(metadata), target, locked

    @staticmethod
    def _validate_metadata(file_id: str, metadata: dict):
        keys = {"file_id", "filename", "path", "size", "media_type", "mime_type"}
        if (not isinstance(metadata, dict) or set(metadata) != keys
                or metadata.get("file_id") != file_id
                or not isinstance(metadata.get("filename"), str)
                or metadata["filename"] != _safe_filename(metadata["filename"])
                or metadata.get("path") != f"attachments/{file_id}/{metadata['filename']}"
                or type(metadata.get("size")) is not int or not 0 <= metadata["size"] <= MAX_MEDIA_BYTES
                or metadata.get("media_type") not in MEDIA_TYPES
                or metadata.get("mime_type") != _mime_type(metadata.get("mime_type"))):
            raise HarnessError("Повреждены метаданные вложения.")

    def read(self, file_id: str) -> tuple[dict, bytes]:
        with self.open(file_id) as (metadata, target, locked):
            content = target.read(MAX_MEDIA_BYTES)
            locked.validate()
            return metadata, content

    def path(self, file_id: str) -> Path:
        """Return a validated absolute path for trusted ingestion code."""
        with self.open(file_id) as (metadata, _target, locked):
            locked.validate()
            return self.workspace.joinpath(*metadata["path"].split("/"))


_MEDIA_SCHEMAS = {
    "telegram_send_file": _function(
        "telegram_send_file",
        "Send one nonempty binary attachment to the user in the current Telegram chat. Supply exactly one local attachment file_id (from incoming messages or http_download) or a relative repository path (requires code access). The recipient is fixed by the application. Maximum 50 MiB (photo: 10 MiB); use document to preserve arbitrary original files. A send error may mean Telegram received it; do not automatically retry.",
        {
            "file_id": {"type": "string", "description": "Local attachment ID returned in this user's messages or by http_download."},
            "path": {"type": "string", "description": "Relative file path inside the selected repository; requires code access."},
            "caption": {"type": "string", "maxLength": 1024, "default": "", "description": "Optional plain-text caption; empty for sticker or video_note."},
            "media_type": {"type": "string", "enum": list(MEDIA_TYPES), "default": "document"},
        }, []),
    "http_download": _function(
        "http_download",
        "Download one public HTTP(S) URL with GET into this user's local attachment store, up to 50 MiB. Returns file_id, filename, size and workspace-relative path, never raw binary. Use telegram_send_file with that file_id to deliver it to the current Telegram user. No redirects, retries, private networks or inherited credentials; only successful 2xx responses are stored.",
        {
            "url": {"type": "string", "description": "Public HTTP(S) URL on its standard port."},
            "headers": {"type": "object", "additionalProperties": {"type": "string"}, "maxProperties": MAX_HTTP_HEADERS, "default": {}, "description": "Explicit request headers, including Authorization if required; transport headers are forbidden."},
            "filename": {"type": "string", "maxLength": 1024, "description": "Optional preferred filename. Directory components and unsafe characters are removed."},
        }, ["url"]),
}


def _download_public(url: str, headers: dict) -> dict:
    return _download_http_public(url, headers, MAX_MEDIA_BYTES)


def _download_name(values: dict, response: dict) -> str:
    if "filename" in values:
        return values["filename"]
    disposition = response.get("headers", {}).get("content-disposition", "")
    message = Message()
    message["Content-Disposition"] = disposition
    return message.get_filename() or unquote(urlsplit(values["url"]).path.rsplit("/", 1)[-1]) or "download.bin"


class TelegramToolHarness(ToolHarness):
    def __init__(self, settings: HarnessSettings, approve: Callable[[ToolRequest], bool],
                 *, store: AttachmentStore, send: Callable[[str, bytes, str, str], dict],
                 protected_names: frozenset[str] = frozenset()):
        super().__init__(settings, approve, protected_names=protected_names)
        if not isinstance(store, AttachmentStore) or not callable(send):
            raise HarnessError("Для Telegram требуются хранилище вложений и обработчик отправки.")
        self.store = store
        self.send = send

    def schemas(self) -> list[dict]:
        schemas = super().schemas() + [copy.deepcopy(_MEDIA_SCHEMAS["telegram_send_file"])]
        if self.settings.http_enabled:
            schemas.append(copy.deepcopy(_MEDIA_SCHEMAS["http_download"]))
        return schemas

    def _media_arguments(self, name: str, raw: str) -> dict:
        if name == "http_download" and not self.settings.http_enabled:
            raise HarnessError("HTTP-запросы отключены для этого чата.")
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
            values = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(HarnessError("Некорректное JSON-число.")))
        except (ValueError, RecursionError) as error:
            raise HarnessError("Некорректные JSON-аргументы.") from error
        schema = _MEDIA_SCHEMAS[name]["function"]["parameters"]
        if not isinstance(values, dict) or set(values) - set(schema["properties"]) or set(schema["required"]) - set(values):
            raise HarnessError("Аргументы не соответствуют схеме инструмента.")
        for key, definition in schema["properties"].items():
            if key not in values:
                if "default" in definition:
                    values[key] = copy.deepcopy(definition["default"])
                continue
            value = values[key]
            if type(value) is not {"string": str, "object": dict}[definition["type"]]:
                raise HarnessError(f"Недопустимый тип аргумента {key}.")
            if isinstance(value, str) and len(value) > definition.get("maxLength", MAX_ARGUMENT_BYTES):
                raise HarnessError(f"Недопустимая длина аргумента {key}.")
            if "enum" in definition and value not in definition["enum"]:
                raise HarnessError(f"Недопустимое значение аргумента {key}.")
        if name == "telegram_send_file":
            if ("file_id" in values) == ("path" in values):
                raise HarnessError("Укажите ровно один аргумент: file_id или path.")
            if "path" in values:
                if not self.settings.code_enabled:
                    raise HarnessError("Для отправки файла репозитория включите инструменты кода.")
                _relative(values["path"])
            elif not _ID.fullmatch(values["file_id"]):
                raise HarnessError("Недопустимый идентификатор вложения.")
            if "\x00" in values["caption"] or len(values["caption"].encode("utf-16-le")) // 2 > 1024:
                raise HarnessError("Подпись ограничена 1024 символами Telegram и не может содержать NUL.")
            if values["media_type"] in {"sticker", "video_note"} and values["caption"]:
                raise HarnessError("Стикеры и видеосообщения не поддерживают подпись.")
        else:
            values["url"] = _url(values["url"])
            _http_arguments({"method": "GET", "headers": values["headers"], "body": ""})
        return values

    def _approved(self, request: ToolRequest, snapshot: HarnessSettings) -> bool:
        try:
            return self.approve(request) is True and self.settings == snapshot
        except (Exception, KeyboardInterrupt):
            return False

    def execute(self, name: str, arguments: str) -> str:
        if name not in _MEDIA_SCHEMAS:
            return super().execute(name, arguments)
        started = False
        try:
            values = self._media_arguments(name, arguments)
            snapshot = replace(self.settings)
            if name == "http_download":
                summary = (f"Загрузить файл: HTTP GET {values['url']}\n"
                           f"Один запрос, без перенаправлений и повторов; лимит {MAX_MEDIA_BYTES} байт.\n"
                           "Заголовки: " + json.dumps(values["headers"], ensure_ascii=False))
                if "filename" in values:
                    summary += "\nИмя файла: " + _safe_filename(values["filename"])
                if not self._approved(ToolRequest(name, copy.deepcopy(values), summary), snapshot):
                    return self._denied(name)
                response = _download_public(values["url"], values["headers"])
                started = True
                if self.settings != snapshot:
                    return self._denied(name)
                metadata = self.store.add(_download_name(values, response), response["content"],
                                          mime_type=response["content_type"])
                result = {**metadata, "url": values["url"], "status_code": response["status_code"]}
            else:
                with ExitStack() as stack:
                    if "file_id" in values:
                        metadata, target, locked = stack.enter_context(self.store.open(values["file_id"]))
                        filename = metadata["filename"]
                        source = f"Вложение {values['file_id']} ({metadata['path']})"
                    else:
                        locked = stack.enter_context(LockedRepository(snapshot.repository, self._identity))
                        parts = _relative(values["path"])
                        locked.directory(parts[:-1])
                        target = stack.enter_context(locked.open_file(parts))
                        filename = _safe_filename(parts[-1])
                        source = str(snapshot.repository.joinpath(*parts))
                    limit = MAX_PHOTO_BYTES if values["media_type"] == "photo" else MAX_MEDIA_BYTES
                    if not 0 < target.size <= limit:
                        raise HarnessError(f"Отправляемый файл должен быть непустым и содержать не более {limit} байт.")
                    summary = (f"Отправить файл текущему пользователю Telegram: {filename}\n"
                               f"Источник: {source}\nРазмер: {target.size} байт. Тип: {values['media_type']}.\n"
                               "Отправка без автоматических повторов.\nПодпись: " + values["caption"])
                    if not self._approved(ToolRequest(name, copy.deepcopy(values), summary), snapshot):
                        return self._denied(name)
                    locked.validate()
                    content = target.read(MAX_MEDIA_BYTES)
                    if self.settings != snapshot:
                        return self._denied(name)
                    started = True
                    try:
                        sent = self.send(filename, content, values["media_type"], values["caption"])
                    except Exception as error:
                        started = getattr(error, "send_started", started) is True
                        raise HarnessError("Не удалось подтвердить отправку файла в Telegram. Файл мог быть отправлен; не повторяйте вызов автоматически.") from error
                    result = {"filename": filename, "size": len(content), "media_type": values["media_type"], "send_started": True}
                    if isinstance(sent, dict):
                        result["telegram"] = sent
            return json.dumps({**result, "status": "ok"}, ensure_ascii=False)
        except (HarnessError, FileAccessError, OSError, UnicodeError, HTTPException, ValueError, TypeError, OverflowError) as error:
            result = {"status": "error", "message": str(error)}
            if name == "telegram_send_file":
                result["send_started"] = started
            elif name == "http_download":
                result["request_started"] = started or getattr(error, "request_started", False) is True
            return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _denied(name: str) -> str:
        result = {"status": "denied", "message": "Пользователь не разрешил использование инструмента или настройки изменились."}
        if name == "telegram_send_file":
            result["send_started"] = False
        return json.dumps(result, ensure_ascii=False)
