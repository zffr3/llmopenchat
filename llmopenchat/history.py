"""Validated chat snapshots and a small, local history index."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import ROOT
from .api import MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_CALLS, validated_tool_call

MAX_SESSION_BYTES = 16 * 1024 * 1024
MAX_TITLE_LENGTH = 80
_SAVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class SessionEntry:
    path: Path
    title: str
    saved_at: datetime
    message_count: int
    model: str | None = None
    target_model: str | None = None
    model_id: str | None = None

    @property
    def display_model(self) -> str:
        return self.target_model or self.model_id or self.model or "Неизвестная модель"


def _validated_messages(data: object) -> list[dict]:
    messages = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(messages, list):
        raise ValueError("Неверный формат диалога: нужен список messages с role и content.")
    validated: list[dict] = []
    pending: set[str] = set()
    identifiers: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("Неверный формат диалога: нужен список messages с role и content.")
        role = message["role"]
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Неверный формат диалога: неподдерживаемая роль сообщения.")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(content, str) or not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("Неверный формат диалога: результат инструмента без соответствующего вызова.")
            pending.remove(call_id)
            validated.append({"role": role, "content": content, "tool_call_id": call_id})
            continue
        if pending:
            raise ValueError("Неверный формат диалога: отсутствует результат вызова инструмента.")
        calls = message.get("tool_calls", []) if role == "assistant" else []
        if not isinstance(calls, list) or len(calls) > MAX_TOOL_CALLS:
            raise ValueError("Неверный формат диалога: неверный список вызовов инструментов.")
        if calls and content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("Неверный формат диалога: нужен список messages с role и content.")
        entry = {"role": role, "content": content}
        if calls:
            clean_calls = []
            argument_bytes = 0
            for value in calls:
                try:
                    call = validated_tool_call(value)
                except ValueError as error:
                    raise ValueError(f"Неверный формат диалога: {error}") from error
                if call["id"] in identifiers:
                    raise ValueError("Неверный формат диалога: повторный идентификатор вызова инструмента.")
                argument_bytes += len(call["function"]["arguments"].encode("utf-8"))
                if argument_bytes > MAX_TOOL_ARGUMENT_BYTES:
                    raise ValueError("Неверный формат диалога: аргументы инструментов превышают 256 КБ.")
                pending.add(call["id"])
                identifiers.add(call["id"])
                clean_calls.append(call)
            entry["tool_calls"] = clean_calls
        validated.append(entry)
    if pending:
        raise ValueError("Неверный формат диалога: отсутствует результат вызова инструмента.")
    return validated


def _read_session(path: Path) -> tuple[object, list[dict], datetime]:
    stat = path.stat()
    if stat.st_size > MAX_SESSION_BYTES:
        raise ValueError("Файл диалога слишком большой (максимум 16 МБ).")
    # Bound the read as well: a file can grow after stat().
    with path.open("rb") as source:
        content = source.read(MAX_SESSION_BYTES + 1)
    if len(content) > MAX_SESSION_BYTES:
        raise ValueError("Файл диалога слишком большой (максимум 16 МБ).")
    try:
        data = json.loads(content.decode("utf-8-sig"))
    except RecursionError as error:
        raise ValueError("Неверный формат диалога: слишком глубоко вложенный JSON.") from error
    return data, _validated_messages(data), datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)


def load_session(path: Path) -> list[dict]:
    """Load current snapshots or legacy message arrays, dropping extra fields."""
    return _read_session(Path(path))[1]


def save_session(
    path: Path,
    messages: list[dict],
    model: str,
    *,
    target_model: str | None = None,
    model_id: str | None = None,
) -> None:
    """Atomically save a snapshot while retaining the original JSON envelope."""
    path = Path(path)
    data = {
        "model": model,
        "saved_at": datetime.now().astimezone().isoformat(),
        "messages": _validated_messages(messages),
    }
    for key, value in (("target_model", target_model), ("model_id", model_id)):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} должен быть строкой.")
            data[key] = value
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if len(content.encode("utf-8")) > MAX_SESSION_BYTES:
        raise ValueError("Файл диалога слишком большой (максимум 16 МБ).")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        # Each writer owns its temporary file, so concurrent saves cannot
        # replace or delete one another's staging files.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as destination:
            temporary = Path(destination.name)
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        # Windows can reject simultaneous replacements of the same target.
        with _SAVE_LOCK:
            temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def new_session_path(root: Path = ROOT) -> Path:
    """Return a fresh autosave path, including when resuming an old snapshot."""
    filename = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex + ".json"
    return Path(root) / ".local" / "sessions" / filename


def _title(messages: list[dict]) -> str:
    for message in messages:
        if message["role"] == "user":
            text = " ".join(message["content"].split())
            # User content is printed by the history menu; remove invisible
            # formatting and terminal control characters from its title.
            text = "".join(character for character in text if not unicodedata.category(character).startswith("C"))
            if not text:
                return "Без названия"
            return text if len(text) <= MAX_TITLE_LENGTH else text[:MAX_TITLE_LENGTH - 1].rstrip() + "…"
    return "Новый диалог"


def _metadata_string(data: dict, key: str) -> str | None:
    value = data.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _saved_at(data: dict, fallback: datetime) -> datetime:
    value = data.get("saved_at")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass
    return fallback


def list_sessions(
    directory: Path | None = None,
    *,
    emit: Callable[[str], None] | None = None,
) -> list[SessionEntry]:
    """List valid snapshots newest first; optionally report skipped files."""
    directory = Path(directory) if directory is not None else ROOT / ".local" / "sessions"
    if not directory.exists():
        return []
    try:
        paths = list(directory.iterdir())
    except OSError as error:
        if emit:
            emit(f"Не удалось прочитать историю {directory}: {error}")
        return []
    entries: list[SessionEntry] = []
    for path in paths:
        if path.suffix.lower() != ".json":
            continue
        try:
            if not path.is_file():
                continue
            data, messages, modified_at = _read_session(path)
            metadata = data if isinstance(data, dict) else {}
            entries.append(SessionEntry(
                path=path,
                title=_title(messages),
                saved_at=_saved_at(metadata, modified_at),
                message_count=len(messages),
                model=_metadata_string(metadata, "model"),
                target_model=_metadata_string(metadata, "target_model"),
                model_id=_metadata_string(metadata, "model_id"),
            ))
        except (OSError, UnicodeError, ValueError, OverflowError) as error:
            if emit:
                emit(f"Пропущен диалог {path.name}: {error}")
    return sorted(entries, key=lambda entry: (entry.saved_at, entry.path.name), reverse=True)
