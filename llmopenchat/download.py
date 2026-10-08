"""Resumable parallel HTTPS downloads with pinned size and SHA256 verification."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request
import uuid


_CHUNK_SIZE = 32 * 1024 * 1024
_MIN_CHUNK_SIZE = 1024 * 1024
_READ_SIZE = 256 * 1024
_RETRIES = 3
_TIMEOUT = 30
_PROGRESS_INTERVAL = 5
_MIB = 1024**2
_GIB = 1024**3


class DownloadError(RuntimeError):
    """An actionable transfer or integrity error, without URL credentials."""


def _digest(path: Path, emit: Callable[[str], None]) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    last_update = time.monotonic()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * _MIB), b""):
            digest.update(block)
            done += len(block)
            if time.monotonic() - last_update >= _PROGRESS_INTERVAL:
                emit(f"Проверка SHA256: {done / total:.0%}")
                last_update = time.monotonic()
    return digest.hexdigest()


def _save_checkpoint(path: Path, metadata: dict, completed: set[int]) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump({**metadata, "completed": sorted(completed)}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    # Loopback HTTP is allowed for the isolated test server only.
    local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if (parsed.scheme != "https" and not local_http) or not parsed.hostname or parsed.username or parsed.password:
        raise DownloadError("Нужен публичный HTTPS-адрес без имени пользователя и пароля.")


def download_file(
    url: str,
    destination: Path,
    size: int,
    sha256: str,
    emit: Callable[[str], None] = print,
    workers: int = 8,
) -> Path:
    """Download exact byte ranges; resume only matching checkpoints; verify SHA256."""
    _validate_url(url)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise DownloadError("Размер загрузки должен быть положительным целым числом.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise DownloadError("Нужна корректная контрольная сумма SHA256.")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise DownloadError("Число соединений должно быть от 1 до 32.")
    sha256 = sha256.lower()
    destination = Path(destination).resolve()
    partial = destination.with_name(destination.name + ".partial")
    checkpoint = destination.with_name(destination.name + ".partial.json")
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != size:
            raise DownloadError(f"Файл {destination} уже существует с другим размером; переименуйте его.")
        emit("Проверка уже загруженного файла SHA256…")
        if _digest(destination, emit) != sha256:
            raise DownloadError(f"Контрольная сумма существующего файла {destination} не совпала; переименуйте его.")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = min(_CHUNK_SIZE, max(_MIN_CHUNK_SIZE, math.ceil(size / (workers * 4))))
    chunk_count = math.ceil(size / chunk_size)
    metadata = {"url": url, "size": size, "sha256": sha256, "chunk_size": chunk_size}
    completed: set[int] = set()
    if partial.exists() or checkpoint.exists():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if not partial.is_file() or partial.stat().st_size != size:
                raise ValueError("partial size")
            if any(saved.get(key) != value for key, value in metadata.items()):
                raise ValueError("checkpoint metadata")
            indices = saved.get("completed")
            if not isinstance(indices, list) or any(type(index) is not int or not 0 <= index < chunk_count for index in indices):
                raise ValueError("completed ranges")
            if len(set(indices)) != len(indices):
                raise ValueError("duplicate completed ranges")
            completed = set(indices)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise DownloadError(
                f"Незавершенная загрузка {partial.name} принадлежит другому файлу или повреждена. "
                f"Переименуйте {partial.name} и {checkpoint.name}, затем повторите."
            ) from exc
    else:
        try:
            with partial.open("xb") as stream:
                stream.truncate(size)
            _save_checkpoint(checkpoint, metadata, completed)
        except OSError as exc:
            raise DownloadError("Не удалось создать файл загрузки; проверьте свободное место и доступ к каталогу.") from exc

    lock = threading.Lock()
    stop = threading.Event()
    active: dict[int, int] = {}
    transferred = 0
    started = time.monotonic()

    def chunk_length(index: int) -> int:
        return min(chunk_size, size - index * chunk_size)

    def progress() -> str:
        with lock:
            done = sum(chunk_length(index) for index in completed) + sum(active.values())
            transferred_now = transferred
        elapsed = max(time.monotonic() - started, 0.001)
        speed = transferred_now / elapsed
        estimate = (size - done) / speed if speed else None
        eta = f", осталось ~{estimate / 60:.0f} мин" if estimate is not None else ""
        return (
            f"Загрузка: {done / size:.1%}, {done / _GIB:.2f}/{size / _GIB:.2f} ГиБ, "
            f"{speed / _MIB:.2f} МиБ/с, прошло {elapsed / 60:.1f} мин{eta}"
        )

    def download_chunk(index: int) -> None:
        nonlocal transferred
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        expected = end - start + 1
        opener = urllib.request.build_opener()
        for attempt in range(_RETRIES):
            if stop.is_set():
                return
            with lock:
                active[index] = 0
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "llmopenchat-local-downloader/1.0",
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
            )
            try:
                with opener.open(request, timeout=_TIMEOUT) as response:
                    _validate_url(response.geturl())
                    if response.status != 206:
                        raise DownloadError("Сервер не поддерживает диапазоны загрузки (ожидался HTTP 206).")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if match is None or tuple(map(int, match.groups())) != (start, end, size):
                        raise DownloadError("Сервер вернул неверный Content-Range; загрузка остановлена.")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and content_length != str(expected):
                        raise DownloadError("Сервер вернул неверный размер диапазона.")
                    read_count = 0
                    with partial.open("r+b") as output:
                        output.seek(start)
                        while read_count < expected:
                            if stop.is_set():
                                return
                            block = response.read1(min(_READ_SIZE, expected - read_count))
                            if not block:
                                raise OSError("Соединение закрыто до завершения диапазона")
                            output.write(block)
                            read_count += len(block)
                            with lock:
                                active[index] = read_count
                                transferred += len(block)
                        if response.read1(1):
                            raise DownloadError("Сервер отправил лишние байты диапазона.")
                        output.flush()
                        os.fsync(output.fileno())
                    with lock:
                        completed.add(index)
                        active.pop(index, None)
                        _save_checkpoint(checkpoint, metadata, completed)
                    return
            except urllib.error.HTTPError as exc:
                exc.close()
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise DownloadError(f"Сервер отклонил диапазон загрузки: HTTP {exc.code}.") from exc
            except DownloadError:
                raise
            except (OSError, urllib.error.URLError, http.client.HTTPException):
                pass
            if attempt + 1 == _RETRIES:
                raise DownloadError("Соединение прервалось после трех попыток. Повторный запуск продолжит загрузку.")
            if stop.wait(0.5 * (attempt + 1)):
                return

    emit(f"Загрузка по {workers} соединениям; сохранено {len(completed)}/{chunk_count} диапазонов.")
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llmopenchat-download")
    pending = set()
    try:
        pending = {executor.submit(download_chunk, index) for index in range(chunk_count) if index not in completed}
        next_update = time.monotonic() + _PROGRESS_INTERVAL
        while pending:
            finished, pending = wait(pending, timeout=_PROGRESS_INTERVAL, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
            if time.monotonic() >= next_update:
                emit(progress())
                next_update = time.monotonic() + _PROGRESS_INTERVAL
    except BaseException:
        stop.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    emit(progress())
    emit("Проверка SHA256 загруженного файла…")
    if _digest(partial, emit) != sha256:
        raise DownloadError(
            f"Контрольная сумма загрузки не совпала. Переименуйте {partial.name} "
            f"и {checkpoint.name}, затем повторите загрузку."
        )
    if destination.exists():
        raise DownloadError(f"Во время загрузки появился файл {destination}; он сохранен без изменений.")
    partial.replace(destination)
    checkpoint.unlink()
    emit(f"Загрузка и проверка завершены: {destination.name}")
    return destination
