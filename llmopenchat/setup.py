"""Install a pinned official llama.cpp Vulkan runtime and one GGUF model."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import time
from typing import Callable
import urllib.error
import urllib.request
import uuid
import zipfile


LLAMA_RELEASE = "b11445"
LLAMA_BACKEND = "vulkan"
DEFAULT_REPO = "mradermacher/Huihui-GLM-4.7-Flash-abliterated-GGUF"
DEFAULT_FILENAME = "Huihui-GLM-4.7-Flash-abliterated.Q4_K_M.gguf"
DEFAULT_REVISION = "2d925ab0fdaa279a87485f4786ca22661bf030b2"
DEFAULT_SHA256 = "ca247d439725435ad5addc43926d311d0b0cd97cb65a32eb7d5b3ac479bf434d"
_GIB = 1024**3
_ASSETS = (
    {
        "name": "llama-b11445-bin-win-vulkan-x64.zip",
        "size": 33337869,
        "sha256": "975a788f5e5a55410fb5b72660f3f5673ad9fccbe99d66a2a2034f775a0bd432",
    },
)


class SetupError(RuntimeError):
    """An actionable local installation error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SetupError(f"Путь выходит за каталог проекта: {candidate}") from exc
    return resolved


def _download_asset(asset: dict, destination: Path, emit: Callable[[str], None]) -> None:
    """Resume an archive and verify the GitHub-published SHA256 before extraction."""
    if destination.is_file():
        if destination.stat().st_size == asset["size"] and _sha256(destination) == asset["sha256"]:
            return
        raise SetupError(f"Поврежден архив {destination}. Удалите этот файл и повторите установку.")
    partial = destination.with_suffix(destination.suffix + ".partial")
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/{asset['name']}"
    if asset["size"] > 1024 * 1024:
        from .download import download_file
        download_file(url, destination, asset["size"], asset["sha256"], emit=emit, workers=16)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > asset["size"]:
            raise SetupError(f"Неверный размер незавершенного архива {partial}; удалите его и повторите.")
        if offset == asset["size"]:
            break
        headers = {"User-Agent": "llmopenchat-local-installer/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        emit(f"Загрузка {asset['name']} ({asset['size'] / _GIB:.2f} ГиБ)…")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status != 206:
                    offset = 0
                elif response.status == 206:
                    range_header = response.headers.get("Content-Range", "")
                    if not range_header.startswith(f"bytes {offset}-"):
                        raise SetupError("Сервер вернул неверный диапазон загрузки.")
                last_update = time.monotonic()
                mode = "ab" if offset else "wb"
                with partial.open(mode) as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                        offset += len(block)
                        if offset > asset["size"]:
                            raise SetupError("Загруженный архив больше указанного размера.")
                        if time.monotonic() - last_update >= 5:
                            emit(f"  {offset / asset['size']:.0%} ({offset / _GIB:.2f} ГиБ)")
                            last_update = time.monotonic()
            if offset != asset["size"]:
                raise OSError("Неполная загрузка архива")
            break
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            if attempt == 2:
                raise SetupError(f"Не удалось скачать {asset['name']}: {exc}. Повторный запуск продолжит загрузку.") from exc
            emit(f"Соединение прервалось, продолжаю загрузку (попытка {attempt + 2}/3)…")
    if _sha256(partial) != asset["sha256"]:
        raise SetupError(f"SHA256 не совпал у {partial}. Удалите этот файл и повторите установку.")
    partial.replace(destination)


def _extract_archive(archive: Path, target: Path) -> None:
    """Reject traversal and symlinks even for the pinned official release assets."""
    with zipfile.ZipFile(archive) as compressed:
        for member in compressed.infolist():
            portable = PurePosixPath(member.filename.replace("\\", "/"))
            if portable.is_absolute() or ".." in portable.parts or any(":" in part for part in portable.parts):
                raise SetupError(f"Недопустимый путь внутри архива: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise SetupError(f"Символическая ссылка внутри архива: {member.filename}")
            output = _inside(target, target.joinpath(*portable.parts))
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with compressed.open(member) as source, output.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=4 * 1024 * 1024)


def _install_runtime(root: Path, emit: Callable[[str], None]) -> Path:
    runtime = _inside(root, root / ".local" / "runtime" / f"llama-{LLAMA_RELEASE}-{LLAMA_BACKEND}")
    marker = runtime / "installation.json"
    if marker.is_file():
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
            executable = _inside(runtime, runtime / recorded["executable"])
            if recorded["release"] == LLAMA_RELEASE and recorded["assets"] == list(_ASSETS) and executable.is_file():
                return executable
        except (OSError, ValueError, KeyError):
            pass
        raise SetupError(f"Неполная установка среды {runtime}. Переименуйте этот каталог и повторите установку.")
    if runtime.exists():
        raise SetupError(f"Каталог {runtime} уже существует без метаданных установки. Переименуйте его и повторите.")
    downloads = _inside(root, root / ".local" / "downloads")
    archives = []
    for asset in _ASSETS:
        archive = downloads / asset["name"]
        _download_asset(asset, archive, emit)
        archives.append(archive)
    staging = _inside(root, root / ".local" / "runtime" / f"staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        emit(f"Установка официального llama.cpp {LLAMA_RELEASE} / Vulkan…")
        for archive in archives:
            _extract_archive(archive, staging)
        executables = list(staging.rglob("llama-server.exe"))
        if len(executables) != 1:
            raise SetupError("В архиве не найден единственный llama-server.exe.")
        executable = executables[0]
        # Release dependency archives may put DLLs in a separate subdirectory.
        for dll in list(staging.rglob("*.dll")):
            sibling = executable.parent / dll.name
            if sibling == dll:
                continue
            if sibling.exists() and _sha256(sibling) != _sha256(dll):
                raise SetupError(f"Конфликт зависимостей среды: {dll.name}")
            if not sibling.exists():
                shutil.copy2(dll, sibling)
        dll_names = {dll.name.lower() for dll in executable.parent.glob("*.dll")}
        if "ggml-vulkan.dll" not in dll_names:
            raise SetupError("В официальном архиве отсутствует ggml-vulkan.dll.")
        recorded = {
            "release": LLAMA_RELEASE,
            "backend": LLAMA_BACKEND,
            "source": f"https://github.com/ggml-org/llama.cpp/releases/tag/{LLAMA_RELEASE}",
            "assets": list(_ASSETS),
            "executable": executable.relative_to(staging).as_posix(),
        }
        (staging / "installation.json").write_text(json.dumps(recorded, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.rename(runtime)
        return runtime / recorded["executable"]
    finally:
        if staging.exists():
            # Confirm the absolute recursive-delete target stays in this project.
            safe_staging = _inside(root, staging)
            shutil.rmtree(safe_staging)


def install_local(
    root: Path,
    repo_id: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILENAME,
    emit: Callable[[str], None] = print,
) -> dict:
    """Download only the selected GGUF, returning workspace-relative config paths."""
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SetupError("Эта автоматическая установка рассчитана на Windows x64 с видеодрайвером Vulkan.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise SetupError("Неверный идентификатор Hugging Face; нужен формат автор/репозиторий.")
    if Path(filename).name != filename or "/" in filename or "\\" in filename or not filename.lower().endswith(".gguf") or ":" in filename:
        raise SetupError("Выберите имя одного GGUF-файла без каталогов.")
    root = Path(root).resolve()
    if not root.is_dir():
        raise SetupError(f"Каталог проекта не существует: {root}")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SetupError("Установите зависимости: python -m pip install -r requirements.txt") from exc
    emit(f"Проверка размера {repo_id}/{filename}…")
    revision = DEFAULT_REVISION if (repo_id, filename) == (DEFAULT_REPO, DEFAULT_FILENAME) else "main"
    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    except Exception as exc:
        raise SetupError(f"Не удалось получить метаданные Hugging Face для {repo_id}: {exc}") from exc
    revision = info.sha
    selected = next((sibling for sibling in info.siblings if sibling.rfilename == filename), None)
    if selected is None or not selected.size:
        raise SetupError(f"Не найден файл с известным размером: {repo_id}/{filename}")
    lfs = selected.lfs
    expected_hash = getattr(lfs, "sha256", None) if lfs is not None else None
    if isinstance(lfs, dict):
        expected_hash = lfs.get("sha256")
    if (repo_id, filename) == (DEFAULT_REPO, DEFAULT_FILENAME):
        if expected_hash != DEFAULT_SHA256 or selected.size != 18132722048:
            raise SetupError("Метаданные выбранной закрепленной модели не совпали с ожидаемыми.")
    model_dir = _inside(root, root / "models" / repo_id.replace("/", "--"))
    model = _inside(root, model_dir / filename)
    already_downloaded = model.is_file() and model.stat().st_size == selected.size
    archive_remaining = sum(
        asset["size"]
        for asset in _ASSETS
        if not (root / ".local" / "downloads" / asset["name"]).is_file()
    )
    required = (0 if already_downloaded else selected.size) + archive_remaining + 4 * _GIB
    free = shutil.disk_usage(root).free
    emit(f"Модель: {selected.size / _GIB:.2f} ГиБ. Свободно: {free / _GIB:.1f} ГиБ.")
    if free < required:
        raise SetupError(f"Недостаточно места на диске: нужно около {required / _GIB:.1f} ГиБ, свободно {free / _GIB:.1f} ГиБ.")
    executable = _install_runtime(root, emit)
    model_dir.mkdir(parents=True, exist_ok=True)
    emit("Загрузка одного GGUF через Hugging Face; прерванную загрузку можно продолжить повторным запуском…")
    try:
        downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=model_dir))
    except Exception as exc:
        raise SetupError(f"Не удалось загрузить модель: {exc}. Повторный запуск продолжит загрузку.") from exc
    downloaded = _inside(root, downloaded)
    if downloaded.stat().st_size != selected.size:
        raise SetupError("Размер загруженной модели не совпал с метаданными.")
    emit("Проверка SHA256 модели…")
    actual_hash = _sha256(downloaded)
    if expected_hash and actual_hash != expected_hash:
        raise SetupError(f"SHA256 модели не совпал. Удалите поврежденный файл {downloaded} и повторите установку.")
    provenance = {
        "repo_id": repo_id,
        "filename": filename,
        "revision": revision,
        "size": selected.size,
        "sha256": actual_hash,
        "source": f"https://huggingface.co/{repo_id}/blob/{revision}/{filename}",
    }
    (model_dir / "download.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "executable": executable.relative_to(root).as_posix(),
        "model_path": downloaded.relative_to(root).as_posix(),
        "repo_id": repo_id,
        "filename": filename,
    }
    lock = {
        **result,
        "runtime_release": LLAMA_RELEASE,
        "runtime_backend": LLAMA_BACKEND,
        "runtime_assets": list(_ASSETS),
        "model": provenance,
    }
    lock_path = _inside(root, root / ".local" / "install.json")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("Модель и среда llama.cpp установлены.")
    return result


OLLAMA_MODEL = "huihui_ai/glm-4.7-flash-abliterated"
OLLAMA_FILENAME = "Huihui-GLM-4.7-Flash-abliterated-ollama.Q4_K_M.gguf"
OLLAMA_SIZE = 18765923488
OLLAMA_SHA256 = "db7192ff754ada80a81f7f8cd4704a273f79d43a3160b1b74f3b5988b09f238c"
OLLAMA_SOURCE = "https://ollama.com/" + OLLAMA_MODEL
OLLAMA_BLOB = "https://registry.ollama.ai/v2/" + OLLAMA_MODEL + "/blobs/sha256:" + OLLAMA_SHA256


def install_ollama_local(root: Path, emit: Callable[[str], None] = print) -> dict:
    """Install the author's pinned Q4_K_M GGUF without installing Ollama."""
    from .download import download_file
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SetupError("Эта автоматическая установка рассчитана на Windows x64 с видеодрайвером Vulkan.")
    root = Path(root).resolve()
    if not root.is_dir():
        raise SetupError(f"Каталог проекта не существует: {root}")
    model_dir = _inside(root, root / "models" / OLLAMA_MODEL.replace("/", "--"))
    model = _inside(root, model_dir / OLLAMA_FILENAME)
    complete = model.is_file() and model.stat().st_size == OLLAMA_SIZE
    # A preallocated resumable partial already reserves its final space.
    partial = model.with_suffix(model.suffix + ".partial")
    reserved = partial.stat().st_size if partial.is_file() else 0
    remaining = 0 if complete else max(0, OLLAMA_SIZE - reserved)
    # Keep the original verified blob and a losslessly converted llama.cpp GGUF.
    converted = model_dir / (model.stem + ".llamacpp.gguf")
    conversion_space = 0 if converted.is_file() else OLLAMA_SIZE + 1024 * 1024
    required = remaining + conversion_space + 2 * _GIB
    free = shutil.disk_usage(root).free
    emit(f"Модель автора: Huihui GLM-4.7-Flash abliterated Q4_K_M, {OLLAMA_SIZE / _GIB:.2f} ГиБ. Свободно: {free / _GIB:.1f} ГиБ.")
    if free < required:
        raise SetupError(f"Недостаточно места: нужно около {required / _GIB:.1f} ГиБ, свободно {free / _GIB:.1f} ГиБ.")
    try:
        executable = _install_runtime(root, emit)
        model_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_file(OLLAMA_BLOB, model, OLLAMA_SIZE, OLLAMA_SHA256, emit=emit, workers=32)
        from .gguf_compat import convert_glm4moelite
        compatibility = convert_glm4moelite(downloaded, converted, emit=emit)
        if compatibility.get("source_sha256") != OLLAMA_SHA256:
            raise SetupError("Исходный GGUF изменился после проверки контрольной суммы; установка остановлена.")
        from .gguf_compat import TEMPLATE_SHA256
        template_path = _inside(root, root / "llmopenchat" / "templates" / "GLM-4.7-Flash.jinja")
        if not template_path.is_file() or _sha256(template_path) != TEMPLATE_SHA256:
            raise SetupError("Шаблон GLM-4.7-Flash отсутствует или изменён. Восстановите файл из проекта.")
    except (OSError, ValueError, RuntimeError) as error:
        if isinstance(error, SetupError):
            raise
        raise SetupError(f"Ошибка установки: {error}. Повторный запуск продолжит загрузку.") from error
    provenance = {
        "model": OLLAMA_MODEL,
        "source": OLLAMA_SOURCE,
        "blob": OLLAMA_BLOB,
        "sha256": OLLAMA_SHA256,
        "size": OLLAMA_SIZE,
        "quantization": "Q4_K_M",
        "llama_cpp_compatibility": compatibility,
    }
    (model_dir / "download.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "executable": executable.relative_to(root).as_posix(),
        "model_path": converted.relative_to(root).as_posix(),
        "repo_id": OLLAMA_MODEL,
        "filename": OLLAMA_FILENAME,
        "chat_template_file": "llmopenchat/templates/GLM-4.7-Flash.jinja",
    }
    lock = {**result, "runtime_release": LLAMA_RELEASE, "runtime_backend": "vulkan", "runtime_assets": list(_ASSETS), "model": provenance}
    lock_path = _inside(root, root / ".local" / "install.json")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("Модель и сервер установлены и проверены.")
    return result
