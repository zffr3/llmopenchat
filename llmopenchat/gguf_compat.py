"""Metadata-only Ollama GLM-4.7-Flash conversion for native llama.cpp.

The conversion follows Ollama's pinned compatibility handler. Tensor descriptors
and the complete tensor payload remain byte-for-byte identical.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time
from typing import BinaryIO, Callable
import uuid


CONVERTER_VERSION = 1
COMPAT_COMMIT = "0d0720e51fb2fd9aa58781c3d720c06d720c2e7b"
COMPAT_SOURCE = f"https://github.com/ollama/ollama/blob/{COMPAT_COMMIT}/llama/compat/llama-ollama-compat.cpp"
TEMPLATE_COMMIT = "58cb9138e46180d45854cf45d4c1e25ad5f7e1c4"
TEMPLATE_SHA256 = "d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375"
TEMPLATE_SOURCE = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{TEMPLATE_COMMIT}/models/templates/GLM-4.7-Flash.jinja"
_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_HEADER_LIMIT = 128 * 1024**2
_BLOCK = 8 * 1024**2


class CompatibilityError(RuntimeError):
    """An invalid source or a conflicting existing conversion."""


@dataclass(frozen=True)
class _Value:
    kind: int
    raw: bytes

    def string(self) -> str:
        if self.kind != 8:
            raise CompatibilityError("Ожидалась строка в метаданных GGUF.")
        length = struct.unpack_from("<Q", self.raw)[0]
        return self.raw[8:8 + length].decode("utf-8")

    def u32(self) -> int:
        if self.kind != 4:
            raise CompatibilityError("Ожидался UINT32 в метаданных GGUF.")
        return struct.unpack("<I", self.raw)[0]


class _Reader:
    def __init__(self, stream: BinaryIO, size: int):
        self.stream = stream
        self.size = size

    def take(self, count: int) -> bytes:
        if count < 0 or self.stream.tell() + count > min(self.size, _HEADER_LIMIT):
            raise CompatibilityError("Поврежденный или слишком большой заголовок GGUF.")
        value = self.stream.read(count)
        if len(value) != count:
            raise CompatibilityError("Неполный заголовок GGUF.")
        return value

    def number(self, fmt: str) -> int:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def skip(self, count: int) -> None:
        if count < 0 or self.stream.tell() + count > min(self.size, _HEADER_LIMIT):
            raise CompatibilityError("Поврежденный или слишком большой заголовок GGUF.")
        self.stream.seek(count, os.SEEK_CUR)

    def string(self) -> str:
        length = self.number("<Q")
        if length > 16384:
            raise CompatibilityError("Слишком длинное имя поля или тензора GGUF.")
        try:
            return self.take(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CompatibilityError("Неверная строка UTF-8 в заголовке GGUF.") from exc

    def skip_value(self, kind: int, depth: int = 0) -> None:
        if depth > 8:
            raise CompatibilityError("Слишком глубокий массив GGUF.")
        if kind in _FORMATS:
            self.skip(struct.calcsize(_FORMATS[kind]))
        elif kind == 8:
            self.skip(self.number("<Q"))
        elif kind == 9:
            element = self.number("<I")
            count = self.number("<Q")
            if count > 20_000_000:
                raise CompatibilityError("Слишком большой массив GGUF.")
            if element in _FORMATS:
                self.skip(count * struct.calcsize(_FORMATS[element]))
            else:
                for _ in range(count):
                    self.skip_value(element, depth + 1)
        else:
            raise CompatibilityError(f"Неизвестный тип GGUF: {kind}.")


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _str_value(value: str) -> _Value:
    return _Value(8, _string(value))


def _u32_value(value: int) -> _Value:
    if not 0 <= value <= 0xFFFFFFFF:
        raise CompatibilityError("Значение метаданных не помещается в UINT32.")
    return _Value(4, struct.pack("<I", value))


def _read_header(stream: BinaryIO, size: int) -> dict:
    reader = _Reader(stream, size)
    if reader.take(4) != b"GGUF" or reader.number("<I") != 3:
        raise CompatibilityError("Для преобразования нужен файл GGUF версии 3.")
    tensor_count = reader.number("<Q")
    key_count = reader.number("<Q")
    if not 0 < tensor_count <= 1_000_000 or not 0 < key_count <= 100_000:
        raise CompatibilityError("Недопустимое число тензоров или полей GGUF.")
    metadata = {}
    for _ in range(key_count):
        key = reader.string()
        if not key or key in metadata:
            raise CompatibilityError("Пустое или повторяющееся имя поля GGUF.")
        kind = reader.number("<I")
        start = stream.tell()
        reader.skip_value(kind)
        end = stream.tell()
        stream.seek(start)
        metadata[key] = _Value(kind, reader.take(end - start))
    tensor_start = stream.tell()
    names = set()
    for _ in range(tensor_count):
        name = reader.string()
        if not name or name in names:
            raise CompatibilityError("Пустое или повторяющееся имя тензора GGUF.")
        names.add(name)
        rank = reader.number("<I")
        if not 1 <= rank <= 4:
            raise CompatibilityError("Недопустимое число измерений тензора GGUF.")
        reader.skip(rank * 8)
        reader.number("<I")  # Quantization type; descriptor is preserved verbatim.
        reader.number("<Q")  # Offset relative to the aligned data region.
    tensor_end = stream.tell()
    stream.seek(tensor_start)
    tensor_info = reader.take(tensor_end - tensor_start)
    alignment = metadata["general.alignment"].u32() if "general.alignment" in metadata else 32
    if not 1 <= alignment <= 1024**2 or alignment & (alignment - 1):
        raise CompatibilityError("Недопустимое выравнивание GGUF.")
    data_offset = (tensor_end + alignment - 1) // alignment * alignment
    if data_offset >= size:
        raise CompatibilityError("В GGUF отсутствует область данных тензоров.")
    return {"metadata": metadata, "tensor_info": tensor_info, "tensor_count": tensor_count,
            "alignment": alignment, "data_offset": data_offset}


def _translate_metadata(source: dict[str, _Value]) -> dict[str, _Value]:
    architecture = source.get("general.architecture")
    if architecture is None or architecture.string() != "glm4moelite":
        raise CompatibilityError("Это преобразование предназначено только для архитектуры glm4moelite.")
    output = dict(source)
    output["general.architecture"] = _str_value("deepseek2")
    # Ollama's rename_kv_prefix copies keys and preserves existing destinations.
    for key, value in source.items():
        if key.startswith("glm4moelite."):
            output.setdefault("deepseek2." + key[len("glm4moelite."):], value)
    output["deepseek2.attention.head_count_kv"] = _u32_value(1)
    kv_rank = output.get("deepseek2.attention.kv_lora_rank")
    rope = output.get("deepseek2.rope.dimension_count")
    if kv_rank is not None and rope is not None:
        rank = kv_rank.u32()
        output["deepseek2.attention.key_length"] = _u32_value(rank + rope.u32())
        output["deepseek2.attention.value_length"] = _u32_value(rank)
    head = source.get("glm4moelite.attention.key_length")
    if head is not None:
        output["deepseek2.attention.key_length_mla"] = _u32_value(head.u32())
        output["deepseek2.attention.value_length_mla"] = _u32_value(head.u32())
    output.setdefault("deepseek2.expert_group_count", _u32_value(1))
    output.setdefault("deepseek2.expert_group_used_count", _u32_value(1))
    output["tokenizer.ggml.pre"] = _str_value("glm4")
    eos = source.get("tokenizer.ggml.eos_token_ids")
    if eos is not None and eos.kind == 9:
        element, count = struct.unpack_from("<IQ", eos.raw)
        if element in (4, 5):
            fmt = "<I" if element == 4 else "<i"
            ids = [item[0] for item in struct.iter_unpack(fmt, eos.raw[12:])]
            if len(ids) != count:
                raise CompatibilityError("Неверный массив стоп-токенов GGUF.")
            ids = [value for value in ids if value >= 0]
            if len(ids) >= 2:
                output.setdefault("tokenizer.ggml.eot_token_id", _u32_value(ids[1]))
            if len(ids) >= 3:
                output.setdefault("tokenizer.ggml.eom_token_id", _u32_value(ids[2]))
    return output


def _sha256(path: Path, emit: Callable[[str], None]) -> str:
    result = hashlib.sha256()
    done = 0
    size = path.stat().st_size
    updated = time.monotonic()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK):
            result.update(block)
            done += len(block)
            if time.monotonic() - updated >= 5:
                emit(f"Проверка SHA256: {done / size:.0%}")
                updated = time.monotonic()
    return result.hexdigest()


def convert_glm4moelite(source: Path, destination: Path, emit: Callable[[str], None] = print) -> dict:
    """Rewrite metadata, copy unchanged tensor bytes, and record exact provenance."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or not source.is_file():
        raise CompatibilityError("Нужны существующий исходный GGUF и отдельный путь для результата.")
    original_size = source.stat().st_size
    with source.open("rb") as stream:
        header = _read_header(stream, original_size)
        stream.seek(0)
        source_prefix = stream.read(header["data_offset"])
    # Derive metadata from the exact prefix snapshot later checked during copy.
    # This covers source changes between parsing the header and hashing the file.
    header = _read_header(io.BytesIO(source_prefix), original_size)
    if header["data_offset"] != len(source_prefix):
        raise CompatibilityError("Заголовок исходного GGUF изменился во время чтения.")
    source_prefix_sha = hashlib.sha256(source_prefix).hexdigest()
    translated = _translate_metadata(header["metadata"])
    emit("Проверка SHA256 исходного GGUF перед преобразованием метаданных…")
    source_sha = _sha256(source, emit)
    identity = {"converter_version": CONVERTER_VERSION, "compat_commit": COMPAT_COMMIT,
                "source_sha256": source_sha, "source_size": original_size,
                "source_architecture": "glm4moelite", "architecture": "deepseek2",
                "tensor_count": header["tensor_count"], "alignment": header["alignment"],
                "source_data_offset": header["data_offset"], "source_header_sha256": source_prefix_sha}
    sidecar = destination.with_name(destination.name + ".json")
    if destination.exists() or sidecar.exists():
        try:
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))
            if any(recorded.get(key) != value for key, value in identity.items()):
                raise ValueError("conversion identity")
            if destination.stat().st_size != recorded["size"] or _sha256(destination, emit) != recorded["sha256"]:
                raise ValueError("converted checksum")
            emit("Уже преобразованный GGUF проверен.")
            return recorded
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise CompatibilityError("Результат преобразования или его метаданные уже существуют и не совпали; переименуйте их.") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".converting")
    if staging.exists():
        raise CompatibilityError(f"Незавершенный файл {staging.name} уже существует; переименуйте его и повторите.")
    new_header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, header["tensor_count"], len(translated)))
    for key, value in translated.items():
        new_header.extend(_string(key))
        new_header.extend(struct.pack("<I", value.kind))
        new_header.extend(value.raw)
    new_header.extend(header["tensor_info"])
    alignment = header["alignment"]
    new_header.extend(b"\0" * (-len(new_header) % alignment))
    converted_sha = hashlib.sha256()
    copied_source_sha = hashlib.sha256()
    payload_sha = hashlib.sha256()
    temporary_sidecar = sidecar.with_name(f"{sidecar.name}.{uuid.uuid4().hex}.tmp")
    created = False
    try:
        emit("Преобразование метаданных GLM; копирование неизмененных тензоров…")
        with source.open("rb") as incoming, staging.open("xb") as outgoing:
            created = True
            copied_prefix = incoming.read(header["data_offset"])
            if hashlib.sha256(copied_prefix).hexdigest() != source_prefix_sha:
                raise CompatibilityError("Заголовок исходного GGUF изменился во время преобразования.")
            copied_source_sha.update(copied_prefix)
            outgoing.write(new_header)
            converted_sha.update(new_header)
            copied = 0
            updated = time.monotonic()
            while block := incoming.read(_BLOCK):
                outgoing.write(block)
                converted_sha.update(block)
                copied_source_sha.update(block)
                payload_sha.update(block)
                copied += len(block)
                if time.monotonic() - updated >= 5:
                    emit(f"Копирование тензоров: {copied / (original_size - header['data_offset']):.0%}")
                    updated = time.monotonic()
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if copied_source_sha.hexdigest() != source_sha:
            raise CompatibilityError("Исходный GGUF изменился во время преобразования.")
        result = {**identity, "source_filename": source.name, "filename": destination.name,
                  "compat_source": COMPAT_SOURCE, "size": staging.stat().st_size,
                  "sha256": converted_sha.hexdigest(), "tensor_payload_sha256": payload_sha.hexdigest(),
                  "data_offset": len(new_header), "tensor_descriptors_unchanged": True,
                  "tensor_payload_unchanged": True, "chat_template_source": TEMPLATE_SOURCE,
                  "chat_template_commit": TEMPLATE_COMMIT, "chat_template_sha256": TEMPLATE_SHA256}
        with temporary_sidecar.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() or sidecar.exists():
            raise CompatibilityError("Во время преобразования появились файлы результата; они сохранены без изменений.")
        staging.replace(destination)
        temporary_sidecar.replace(sidecar)
        emit("Совместимый GGUF создан; веса и квантование сохранены.")
        return result
    finally:
        # Only unlink a file created by this call, inside the exact output folder.
        if created and staging.exists() and staging.resolve().parent == destination.parent:
            staging.unlink()
        if temporary_sidecar.exists() and temporary_sidecar.resolve().parent == destination.parent:
            temporary_sidecar.unlink()
