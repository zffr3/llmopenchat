"""Bounded local image preparation for explicit multimodal user messages.

Pillow is imported only when actual image bytes need processing. Text-only
chat, history previews and disabled vision therefore work without its codecs.
No remote image URLs, local paths or unchecked base64 reach the model API.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
import warnings
from dataclasses import dataclass, fields
from io import BytesIO


MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_SIDE = 2048
MAX_IMAGE_BYTES = 1024 * 1024
MAX_HISTORY_IMAGES = 4
MAX_CONTENT_PARTS = 32
IMAGE_PLACEHOLDER = "[Изображение]"
REMOVED_IMAGE_PLACEHOLDER = "[Изображение удалено из контекста по лимиту; при необходимости отправьте снова.]"
_FORMATS = {"JPEG": "jpeg", "PNG": "png", "WEBP": "webp"}
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
_DATA_URL = re.compile(r"data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/]*={0,2})\Z")


class VisionError(ValueError):
    pass


@dataclass(frozen=True)
class VisionSettings:
    enabled: bool = False
    max_input_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 24_000_000
    max_side: int = 1536
    max_image_bytes: int = 512 * 1024
    max_history_images: int = 4

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise VisionError("vision.enabled должен быть true или false.")
        bounds = {
            "max_input_bytes": (1, MAX_INPUT_BYTES), "max_pixels": (1, MAX_PIXELS),
            "max_side": (1, MAX_SIDE), "max_image_bytes": (1, MAX_IMAGE_BYTES),
            "max_history_images": (1, MAX_HISTORY_IMAGES),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise VisionError(f"vision.{name} должен быть целым числом от {minimum} до {maximum}.")

    @classmethod
    def from_config(cls, config: dict) -> VisionSettings:
        if not isinstance(config, dict):
            raise VisionError("Конфигурация должна быть JSON-объектом.")
        values = config.get("vision", {})
        if not isinstance(values, dict):
            raise VisionError("vision должен быть JSON-объектом.")
        if set(values) - {field.name for field in fields(cls)}:
            raise VisionError("Неизвестный параметр vision.")
        return cls(**values)


def image_candidate(media_type: str, filename: str, mime_type: str) -> bool:
    if media_type == "photo":
        return True
    if media_type != "document":
        return False
    name = filename.lower() if isinstance(filename, str) else ""
    mime = mime_type.lower().split(";", 1)[0].strip() if isinstance(mime_type, str) else ""
    return name.endswith(_IMAGE_SUFFIXES) or mime in {"image/jpeg", "image/png", "image/webp"}


def incoming_image(message: dict) -> dict | None:
    """Select the largest native photo or supported image document metadata."""
    if not isinstance(message, dict):
        return None
    if isinstance(message.get("animation"), dict) or isinstance(message.get("sticker"), dict):
        return None
    photos = message.get("photo")
    if isinstance(photos, list):
        candidates = [photo for photo in photos if isinstance(photo, dict)
                      and isinstance(photo.get("file_id"), str) and photo["file_id"]]
        if candidates:
            def rank(photo):
                width, height, size = (photo.get(key, 0) for key in ("width", "height", "file_size"))
                area = width * height if type(width) is int and type(height) is int and width > 0 and height > 0 else 0
                return area, size if type(size) is int and size >= 0 else 0
            return {**max(candidates, key=rank), "media_type": "photo"}
    document = message.get("document")
    if (isinstance(document, dict) and isinstance(document.get("file_id"), str) and document["file_id"]
            and image_candidate("document", document.get("file_name", ""), document.get("mime_type", ""))):
        return {**document, "media_type": "document"}
    return None


def _pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise VisionError('Для обработки изображений установите Pillow: python -m pip install "Pillow>=13.0.0,<14".') from error
    return Image, ImageOps


def _decode_image(content: bytes, *, max_pixels: int, max_side: int | None = None,
                  expected_format: str | None = None):
    Image, _ = _pillow()

    def check(image):
        if image.format not in _FORMATS or (expected_format is not None and _FORMATS[image.format] != expected_format):
            raise VisionError("Нужен файл JPEG, PNG или WebP с соответствующим типом содержимого.")
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise VisionError(f"Изображение превышает лимит {max_pixels} пикселей.")
        if max_side is not None and max(width, height) > max_side:
            raise VisionError(f"Сторона изображения превышает лимит {max_side} пикселей.")
        if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
            raise VisionError("Анимированные изображения не поддерживаются; отправьте один кадр JPEG, PNG или WebP.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content), formats=list(_FORMATS)) as probe:
                check(probe)
                probe.verify()
            # verify() checks the container; load() separately forces the actual
            # decoder to reject truncated or malformed pixel data before use.
            with Image.open(BytesIO(content), formats=list(_FORMATS)) as image:
                check(image)
                image.load()
                return image.copy()
    except VisionError:
        raise
    except (OSError, ValueError, SyntaxError, EOFError, OverflowError, MemoryError,
            Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise VisionError("Не удалось безопасно прочитать изображение JPEG, PNG или WebP: файл повреждён или превышает лимиты.") from error


def prepare_image(content: bytes, settings: VisionSettings) -> dict:
    if not isinstance(settings, VisionSettings):
        raise VisionError("Нужны проверенные настройки vision.")
    if not settings.enabled:
        raise VisionError("Обработка изображений отключена: включите vision.enabled и выберите модель с поддержкой изображений.")
    if type(content) is not bytes or not content or len(content) > settings.max_input_bytes:
        raise VisionError(f"Изображение должно быть непустым файлом размером до {settings.max_input_bytes} байт.")
    Image, ImageOps = _pillow()
    source = _decode_image(content, max_pixels=settings.max_pixels)
    try:
        oriented = ImageOps.exif_transpose(source)
        try:
            oriented.thumbnail((settings.max_side, settings.max_side), Image.Resampling.LANCZOS)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                try:
                    rgb = Image.new("RGB", rgba.size, "white")
                    rgb.paste(rgba, mask=rgba.getchannel("A"))
                finally:
                    rgba.close()
            else:
                rgb = oriented.convert("RGB")
        finally:
            oriented.close()
        try:
            # Metadata and profiles are neither forwarded nor inferred by save.
            rgb.info.clear()
            encoded = None
            for _ in range(8):
                for quality in (85, 70, 55, 40):
                    with BytesIO() as output:
                        rgb.save(output, format="JPEG", quality=quality, optimize=True)
                        encoded = output.getvalue()
                    if len(encoded) <= settings.max_image_bytes:
                        return {"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
                        }}
                if rgb.size == (1, 1):
                    break
                factor = min(0.8, max(0.01, math.sqrt(settings.max_image_bytes / len(encoded)) * 0.9))
                size = tuple(max(1, int(side * factor)) for side in rgb.size)
                smaller = rgb.resize(size, Image.Resampling.LANCZOS)
                rgb.close()
                rgb = smaller
            raise VisionError("Изображение невозможно уменьшить до заданного vision.max_image_bytes.")
        finally:
            rgb.close()
    except VisionError:
        raise
    except (OSError, ValueError, SyntaxError, OverflowError, MemoryError) as error:
        raise VisionError("Не удалось безопасно подготовить изображение для модели.") from error
    finally:
        source.close()


def image_message(text: str, content: bytes, config: dict) -> list[dict]:
    if not isinstance(text, str):
        raise VisionError("Текст сообщения с изображением должен быть строкой.")
    return [{"type": "text", "text": text}, prepare_image(content, VisionSettings.from_config(config))]


def has_images(content: object) -> bool:
    return isinstance(content, list) and any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)


def content_text(content: object) -> str:
    """Safe display/RAG text; image data URLs are never returned."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif part.get("type") == "image_url":
            parts.append(IMAGE_PLACEHOLDER)
    return "\n".join(parts)


def validate_content(content: object) -> list[dict]:
    """Validate saved/outbound user parts and return a detached, exact copy."""
    if not isinstance(content, list) or not 1 <= len(content) <= MAX_CONTENT_PARTS:
        raise VisionError(f"Сообщение должно содержать от 1 до {MAX_CONTENT_PARTS} частей.")
    clean = []
    images = 0
    for part in content:
        if not isinstance(part, dict):
            raise VisionError("Неверная часть сообщения с изображением.")
        if part.get("type") == "text" and set(part) == {"type", "text"} and isinstance(part["text"], str):
            clean.append({"type": "text", "text": part["text"]})
            continue
        if part.get("type") != "image_url" or set(part) != {"type", "image_url"}:
            raise VisionError("Разрешены только части text и image_url.")
        image_url = part["image_url"]
        if not isinstance(image_url, dict) or set(image_url) != {"url"} or not isinstance(image_url.get("url"), str):
            raise VisionError("Неверная часть image_url.")
        url = image_url["url"]
        if len(url) > 32 + 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise VisionError("Изображение в сообщении превышает лимит 1 МиБ.")
        match = _DATA_URL.fullmatch(url)
        if match is None:
            raise VisionError("Изображение должно быть встроенным data:image JPEG, PNG или WebP; внешние URL и пути запрещены.")
        images += 1
        if images > MAX_HISTORY_IMAGES:
            raise VisionError(f"В сообщении допускается не более {MAX_HISTORY_IMAGES} изображений.")
        try:
            decoded = base64.b64decode(match[2], validate=True)
        except (ValueError, binascii.Error) as error:
            raise VisionError("Некорректное base64-изображение.") from error
        if not decoded or len(decoded) > MAX_IMAGE_BYTES or base64.b64encode(decoded).decode("ascii") != match[2]:
            raise VisionError("Некорректное или слишком большое base64-изображение.")
        image = _decode_image(decoded, max_pixels=MAX_PIXELS, max_side=MAX_SIDE, expected_format=match[1])
        image.close()
        clean.append({"type": "image_url", "image_url": {"url": url}})
    return clean


def prune_images(messages: list[dict], config: dict) -> None:
    """Keep the last configured image count across the entire chat in place."""
    settings = VisionSettings.from_config(config)
    remaining = settings.max_history_images
    for message in reversed(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for index in range(len(content) - 1, -1, -1):
            part = content[index]
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            if remaining and message.get("role") == "user":
                remaining -= 1
            else:
                content[index] = {"type": "text", "text": REMOVED_IMAGE_PLACEHOLDER}
