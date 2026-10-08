"""Small stdlib-only client for the official Telegram Bot API."""

from __future__ import annotations

import json
import math
import re
import secrets
import time
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_MESSAGE_UNITS = 4096
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_RATE_LIMIT_RETRIES = 2
MAX_RATE_LIMIT_WAIT = 5
_TOKEN = re.compile(r"[0-9]+:[A-Za-z0-9_-]+\Z")
_METHOD = re.compile(r"[A-Za-z][A-Za-z0-9]*\Z")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)


class TelegramError(RuntimeError):
    """An API/transport error whose message is safe to show to the user."""

    def __init__(
        self,
        message: str,
        *,
        error_code: int | None = None,
        retry_after: int | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.retry_after = retry_after


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The secret is part of the API URL. Never follow a server redirect.
        return None


def _split_text(text: str, limit: int) -> list[str]:
    """Preserve text while bounding UTF-16 length and keeping Unicode scalars whole."""
    if not isinstance(text, str):
        raise TelegramError("Текст сообщения Telegram должен быть строкой.")
    chunks: list[str] = []
    start = units = 0
    for index, char in enumerate(text):
        codepoint = ord(char)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise TelegramError("Текст сообщения Telegram содержит некорректный Unicode.")
        char_units = 2 if codepoint > 0xFFFF else 1
        if units + char_units > limit:
            chunks.append(text[start:index])
            start, units = index, 0
        units += char_units
    chunks.append(text[start:])
    return chunks


class TelegramAPI:
    def __init__(self, token: str):
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise TelegramError("Неверный формат токена Telegram-бота в credits.txt.")
        self._token = token
        self._opener = build_opener(_NoRedirect())

    def _description(self, value: object) -> str:
        if not isinstance(value, str):
            return ""
        # Do not expose credentials even if an upstream error echoes its URL.
        value = value.replace(self._token, "[токен скрыт]")
        value = value.replace(quote(self._token, safe=""), "[токен скрыт]")
        value = _URL.sub("[URL скрыт]", value)
        return " ".join(value.split())[:500]

    def _request(self, request: Request, timeout: float) -> tuple[dict, int | None]:
        status = None
        try:
            try:
                response = self._opener.open(request, timeout=timeout)
            except HTTPError as exc:
                status, response = exc.code, exc
            with response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except (URLError, OSError, HTTPException, ValueError):
            # urllib exception messages (and chained exceptions) can contain the token.
            raise TelegramError("Не удалось подключиться к Telegram Bot API.") from None
        if len(body) > MAX_RESPONSE_BYTES:
            raise TelegramError("Ответ Telegram Bot API превышает допустимый размер.")
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeError, ValueError, TypeError, RecursionError):
            if status is not None:
                raise TelegramError(
                    f"Ошибка Telegram Bot API: HTTP {status}.", error_code=status
                ) from None
            raise TelegramError("Telegram Bot API вернул некорректный JSON.") from None
        if not isinstance(data, dict) or type(data.get("ok")) is not bool:
            raise TelegramError("Telegram Bot API вернул некорректный ответ.")
        return data, status

    def call(self, method: str, payload: dict | None = None, timeout: float = 30) -> Any:
        if not isinstance(method, str) or not _METHOD.fullmatch(method):
            raise TelegramError("Неверное имя метода Telegram Bot API.")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise TelegramError("Таймаут Telegram Bot API должен быть положительным числом.")
        if payload is not None and not isinstance(payload, dict):
            raise TelegramError("Параметры Telegram Bot API должны быть объектом.")
        try:
            body = json.dumps(payload or {}, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise TelegramError("Невозможно преобразовать параметры Telegram Bot API в JSON.") from None
        request = Request(
            f"https://api.telegram.org/bot{self._token}/{method}",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        return self._perform(request, timeout)

    def _perform(self, request: Request, timeout: float) -> Any:
        waited = 0
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            data, status = self._request(request, timeout)
            if data["ok"] and status is None:
                if "result" not in data:
                    raise TelegramError("Telegram Bot API вернул ответ без результата.")
                return data["result"]
            code = data.get("error_code", status)
            if type(code) is not int:
                code = status
            parameters = data.get("parameters")
            retry_after = parameters.get("retry_after") if isinstance(parameters, dict) else None
            if type(retry_after) is not int or retry_after < 0:
                retry_after = None
            if (
                code == 429
                and retry_after is not None
                and attempt < MAX_RATE_LIMIT_RETRIES
                and waited + retry_after <= MAX_RATE_LIMIT_WAIT
            ):
                time.sleep(retry_after)
                waited += retry_after
                continue
            detail = self._description(data.get("description"))
            label = f"Ошибка Telegram Bot API ({code})" if code is not None else "Ошибка Telegram Bot API"
            raise TelegramError(
                f"{label}: {detail}" if detail else f"{label}.",
                error_code=code,
                retry_after=retry_after,
            )
        raise TelegramError("Превышен лимит повторов Telegram Bot API.")

    def get_updates(self, offset: int | None = None, timeout: int = 25) -> list[dict]:
        if type(timeout) is not int or timeout < 0:
            raise TelegramError("Таймаут опроса Telegram должен быть целым неотрицательным числом.")
        payload: dict = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            if type(offset) is not int:
                raise TelegramError("Смещение обновлений Telegram должно быть целым числом.")
            payload["offset"] = offset
        result = self.call("getUpdates", payload, timeout=max(10, timeout + 10))
        if not isinstance(result, list) or not all(isinstance(update, dict) for update in result):
            raise TelegramError("Telegram Bot API вернул некорректный список обновлений.")
        return result

    def send_message(self, chat_id: int | str, text: str, reply_markup: dict | None = None) -> dict:
        chunks = _split_text(text, MAX_MESSAGE_UNITS)
        if not text:
            raise TelegramError("Telegram не принимает пустое сообщение.")
        result: dict = {}
        for index, chunk in enumerate(chunks):
            payload: dict = {"chat_id": chat_id, "text": chunk}
            if reply_markup is not None and index == len(chunks) - 1:
                payload["reply_markup"] = reply_markup
            response = self.call("sendMessage", payload)
            if not isinstance(response, dict):
                raise TelegramError("Telegram Bot API вернул некорректное сообщение.")
            result = response
        return result

    def answer_callback_query(
        self, callback_query_id: str, text: str = "", show_alert: bool = False
    ) -> Any:
        return self.call(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query_id,
                "text": _split_text(text, 200)[0],
                "show_alert": show_alert,
            },
        )

    def download_document(self, file_id: str, *, max_bytes: int = 20 * 1024 * 1024) -> bytes:
        """Download only a bounded file from the official getFile endpoint."""
        if (not isinstance(file_id, str) or not file_id or len(file_id) > 1024
                or type(max_bytes) is not int or not 1 <= max_bytes <= 20 * 1024 * 1024):
            raise TelegramError("Неверный ID документа или лимит загрузки Telegram.")
        result = self.call("getFile", {"file_id": file_id})
        if not isinstance(result, dict):
            raise TelegramError("Telegram вернул неверные сведения о документе.")
        path = result.get("file_path")
        size = result.get("file_size")
        if (not isinstance(path, str) or not path or len(path) > 1024
                or not re.fullmatch(r"[A-Za-z0-9_./-]+", path)
                or any(part in {"", ".", ".."} for part in path.split("/"))):
            raise TelegramError("Telegram вернул неверный путь документа.")
        if size is not None and (type(size) is not int or size < 0 or size > max_bytes):
            raise TelegramError("Документ Telegram превышает допустимый размер загрузки.")
        request = Request(f"https://api.telegram.org/file/bot{self._token}/{path}", method="GET")
        try:
            with self._opener.open(request, timeout=60) as response:
                content = response.read(max_bytes + 1)
        except (URLError, OSError, HTTPException, ValueError):
            raise TelegramError("Не удалось скачать документ из Telegram.") from None
        if len(content) > max_bytes:
            raise TelegramError("Документ Telegram превышает допустимый размер загрузки.")
        if size is not None and len(content) != size:
            raise TelegramError("Документ Telegram загружен не полностью.")
        return content

    def send_document(
        self, chat_id: int | str, filename: str, content: bytes, caption: str = ""
    ) -> dict:
        """Upload supplied bytes; the filename is a label, never a local file path."""
        if not isinstance(content, bytes) or len(content) > MAX_DOCUMENT_BYTES:
            raise TelegramError("Документ Telegram должен содержать bytes размером до 50 МБ.")
        if not isinstance(filename, str) or not filename:
            raise TelegramError("Документ Telegram должен иметь имя файла.")
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        filename = "".join("_" if char == '"' or ord(char) < 32 else char for char in filename)
        if not filename:
            raise TelegramError("Документ Telegram должен иметь имя файла.")
        caption = _split_text(caption, 1024)[0]
        try:
            fields = {"chat_id": str(chat_id).encode("utf-8"), "caption": caption.encode("utf-8")}
            file_header = (
                f'Content-Disposition: form-data; name="document"; filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("utf-8")
        except UnicodeError:
            raise TelegramError("Документ Telegram содержит некорректный Unicode.") from None
        boundary = "llmopenchat-" + secrets.token_hex(16)
        delimiter = boundary.encode("ascii")
        while delimiter in content or delimiter in file_header or any(delimiter in value for value in fields.values()):
            boundary = "llmopenchat-" + secrets.token_hex(16)
            delimiter = boundary.encode("ascii")
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.extend(
                [
                    b"--" + delimiter + b"\r\n",
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
                    value,
                    b"\r\n",
                ]
            )
        parts.extend([b"--" + delimiter + b"\r\n", file_header, content, b"\r\n--" + delimiter + b"--\r\n"])
        request = Request(
            f"https://api.telegram.org/bot{self._token}/sendDocument",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        result = self._perform(request, timeout=30)
        if not isinstance(result, dict):
            raise TelegramError("Telegram Bot API вернул некорректные данные документа.")
        return result

    def edit_message_reply_markup(
        self, chat_id: int | str, message_id: int, reply_markup: dict | None = None
    ) -> Any:
        return self.call(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": reply_markup if reply_markup is not None else {"inline_keyboard": []},
            },
        )

    def set_commands(self, commands: list[dict]) -> Any:
        return self.call("setMyCommands", {"commands": commands})

    def get_me(self) -> dict:
        result = self.call("getMe")
        if not isinstance(result, dict):
            raise TelegramError("Telegram Bot API вернул некорректные данные бота.")
        return result
