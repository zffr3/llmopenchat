"""A bounded, authenticated HTTP entry point for one trusted scenario."""

from __future__ import annotations

import hmac
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .scenarios import parse_json_document


REQUEST_IO_TIMEOUT = 10.0
JSON_ENVELOPE_BYTES = 1024
_TOKEN = re.compile(r"[\x21-\x7e]{1,4096}\Z")
_LENGTH = re.compile(r"[0-9]{1,10}\Z")


class _WebhookServer(HTTPServer):
    # HTTPServer deliberately executes one request at a time. In particular,
    # simultaneous callers cannot share an active model stream or tool harness.
    runner: Any
    authorization: bytes
    input_max_bytes: int
    body_max_bytes: int

    def handle_error(self, request, client_address):
        # BaseServer prints tracebacks that can include credentials or model text.
        # Request failures are represented by the handler's public JSON errors.
        pass


class _WebhookHandler(BaseHTTPRequestHandler):
    server: _WebhookServer
    protocol_version = "HTTP/1.0"

    def setup(self):
        self.request.settimeout(REQUEST_IO_TIMEOUT)
        super().setup()

    def log_message(self, format, *args):
        # Do not write request paths, headers, input, or model output to stdout.
        pass

    def _respond(self, status: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except OSError:
            # A disconnected caller must not cause a traceback containing input.
            pass

    def _error(self, status: int, message: str) -> None:
        self._respond(status, {"error": message})

    def send_error(self, code, message=None, explain=None):
        # Ignore BaseHTTPRequestHandler's message, which can echo request data.
        self._error(code, "Некорректный HTTP-запрос.")

    def do_POST(self):
        if self.path != "/run":
            self._error(404, "Маршрут не найден.")
            return
        authorization = self.headers.get_all("Authorization", [])
        if len(authorization) != 1 or not hmac.compare_digest(
            authorization[0].encode("utf-8"), self.server.authorization
        ):
            self._error(401, "Требуется авторизация Bearer.")
            return
        if self.headers.get_all("Transfer-Encoding", []):
            self._error(400, "Transfer-Encoding не поддерживается.")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            self._error(411, "Требуется Content-Length.")
            return
        if len(lengths) != 1 or not _LENGTH.fullmatch(lengths[0]):
            self._error(400, "Некорректный Content-Length.")
            return
        length = int(lengths[0])
        if length > self.server.body_max_bytes:
            self._error(413, "Тело запроса превышает допустимый размер.")
            return
        content_types = self.headers.get_all("Content-Type", [])
        charset = self.headers.get_content_charset()
        if (
            len(content_types) != 1
            or self.headers.get_content_type() != "application/json"
            or (charset is not None and charset.lower() not in {"utf-8", "utf8"})
        ):
            self._error(415, "Требуется application/json в UTF-8.")
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                self._error(400, "Неполное тело запроса.")
                return
            payload = parse_json_document(body.decode("utf-8"), max_bytes=self.server.body_max_bytes)
            if not isinstance(payload, dict) or set(payload) != {"text"} or not isinstance(payload["text"], str):
                self._error(400, 'Требуется JSON-объект с единственным строковым полем "text".')
                return
            text = payload["text"]
            if not text.strip():
                self._error(400, "Входной текст не должен быть пустым.")
                return
            if len(text.encode("utf-8")) > self.server.input_max_bytes:
                self._error(413, "Текст превышает допустимый размер.")
                return
        except (OSError, ValueError, UnicodeError, RecursionError):
            self._error(400, "Некорректное тело JSON-запроса.")
            return
        try:
            result = self.server.runner.run(text)
            # Preflight encoding ensures serialization failure still yields 502.
            json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except Exception:
            self._error(502, "Не удалось выполнить сценарий.")
            return
        self._respond(200, result)

    def _unsupported(self):
        self._error(404, "Маршрут не найден.")

    do_GET = do_HEAD = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _unsupported


def build_webhook_server(
    scenario: Any,
    runner: Any,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
) -> HTTPServer:
    """Bind a serial server; the caller owns serve_forever and server_close.

    The trusted scenario and output destination are fixed when this server is
    created. A request can provide only input text. Authentication is mandatory,
    including on localhost; callers should obtain the secret from their environment.
    """
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise ValueError("Для webhook требуется непустой токен Bearer без пробелов.")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("Для webhook требуется непустой адрес прослушивания.")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Порт webhook должен быть целым числом от 0 до 65535.")
    input_max_bytes = scenario.input_max_bytes
    if type(input_max_bytes) is not int or input_max_bytes <= 0:
        raise ValueError("Некорректный лимит входного текста сценария.")
    server = _WebhookServer((host, port), _WebhookHandler)
    server.runner = runner
    server.authorization = ("Bearer " + token).encode("ascii")
    server.input_max_bytes = input_max_bytes
    # JSON may represent a one-byte character as a six-byte \uXXXX escape.
    # The separate decoded-text limit preserves the scenario's actual input bound.
    server.body_max_bytes = 6 * input_max_bytes + JSON_ENVELOPE_BYTES
    return server
