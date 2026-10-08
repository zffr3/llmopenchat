"""Small OpenAI-compatible streaming client. Requests go to the configured URL."""

from __future__ import annotations

import json
import os
import re
import socket
import threading
from http.client import HTTPException
from dataclasses import dataclass
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener
import ipaddress


class ApiError(RuntimeError):
    pass


MAX_TOOL_CALLS = 16
MAX_TOOL_ARGUMENT_BYTES = 256 * 1024
_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_TOOL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_TOOL_REQUEST_FIELDS = {"tools", "tool_choice", "parallel_tool_calls"}


def validated_tool_call(value: object) -> dict:
    """Copy only OpenAI tool-call fields; this never executes anything."""
    if not isinstance(value, dict) or value.get("type") != "function":
        raise ValueError("Неверный тип вызова инструмента.")
    call_id = value.get("id")
    function = value.get("function")
    if not isinstance(call_id, str) or not _TOOL_ID.fullmatch(call_id):
        raise ValueError("Неверный идентификатор вызова инструмента.")
    if not isinstance(function, dict):
        raise ValueError("Неверный формат function в вызове инструмента.")
    name, arguments = function.get("name"), function.get("arguments")
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise ValueError("Неверное имя инструмента.")
    if not isinstance(arguments, str):
        raise ValueError("Аргументы инструмента должны быть строкой JSON.")
    try:
        if len(arguments.encode("utf-8")) > MAX_TOOL_ARGUMENT_BYTES:
            raise ValueError("Аргументы инструмента слишком большие (максимум 256 КБ).")
        parsed = json.loads(arguments, parse_constant=_invalid_constant)
        # Reject escaped lone surrogates and numeric overflow as well as NaN.
        json.dumps(parsed, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Аргументы инструмента должны быть корректным JSON-объектом.") from error
    if not isinstance(parsed, dict):
        raise ValueError("Аргументы инструмента должны быть JSON-объектом.")
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _invalid_constant(value: str):
    raise ValueError(f"Недопустимая JSON-константа в аргументах: {value}.")


class _ToolCallBuffer:
    def __init__(self, allowed_names: set[str], existing_ids: set[str]):
        self.allowed_names = allowed_names
        self.existing_ids = existing_ids
        self.calls: dict[int, dict] = {}
        self.argument_bytes = 0

    def add(self, deltas: object) -> None:
        if not isinstance(deltas, list):
            raise ApiError("Неверный формат tool_calls в ответе сервера.")
        if deltas and not self.allowed_names:
            raise ApiError("Модель запросила инструмент, который не подключён в этом чате.")
        for delta in deltas:
            if not isinstance(delta, dict):
                raise ApiError("Неверный формат вызова инструмента в ответе сервера.")
            index = delta.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_TOOL_CALLS:
                raise ApiError("Неверный индекс вызова инструмента (максимум 16 вызовов).")
            call = self.calls.setdefault(index, {"id": "", "type": None, "function": {"name": "", "arguments": ""}})
            if "type" in delta:
                if delta["type"] != "function":
                    raise ApiError("Модель запросила неподдерживаемый тип инструмента.")
                call["type"] = "function"
            if "id" in delta:
                fragment = delta["id"]
                if not isinstance(fragment, str) or len(call["id"]) + len(fragment) > 128:
                    raise ApiError("Неверный идентификатор вызова инструмента.")
                call["id"] += fragment
            if "function" not in delta:
                continue
            function = delta["function"]
            if not isinstance(function, dict):
                raise ApiError("Неверный формат function в ответе сервера.")
            for key in ("name", "arguments"):
                if key not in function:
                    continue
                fragment = function[key]
                if not isinstance(fragment, str):
                    raise ApiError("Неверный формат имени или аргументов инструмента.")
                if key == "name" and len(call["function"][key]) + len(fragment) > 64:
                    raise ApiError("Неверное имя инструмента.")
                if key == "arguments":
                    try:
                        self.argument_bytes += len(fragment.encode("utf-8"))
                    except UnicodeError as error:
                        raise ApiError("Неверная кодировка аргументов инструмента.") from error
                    if self.argument_bytes > MAX_TOOL_ARGUMENT_BYTES:
                        raise ApiError("Аргументы инструментов слишком большие (максимум 256 КБ).")
                call["function"][key] += fragment

    def complete(self, finish_reason: str | None) -> list[dict]:
        if not self.calls:
            if finish_reason == "tool_calls":
                raise ApiError("Модель завершила вызов инструментов без самих вызовов.")
            return []
        if finish_reason != "tool_calls":
            raise ApiError("Поток вызовов инструментов не завершён полностью; вызовы отменены.")
        if set(self.calls) != set(range(len(self.calls))):
            raise ApiError("В потоке отсутствует часть вызовов инструментов.")
        calls, identifiers = [], set(self.existing_ids)
        for index in range(len(self.calls)):
            try:
                call = validated_tool_call(self.calls[index])
            except ValueError as error:
                raise ApiError(str(error)) from error
            if call["function"]["name"] not in self.allowed_names:
                raise ApiError("Модель запросила инструмент, который не подключён в этом чате.")
            if call["id"] in identifiers:
                raise ApiError("Повторный идентификатор вызова инструмента.")
            identifiers.add(call["id"])
            calls.append(call)
        return calls


@dataclass
class StreamEvent:
    kind: str
    value: str | dict


def _events(lines) -> Iterator[str]:
    """Decode SSE data records, including multiline records and comments."""
    data: list[str] = []
    for raw in lines:
        try:
            line = raw.decode("utf-8").rstrip("\r\n") if isinstance(raw, bytes) else raw.rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ApiError("Сервер вернул текст в неверной кодировке UTF-8.") from error
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


class ApiClient:
    def __init__(self, config: dict):
        self.config = config
        self._cancelled = threading.Event()
        self._response_lock = threading.Lock()
        self._response = None
        host = urlparse(config["base_url"]).hostname
        try:
            local = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        self.opener = build_opener(ProxyHandler({})) if local else build_opener()

    def cancel(self) -> None:
        """Wake a streaming read when its UI cancels this client's request."""
        self._cancelled.set()
        with self._response_lock:
            response = self._response
            raw = getattr(getattr(response, "fp", None), "raw", None)
            connection = getattr(raw, "_sock", None)
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                # urllib retains a makefile() reference, so socket.close()
                # defers native close. Windows can keep its timed read blocked
                # even after shutdown(); detach and close the socket handle to
                # wake it without acquiring the BufferedReader's read lock.
                try:
                    descriptor = connection.detach()
                    if descriptor != -1:
                        socket.close(descriptor)
                except OSError:
                    pass

    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None,
                    tool_choice: str | None = None) -> Iterator[StreamEvent]:
        if self._cancelled.is_set():
            raise ApiError("Запрос отменён.")
        allowed_names: set[str] = set()
        if tools is not None:
            if not isinstance(tools, list):
                raise ApiError("Список подключённых инструментов должен быть списком.")
            for tool in tools:
                function = tool.get("function") if isinstance(tool, dict) else None
                name = function.get("name") if isinstance(function, dict) else None
                if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
                    raise ApiError("Неверная схема подключённого инструмента.")
                if name in allowed_names:
                    raise ApiError("Повторное имя подключённого инструмента.")
                allowed_names.add(name)
        if tool_choice is not None and (not isinstance(tool_choice, str) or tool_choice not in {"auto", "required", "none"}):
            raise ApiError("tool_choice должен быть auto, required или none.")
        if not allowed_names and tool_choice not in (None, "none"):
            raise ApiError("Выбор инструмента требует подключённых в этом чате инструментов.")
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
            "stream": True,
            "stream_options": {"include_usage": True},
            **{key: value for key, value in self.config.get("request_extra", {}).items() if key not in _TOOL_REQUEST_FIELDS},
            "tool_choice": (tool_choice or "auto") if allowed_names else "none",
        }
        if allowed_names:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = False
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if api_key := os.environ.get("LLMOPENCHAT_API_KEY"):
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(
            self.config["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        finish_reason = None
        existing_ids = {
            call["id"]
            for message in messages if isinstance(message, dict) and isinstance(message.get("tool_calls"), list)
            for call in message["tool_calls"] if isinstance(call, dict) and isinstance(call.get("id"), str)
        }
        tool_calls = _ToolCallBuffer(allowed_names if payload["tool_choice"] != "none" else set(), existing_ids)
        try:
            with self.opener.open(request, timeout=self.config["request_timeout"]) as response:
                with self._response_lock:
                    self._response = response
                if self._cancelled.is_set():
                    raise ApiError("Запрос отменён.")
                for record in _events(response):
                    if self._cancelled.is_set():
                        raise ApiError("Запрос отменён.")
                    if record == "[DONE]":
                        for call in tool_calls.complete(finish_reason):
                            yield StreamEvent("tool_call", call)
                        yield StreamEvent("done", {})
                        return
                    try:
                        chunk = json.loads(record)
                    except ValueError as error:
                        raise ApiError("Сервер вернул поврежденный поток JSON.") from error
                    if not isinstance(chunk, dict):
                        raise ApiError("Неверный формат ответа сервера.")
                    if chunk.get("error"):
                        raise ApiError(str(chunk["error"]))
                    choices = chunk.get("choices", [])
                    if not isinstance(choices, list):
                        raise ApiError("Неверный формат choices в ответе сервера.")
                    if len(choices) > 1:
                        raise ApiError("Поддерживается только один вариант ответа модели.")
                    for choice in choices:
                        if not isinstance(choice, dict):
                            raise ApiError("Неверный формат элемента choices в ответе сервера.")
                        index = choice.get("index", 0)
                        if isinstance(index, bool) or not isinstance(index, int) or index != 0:
                            raise ApiError("Поддерживается только вариант ответа с индексом 0.")
                        delta = choice.get("delta")
                        if delta is None:
                            delta = {}
                        if not isinstance(delta, dict):
                            raise ApiError("Неверный формат delta в ответе сервера.")
                        if finish_reason is not None and (delta or choice.get("finish_reason")):
                            raise ApiError("Сервер продолжил ответ после его завершения.")
                        if "tool_calls" in delta:
                            tool_calls.add(delta["tool_calls"])
                        if delta.get("function_call"):
                            raise ApiError("Устаревший формат function_call не поддерживается.")
                        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning:
                            yield StreamEvent("reasoning", reasoning)
                        content = delta.get("content")
                        if content is not None and not isinstance(content, str):
                            raise ApiError("Неверный формат content в ответе сервера.")
                        if isinstance(content, str) and content:
                            yield StreamEvent("content", content)
                        refusal = delta.get("refusal")
                        if refusal is not None and not isinstance(refusal, str):
                            raise ApiError("Неверный формат refusal в ответе сервера.")
                        if refusal:
                            yield StreamEvent("refusal", refusal)
                        if reason := choice.get("finish_reason"):
                            if not isinstance(reason, str):
                                raise ApiError("Неверный формат finish_reason в ответе сервера.")
                            finish_reason = reason
                            yield StreamEvent("finish", reason)
                    if usage := chunk.get("usage"):
                        if not isinstance(usage, dict):
                            raise ApiError("Неверный формат usage в ответе сервера.")
                        yield StreamEvent("usage", usage)
                if finish_reason is None:
                    raise ApiError("Соединение оборвалось до завершения ответа. Попробуйте снова.")
                for call in tool_calls.complete(finish_reason):
                    yield StreamEvent("tool_call", call)
                yield StreamEvent("done", {})
        except HTTPError as error:
            try:
                detail = error.read(8192).decode("utf-8", errors="replace")
            finally:
                error.close()
            raise ApiError(f"HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, OSError, HTTPException) as error:
            raise ApiError(f"Не удалось получить ответ модели: {error}") from error
        finally:
            with self._response_lock:
                self._response = None
