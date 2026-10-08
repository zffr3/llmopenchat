from __future__ import annotations

import copy
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from llmopenchat.api import MAX_TOOL_ARGUMENT_BYTES, ApiClient, ApiError
from llmopenchat.config import DEFAULT_CONFIG


def sse_record(value) -> bytes:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return ("data: " + value + "\r\n\r\n").encode("utf-8")


TOOLS = [{"type": "function", "function": {"name": "repository_read", "parameters": {"type": "object"}}}]


def tool_delta(index=0, call_id="call_read", name="repository_read", arguments='{"path":"README.md"}'):
    return {"index": index, "id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class ApiClientTests(unittest.TestCase):
    def setUp(self):
        self.status = 200
        self.chunked = False
        self.response = sse_record("[DONE]")
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append(
                    {"path": self.path, "headers": self.headers, "payload": json.loads(body)}
                )
                self.send_response(owner.status)
                self.send_header(
                    "Content-Type",
                    "text/event-stream; charset=utf-8" if owner.status == 200 else "application/json",
                )
                if owner.chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                # Split characters across writes to exercise actual HTTP UTF-8 decoding.
                response = owner.response
                try:
                    for start in range(0, len(response), 7):
                        self.wfile.write(response[start : start + 7])
                    self.wfile.flush()
                except ConnectionError:
                    # Rejecting an invalid record closes the response immediately.
                    pass

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["base_url"] = f"http://127.0.0.1:{self.server.server_port}/v1/"
        self.config["request_timeout"] = 5
        self.environment = patch.dict("os.environ", {"LLMOPENCHAT_API_KEY": ""})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def events(self, messages=None, tools=None, tool_choice=None):
        return list(ApiClient(self.config).stream_chat(messages or [{"role": "user", "content": "Привет"}], tools=tools, tool_choice=tool_choice))

    def tool_response(self, deltas, finish="tool_calls", done=True):
        records = [sse_record({"choices": [{"index": 0, "delta": {"tool_calls": delta}}]}) for delta in deltas]
        if finish is not None:
            records.append(sse_record({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}))
        if done:
            records.append(sse_record("[DONE]"))
        self.response = b"".join(records)

    def assert_no_tool_event(self, tools=TOOLS):
        seen = []
        with self.assertRaises(ApiError):
            for event in ApiClient(self.config).stream_chat([{"role": "user", "content": "x"}], tools=tools):
                seen.append(event)
        self.assertFalse(any(event.kind == "tool_call" for event in seen))

    def test_streams_russian_reasoning_content_finish_usage_and_done(self):
        self.response = b": keepalive\r\nevent: message\r\n\r\n" + b"".join(
            [
                sse_record({"choices": [{"delta": {"reasoning": "Обдумаю вопрос. "}}]}),
                sse_record({"choices": [{"delta": {"reasoning_content": "Продолжение мысли."}}]}),
                sse_record({"choices": [{"delta": {"content": "Здравствуйте, "}}]}),
                sse_record({"choices": [{"delta": {"content": "мир!"}}]}),
                sse_record({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                sse_record({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4}}),
                sse_record("[DONE]"),
            ]
        )
        self.assertEqual(
            [(event.kind, event.value) for event in self.events()],
            [
                ("reasoning", "Обдумаю вопрос. "),
                ("reasoning", "Продолжение мысли."),
                ("content", "Здравствуйте, "),
                ("content", "мир!"),
                ("finish", "stop"),
                ("usage", {"prompt_tokens": 11, "completion_tokens": 4}),
                ("done", {}),
            ],
        )

    def test_preserves_messages_and_request_settings(self):
        messages = [
            {"role": "system", "content": "Отвечай по-русски."},
            {"role": "user", "content": "Сколько будет два плюс два?"},
        ]
        self.config.update(model="test-local", temperature=0.25, max_tokens=73)
        self.config["request_extra"] = {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 0.9}
        self.events(messages)
        request = self.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(
            request["payload"],
            {
                "model": "test-local",
                "messages": messages,
                "temperature": 0.25,
                "max_tokens": 73,
                "stream": True,
                "stream_options": {"include_usage": True},
                "tool_choice": "none",
                "chat_template_kwargs": {"enable_thinking": False},
                "top_p": 0.9,
            },
        )
        self.assertIsNone(request["headers"].get("Authorization"))

    def test_optional_api_key_is_sent(self):
        with patch.dict("os.environ", {"LLMOPENCHAT_API_KEY": "test-token"}):
            self.events()
        self.assertEqual(self.requests[0]["headers"]["Authorization"], "Bearer test-token")

    def test_enabled_tools_are_supplied_only_by_live_harness(self):
        self.config["request_extra"] = {"tools": [{"malicious": True}], "tool_choice": "required", "parallel_tool_calls": True}
        self.events(tools=TOOLS)
        payload = self.requests[-1]["payload"]
        self.assertEqual(payload["tools"], TOOLS)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertFalse(payload["parallel_tool_calls"])
        self.events(tools=[])
        disabled = self.requests[-1]["payload"]
        self.assertEqual(disabled["tool_choice"], "none")
        self.assertNotIn("tools", disabled)
        self.assertNotIn("parallel_tool_calls", disabled)

    def test_explicit_tool_choice_requires_live_schemas_and_cannot_be_injected(self):
        self.config["request_extra"] = {"tool_choice": "none", "tools": []}
        self.events(tools=TOOLS, tool_choice="required")
        self.assertEqual(self.requests[-1]["payload"]["tool_choice"], "required")
        self.assertEqual(self.requests[-1]["payload"]["tools"], TOOLS)
        self.events(tools=TOOLS, tool_choice="none")
        self.assertEqual(self.requests[-1]["payload"]["tool_choice"], "none")
        count = len(self.requests)
        for tools, choice in ((None, "required"), ([], "auto"), (TOOLS, {}), (TOOLS, "invalid")):
            with self.subTest(tools=tools, choice=choice), self.assertRaises(ApiError):
                self.events(tools=tools, tool_choice=choice)
        self.assertEqual(len(self.requests), count)

    def test_interleaved_deltas_are_buffered_and_emitted_in_index_order_at_done(self):
        self.tool_response([
            [tool_delta(index=1, call_id="call_second", name="repository_", arguments='{"path":"В')],
            [tool_delta(call_id="call_", name="repository_", arguments='{"path":')],
            [{"index": 1, "function": {"name": "read", "arguments": 'опрос.md"}'}}],
            [{"index": 0, "id": "first", "function": {"name": "read", "arguments": '"README.md"}'}}],
        ])
        self.response = self.response.removesuffix(sse_record("[DONE]")) + sse_record({"usage": {"completion_tokens": 8}}) + sse_record("[DONE]")
        events = self.events(tools=TOOLS)
        self.assertEqual([event.kind for event in events], ["finish", "usage", "tool_call", "tool_call", "done"])
        self.assertEqual(events[2].value, {"id": "call_first", "type": "function", "function": {"name": "repository_read", "arguments": '{"path":"README.md"}'}})
        self.assertEqual(events[3].value["id"], "call_second")
        self.assertEqual(json.loads(events[3].value["function"]["arguments"]), {"path": "Вопрос.md"})

    def test_complete_tool_stream_can_finish_at_clean_eof(self):
        self.tool_response([[tool_delta()]], done=False)
        self.assertEqual([event.kind for event in self.events(tools=TOOLS)], ["finish", "tool_call", "done"])

    def test_unfinished_or_truncated_calls_never_emit_tool_events(self):
        for finish, done in ((None, True), (None, False), ("length", True), ("stop", True), ("content_filter", True)):
            with self.subTest(finish=finish, done=done):
                self.tool_response([[tool_delta()]], finish=finish, done=done)
                self.assert_no_tool_event()
        self.tool_response([[tool_delta()]], done=False)
        self.chunked = True
        response = self.response
        self.response = f"{len(response) + 64:x}\r\n".encode("ascii") + response
        self.assert_no_tool_event()

    def test_disabled_and_unknown_tools_are_rejected(self):
        self.tool_response([[tool_delta()]])
        self.assert_no_tool_event(tools=None)
        self.tool_response([[tool_delta(name="repository_write")]])
        self.assert_no_tool_event()
        self.tool_response([], finish="tool_calls")
        self.assert_no_tool_event()

    def test_invalid_call_shapes_ids_arguments_and_duplicate_ids_are_rejected(self):
        invalid = [
            [{**tool_delta(), "index": True}],
            [{**tool_delta(), "index": 16}],
            [{**tool_delta(), "index": -1}],
            [{**tool_delta(), "index": None}],
            [{**tool_delta(), "type": "shell"}],
            [{**tool_delta(), "function": "invalid"}],
            [tool_delta(call_id="call with spaces")],
            [tool_delta(call_id="")],
            [tool_delta(name="name.with.dot")],
            [tool_delta(arguments={"path": "README.md"})],
            [tool_delta(arguments='{"path":')],
            [tool_delta(arguments='[]')],
            [tool_delta(arguments='{"path":NaN}')],
            [tool_delta(arguments='{"path":"\\ud800"}')],
            [tool_delta(arguments='{"value":1e10000}')],
            [tool_delta(index=1)],
            [tool_delta(), tool_delta(index=1)],
            [None],
        ]
        for calls in invalid:
            with self.subTest(calls=calls):
                self.tool_response([calls])
                self.assert_no_tool_event()

    def test_bounded_tool_argument_buffer_and_sixteen_call_limit(self):
        # UTF-8 byte bounds apply across fragments and calls, not character counts.
        with patch("llmopenchat.api.MAX_TOOL_ARGUMENT_BYTES", 64):
            self.tool_response([
                [tool_delta(arguments='{"path":"' + "я" * 15)],
                [{"index": 0, "function": {"arguments": "я" * 15 + '"}'}}],
            ])
            self.assert_no_tool_event()
        self.tool_response([[tool_delta(arguments='{"path":"' + "x" * MAX_TOOL_ARGUMENT_BYTES + '"}')]])
        self.assert_no_tool_event()
        self.tool_response([[tool_delta(index=index, call_id=f"call_{index}") for index in range(16)]])
        self.assertEqual(sum(event.kind == "tool_call" for event in self.events(tools=TOOLS)), 16)
        self.tool_response([[tool_delta(index=index, call_id=f"call_{index}") for index in range(17)]])
        self.assert_no_tool_event()

    def test_invalid_trailing_records_cancel_all_buffered_calls(self):
        self.tool_response([[tool_delta()]], done=False)
        self.response += sse_record({"usage": "bad"}) + sse_record("[DONE]")
        self.assert_no_tool_event()

    def test_call_id_cannot_reuse_an_earlier_history_round(self):
        self.tool_response([[tool_delta()]])
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{key: value for key, value in tool_delta().items() if key != "index"}]},
            {"role": "tool", "tool_call_id": "call_read", "content": "Earlier result"},
            {"role": "user", "content": "Again"},
        ]
        seen = []
        with self.assertRaisesRegex(ApiError, "Повторный идентификатор"):
            for event in ApiClient(self.config).stream_chat(messages, tools=TOOLS):
                seen.append(event)
        self.assertFalse(any(event.kind == "tool_call" for event in seen))

    def test_multiple_choices_and_nonzero_choice_indices_are_rejected(self):
        for choices in (
            [{"index": 1, "delta": {"content": "x"}}],
            [{"index": True, "delta": {}}],
            [{"index": 0, "delta": {}}, {"index": 1, "delta": {}}],
        ):
            with self.subTest(choices=choices):
                self.response = sse_record({"choices": choices}) + sse_record("[DONE]")
                self.assert_no_tool_event()

    def test_invalid_tool_schemas_fail_before_http_request(self):
        for tools in ({}, [None], [{"type": "shell"}], TOOLS + TOOLS):
            with self.subTest(tools=tools):
                with self.assertRaises(ApiError):
                    self.events(tools=tools)
        self.assertEqual(self.requests, [])

    def test_multiline_sse_record_and_finish_without_done(self):
        self.response = (
            'data: {"choices": [\n'
            'data: {"delta": {"content": "Ответ"}, "finish_reason": "length"}\n'
            'data: ]}\n\n'
        ).encode("utf-8")
        self.assertEqual(
            [(event.kind, event.value) for event in self.events()],
            [("content", "Ответ"), ("finish", "length"), ("done", {})],
        )

    def test_stream_ending_before_finish_raises(self):
        self.response = sse_record({"choices": [{"delta": {"content": "Незавершённый ответ"}}]})
        with self.assertRaisesRegex(ApiError, "оборвалось"):
            self.events()

    def test_truncated_http_chunk_raises_api_error(self):
        self.chunked = True
        record = sse_record({"choices": [{"delta": {"content": "Ответ"}}]})
        # The advertised HTTP chunk is larger than the bytes sent; EOF is an error.
        self.response = f"{len(record) + 64:x}\r\n".encode("ascii") + record
        with self.assertRaises(ApiError):
            self.events()

    def test_corrupt_json_stream_raises_api_error(self):
        self.response = sse_record('{"choices":')
        with self.assertRaisesRegex(ApiError, "JSON"):
            self.events()

    def test_invalid_utf8_stream_raises_api_error(self):
        self.response = b'data: {"choices": [], "text": "\xff"}\n\n'
        with self.assertRaises(ApiError):
            self.events()

    def test_malformed_choices_or_delta_raises_api_error(self):
        for chunk in (
            {"choices": "invalid"},
            {"choices": [None]},
            {"choices": [{"delta": ["unexpected"]}]},
            {"choices": [{"delta": []}]},
            {"choices": [{"delta": ""}]},
        ):
            with self.subTest(chunk=chunk):
                self.response = sse_record(chunk) + sse_record("[DONE]")
                with self.assertRaises(ApiError):
                    self.events()

    def test_refusal_is_exposed_to_noninteractive_consumers(self):
        self.response = b"".join([
            sse_record({"choices": [{"delta": {"refusal": "Cannot process"}}]}),
            sse_record({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            sse_record("[DONE]"),
        ])
        self.assertEqual([(event.kind, event.value) for event in self.events()],
                         [("refusal", "Cannot process"), ("finish", "stop"), ("done", {})])

    def test_malformed_content_or_refusal_is_rejected(self):
        for delta in ({"content": []}, {"refusal": True}):
            with self.subTest(delta=delta):
                self.response = sse_record({"choices": [{"delta": delta}]}) + sse_record("[DONE]")
                with self.assertRaises(ApiError):
                    self.events()

    def test_stream_error_preserves_server_detail(self):
        self.response = sse_record({"error": {"message": "Не хватает памяти", "type": "server_error"}})
        with self.assertRaisesRegex(ApiError, "Не хватает памяти"):
            self.events()

    def test_http_error_preserves_status_and_russian_detail(self):
        self.status = 503
        self.response = json.dumps({"error": {"message": "Модель ещё загружается"}}, ensure_ascii=False).encode("utf-8")
        with self.assertRaises(ApiError) as raised:
            self.events()
        self.assertIn("HTTP 503", str(raised.exception))
        self.assertIn("Модель ещё загружается", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
