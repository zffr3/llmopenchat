"""Shared generation preserves tool limits, cancellation, and saved protocol."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from llmopenchat.api import ApiError, StreamEvent
from llmopenchat.conversation import (
    MAX_TOOL_CALLS, MAX_TOOL_ROUNDS, TOOL_LIMIT_MESSAGE,
    GenerationCancelled, generate_response, initial_messages,
)
from llmopenchat.harness import HarnessError
from llmopenchat.history import load_session, save_session


def call(identifier="call-1", name="read_file", arguments='{"path":"main.py"}'):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_round(*calls, content=""):
    return [StreamEvent("content", content), StreamEvent("finish", "tool_calls"),
            *(StreamEvent("tool_call", value) for value in calls), StreamEvent("done", {})]


def final_round(answer="Готово."):
    return [StreamEvent("content", answer), StreamEvent("finish", "stop"), StreamEvent("done", {})]


class Client:
    def __init__(self, *rounds):
        self.rounds = iter(rounds)
        self.requests = []
        self.closed = 0
        self.choices = []

    def stream_chat(self, messages, tools=None, tool_choice=None):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        self.choices.append(tool_choice)
        try:
            for event in next(self.rounds):
                if isinstance(event, BaseException):
                    raise event
                yield event
        finally:
            self.closed += 1


class Harness:
    def __init__(self, execute=None, names=None, settings=None):
        self.executed = []
        self.action = execute
        self.names = names or ["read_file"]
        self.settings = settings

    def schemas(self):
        return [{"type": "function", "function": {"name": name}} for name in self.names]

    def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return self.action(name, arguments) if self.action else '{"status":"ok","content":"print(1)"}'


class ConversationTests(unittest.TestCase):
    def control_events(self, events):
        return [event for event in events if event.kind in {"harness_status", "harness_summary", "tool_start", "tool_result"}]

    def code_harness(self, execute=None):
        return Harness(execute, ["read_file", "list_files", "write_file", "create_directory"],
                       SimpleNamespace(code_enabled=True, web_enabled=False, repository=Path("CodeRepo")))

    def progress(self, events):
        summaries = [event.value for event in events if event.kind == "harness_summary"]
        self.assertEqual(len(summaries), 1)
        return summaries[0]

    def assert_saveable(self, messages):
        with tempfile.TemporaryDirectory(prefix="llmopenchat-conversation-") as directory:
            path = Path(directory) / "session.json"
            save_session(path, messages, "test")
            self.assertEqual(load_session(path), messages)

    def test_initial_messages_is_shared_and_has_no_empty_instruction(self):
        self.assertEqual(initial_messages({"system_prompt": ""}), [])
        self.assertEqual(initial_messages({"system_prompt": "Инструкция"}),
                         [{"role": "system", "content": "Инструкция"}])

    def test_plain_answer_forwards_all_events_and_ui_owns_final_history(self):
        events = [StreamEvent("reasoning", "Обдумываю."), StreamEvent("content", "Ответ"),
                  StreamEvent("usage", {"completion_tokens": 3}), StreamEvent("finish", "length"),
                  StreamEvent("done", {})]
        messages = [{"role": "user", "content": "Вопрос"}]
        received = []
        client = Client(events)
        result = generate_response(client, messages, on_event=received.append)
        self.assertEqual([event for event in received if event not in self.control_events(received)], events)
        self.assertEqual(self.progress(received), {"requested": 0, "succeeded": 0, "denied": 0, "failed": 0,
                         "files_written": [], "directories_created": [], "no_changes": True, "uncertain_changes": False, "scripts_run": []})
        self.assertEqual(result.answer, "Ответ")
        self.assertEqual(result.usage, {"completion_tokens": 3})
        self.assertEqual(result.finish, "length")
        self.assertGreaterEqual(result.elapsed, 0)
        self.assertEqual(messages, [{"role": "user", "content": "Вопрос"}])
        self.assertEqual(client.requests[0][1], None)
        self.assertEqual(client.closed, 1)

    def test_tools_have_all_placeholder_results_before_first_execution(self):
        messages = [{"role": "user", "content": "Проверь два файла"}]

        def execute(name, arguments):
            self.assertEqual([message["role"] for message in messages],
                             ["user", "assistant", "tool", "tool"])
            if len(harness.executed) == 1:
                self.assertTrue(all(json.loads(message["content"])["status"] == "error"
                                    for message in messages[-2:]))
            self.assert_saveable(messages)
            return '{"status":"ok"}'

        harness = Harness(execute)
        first = tool_round(call(), call("call-2"), content="Проверю.")
        final = final_round()
        received = []
        client = Client(first, final)
        result = generate_response(client, messages, harness, received.append)
        self.assertEqual(result.answer, "Готово.")
        streamed = [event for event in received if event not in self.control_events(received)]
        self.assertEqual(streamed[:len(first)], first)
        self.assertEqual(streamed[len(first)], StreamEvent("tool_round", {"round": 1, "calls": 2}))
        self.assertEqual(streamed[len(first) + 1:], final)
        self.assertEqual(client.requests[1][0][1:], messages)
        self.assertEqual(len(harness.executed), 2)

    def test_cancellation_before_request_opens_no_stream(self):
        client = Client(final_round())
        messages = [{"role": "user", "content": "Вопрос"}]
        with self.assertRaises(GenerationCancelled):
            generate_response(client, messages, cancelled=lambda: True)
        self.assertEqual(client.requests, [])
        self.assertEqual(len(messages), 1)

    def test_cancellation_during_stream_closes_it_and_discards_partial_answer(self):
        state = {"cancelled": False}
        received = []

        def receive(event):
            received.append(event)
            if event.kind == "content":
                state["cancelled"] = True

        client = Client(final_round())
        messages = [{"role": "user", "content": "Вопрос"}]
        with self.assertRaises(GenerationCancelled):
            generate_response(client, messages, on_event=receive, cancelled=lambda: state["cancelled"])
        self.assertEqual([event.kind for event in received if event not in self.control_events(received)], ["content"])
        self.assertEqual(client.closed, 1)
        self.assertEqual(len(messages), 1)

    def test_cancellation_between_tools_keeps_first_result_and_second_placeholder(self):
        state = {"cancelled": False}

        def execute(name, arguments):
            state["cancelled"] = True
            return '{"status":"ok","content":"first result"}'

        messages = [{"role": "user", "content": "Вопрос"}]
        harness = Harness(execute)
        client = Client(tool_round(call(), call("call-2")))
        with self.assertRaises(GenerationCancelled):
            generate_response(client, messages, harness, cancelled=lambda: state["cancelled"])
        self.assertEqual(len(harness.executed), 1)
        self.assertEqual(json.loads(messages[-2]["content"])["status"], "ok")
        self.assertEqual(json.loads(messages[-1]["content"])["status"], "error")
        self.assertEqual(len(client.requests), 1)
        self.assert_saveable(messages)

    def test_keyboard_interrupt_records_partial_action_and_complete_protocol(self):
        def interrupted(name, arguments):
            raise KeyboardInterrupt

        messages = [{"role": "user", "content": "Вопрос"}]
        harness = Harness(interrupted)
        with self.assertRaises(KeyboardInterrupt):
            generate_response(Client(tool_round(call(), call("call-2"))), messages, harness)
        self.assertEqual(len(harness.executed), 1)
        self.assertIn("частично", json.loads(messages[-2]["content"])["error"])
        self.assert_saveable(messages)

    def test_harness_error_becomes_result_and_model_can_explain_it(self):
        def failed(name, arguments):
            raise HarnessError("Файл недоступен")

        messages = [{"role": "user", "content": "Вопрос"}]
        result = generate_response(Client(tool_round(call()), final_round("Файл недоступен.")),
                                   messages, Harness(failed))
        self.assertEqual(result.answer, "Файл недоступен.")
        self.assertEqual(json.loads(messages[-1]["content"])["error"], "Файл недоступен")
        self.assert_saveable(messages)

    def test_invalid_or_incomplete_calls_never_execute_or_mutate_history(self):
        cases = [
            (tool_round(call()), False),
            ([StreamEvent("tool_call", call()), ApiError("broken stream")], True),
            ([StreamEvent("tool_call", call()), StreamEvent("finish", "length")], True),
            (tool_round(call(), call()), True),
            (tool_round(call(name="write_file")), True),
            (tool_round(call(arguments="[]")), True),
            (tool_round(call(arguments="{invalid")), True),
            (tool_round({"id": "wrong"}), True),
        ]
        for events, enabled in cases:
            with self.subTest(events=events, enabled=enabled):
                messages = [{"role": "user", "content": "Вопрос"}]
                harness = Harness()
                with self.assertRaises(ApiError):
                    generate_response(Client(events), messages, harness if enabled else None)
                self.assertEqual(harness.executed, [])
                self.assertEqual(len(messages), 1)

    def test_repeated_historical_id_is_rejected_without_reexecution(self):
        messages = [{"role": "user", "content": "Вопрос"}]
        harness = Harness()
        with self.assertRaisesRegex(ApiError, "повторный"):
            generate_response(Client(tool_round(call()), tool_round(call())), messages, harness)
        self.assertEqual(len(harness.executed), 1)
        self.assert_saveable(messages)

    def test_total_call_budget_declines_whole_next_round_and_keeps_it_saveable(self):
        first = [call(f"first-{index}") for index in range(MAX_TOOL_CALLS - 1)]
        messages = [{"role": "user", "content": "Вопрос"}]
        harness = Harness()
        client = Client(tool_round(*first), tool_round(call("last-1"), call("last-2")))
        result = generate_response(client, messages, harness)
        self.assertEqual(result.answer, TOOL_LIMIT_MESSAGE)
        self.assertEqual(len(harness.executed), MAX_TOOL_CALLS - 1)
        self.assertEqual(json.loads(messages[-1]["content"])["status"], "error")
        self.assertEqual(json.loads(messages[-2]["content"])["status"], "error")
        self.assert_saveable(messages)

    def test_exact_call_budget_can_finish_normally(self):
        calls = [call(f"call-{index}") for index in range(MAX_TOOL_CALLS)]
        messages = [{"role": "user", "content": "Вопрос"}]
        harness = Harness()
        result = generate_response(Client(tool_round(*calls), final_round()), messages, harness)
        self.assertEqual(result.answer, "Готово.")
        self.assertEqual(len(harness.executed), MAX_TOOL_CALLS)
        self.assert_saveable(messages)

    def test_round_budget_stops_with_placeholder_results(self):
        client = Client(*(tool_round(call(f"call-{index}")) for index in range(MAX_TOOL_ROUNDS + 1)))
        harness = Harness()
        messages = [{"role": "user", "content": "Вопрос"}]
        result = generate_response(client, messages, harness)
        self.assertEqual(result.answer, TOOL_LIMIT_MESSAGE)
        self.assertEqual(len(harness.executed), MAX_TOOL_ROUNDS)
        self.assertEqual(json.loads(messages[-1]["content"])["status"], "error")
        self.assert_saveable(messages)

    def test_api_failure_after_completed_tool_round_keeps_history_saveable(self):
        messages = [{"role": "user", "content": "Вопрос"}]
        harness = Harness()
        with self.assertRaisesRegex(ApiError, "broken stream"):
            generate_response(Client(tool_round(call()), [ApiError("broken stream")]), messages, harness)
        self.assertEqual(len(harness.executed), 1)
        self.assertEqual(json.loads(messages[-1]["content"])["status"], "ok")
        self.assert_saveable(messages)

    def test_creation_request_forces_only_first_tool_request_and_preserves_runtime_boundary(self):
        source = 'Console.WriteLine("Hello, World!");\n'
        arguments = json.dumps({"path": "src/Program.cs", "content": source})
        client = Client(tool_round(call(name="write_file", arguments=arguments)), final_round())
        harness = self.code_harness(lambda name, raw: '{"status":"ok","path":"src/Program.cs","directories_created":["src"]}')
        messages = [{"role": "system", "content": "Обычная инструкция"},
                    {"role": "user", "content": "Напиши программу HelloWorld на C#"}]
        events = []
        result = generate_response(client, messages, harness, events.append)
        self.assertEqual(result.answer, "Готово.")
        self.assertEqual(client.choices, ["required", None])
        runtime = client.requests[0][0][1]
        self.assertEqual(runtime["role"], "system")
        self.assertIn("CodeRepo", runtime["content"])
        self.assertIn("relative", runtime["content"])
        self.assertNotIn(runtime, messages)
        self.assertEqual([message for message in client.requests[1][0] if message != runtime], messages)
        self.assert_saveable(messages)
        self.assertEqual(self.progress(events), {"requested": 1, "succeeded": 1, "denied": 0, "failed": 0,
                         "files_written": ["src/Program.cs"], "directories_created": ["src"],
                         "no_changes": False, "uncertain_changes": False, "scripts_run": []})
        self.assertEqual(next(event.value for event in events if event.kind == "tool_start"),
                         {"id": "call-1", "name": "write_file", "arguments": {"path": "src/Program.cs", "content": source}})
        self.assertEqual(next(event.value for event in events if event.kind == "tool_result")["status"], "ok")

    def test_creation_intent_is_conservative_for_russian_english_examples_and_fences(self):
        for prompt in ("Напиши пример программы", "Объясни, как создать проект", "Show an example program",
                       "Write a sample file", "Создай пример проекта", "Напиши программу\n```csharp\nConsole.WriteLine();\n```",
                       "Напиши программу, но не создавай файлов", "Напиши программу только в чате",
                       "Write a script without saving", "Create a program, do not create files"):
            with self.subTest(prompt=prompt):
                client = Client(final_round("Пример в чате."))
                generate_response(client, [{"role": "user", "content": prompt}], self.code_harness())
                self.assertEqual(client.choices, [None])
        for prompt in ("Create a C# project", "Write a script", "Создай файл Program.cs", "Напиши HelloWorld на C#", "Запиши main.py", "Сохрани файл Program.cs"):
            with self.subTest(prompt=prompt):
                client = Client(final_round("Created."), final_round("Created."))
                result = generate_response(client, [{"role": "user", "content": prompt}], self.code_harness())
                self.assertEqual(client.choices, ["required", "required"])
                self.assertIn("Файлы не созданы", result.answer)

    def test_false_creation_claim_gets_one_bounded_correction_and_clear_no_changes(self):
        client = Client(final_round("Файл создан!"), final_round("Готово, всё записано!"))
        messages = [{"role": "user", "content": "Создай скрипт Python"}]
        events = []
        result = generate_response(client, messages, self.code_harness(), events.append)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.choices, ["required", "required"])
        self.assertIn("Изменений в репозитории нет", result.answer)
        self.assertIn("retrying", [event.value["phase"] for event in events if event.kind == "harness_status"])
        self.assertIn("corrective attempt", client.requests[1][0][0]["content"])
        self.assertEqual(messages, [{"role": "user", "content": "Создай скрипт Python"}])
        self.assertTrue(self.progress(events)["no_changes"])

    def test_read_then_text_can_correct_once_before_mutation(self):
        arguments = '{"path":"Program.cs","content":"Console.WriteLine(1);"}'
        client = Client(tool_round(call()), final_round("Сделал"),
                        tool_round(call("write-1", "write_file", arguments)), final_round())
        events = []
        messages = [{"role": "user", "content": "Создай программу на C#"}]
        generate_response(client, messages, self.code_harness(), events.append)
        self.assertEqual(client.choices, ["required", None, "required", None])
        self.assertEqual(self.progress(events)["requested"], 2)
        self.assertEqual(self.progress(events)["files_written"], ["Program.cs"])
        self.assert_saveable(messages)

    def test_any_denial_or_error_suppresses_corrective_force_and_false_claim(self):
        for name in ("read_file", "write_file"):
            for status in ("denied", "error"):
                with self.subTest(name=name, status=status):
                    harness = self.code_harness(lambda name, raw: json.dumps({"status": status, "message": "Не выполнено"}))
                    arguments = '{"path":"Program.cs","content":"x"}' if name == "write_file" else '{"path":"Program.cs"}'
                    client = Client(tool_round(call(name=name, arguments=arguments)), final_round("Файл создан!"))
                    events = []
                    result = generate_response(client, [{"role": "user", "content": "Создай файл Program.cs"}], harness, events.append)
                    self.assertEqual(client.choices, ["required", None])
                    self.assertEqual(len(harness.executed), 1)
                    self.assertNotIn("Файл создан!", result.answer)
                    progress = self.progress(events)
                    self.assertEqual(progress["denied" if status == "denied" else "failed"], 1)
                    self.assertEqual(progress["uncertain_changes"], name == "write_file" and status == "error")
                    self.assertFalse(any(event.kind == "harness_status" and event.value["phase"] == "retrying" for event in events))

    def test_directory_results_are_summarized_without_claiming_file_writes(self):
        client = Client(tool_round(call(name="create_directory", arguments='{"path":"src"}')), final_round("Создана папка"))
        harness = self.code_harness(lambda name, raw: '{"status":"ok","path":"src","created":true,"directories_created":["src"]}')
        events = []
        result = generate_response(client, [{"role": "user", "content": "Создай проект"}], harness, events.append)
        self.assertIn("Файлы не записаны", result.answer)
        progress = self.progress(events)
        self.assertEqual(progress["directories_created"], ["src"])
        self.assertEqual(progress["files_written"], [])
        self.assertFalse(progress["no_changes"])

    def test_interrupted_mutation_reports_uncertainty_and_remaining_placeholders(self):
        def interrupted(name, raw):
            raise KeyboardInterrupt
        events = []
        messages = [{"role": "user", "content": "Создай программу"}]
        client = Client(tool_round(call(name="write_file", arguments='{"path":"main.py","content":"x"}'),
                                  call("second", "read_file")))
        with self.assertRaises(KeyboardInterrupt):
            generate_response(client, messages, self.code_harness(interrupted), events.append)
        progress = self.progress(events)
        self.assertEqual(progress["requested"], 2)
        self.assertEqual(progress["failed"], 2)
        self.assertTrue(progress["uncertain_changes"])
        self.assertFalse(progress["no_changes"])
        self.assertEqual(len(client.requests), 1)
        self.assert_saveable(messages)

    def test_malformed_harness_result_is_error_and_never_confirms_files(self):
        events = []
        client = Client(tool_round(call(name="write_file", arguments='{"path":"main.py","content":"x"}')), final_round())
        messages = [{"role": "user", "content": "Создай программу"}]
        generate_response(client, messages, self.code_harness(lambda name, raw: '{"status":[]}'), events.append)
        self.assertEqual(self.progress(events)["failed"], 1)
        self.assertEqual(self.progress(events)["files_written"], [])
        self.assertEqual(json.loads(messages[-1]["content"])["status"], "error")

    def test_powershell_execution_is_required_initially_and_summarizes_only_success(self):
        harness = Harness(lambda name, raw: '{"status":"ok","path":"hello.ps1","stdout":"Hello","stderr":"","exit_code":0}',
                          ["powershell_run"], SimpleNamespace(code_enabled=False, web_enabled=False,
                                                            powershell_enabled=True, repository=Path("CodeRepo")))
        client = Client(tool_round(call(name="powershell_run", arguments='{"path":"hello.ps1"}')), final_round("Hello"))
        events = []
        messages = [{"role": "user", "content": "Запусти hello.ps1"}]
        result = generate_response(client, messages, harness, events.append)
        self.assertEqual(client.choices, ["required", None])
        self.assertEqual(result.answer, "Hello")
        self.assertEqual(self.progress(events)["scripts_run"], ["hello.ps1"])
        self.assertTrue(self.progress(events)["no_changes"])
        self.assertTrue(next(event.value for event in events if event.kind == "harness_status")["powershell_enabled"])
        self.assertIn("C# compilation/execution are unavailable", client.requests[0][0][0]["content"])
        self.assert_saveable(messages)

    def test_powershell_text_only_execution_gets_one_correction_and_clear_failure(self):
        harness = Harness(names=["powershell_run"], settings=SimpleNamespace(code_enabled=False, web_enabled=False,
                           powershell_enabled=True, repository=Path("CodeRepo")))
        client = Client(final_round("Ran it!"), final_round("Ran it!"))
        result = generate_response(client, [{"role": "user", "content": "Run hello.ps1"}], harness)
        self.assertEqual(client.choices, ["required", "required"])
        self.assertIn("Скрипт не запущен", result.answer)
        self.assertEqual(harness.executed, [])

    def test_powershell_denied_or_failed_execution_never_forces_retry_or_claims_success(self):
        for value in ({"status": "denied"}, {"status": "error", "execution_started": False},
                      {"status": "ok", "exit_code": 1, "stdout": "failed"}):
            with self.subTest(value=value):
                harness = Harness(lambda name, raw: json.dumps(value), ["powershell_run"],
                                  SimpleNamespace(code_enabled=False, web_enabled=False, powershell_enabled=True,
                                                  repository=Path("CodeRepo")))
                client = Client(tool_round(call(name="powershell_run", arguments='{"path":"hello.ps1"}')), final_round("Ran it!"))
                events = []
                result = generate_response(client, [{"role": "user", "content": "Run hello.ps1"}], harness, events.append)
                self.assertEqual(client.choices, ["required", None])
                self.assertIn("не подтверждено", result.answer)
                self.assertEqual(self.progress(events)["scripts_run"], [])
                self.assertEqual(self.progress(events)["succeeded"], 0)


if __name__ == "__main__":
    unittest.main()
