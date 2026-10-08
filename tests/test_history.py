from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from llmopenchat.history import MAX_SESSION_BYTES, list_sessions, load_session, new_session_path, save_session
from llmopenchat.api import MAX_TOOL_ARGUMENT_BYTES


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-history-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.messages = [
            {"role": "system", "content": "Системная инструкция."},
            {"role": "user", "content": "Как настроить\n  сервер, ёж?"},
            {"role": "assistant", "content": "Помогу."},
        ]

    def write(self, name, payload):
        path = self.directory / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def tool_call(self, call_id="call_read", name="repository_read", arguments='{"path":"README.md"}'):
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}

    def tool_round(self):
        return [
            {"role": "user", "content": "Прочитай README.md"},
            {"role": "assistant", "content": "", "tool_calls": [self.tool_call()]},
            {"role": "tool", "tool_call_id": "call_read", "content": '{"text":"Описание проекта"}'},
            {"role": "assistant", "content": "Это консольный чат."},
        ]

    def test_list_orders_saved_timestamps_and_exposes_model_metadata(self):
        older = self.write("old.json", {"messages": self.messages, "saved_at": "2026-10-07T09:00:00+05:00", "model": "glm-local"})
        newer = self.write("new.json", {
            "messages": self.messages, "saved_at": "2026-10-07T05:00:00Z",
            "model": "glm-local", "target_model": "huihui_ai/glm-4.7-flash-abliterated", "model_id": "flash-uncensored-q4",
        })
        # Saved timestamps, not modification dates or filename order, decide recency.
        os.utime(older, (1900000000, 1900000000))
        os.utime(newer, (1600000000, 1600000000))
        entries = list_sessions(self.directory)
        self.assertEqual([entry.path for entry in entries], [newer, older])
        self.assertEqual(entries[0].saved_at, datetime(2026, 10, 7, 5, tzinfo=timezone.utc))
        self.assertEqual(entries[0].title, "Как настроить сервер, ёж?")
        self.assertEqual(entries[0].message_count, 3)
        self.assertEqual(entries[0].model, "glm-local")
        self.assertEqual(entries[0].model_id, "flash-uncensored-q4")
        self.assertEqual(entries[0].display_model, "huihui_ai/glm-4.7-flash-abliterated")

    def test_legacy_arrays_and_invalid_dates_use_file_timestamp(self):
        legacy = self.write("legacy.json", self.messages)
        invalid_date = self.write("invalid-date.json", {"messages": [], "saved_at": "invalid", "model": 7})
        os.utime(legacy, (1700000000, 1700000000))
        os.utime(invalid_date, (1600000000, 1600000000))
        entries = list_sessions(self.directory)
        self.assertEqual([entry.path for entry in entries], [legacy, invalid_date])
        self.assertEqual(entries[0].saved_at.timestamp(), 1700000000)
        self.assertEqual(entries[1].title, "Новый диалог")
        self.assertEqual(entries[1].display_model, "Неизвестная модель")
        self.assertEqual(load_session(legacy), self.messages)

    def test_corrupt_invalid_utf8_and_oversized_files_are_reported_and_skipped(self):
        self.write("valid.json", {"messages": self.messages})
        (self.directory / "corrupt.json").write_text("{", encoding="utf-8")
        (self.directory / "encoding.json").write_bytes(b"\xff\xfe")
        oversized = self.directory / "oversized.json"
        with oversized.open("wb") as destination:
            destination.truncate(MAX_SESSION_BYTES + 1)
        self.write("invalid.json", {"messages": [{"role": "tool", "content": "unsupported"}]})
        self.write("unrelated.txt", {})
        (self.directory / "folder.json").mkdir()
        reports = []
        entries = list_sessions(self.directory, emit=reports.append)
        self.assertEqual([entry.path.name for entry in entries], ["valid.json"])
        self.assertEqual(len(reports), 4)
        self.assertTrue(any("oversized.json" in report and "16 МБ" in report for report in reports))
        with self.assertRaisesRegex(ValueError, "16 МБ"):
            load_session(oversized)

    def test_load_keeps_validation_and_drops_extra_message_fields(self):
        valid = self.write("bom.json", {"messages": [{"role": "user", "content": "Привет", "name": "untrusted", "tool_calls": []}]})
        valid.write_text("\ufeff" + valid.read_text(encoding="utf-8"), encoding="utf-8")
        self.assertEqual(load_session(valid), [{"role": "user", "content": "Привет"}])
        for payload in (None, {}, {"messages": {}}, [{"role": "tool", "content": "x"}], [{"role": [], "content": "x"}], [{"role": "user", "content": 7}], [None]):
            with self.subTest(payload=payload):
                path = self.write("invalid.json", payload)
                with self.assertRaisesRegex(ValueError, "Неверный формат диалога"):
                    load_session(path)

    def test_tool_rounds_survive_save_and_load_without_replaying_or_restoring_permissions(self):
        messages = self.tool_round()
        messages += [
            {"role": "user", "content": "Теперь исправь код"},
            {"role": "assistant", "content": "", "tool_calls": [self.tool_call("call_write", "repository_write", '{"path":"main.py","content":"print(1)"}')]},
            {"role": "tool", "tool_call_id": "call_write", "content": '{"error":"Пользователь отклонил вызов"}'},
            {"role": "assistant", "content": "Вызов отклонён."},
        ]
        path = self.directory / "tools.json"
        save_session(path, messages, "glm-local")
        self.assertEqual(load_session(path), messages)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(harness={"web": True, "coding": True, "repository": "C:/", "approved": True})
        payload["messages"][1]["approved"] = True
        payload["messages"][1]["tool_calls"][0]["execute"] = True
        payload["messages"][1]["tool_calls"][0]["function"]["permission"] = "always"
        payload["messages"][2]["name"] = "untrusted"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(load_session(path), messages)
        self.assertEqual(list_sessions(self.directory)[0].message_count, len(messages))

    def test_multi_call_round_accepts_results_in_either_order_and_normalizes_null_content(self):
        calls = [self.tool_call("first"), self.tool_call("second")]
        messages = [
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "second", "content": "Второй результат"},
            {"role": "tool", "tool_call_id": "first", "content": "Первый результат"},
        ]
        path = self.write("multi.json", messages)
        self.assertEqual(load_session(path), [{**messages[0], "content": ""}, *messages[1:]])
        # Old text-only snapshots sometimes include an empty tool_calls field.
        path = self.write("legacy-empty.json", [{"role": "assistant", "content": "Ответ", "tool_calls": []}])
        self.assertEqual(load_session(path), [{"role": "assistant", "content": "Ответ"}])

    def test_incomplete_mismatched_duplicate_or_out_of_order_rounds_are_rejected(self):
        round_messages = self.tool_round()
        invalid = [
            round_messages[:2],
            [round_messages[2]],
            [round_messages[1], round_messages[3], round_messages[2]],
            [round_messages[1], {**round_messages[2], "tool_call_id": "unknown"}],
            [round_messages[1], round_messages[2], round_messages[2]],
            [round_messages[1], {**round_messages[2], "content": None}],
            [round_messages[1], {"role": "tool", "tool_call_id": [], "content": "x"}],
            [round_messages[1], round_messages[2], *round_messages[1:]],
            [{"role": "assistant", "content": "", "tool_calls": [self.tool_call(), self.tool_call()]}, round_messages[2]],
            [{"role": "assistant", "content": "", "tool_calls": [self.tool_call("first"), self.tool_call("second")]}, {"role": "tool", "tool_call_id": "first", "content": "x"}],
        ]
        for messages in invalid:
            with self.subTest(messages=messages):
                path = self.write("invalid-tools.json", messages)
                with self.assertRaisesRegex(ValueError, "Неверный формат диалога"):
                    load_session(path)

    def test_malformed_or_oversized_tool_calls_are_rejected(self):
        call = self.tool_call()
        invalid = [
            [{**call, "type": "shell"}],
            [{**call, "id": ""}],
            [{**call, "id": "invalid id"}],
            [{**call, "function": []}],
            [self.tool_call(name="name.with.dot")],
            [self.tool_call(arguments={"path": "README.md"})],
            [self.tool_call(arguments="{broken")],
            [self.tool_call(arguments="[]")],
            [self.tool_call(arguments='{"x":NaN}')],
            [self.tool_call(arguments='{"x":"' + "x" * MAX_TOOL_ARGUMENT_BYTES + '"}')],
            [self.tool_call(f"call_{index}") for index in range(17)],
            [None],
            "invalid",
        ]
        for calls in invalid:
            with self.subTest(calls=str(calls)[:100]):
                path = self.write("invalid-tools.json", [
                    {"role": "assistant", "content": "", "tool_calls": calls},
                    {"role": "tool", "tool_call_id": "call_read", "content": "x"},
                ])
                with self.assertRaisesRegex(ValueError, "Неверный формат диалога"):
                    load_session(path)

    def test_incomplete_tool_save_preserves_previous_snapshot(self):
        path = self.write("saved.json", self.messages)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "отсутствует результат"):
            save_session(path, self.tool_round()[:2], "glm-local")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_atomic_save_keeps_metadata_and_concurrent_writers_leave_no_temporary_files(self):
        path = self.directory / "новая папка" / "диалог.json"
        save_session(path, self.messages, "glm-local", target_model="GLM-4.7-Flash", model_id="flash-q4")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["target_model"], "GLM-4.7-Flash")
        self.assertEqual(payload["model_id"], "flash-q4")
        self.assertIsNotNone(datetime.fromisoformat(payload["saved_at"]).tzinfo)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(save_session, path, [{"role": "user", "content": str(index)}], "glm-local") for index in range(12)]
            for future in futures:
                future.result()
        self.assertIn(load_session(path)[0]["content"], {str(index) for index in range(12)})
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_replace_preserves_snapshot_and_removes_owned_temporary_file(self):
        path = self.write("saved.json", {"messages": self.messages})
        original = path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("failed replace")):
            with self.assertRaisesRegex(OSError, "failed replace"):
                save_session(path, [{"role": "user", "content": "Новое"}], "glm-local")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_oversized_save_preserves_previous_snapshot(self):
        path = self.write("saved.json", {"messages": self.messages})
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "16 МБ"):
            save_session(path, [{"role": "user", "content": "x" * MAX_SESSION_BYTES}], "glm-local")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_default_directory_and_fresh_resume_paths(self):
        sessions = self.directory / ".local" / "sessions"
        sessions.mkdir(parents=True)
        save_session(sessions / "seed.json", self.messages, "glm-local")
        with patch("llmopenchat.history.ROOT", self.directory):
            self.assertEqual(list_sessions()[0].path, sessions / "seed.json")
        paths = [new_session_path(self.directory) for _ in range(20)]
        self.assertEqual(len(set(paths)), 20)
        self.assertTrue(all(path.parent == sessions and path != sessions / "seed.json" for path in paths))
        self.assertEqual(list_sessions(self.directory / "missing"), [])

    def test_long_titles_are_shortened_and_control_characters_removed(self):
        self.write("long.json", [{"role": "user", "content": "\x1b\u202e" + "А" * 100}])
        title = list_sessions(self.directory)[0].title
        self.assertEqual(title, "А" * 79 + "…")


if __name__ == "__main__":
    unittest.main()
