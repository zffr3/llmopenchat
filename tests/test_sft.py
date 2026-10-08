from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llmopenchat.history import save_session
from llmopenchat.sft import SftError, from_session, prepare_dataset, read_dataset, validate_dataset, write_template


class SftTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-sft-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def record(self, identifier="a", *, status="approved", question="Вопрос", answer="Ответ", group=None):
        value = {"schema_version": 1, "id": identifier, "task": "chat", "status": status,
                 "messages": [{"role": "system", "content": "Отвечай по-русски."},
                              {"role": "user", "content": question},
                              {"role": "assistant", "content": answer}]}
        if group is not None:
            value["group_id"] = group
        return value

    def write(self, records, *, array=False, filename="dataset.jsonl"):
        path = self.root / filename
        text = json.dumps(records, ensure_ascii=False) if array else "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in records)
        path.write_text(text, encoding="utf-8")
        return path

    def prepare(self, records, **options):
        source = self.write(records)
        job_path = prepare_dataset(source, self.root / "job", "Qwen/Qwen3-0.6B", **options)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        return job, self.rows("train.jsonl"), self.rows("validation.jsonl")

    def rows(self, name, directory="job"):
        return [json.loads(line) for line in (self.root / directory / name).read_text(encoding="utf-8").splitlines()]

    def test_jsonl_array_bom_and_status_default_are_equivalent(self):
        record = self.record()
        del record["status"]
        jsonl = self.write([record])
        jsonl.write_text("\ufeff\n" + jsonl.read_text(encoding="utf-8"), encoding="utf-8")
        array = self.write([record], array=True, filename="array.json")
        self.assertEqual(read_dataset(jsonl), read_dataset(array))
        self.assertEqual(read_dataset(jsonl)[0]["status"], "draft")
        stats = validate_dataset(jsonl)
        self.assertEqual(stats["draft"], 1)
        self.assertEqual(stats["exportable_targets"], 0)

    def test_only_approved_records_are_exported_and_input_is_unchanged(self):
        records = [self.record("approved"), self.record("draft", status="draft", answer="Сомнительно"),
                   self.record("rejected", status="rejected", answer="Неправильно")]
        source = self.write(records)
        original = source.read_bytes()
        path = prepare_dataset(source, self.root / "job", "Qwen/Qwen3-0.6B")
        job = json.loads(path.read_text(encoding="utf-8"))
        rows = self.rows("train.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["completion"], [{"role": "assistant", "content": "Ответ"}])
        self.assertEqual(job["stats"]["approved"], 1)
        self.assertEqual(job["stats"]["skipped"], 2)
        self.assertEqual(source.read_bytes(), original)

    def test_each_target_is_separate_and_untrained_assistants_remain_context(self):
        record = self.record()
        record["messages"][2]["train"] = False
        record["messages"].extend([
            {"role": "user", "content": "Уточни"},
            {"role": "assistant", "content": "Исправленный ответ", "train": True},
            {"role": "user", "content": "Продолжи"},
            {"role": "assistant", "content": "Продолжение"}])
        job, rows, validation = self.prepare([record])
        self.assertEqual(len(rows), 2)
        self.assertEqual(validation, [])
        self.assertEqual(rows[0]["completion"], [{"role": "assistant", "content": "Исправленный ответ"}])
        self.assertEqual(rows[0]["prompt"][2], {"role": "assistant", "content": "Ответ"})
        self.assertEqual(rows[1]["prompt"][4]["content"], "Исправленный ответ")
        self.assertNotIn("train", json.dumps(rows))
        self.assertEqual(job["validation_file"], None)
        self.assertTrue(job["stats"]["warnings"])

    def test_all_targets_from_one_conversation_stay_together(self):
        record = self.record("multi", question="multi 1")
        record["messages"].extend([{"role": "user", "content": "multi 2"},
                                   {"role": "assistant", "content": "Ответ 2"}])
        _, train, validation = self.prepare([record, self.record("other", question="other")], validation_ratio=0.5)
        for split in (train, validation):
            multi = [row for row in split if row["prompt"][1]["content"] == "multi 1"]
            self.assertIn(len(multi), (0, 2))
        self.assertTrue(train)
        self.assertTrue(validation)

    def test_explicit_groups_and_identical_prompts_are_joined_transitively(self):
        records = [self.record("a", question="shared", answer="A", group="g"),
                   self.record("b", question="different", answer="B", group="g"),
                   self.record("c", question="shared", answer="C", group="h"),
                   self.record("d", question="last", answer="D", group="h"),
                   self.record("e", question="independent", answer="E")]
        job, train, validation = self.prepare(records, validation_ratio=0.5)
        self.assertEqual(job["stats"]["groups"], 2)
        for split in (train, validation):
            answers = {row["completion"][0]["content"] for row in split}
            self.assertTrue({"A", "B", "C", "D"}.issubset(answers) or not answers & {"A", "B", "C", "D"})
        train_prompts = {json.dumps(row["prompt"], sort_keys=True) for row in train}
        validation_prompts = {json.dumps(row["prompt"], sort_keys=True) for row in validation}
        self.assertFalse(train_prompts & validation_prompts)

    def test_identical_targets_are_deduplicated_without_losing_groups(self):
        records = [self.record("a", group="first"), self.record("b", group="second"),
                   self.record("c", question="Еще", answer="Еще ответ", group="second")]
        job, train, validation = self.prepare(records)
        self.assertEqual(len(train), 2)
        self.assertEqual(validation, [])
        self.assertEqual(job["stats"]["duplicate_targets"], 1)
        self.assertEqual(job["stats"]["groups"], 1)

    def test_split_is_reproducible_and_files_have_integrity_hashes(self):
        source = self.write([self.record(str(index), question=str(index)) for index in range(10)])
        first = prepare_dataset(source, self.root / "one", "Qwen/Qwen3-0.6B", seed=17, validation_ratio=0.2)
        second = prepare_dataset(source, self.root / "two", "Qwen/Qwen3-0.6B", seed=17, validation_ratio=0.2)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        job = json.loads(first.read_text(encoding="utf-8"))
        self.assertEqual(job["stats"]["validation"], 2)
        for field, filename in (("train_sha256", "train.jsonl"), ("validation_sha256", "validation.jsonl")):
            self.assertEqual(job[field], hashlib.sha256((first.parent / filename).read_bytes()).hexdigest())

    def test_no_approved_targets_produces_no_job(self):
        for record in (self.record(status="draft"), self.record(status="rejected")):
            with self.subTest(status=record["status"]):
                source = self.write([record])
                with self.assertRaisesRegex(SftError, "Нет одобренных"):
                    prepare_dataset(source, self.root / "job", "Qwen/Qwen3-0.6B")
                self.assertFalse((self.root / "job").exists())

    def test_template_and_session_import_require_review_and_never_overwrite(self):
        template = write_template(self.root / "template.jsonl")
        self.assertEqual(read_dataset(template)[0]["status"], "draft")
        original = template.read_bytes()
        with self.assertRaises(SftError):
            write_template(template)
        self.assertEqual(template.read_bytes(), original)
        source = self.root / "session.json"
        save_session(source, self.record()["messages"], "local")
        output = from_session(source, self.root / "imported.jsonl", task="style")
        imported = read_dataset(output)[0]
        self.assertEqual(imported["status"], "draft")
        self.assertEqual(imported["task"], "style")
        self.assertTrue(imported["group_id"].startswith("session-"))
        with self.assertRaises(SftError):
            from_session(source, output)

    def test_import_rejects_tools_and_incomplete_conversations(self):
        tool_call = {"id": "c", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        for messages, expected in (([{"role": "user", "content": "Вопрос"}], "нет ответов"),
                                   ([{"role": "user", "content": "Вопрос"},
                                     {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                                     {"role": "tool", "tool_call_id": "c", "content": "ok"}], "инструментами")):
            path = self.root / "session.json"
            save_session(path, messages, "local")
            with self.assertRaisesRegex(SftError, expected):
                from_session(path, self.root / "imported.jsonl")
            self.assertFalse((self.root / "imported.jsonl").exists())

    def test_duplicate_ids_and_invalid_schema_are_rejected(self):
        invalid = [self.record(), self.record()]
        with self.assertRaisesRegex(SftError, "повторный id"):
            read_dataset(self.write(invalid))
        cases = [("schema_version", True), ("schema_version", 2), ("id", " "),
                 ("task", None), ("status", "correct"), ("messages", []), ("group_id", "")]
        for field, value in cases:
            record = self.record()
            record[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SftError):
                read_dataset(self.write([record]))

    def test_multimodal_tools_train_flags_and_empty_targets_are_rejected(self):
        messages = [{"role": "tool", "content": "ok"},
                    {"role": "assistant", "content": "ok", "tool_calls": []},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                    {"role": "assistant", "content": "ok", "train": 1},
                    {"role": "assistant", "content": " "},
                    {"role": "user", "content": "ok", "train": True}]
        for message in messages:
            record = self.record()
            record["messages"][-1] = message
            with self.subTest(message=message), self.assertRaises(SftError):
                read_dataset(self.write([record]))
        record = self.record()
        record["messages"] = [{"role": "assistant", "content": "Нет запроса"}]
        with self.assertRaisesRegex(SftError, "запроса user"):
            read_dataset(self.write([record]))
        record = self.record()
        record["messages"].append({"role": "assistant", "content": "Без нового запроса"})
        with self.assertRaisesRegex(SftError, "последнее сообщение"):
            read_dataset(self.write([record]))
        record = self.record()
        record["messages"][0]["content"] = " "
        with self.assertRaisesRegex(SftError, "пустым"):
            read_dataset(self.write([record]))
        record = self.record()
        record["messages"][-1]["train"] = False
        with self.assertRaisesRegex(SftError, "хотя бы одного"):
            read_dataset(self.write([record]))

    def test_bounded_reads_corrupt_json_and_nonfinite_json_are_rejected(self):
        source = self.write([self.record()])
        with patch("llmopenchat.sft.MAX_DATASET_BYTES", 10), self.assertRaisesRegex(SftError, "64 МиБ"):
            read_dataset(source)
        with patch("llmopenchat.sft.MAX_RECORD_BYTES", 10), self.assertRaisesRegex(SftError, "2 МиБ"):
            read_dataset(source)
        for text in ("", "{", "null", "[]", '{"schema_version":NaN}', '{"schema_version":1e400}', "[[[[[["):
            source.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(SftError):
                read_dataset(source)

    def test_unknown_quality_labels_cannot_be_silently_ignored(self):
        record = self.record()
        record["correct"] = False
        with self.assertRaisesRegex(SftError, "status"):
            read_dataset(self.write([record]))
        del record["correct"]
        record["metadata"] = {"source": "manual", "language": "ru"}
        self.assertEqual(read_dataset(self.write([record]))[0]["metadata"], record["metadata"])

    def test_session_import_applies_record_size_limit_before_writing(self):
        source = self.root / "session.json"
        save_session(source, self.record()["messages"], "local")
        with patch("llmopenchat.sft.MAX_RECORD_BYTES", 10), self.assertRaisesRegex(SftError, "2 МиБ"):
            from_session(source, self.root / "imported.jsonl")
        self.assertFalse((self.root / "imported.jsonl").exists())

    def test_invalid_training_options_fail_before_writing(self):
        source = self.write([self.record()])
        cases = [{"method": "full"}, {"validation_ratio": 1}, {"validation_ratio": float("nan")},
                 {"seed": True}, {"seed": -1}, {"max_length": 0}, {"max_length": 1.5},
                 {"epochs": 0}, {"epochs": float("inf")}, {"learning_rate": True},
                 {"learning_rate": -0.1}, {"lora_rank": 0}, {"lora_alpha": 0},
                 {"lora_dropout": 1}, {"revision": " "}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(SftError):
                prepare_dataset(source, self.root / "job", "Qwen/Qwen3-0.6B", **options)
            self.assertFalse((self.root / "job").exists())

    def test_explicit_transformers_model_is_required(self):
        source = self.write([self.record()])
        for model in ("model.gguf", "author/Model-GGUF", "", " guessed ", "model.safetensors", "https://host/model"):
            with self.subTest(model=model), self.assertRaises(SftError):
                prepare_dataset(source, self.root / "job", model)
        local = self.root / "checkpoint"
        local.mkdir()
        (local / "config.json").write_text("{}", encoding="utf-8")
        job = prepare_dataset(source, self.root / "job", str(local), method="lora", validation_ratio=0)
        self.assertEqual(json.loads(job.read_text(encoding="utf-8"))["base_model"], str(local))

    def test_output_must_be_new_or_empty_and_existing_content_is_preserved(self):
        source = self.write([self.record()])
        output = self.root / "job"
        output.mkdir()
        existing = output / "keep.txt"
        existing.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(SftError, "новым или пустым"):
            prepare_dataset(source, output, "Qwen/Qwen3-0.6B")
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
        self.assertEqual(list(output.iterdir()), [existing])
        existing.unlink()
        self.assertTrue(prepare_dataset(source, output, "Qwen/Qwen3-0.6B").is_file())
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(SftError):
            prepare_dataset(source, output, "Qwen/Qwen3-0.6B")
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_failure_during_job_creation_cleans_only_created_files(self):
        source = self.write([self.record()])
        from llmopenchat.sft import _write_new

        def fail_second_file(path, content):
            if path.name == "validation.jsonl":
                raise SftError("simulated write failure")
            return _write_new(path, content)

        with patch("llmopenchat.sft._write_new", side_effect=fail_second_file), self.assertRaises(SftError):
            prepare_dataset(source, self.root / "job", "Qwen/Qwen3-0.6B")
        self.assertFalse((self.root / "job").exists())
        self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
