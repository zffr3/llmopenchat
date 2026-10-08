import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from llmopenchat import gguf_compat as compat


def string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def scalar(kind, value):
    return compat._Value(kind, struct.pack(compat._FORMATS[kind], value))


def text(value):
    return compat._Value(8, string(value))


def array(kind, items):
    values = [string(item) if kind == 8 else struct.pack(compat._FORMATS[kind], item) for item in items]
    return compat._Value(9, struct.pack("<IQ", kind, len(items)) + b"".join(values))


def make_source(path, changes=None, alignment=64):
    metadata = {
        "general.architecture": text("glm4moelite"),
        "general.alignment": scalar(4, alignment),
        "glm4moelite.attention.head_count": scalar(4, 20),
        "glm4moelite.attention.head_count_kv": scalar(4, 20),
        "glm4moelite.attention.key_length": scalar(4, 256),
        "glm4moelite.attention.key_length_mla": scalar(4, 576),
        "glm4moelite.attention.value_length_mla": scalar(4, 512),
        "glm4moelite.attention.kv_lora_rank": scalar(4, 512),
        "glm4moelite.attention.q_lora_rank": scalar(4, 768),
        "glm4moelite.rope.dimension_count": scalar(4, 64),
        "glm4moelite.block_count": scalar(4, 47),
        "glm4moelite.expert_weights_scale": scalar(6, 1.8),
        "glm4moelite.expert_weights_norm": scalar(7, True),
        "tokenizer.ggml.pre": text("chatglm-bpe"),
        "tokenizer.ggml.eos_token_id": scalar(4, 154820),
        "tokenizer.ggml.eos_token_ids": array(4, [154820, 154827, 154829]),
        "tokenizer.ggml.tokens": array(8, ["", "слово", "<|user|>"]),
        "vendor.opaque": compat._Value(9, struct.pack("<IQ", 9, 2)
            + struct.pack("<IQ", 3, 2) + struct.pack("<hh", -17, 42)
            + struct.pack("<IQ", 8, 2) + string("α") + string("z")),
        "vendor.signed64": scalar(11, -9223372036854775807),
    }
    for key, value in (changes or {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    tensor_info = (string("token_embd.weight") + struct.pack("<IQQIQ", 2, 32, 2, 0, 0)
                   + string("blk.0.attn_k_b.weight") + struct.pack("<IQQQIQ", 3, 8, 4, 20, 12, 128))
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 2, len(metadata)))
    for key, value in metadata.items():
        header.extend(string(key) + struct.pack("<I", value.kind) + value.raw)
    header.extend(tensor_info)
    header.extend(b"\xbb" * (-len(header) % alignment))
    payload = bytes(range(256)) * 2 + b"exact tensor bytes\0\xff"
    path.write_bytes(header + payload)
    return metadata, tensor_info, bytes(payload), len(header)


class GGUFCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "original.gguf"
        self.destination = self.root / "original.llamacpp.gguf"
        self.messages = []

    def convert(self):
        return compat.convert_glm4moelite(self.source, self.destination, self.messages.append)

    def header(self, path):
        with path.open("rb") as stream:
            return compat._read_header(stream, path.stat().st_size)

    def test_conversion_preserves_tensors_alignment_and_raw_unknown_values(self):
        before, tensor_info, payload, old_offset = make_source(self.source)
        original = self.source.read_bytes()
        result = self.convert()
        after = self.header(self.destination)
        self.assertEqual(after["tensor_info"], tensor_info)
        self.assertEqual(self.destination.read_bytes()[after["data_offset"]:], payload)
        self.assertEqual(after["data_offset"] % 64, 0)
        self.assertNotEqual(after["data_offset"], old_offset)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(after["tensor_count"], 2)
        for key, value in before.items():
            if key not in ("general.architecture", "tokenizer.ggml.pre"):
                self.assertEqual(after["metadata"][key], value)
        self.assertEqual(result["sha256"], hashlib.sha256(self.destination.read_bytes()).hexdigest())
        self.assertEqual(result["tensor_payload_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertTrue(result["tensor_payload_unchanged"])
        self.assertEqual(json.loads(self.destination.with_suffix(".gguf.json").read_text(encoding="utf-8")), result)

    def test_exact_official_mla_and_tokenizer_metadata(self):
        make_source(self.source)
        self.convert()
        values = self.header(self.destination)["metadata"]
        self.assertEqual(values["general.architecture"].string(), "deepseek2")
        self.assertEqual(values["tokenizer.ggml.pre"].string(), "glm4")
        expected = {
            "deepseek2.attention.head_count": 20,
            "deepseek2.attention.head_count_kv": 1,
            "deepseek2.attention.key_length": 576,
            "deepseek2.attention.value_length": 512,
            "deepseek2.attention.key_length_mla": 256,
            "deepseek2.attention.value_length_mla": 256,
            "deepseek2.attention.q_lora_rank": 768,
            "deepseek2.expert_group_count": 1,
            "deepseek2.expert_group_used_count": 1,
            "tokenizer.ggml.eot_token_id": 154827,
            "tokenizer.ggml.eom_token_id": 154829,
        }
        for key, value in expected.items():
            self.assertEqual(values[key].kind, 4, key)
            self.assertEqual(values[key].u32(), value, key)
        self.assertEqual(values["deepseek2.expert_weights_scale"], values["glm4moelite.expert_weights_scale"])
        self.assertEqual(values["deepseek2.expert_weights_norm"], values["glm4moelite.expert_weights_norm"])

    def test_prefix_copy_preserves_existing_targets_except_explicit_overrides(self):
        make_source(self.source, {
            "deepseek2.block_count": scalar(4, 99),
            "deepseek2.expert_group_count": scalar(4, 8),
            "deepseek2.attention.key_length": scalar(4, 99),
            "deepseek2.attention.kv_lora_rank": scalar(4, 256),
        })
        self.convert()
        values = self.header(self.destination)["metadata"]
        self.assertEqual(values["deepseek2.block_count"].u32(), 99)
        self.assertEqual(values["deepseek2.expert_group_count"].u32(), 8)
        self.assertEqual(values["deepseek2.attention.key_length"].u32(), 320)
        self.assertEqual(values["deepseek2.attention.value_length"].u32(), 256)

    def test_signed_eos_negative_ids_are_filtered_before_eog_selection(self):
        make_source(self.source, {"tokenizer.ggml.eos_token_ids": array(5, [-1, 154820, -2, 154827, 154829])})
        self.convert()
        values = self.header(self.destination)["metadata"]
        self.assertEqual(values["tokenizer.ggml.eot_token_id"].u32(), 154827)
        self.assertEqual(values["tokenizer.ggml.eom_token_id"].u32(), 154829)

    def test_existing_eog_ids_are_preserved_and_unsupported_arrays_ignored(self):
        for eos in (array(4, [1, 2, 3]), array(10, [1, 2, 3])):
            with self.subTest(eos=eos.kind):
                values, _, _, _ = make_source(self.source, {
                    "tokenizer.ggml.eos_token_ids": eos,
                    "tokenizer.ggml.eot_token_id": scalar(4, 7),
                    "tokenizer.ggml.eom_token_id": scalar(4, 9),
                })
                converted = compat._translate_metadata(values)
                self.assertEqual(converted["tokenizer.ggml.eot_token_id"].u32(), 7)
                self.assertEqual(converted["tokenizer.ggml.eom_token_id"].u32(), 9)
        values["tokenizer.ggml.eot_token_id"] = values["tokenizer.ggml.eom_token_id"] = None
        del values["tokenizer.ggml.eot_token_id"], values["tokenizer.ggml.eom_token_id"]
        converted = compat._translate_metadata(values)
        self.assertNotIn("tokenizer.ggml.eot_token_id", converted)
        self.assertNotIn("tokenizer.ggml.eom_token_id", converted)

    def test_verified_existing_conversion_skips_copy(self):
        make_source(self.source)
        first = self.convert()
        with mock.patch.object(compat.os, "fsync", side_effect=AssertionError("must not copy")):
            second = self.convert()
        self.assertEqual(first, second)
        self.assertFalse(self.destination.with_suffix(".gguf.converting").exists())

    def test_existing_output_tampering_is_rejected_and_preserved(self):
        make_source(self.source)
        self.convert()
        with self.destination.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x17")
        tampered = self.destination.read_bytes()
        with self.assertRaisesRegex(compat.CompatibilityError, "не совпали"):
            self.convert()
        self.assertEqual(self.destination.read_bytes(), tampered)

    def test_sidecar_identity_mismatch_is_rejected(self):
        make_source(self.source)
        self.convert()
        sidecar = self.destination.with_suffix(".gguf.json")
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        record["source_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(compat.CompatibilityError):
            self.convert()
        self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8"))["source_sha256"], "0" * 64)

    def test_invalid_architecture_type_version_and_alignment_are_rejected(self):
        cases = [
            {"general.architecture": text("glm4moe")},
            {"glm4moelite.attention.kv_lora_rank": scalar(5, 512)},
            {"general.alignment": scalar(4, 48)},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                make_source(self.source, changes)
                with self.assertRaises(compat.CompatibilityError):
                    self.convert()
                self.assertFalse(self.destination.exists())
        make_source(self.source)
        with self.source.open("r+b") as stream:
            stream.seek(4)
            stream.write(struct.pack("<I", 2))
        with self.assertRaises(compat.CompatibilityError):
            self.convert()

    def test_existing_staging_file_is_never_overwritten_or_removed(self):
        make_source(self.source)
        staging = self.destination.with_suffix(".gguf.converting")
        staging.write_bytes(b"user data")
        with self.assertRaisesRegex(compat.CompatibilityError, "Незавершенный"):
            self.convert()
        self.assertEqual(staging.read_bytes(), b"user data")

    def test_changed_source_during_copy_is_rejected_and_own_stage_removed(self):
        make_source(self.source)
        sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        def mutate_after_hash(path, emit):
            with path.open("ab") as stream:
                stream.write(b"changed")
            return sha
        with mock.patch.object(compat, "_sha256", side_effect=mutate_after_hash):
            with self.assertRaisesRegex(compat.CompatibilityError, "изменился"):
                self.convert()
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.destination.with_suffix(".gguf.converting").exists())

    def test_header_change_before_source_hash_is_rejected(self):
        make_source(self.source)
        real_hash = compat._sha256
        def mutate_before_hash(path, emit):
            data = path.read_bytes()
            self.assertIn(b"glm4moelite", data)
            path.write_bytes(data.replace(b"glm4moelite", b"bad4moelite", 1))
            return real_hash(path, emit)
        with mock.patch.object(compat, "_sha256", side_effect=mutate_before_hash):
            with self.assertRaisesRegex(compat.CompatibilityError, "Заголовок.*изменился"):
                self.convert()
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.destination.with_suffix(".gguf.converting").exists())

    def test_copy_failure_cleans_only_owned_staging_files(self):
        make_source(self.source)
        with mock.patch.object(compat.os, "fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.convert()
        self.assertTrue(self.source.is_file())
        self.assertEqual(list(self.root.iterdir()), [self.source])

    def test_pinned_template_sha_and_nonthinking_switch(self):
        path = Path(compat.__file__).parent / "templates" / "GLM-4.7-Flash.jinja"
        contents = path.read_bytes()
        self.assertEqual(hashlib.sha256(contents).hexdigest(), compat.TEMPLATE_SHA256)
        self.assertIn(b"enable_thinking is defined and not enable_thinking", contents)
        self.assertIn(compat.TEMPLATE_COMMIT, compat.TEMPLATE_SOURCE)
        self.assertIn(compat.COMPAT_COMMIT, compat.COMPAT_SOURCE)


if __name__ == "__main__":
    unittest.main()
