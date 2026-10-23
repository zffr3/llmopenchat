from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmopenchat.config import ConfigError, DEFAULT_CONFIG, load_config, validate_config, write_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "config.json"

    def write_json(self, values):
        self.path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")

    def test_absent_config_returns_independent_defaults(self):
        loaded = load_config(self.path)
        loaded["server"]["threads"] = 1
        loaded["request_extra"]["chat_template_kwargs"]["enable_thinking"] = True
        loaded["telegram"]["admin_user_id"] = 123456789
        loaded["telegram"]["business_enabled"] = True
        self.assertEqual(load_config(self.path), DEFAULT_CONFIG)

    def test_partial_server_settings_keep_other_defaults(self):
        self.write_json({"temperature": 0.3, "server": {"context_size": 4096}})
        loaded = load_config(self.path)
        self.assertEqual(loaded["temperature"], 0.3)
        self.assertEqual(loaded["server"]["context_size"], 4096)
        self.assertEqual(loaded["server"]["model_path"], DEFAULT_CONFIG["server"]["model_path"])

    def test_empty_request_extra_removes_default_template_settings(self):
        self.write_json({"request_extra": {}})
        self.assertEqual(load_config(self.path)["request_extra"], {})

    def test_request_extra_is_replaced_whole(self):
        self.write_json({"request_extra": {"top_p": 0.8}})
        self.assertEqual(load_config(self.path)["request_extra"], {"top_p": 0.8})

    def test_vision_requires_explicit_boolean_and_bounded_settings(self):
        self.assertFalse(load_config(self.path)["vision"]["enabled"])
        for settings in ({"enabled": "true"}, {"enabled": 1}, {"max_pixels": 100_000_000},
                         {"max_input_bytes": 100_000_000}, {"max_side": 0},
                         {"max_image_bytes": 100_000_000}, {"max_history_images": 100}):
            with self.subTest(settings=settings):
                self.write_json({"vision": settings})
                with self.assertRaisesRegex(ConfigError, "vision"):
                    load_config(self.path)
        self.write_json({"vision": {"enabled": True, "max_history_images": 2}})
        config = load_config(self.path)
        self.assertTrue(config["vision"]["enabled"])
        self.assertEqual(config["vision"]["max_history_images"], 2)
        write_config(self.path, config)
        self.assertEqual(load_config(self.path), config)

    def test_telegram_legacy_defaults_and_partial_admin_settings(self):
        self.assertEqual(load_config(self.path)["telegram"], {
            "admin_user_id": None, "repository": None, "business_enabled": False,
        })
        self.write_json({"telegram": {"admin_user_id": 123456789}})
        self.assertEqual(load_config(self.path)["telegram"], {
            "admin_user_id": 123456789, "repository": None, "business_enabled": False,
        })

    def test_telegram_business_mode_requires_explicit_boolean_and_owner(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(business_enabled=value):
                self.write_json({"telegram": {
                    "admin_user_id": 123456789, "business_enabled": value,
                }})
                with self.assertRaisesRegex(ConfigError, "telegram.business_enabled"):
                    load_config(self.path)
        self.write_json({"telegram": {"business_enabled": True}})
        with self.assertRaisesRegex(ConfigError, "telegram.admin_user_id"):
            load_config(self.path)

    def test_telegram_business_mode_round_trips_with_owner_and_legacy_settings(self):
        self.write_json({"telegram": {
            "admin_user_id": 123456789,
            "business_enabled": True,
            "repository": "projects/shared",
        }})
        config = load_config(self.path)
        self.assertTrue(config["telegram"]["business_enabled"])
        write_config(self.path, config)
        self.assertEqual(load_config(self.path), config)

    def test_telegram_repository_preserves_absolute_or_relative_path(self):
        for repository in ("projects/shared", str(self.path.parent / "репозиторий")):
            with self.subTest(repository=repository):
                self.write_json({"telegram": {
                    "admin_user_id": 123456789, "repository": repository,
                }})
                self.assertEqual(load_config(self.path)["telegram"]["repository"], repository)

    def test_invalid_telegram_roles_and_repository_settings(self):
        invalid = [
            None, [], "admin", 123,
            {"admin_user_id": True}, {"admin_user_id": False},
            {"admin_user_id": 0}, {"admin_user_id": -1},
            {"admin_user_id": 1.5}, {"admin_user_id": "123456789"},
            {"repository": "projects/shared"},
            {"admin_user_id": 123456789, "repository": ""},
            {"admin_user_id": 123456789, "repository": "   "},
            {"admin_user_id": 123456789, "repository": "repo\x00hidden"},
            {"admin_user_id": 123456789, "repository": True},
            {"admin_user_id": 123456789, "repository": []},
        ]
        for telegram in invalid:
            with self.subTest(telegram=telegram):
                self.write_json({"telegram": telegram})
                with self.assertRaisesRegex(ConfigError, "telegram"):
                    load_config(self.path)

    def test_invalid_json_and_non_object_are_config_errors(self):
        for source in ("{broken", "[]", "null", '"value"'):
            with self.subTest(source=source):
                self.path.write_text(source, encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(self.path)

    def test_invalid_values_fail_before_request_or_server_start(self):
        invalid = [
            {"backend": "unknown"},
            {"base_url": "file:///tmp/model"},
            {"base_url": "http://user:password@localhost:8081/v1"},
            {"model": "  "},
            {"model": 7},
            {"max_tokens": 0},
            {"max_tokens": True},
            {"max_tokens": 1.5},
            {"request_timeout": -1},
            {"request_timeout": float("nan")},
            {"request_timeout": float("inf")},
            {"temperature": -0.1},
            {"temperature": 2.1},
            {"temperature": True},
            {"system_prompt": []},
            {"request_extra": []},
            {"request_extra": {"messages": []}},
            {"request_extra": {"model": "override"}},
            {"request_extra": {"stream": False}},
            {"request_extra": {"tools": []}},
            {"request_extra": {"tool_choice": "auto"}},
            {"request_extra": {"parallel_tool_calls": True}},
            {"server": []},
            {"server": {"context_size": 0}},
            {"server": {"threads": False}},
            {"server": {"startup_timeout": 1.5}},
            {"server": {"extra_args": "--flag"}},
            {"server": {"extra_args": ["--flag", 42]}},
        ]
        for values in invalid:
            with self.subTest(values=values):
                self.write_json(values)
                with self.assertRaises(ConfigError):
                    load_config(self.path)

    def test_utf8_bom_is_accepted(self):
        self.path.write_text('{"system_prompt": "Русский текст"}', encoding="utf-8-sig")
        self.assertEqual(load_config(self.path)["system_prompt"], "Русский текст")

    def test_unicode_atomic_save_load(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["system_prompt"] = "Ты собеседник. Привет, мир! 🦙"
        config["server"]["model_path"] = "models/модель.gguf"
        destination = self.path.parent / "настройки" / "config.json"
        write_config(destination, config)
        self.assertEqual(load_config(destination), config)
        self.assertIn("Привет, мир! 🦙", destination.read_text(encoding="utf-8"))
        self.assertFalse(destination.with_suffix(".json.tmp").exists())

    def test_failed_atomic_replace_preserves_original_config(self):
        original = copy.deepcopy(DEFAULT_CONFIG)
        write_config(self.path, original)
        updated = copy.deepcopy(original)
        updated["system_prompt"] = "Новая настройка"
        with patch.object(Path, "replace", side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                write_config(self.path, updated)
        self.assertEqual(load_config(self.path), original)

    def test_invalid_save_does_not_replace_valid_file(self):
        original = copy.deepcopy(DEFAULT_CONFIG)
        write_config(self.path, original)
        broken = copy.deepcopy(original)
        broken["max_tokens"] = 0
        with self.assertRaises(ConfigError):
            write_config(self.path, broken)
        self.assertEqual(load_config(self.path), original)


if __name__ == "__main__":
    unittest.main()
