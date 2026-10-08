"""Telegram launch routing without a live bot or inference server."""

from __future__ import annotations

import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console

from llmopenchat import cli, menu
from llmopenchat.config import DEFAULT_CONFIG, write_config


class TelegramEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-telegram-entry-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "config.json"
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        write_config(self.path, self.config)
        self.output = io.StringIO()
        self.console = Console(file=self.output, color_system=None, width=160)

    def invoke(self, arguments, choices=(), status=0):
        with (
            patch.object(cli, "ROOT", self.root),
            patch.object(cli, "console", self.console),
            patch.object(cli, "run_telegram", return_value=status) as bot,
            patch.object(cli, "ManagedServer") as server,
            patch.object(cli, "chat") as chat,
            patch("builtins.input", side_effect=choices),
        ):
            result = cli.main(["--config", str(self.path), *arguments])
        return result, bot, server, chat

    def test_direct_launch_returns_runtime_status_without_console_chat(self):
        result, bot, server, chat = self.invoke(["telegram"], status=1)
        self.assertEqual(result, 1)
        bot.assert_called_once_with(self.config, self.root, whitelist_path=None,
                                    credits_path=None, emit=cli.say)
        server.assert_not_called()
        chat.assert_not_called()

    def test_launch_passes_file_paths_and_validated_chat_options(self):
        result, bot, server, chat = self.invoke([
            "telegram", "--whitelist", "allowed-users.txt", "--credits", "bot-token.txt",
            "--no-autostart", "--max-tokens", "123", "--show-reasoning",
        ])
        self.assertEqual(result, 0)
        args, keywords = bot.call_args
        self.assertEqual(args[1], self.root)
        self.assertEqual(args[0]["backend"], "external")
        self.assertEqual(args[0]["max_tokens"], 123)
        self.assertTrue(args[0]["show_reasoning"])
        self.assertEqual(keywords["whitelist_path"], Path("allowed-users.txt"))
        self.assertEqual(keywords["credits_path"], Path("bot-token.txt"))
        self.assertEqual(self.config["backend"], "llama_cpp")
        server.assert_not_called()
        chat.assert_not_called()

    def test_invalid_generation_options_fail_before_launch(self):
        result, bot, server, chat = self.invoke(["telegram", "--max-tokens", "0"])
        self.assertEqual(result, 1)
        self.assertIn("max_tokens", self.output.getvalue())
        bot.assert_not_called()
        server.assert_not_called()
        chat.assert_not_called()

    def test_menu_launches_bot_and_returns_to_menu(self):
        result, bot, server, chat = self.invoke([], ["4", "0"], status=130)
        self.assertEqual(result, 0)
        bot.assert_called_once_with(self.config, self.root, emit=cli.say)
        server.assert_not_called()
        chat.assert_not_called()
        self.assertIn("4 — Telegram-бот", self.output.getvalue())
        self.assertIn("До встречи.", self.output.getvalue())

    def test_exiting_menu_does_not_launch_bot_or_model(self):
        result, bot, server, chat = self.invoke([], ["0"])
        self.assertEqual(result, 0)
        bot.assert_not_called()
        server.assert_not_called()
        chat.assert_not_called()

    def test_telegram_help_does_not_load_config_or_start_runtime(self):
        with (
            patch.object(cli, "console", self.console),
            patch.object(cli, "load_config") as load,
            patch.object(cli, "run_telegram") as bot,
            patch.object(cli, "ManagedServer") as server,
            patch.object(cli.sys, "stdout", self.output),
        ):
            with self.assertRaises(SystemExit) as stopped:
                cli.main(["telegram", "--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--whitelist", self.output.getvalue())
        self.assertIn("Telegram username", self.output.getvalue())
        self.assertIn("--credits", self.output.getvalue())
        load.assert_not_called()
        bot.assert_not_called()
        server.assert_not_called()

    def test_menu_bot_uses_config_selected_in_model_manager(self):
        updated = copy.deepcopy(self.config)
        updated["target_model"] = "author/another-model"
        bot = MagicMock(return_value=0)
        chat = MagicMock()
        with (
            patch.object(menu, "models_menu", return_value=updated),
            patch("builtins.input", side_effect=["3", "4", "0"]),
        ):
            result = menu.main_menu(self.config, self.path, self.root, self.console, chat,
                                    run_telegram=bot)
        self.assertEqual(result, 0)
        bot.assert_called_once_with(updated)
        chat.assert_not_called()

    def test_chat_return_to_menu_keeps_bot_launch_option(self):
        with (
            patch.object(cli, "ROOT", self.root),
            patch.object(cli, "console", self.console),
            patch.object(cli, "run_telegram", return_value=0) as bot,
            patch.object(cli, "ManagedServer") as server,
            patch.object(cli, "chat", return_value=menu.BACK_TO_MENU) as chat,
            patch("builtins.input", side_effect=["4", "0"]),
        ):
            result = cli.main(["--config", str(self.path), "chat"])
        self.assertEqual(result, 0)
        self.assertEqual(server.call_count, 1)
        self.assertEqual(chat.call_count, 1)
        bot.assert_called_once_with(self.config, self.root, emit=cli.say)


if __name__ == "__main__":
    unittest.main()
