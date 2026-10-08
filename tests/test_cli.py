from __future__ import annotations

import copy
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rich.console import Console

from llmopenchat import cli
from llmopenchat.config import DEFAULT_CONFIG, ROOT, write_config
from llmopenchat.models import ModelError, ModelManager


class CliInstallTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="llmopenchat-cli-install-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "config.json"
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        write_config(self.path, self.config)
        self.before = self.path.read_bytes()
        self.output = io.StringIO()
        self.console = Console(file=self.output, color_system=None, width=160)
        self.manager = ModelManager(self.root, emit=lambda message: None)
        self.record = {
            "id": "tiny-text-q4", "name": "Tiny Text", "repo_id": "example/tiny-text-GGUF",
            "revision": "a" * 40, "quantization": "Q4_K_M", "managed": False,
            "files": [], "model_path": "models/tiny-text.gguf", "executable": "llama-server.exe",
            "family": "tiny-text", "context_length": 4096,
        }

    def invoke(self, *arguments):
        with (
            patch.object(cli, "ROOT", self.root),
            patch.object(cli, "console", self.console),
            patch.object(cli, "ModelManager", return_value=self.manager),
            patch.object(cli, "ManagedServer") as server,
        ):
            result = cli.main(["--config", str(self.path), "install", *arguments])
        server.assert_not_called()
        return result

    def test_catalog_install_activates_and_saves_config_with_default_or_selected_quant(self):
        for options, quant in (([], "Q4_K_M"), (["--quant", "Q5_K_M"], "Q5_K_M")):
            with self.subTest(quant=quant):
                write_config(self.path, self.config)
                with (
                    patch.object(self.manager, "install", return_value=self.record) as install,
                    patch.object(self.manager, "activate", wraps=self.manager.activate) as activate,
                    patch.object(self.manager, "_load", return_value={"packages": {self.record["id"]: self.record}}),
                    patch.object(self.manager, "_legacy", return_value=None),
                    patch.object(self.manager, "_state", return_value="ready"),
                    patch("llmopenchat.setup.install_ollama_local") as legacy,
                ):
                    self.assertEqual(self.invoke("--model", "tiny-text", *options), 0)
                install.assert_called_once_with("tiny-text", quant)
                activate.assert_called_once_with(self.record["id"], self.config, self.path)
                legacy.assert_not_called()
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                self.assertEqual(saved["model_package"], self.record["id"])
                self.assertEqual(saved["target_model"], self.record["repo_id"])
                self.assertEqual(saved["server"]["model_path"], self.record["model_path"])
                self.assertEqual(saved["server"]["context_size"], 4096)
        self.assertIn("Установлена и выбрана модель: Tiny Text", self.output.getvalue())

    def test_catalog_install_rejects_conflicting_options_before_download(self):
        cases = (
            ["--model", "tiny-text", "--source", "huggingface"],
            ["--model", "tiny-text", "--repo", "example/other"],
            ["--model", "tiny-text", "--file", "other.gguf"],
            ["--quant", "Q5_K_M"],
        )
        with patch.object(self.manager, "install") as install:
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    self.assertEqual(self.invoke(*arguments), 1)
                    self.assertEqual(self.path.read_bytes(), self.before)
        install.assert_not_called()

    def test_catalog_download_failure_preserves_selected_config(self):
        with (
            patch.object(self.manager, "install", side_effect=ModelError("Недостаточно места")),
            patch.object(self.manager, "activate") as activate,
        ):
            self.assertEqual(self.invoke("--model", "tiny-text"), 1)
        activate.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.before)
        self.assertIn("Недостаточно места", self.output.getvalue())


class CliIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name) / "проверка клиента"
        self.workspace.mkdir()
        self.requests = []
        self.model_requests = 0
        self.saved_paths = set()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                owner.model_requests += 1
                payload = json.dumps({"data": [{"id": "cli-test-model"}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(payload)
                prompt = payload["messages"][-1]["content"]
                chunks = []
                if prompt == "Оборванный запрос":
                    chunks.append({"choices": [{"delta": {"content": "Часть ответа"}}]})
                else:
                    chunks.extend(
                        [
                            {"choices": [{"delta": {"reasoning_content": "Думаю по-русски."}}]},
                            {"choices": [{"delta": {"content": "Ответ: " + prompt}}]},
                            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                            {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
                        ]
                    )
                response = "".join(
                    "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks
                )
                if prompt != "Оборванный запрос":
                    response += "data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                self.wfile.write(response.encode("utf-8"))

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()
        self.config_path = self.workspace / "конфигурация.json"
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config.update(
            backend="external",
            base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
            model="cli-test-model",
            target_model="Локальная проверочная модель",
            temperature=0.21,
            max_tokens=25,
            system_prompt="Проверочная системная инструкция.",
        )
        # An external connection must work independently of local model files.
        self.config["server"].update(executable="absent-server.exe", model_path="absent-model.gguf")
        write_config(self.config_path, self.config)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        sessions = (ROOT / ".local" / "sessions").resolve()
        # Delete only exact session paths explicitly reported by our own process.
        for path in self.saved_paths:
            resolved = path.resolve()
            if resolved.parent == sessions:
                resolved.unlink(missing_ok=True)

    def invoke(self, *arguments, input_text=None):
        environment = dict(os.environ)
        environment.update(
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            LLMOPENCHAT_API_KEY="",
            COLUMNS="300",
            NO_COLOR="1",
            PYTHONPATH=str(ROOT) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
        )
        result = subprocess.run(
            [sys.executable, "-m", "llmopenchat", "--config", str(self.config_path), *arguments],
            input=input_text,
            text=True,
            encoding="utf-8",
            capture_output=True,
            cwd=self.workspace,
            env=environment,
            timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.saved_paths.update(Path(path.strip()) for path in re.findall(r"Сохранено: ([^\r\n]+)", result.stdout))
        return result

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        self.assertEqual(result.stderr, "")

    def test_ask_uses_explicit_config_from_another_working_directory(self):
        result = self.invoke("ask", "Как дела, ёж?", "--max-tokens", "9", "--show-reasoning")
        self.assert_success(result)
        self.assertIn("Ответ: Как дела, ёж?", result.stdout)
        self.assertIn("Думаю по-русски.", result.stdout)
        self.assertEqual(self.model_requests, 1)
        self.assertEqual(len(self.requests), 1)
        payload = self.requests[0]
        self.assertEqual(payload["model"], "cli-test-model")
        self.assertEqual(payload["max_tokens"], 9)
        self.assertEqual(payload["temperature"], 0.21)
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "Проверочная системная инструкция."},
                {"role": "user", "content": "Как дела, ёж?"},
            ],
        )

    def test_default_menu_exits_without_contacting_model_api(self):
        result = self.invoke(input_text="0\n")
        self.assert_success(result)
        self.assertIn("История чатов", result.stdout)
        self.assertIn("Модели: загрузка и выбор", result.stdout)
        self.assertEqual(self.model_requests, 0)
        self.assertEqual(self.requests, [])

    def test_menu_can_open_two_separate_chats_with_real_streaming_api(self):
        result = self.invoke("menu", input_text="1\nПервый чат\n/save\n/menu\n1\nВторой чат\n/save\n/quit\n")
        self.assert_success(result)
        self.assertEqual(self.model_requests, 2)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[1]["messages"], [
            {"role": "system", "content": "Проверочная системная инструкция."},
            {"role": "user", "content": "Второй чат"},
        ])
        self.assertIn("Ответ: Первый чат", result.stdout)
        self.assertIn("Ответ: Второй чат", result.stdout)

    def test_chat_clear_load_save_and_quit_preserve_distinct_sessions(self):
        seed = self.workspace / "сохранённый диалог.json"
        seeded_messages = [
            {"role": "system", "content": "Загруженная инструкция."},
            {"role": "user", "content": "Старый вопрос"},
            {"role": "assistant", "content": "Старый ответ"},
        ]
        seed.write_text(json.dumps({"messages": seeded_messages}, ensure_ascii=False), encoding="utf-8")
        result = self.invoke(
            "chat",
            input_text=f'Привет\n/save\n/clear\nПосле очистки\n/save\n/load "{seed}"\nПосле загрузки\n/save\n/quit\n',
        )
        self.assert_success(result)
        self.assertIn("Ответ: Привет", result.stdout)
        self.assertIn("Начат новый диалог.", result.stdout)
        self.assertIn("Загружено сообщений: 3", result.stdout)
        self.assertIn("Ответ: После загрузки", result.stdout)
        self.assertIn("До встречи.", result.stdout)
        self.assertNotIn("Думаю по-русски.", result.stdout)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(
            self.requests[1]["messages"],
            [
                {"role": "system", "content": "Проверочная системная инструкция."},
                {"role": "user", "content": "После очистки"},
            ],
        )
        self.assertEqual(
            self.requests[2]["messages"],
            seeded_messages + [{"role": "user", "content": "После загрузки"}],
        )
        paths = [Path(path.strip()) for path in re.findall(r"Сохранено: ([^\r\n]+)", result.stdout)]
        self.assertEqual(len(paths), 3, result.stdout)
        self.assertEqual(len(set(paths)), 3, "Загрузка диалога не должна перезаписывать предыдущую автосессию.")
        saved = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        self.assertEqual(saved[0]["messages"][-1]["content"], "Ответ: Привет")
        self.assertEqual(saved[1]["messages"][-1]["content"], "Ответ: После очистки")
        self.assertEqual(saved[2]["messages"][-1]["content"], "Ответ: После загрузки")
        self.assertEqual(json.loads(seed.read_text(encoding="utf-8"))["messages"], seeded_messages)

    def test_invalid_load_retains_current_conversation(self):
        seed = self.workspace / "неверный диалог.json"
        seed.write_text('{"messages": [{"role": "tool", "content": "invalid"}]}', encoding="utf-8")
        result = self.invoke(
            "chat", input_text=f'Первый вопрос\n/load "{seed}"\nВторой вопрос\n/save\n/quit\n'
        )
        self.assert_success(result)
        self.assertIn("Неверный формат диалога", result.stdout)
        self.assertEqual(
            self.requests[1]["messages"],
            [
                {"role": "system", "content": "Проверочная системная инструкция."},
                {"role": "user", "content": "Первый вопрос"},
                {"role": "assistant", "content": "Ответ: Первый вопрос"},
                {"role": "user", "content": "Второй вопрос"},
            ],
        )

    def test_incomplete_response_is_reported_and_next_request_can_succeed(self):
        result = self.invoke("chat", input_text="Оборванный запрос\nНовый запрос\n/save\n/quit\n")
        self.assert_success(result)
        self.assertIn("Соединение оборвалось", result.stdout)
        self.assertIn("Ответ: Новый запрос", result.stdout)
        self.assertEqual(
            self.requests[1]["messages"],
            [
                {"role": "system", "content": "Проверочная системная инструкция."},
                {"role": "user", "content": "Новый запрос"},
            ],
        )

    def test_invalid_token_override_fails_before_contacting_endpoint(self):
        result = self.invoke("ask", "Привет", "--max-tokens", "0")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("max_tokens", result.stdout)
        self.assertEqual(self.model_requests, 0)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
