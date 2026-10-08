"""Retrieved context is transient; answers and source mappings remain saveable."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from llmopenchat.api import ApiError, StreamEvent
from llmopenchat.config import DEFAULT_CONFIG, ConfigError, load_config
from llmopenchat.conversation import GenerationCancelled, generate_response
from llmopenchat.history import load_session, save_session
from llmopenchat.rag import RagError, RagSession


class Client:
    def __init__(self):
        self.requests = []

    def stream_chat(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        yield StreamEvent("content", "Запуск в 09:30 [1].")
        yield StreamEvent("finish", "stop")


class RagConversationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="llmopenchat-rag-generation-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.rag = RagSession(self.root, copy.deepcopy(DEFAULT_CONFIG))
        (self.root / "manual.txt").write_text("Система Орион запускается в 09:30 ежедневно.", encoding="utf-8")
        self.rag.command("add manual.txt")

    def test_retrieved_fragments_only_go_to_outbound_request(self):
        self.rag.command("on")
        messages = [{"role": "system", "content": "Помоги."}, {"role": "user", "content": "Когда запускается Орион?"}]
        original = copy.deepcopy(messages)
        client, events = Client(), []
        result = generate_response(client, messages, on_event=events.append, rag=self.rag)
        self.assertEqual(messages, original)
        self.assertEqual(len(client.requests), 1)
        self.assertIn("09:30", client.requests[0][1]["content"])
        self.assertEqual(client.requests[0][1]["role"], "system")
        self.assertIn("Источники RAG:", result.answer)
        self.assertIn("manual.txt", result.answer)
        self.assertEqual("".join(e.value for e in events if e.kind == "content"), result.answer)
        sources = [e.value for e in events if e.kind == "rag_sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["sources"][0]["id"], 1)
        messages.append({"role": "assistant", "content": result.answer})
        path = self.root / "chat.json"
        save_session(path, messages, "test")
        self.assertEqual(load_session(path), messages)
        self.assertNotIn("ежедневно", path.read_text(encoding="utf-8"))

    def test_no_matches_does_not_call_model(self):
        self.rag.command("on")
        client, events = Client(), []
        result = generate_response(client, [{"role": "user", "content": "Термоядерный звездолёт"}], on_event=events.append, rag=self.rag)
        self.assertEqual(client.requests, [])
        self.assertIn("не найдены", result.answer)
        self.assertNotIn("Источники RAG:", result.answer)

    def test_disabled_rag_leaves_request_unchanged(self):
        client = Client()
        messages = [{"role": "user", "content": "Привет"}]
        result = generate_response(client, messages, rag=self.rag)
        self.assertEqual(client.requests, [messages])
        self.assertNotIn("Источники RAG:", result.answer)

    def test_retrieval_failure_and_cancellation_open_no_stream(self):
        class Broken:
            enabled = True

            def prepare(self, messages):
                raise RagError("Индекс повреждён")

        client = Client()
        with self.assertRaisesRegex(ApiError, "Индекс повреждён"):
            generate_response(client, [], rag=Broken())
        with self.assertRaises(GenerationCancelled):
            generate_response(client, [], rag=self.rag, cancelled=lambda: True)
        self.assertEqual(client.requests, [])

    def test_invalid_rag_config_rejected_and_partial_defaults_merged(self):
        path = self.root / "config.json"
        for rag in ([], {"top_k": True}, {"top_k": 0}, {"chunk_size": 200, "chunk_overlap": 200}, {"max_context_chars": 100000}):
            with self.subTest(rag=rag):
                path.write_text(json.dumps({"rag": rag}), encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(path)
        path.write_text(json.dumps({"rag": {"top_k": 2}}), encoding="utf-8")
        config = load_config(path)
        self.assertEqual(config["rag"]["top_k"], 2)
        self.assertEqual(config["rag"]["chunk_size"], 1000)


if __name__ == "__main__":
    unittest.main()
