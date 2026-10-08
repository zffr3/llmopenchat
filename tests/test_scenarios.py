from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat._winfiles import LockedHandle, supported
from llmopenchat.harness import HarnessSettings, ToolRequest
from llmopenchat.scenarios import (
    MAX_HOSTS_BYTES, MAX_INPUT_BYTES, MAX_JSON_BYTES, ScenarioError, load_hosts, load_scenario, parse_json_document,
)


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "news.json"
        self.path.parent.joinpath("hosts.txt").write_text("http://127.0.0.1:9000\nhttp://10.0.0.4\nhttps://backend.internal:8443\n", encoding="utf-8")
        self.definition = {
            "schema_version": 1,
            "name": "Новости",
            "instruction": "Выдели ключевую мысль и оцени тональность. Ответь JSON.",
            "response_schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "minLength": 1},
                    "sentiment": {"enum": ["positive", "neutral", "negative"]},
                },
                "required": ["summary", "sentiment"],
                "additionalProperties": False,
            },
        }

    def load(self, definition=None):
        self.path.write_text(json.dumps(self.definition if definition is None else definition, ensure_ascii=False), encoding="utf-8")
        return load_scenario(self.path)

    def test_minimal_definition_defaults_to_bounded_stdout_and_no_tools(self):
        scenario = self.load()
        self.assertEqual(scenario.schema_version, 1)
        self.assertEqual(scenario.name, "Новости")
        self.assertEqual(scenario.instruction, self.definition["instruction"])
        self.assertEqual(scenario.input_max_bytes, MAX_JSON_BYTES)
        self.assertEqual(scenario.output.type, "stdout")
        self.assertEqual(scenario.harness_settings(), HarnessSettings())
        self.assertEqual(scenario.allowed_tools, frozenset())
        self.assertFalse(scenario.approval(ToolRequest("read_file", {"path": "news"}, "read")))

    def test_config_bytes_and_utf8_bom(self):
        self.path.write_text(json.dumps(self.definition, ensure_ascii=False), encoding="utf-8-sig")
        self.assertEqual(load_scenario(self.path).name, "Новости")
        self.path.write_bytes(b" " * (MAX_JSON_BYTES + 1))
        with self.assertRaises(ScenarioError):
            load_scenario(self.path)

    def test_missing_invalid_utf8_and_deeply_nested_file_are_scenario_errors(self):
        with self.assertRaises(ScenarioError):
            load_scenario(self.path)
        for raw in (b"\xff", b"[" * 2000 + b"0" + b"]" * 2000):
            with self.subTest(raw=raw[:8]):
                self.path.write_bytes(raw)
                with self.assertRaises(ScenarioError):
                    load_scenario(self.path)

    def test_schema_version_is_required_and_not_boolean(self):
        for version in (None, True, "1", 1.0, 2):
            with self.subTest(version=version):
                values = copy.deepcopy(self.definition)
                if version is None:
                    del values["schema_version"]
                else:
                    values["schema_version"] = version
                with self.assertRaises(ScenarioError):
                    self.load(values)

    def test_required_strings_and_schema(self):
        for field in ("name", "instruction", "response_schema"):
            for invalid in (None, "", [], 42, True):
                with self.subTest(field=field, invalid=invalid):
                    values = copy.deepcopy(self.definition)
                    values[field] = invalid
                    with self.assertRaises(ScenarioError):
                        self.load(values)

    def test_unknown_fields_are_rejected_at_each_policy_level(self):
        mutations = [
            {"model": "override"}, {"instruction_extra": "override"},
            {"generation": {"tools": []}}, {"harness": {"auto_approve_all": True}},
            {"input": {"url": "http://example.com"}}, {"output": {"headers": {}}},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ScenarioError):
                self.load({**self.definition, **mutation})

    def test_generation_rejects_owned_request_fields(self):
        for field in ("messages", "model", "stream", "stream_options", "temperature", "max_tokens", "tools", "tool_choice", "parallel_tool_calls", "response_format"):
            with self.subTest(field=field), self.assertRaises(ScenarioError):
                self.load({**self.definition, "generation": {"request_extra": {field: None}}})

    def test_generation_numbers_and_modes_are_validated(self):
        for generation in ({"temperature": True}, {"temperature": -1}, {"temperature": 2.1},
                           {"max_tokens": False}, {"max_tokens": 0}, {"max_tokens": 1.5},
                           {"max_tokens": 1048577}, {"request_extra": []}, {"response_mode": "yaml"}):
            with self.subTest(generation=generation), self.assertRaises(ScenarioError):
                self.load({**self.definition, "generation": generation})

    def test_make_config_copies_and_owns_instruction_schema_and_tools(self):
        scenario = self.load({**self.definition, "generation": {"temperature": 0, "max_tokens": 256}})
        base = copy.deepcopy(DEFAULT_CONFIG)
        base["request_extra"]["tools"] = [{"function": {"name": "danger"}}]
        base["request_extra"]["response_format"] = {"type": "text"}
        original = copy.deepcopy(base)
        config = scenario.make_config(base)
        self.assertEqual(base, original)
        self.assertEqual(config["system_prompt"], scenario.instruction)
        self.assertEqual(config["temperature"], 0)
        self.assertEqual(config["max_tokens"], 256)
        self.assertNotIn("tools", config["request_extra"])
        self.assertEqual(config["request_extra"]["chat_template_kwargs"], {"enable_thinking": False})
        response_format = config["request_extra"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "scenario")
        self.assertEqual(response_format["json_schema"]["schema"], scenario.response_schema)
        response_format["json_schema"]["schema"].clear()
        self.assertEqual(scenario.response_schema, self.definition["response_schema"])

    def test_explicit_request_extra_replaces_base_and_response_modes(self):
        for mode in ("json_schema", "json_object", "prompt"):
            with self.subTest(mode=mode):
                scenario = self.load({**self.definition, "generation": {"request_extra": {"top_p": 0.5}, "response_mode": mode}})
                config = scenario.make_config(DEFAULT_CONFIG)
                self.assertNotIn("chat_template_kwargs", config["request_extra"])
                self.assertEqual(config["request_extra"]["top_p"], 0.5)
                if mode == "prompt":
                    self.assertNotIn("response_format", config["request_extra"])
                else:
                    self.assertEqual(config["request_extra"]["response_format"]["type"], mode)

    def test_relative_repository_is_anchored_to_definition_not_cwd(self):
        self.path.parent.joinpath("scenarios").mkdir()
        self.path = self.path.parent / "scenarios" / "news.json"
        scenario = self.load({**self.definition, "harness": {"code_enabled": True, "repository": "../repo"}})
        self.assertEqual(scenario.harness.repository, self.path.parent.parent / "repo")
        self.assertEqual(scenario.harness_settings().repository, scenario.harness.repository)
        self.assertFalse(scenario.approval(ToolRequest("read_file", {}, "read")))
        self.assertFalse(scenario.approval(ToolRequest("write_file", {}, "write")))
        self.assertFalse(scenario.approval(ToolRequest("web_fetch", {}, "web")))

    def test_autoapproval_requires_explicit_matching_capability(self):
        invalid = [
            {"web_enabled": "true"}, {"code_enabled": 1}, {"powershell_enabled": []},
            {"auto_approve": True}, {"auto_approve": ["web_fetch"]},
            {"web_enabled": True, "auto_approve": ["shell"]},
            {"web_enabled": True, "auto_approve": ["web_fetch", "web_fetch"]},
            {"web_enabled": True, "auto_approve": [42]},
            {"code_enabled": True}, {"powershell_enabled": True},
            {"repository": "", "web_enabled": True},
        ]
        for harness in invalid:
            with self.subTest(harness=harness), self.assertRaises(ScenarioError):
                self.load({**self.definition, "harness": harness})

    def test_powershell_and_repository_tools_are_separate_capabilities(self):
        scenario = self.load({**self.definition, "harness": {"powershell_enabled": True, "repository": "repo"}})
        self.assertEqual(scenario.allowed_tools, frozenset())
        self.assertFalse(scenario.harness_settings().code_enabled)
        self.assertTrue(scenario.harness_settings().powershell_enabled)
        with self.assertRaises(ScenarioError):
            self.load({**self.definition, "harness": {"powershell_enabled": True, "repository": "repo", "auto_approve": ["read_file"]}})

    def test_input_limit_is_bounded_integer(self):
        scenario = self.load({**self.definition, "input": {"max_bytes": MAX_INPUT_BYTES}})
        self.assertEqual(scenario.input_max_bytes, MAX_INPUT_BYTES)
        for limit in (True, "1024", 0, -1, MAX_INPUT_BYTES + 1):
            with self.subTest(limit=limit), self.assertRaises(ScenarioError):
                self.load({**self.definition, "input": {"max_bytes": limit}})

    def test_post_uses_fixed_internal_url_env_headers_and_timeout(self):
        for url in ("http://127.0.0.1:9000/results", "http://10.0.0.4/results", "https://backend.internal:8443/results?q=news"):
            with self.subTest(url=url):
                scenario = self.load({**self.definition, "output": {"type": "post", "url": url, "headers_env": {"Authorization": "NEWS_AUTH"}, "timeout_seconds": 30}})
                self.assertEqual(scenario.output.url, url)
                self.assertEqual(scenario.output.headers_env, {"Authorization": "NEWS_AUTH"})
                self.assertEqual(scenario.output.timeout_seconds, 30)

    def test_post_unicode_host_path_and_query_are_normalized_for_urllib(self):
        self.path.parent.joinpath("hosts.txt").write_text("https://пример.рф\nhttp://[::1]:9000\n", encoding="utf-8")
        scenario = self.load({**self.definition, "output": {"type": "post", "url": "https://пример.рф/новости/%D1%8F?текст=мир&n=1"}})
        self.assertEqual(scenario.output.url, "https://xn--e1afmkfd.xn--p1ai/%D0%BD%D0%BE%D0%B2%D0%BE%D1%81%D1%82%D0%B8/%D1%8F?%D1%82%D0%B5%D0%BA%D1%81%D1%82=%D0%BC%D0%B8%D1%80&n=1")
        self.assertTrue(scenario.output.url.isascii())
        self.assertTrue(scenario.network_allowed(scenario.output.url))
        scenario = self.load({**self.definition, "output": {"type": "post", "url": "http://[0:0:0:0:0:0:0:1]:9000/новости"}})
        self.assertEqual(scenario.output.url, "http://[::1]:9000/%D0%BD%D0%BE%D0%B2%D0%BE%D1%81%D1%82%D0%B8")
        self.assertTrue(scenario.network_allowed(scenario.output.url))

    def test_post_rejects_malformed_urls_and_stdout_post_settings(self):
        for url in ("file:///tmp/result", "http://", "http://u:p@localhost/out", "http://u@localhost/out", "http://localhost/out#tag", "http://localhost/out#", "http://localhost:bad/out", "http://localhost:0/out", "http://localhost:70000/out", "http://local host/out", "http://localhost/out\nX:bad", "http://localhost\\other/out"):
            with self.subTest(url=url), self.assertRaises(ScenarioError):
                self.load({**self.definition, "output": {"type": "post", "url": url}})
        for output in ({"type": "stdout", "url": "http://localhost"}, {"type": "post"}, {"type": "unknown"}):
            with self.subTest(output=output), self.assertRaises(ScenarioError):
                self.load({**self.definition, "output": output})

    def test_header_names_env_names_and_timeout_are_validated(self):
        invalid = [
            {"headers_env": []}, {"headers_env": {"Bad\r\nHeader": "TOKEN"}},
            {"headers_env": {"Host": "HOST"}}, {"headers_env": {"Content-Length": "LENGTH"}},
            {"headers_env": {"Content-Type": "TYPE"}},
            {"headers_env": {"Authorization": "TOKEN", "authorization": "OTHER"}},
            {"headers_env": {"Authorization": "Bearer token"}},
            {"headers_env": {"Authorization": None}}, {"timeout_seconds": True},
            {"timeout_seconds": 0}, {"timeout_seconds": 301},
        ]
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ScenarioError):
                self.load({**self.definition, "output": {"type": "post", "url": "http://localhost/result", **params}})

    def test_network_requires_sibling_hosts_and_post_origin_must_match(self):
        self.path.parent.joinpath("hosts.txt").unlink()
        for params in ({"harness": {"web_enabled": True}}, {"output": {"type": "post", "url": "http://localhost/result"}}):
            with self.subTest(params=params), self.assertRaises(ScenarioError):
                self.load({**self.definition, **params})
        self.path.parent.joinpath("hosts.txt").write_text("http://localhost:9000\n", encoding="utf-8")
        with self.assertRaises(ScenarioError):
            self.load({**self.definition, "output": {"type": "post", "url": "http://localhost:9001/result"}})

    def test_stdout_also_validates_existing_hosts_but_allows_missing_policy(self):
        self.assertIn("http://127.0.0.1:9000", self.load().allowed_origins)
        hosts_path = self.path.parent / "hosts.txt"
        hosts_path.write_text("https://*.example.com", encoding="utf-8")
        with self.assertRaises(ScenarioError):
            self.load()
        hosts_path.unlink()
        self.assertEqual(self.load().allowed_origins, frozenset())
        try:
            hosts_path.symlink_to(hosts_path.with_name("missing-policy.txt"))
        except (OSError, NotImplementedError):
            return
        with self.assertRaises(ScenarioError):
            self.load()

    def test_only_network_requests_to_allowlisted_origins_are_autoapproved(self):
        self.path.parent.joinpath("hosts.txt").write_text("https://news.example\nhttps://html.duckduckgo.com\n", encoding="utf-8")
        scenario = self.load({**self.definition, "harness": {"web_enabled": True, "auto_approve": ["web_fetch", "web_search"]}})
        self.assertTrue(scenario.approval(ToolRequest("web_fetch", {"url": "https://news.example/article"}, "web")))
        self.assertFalse(scenario.approval(ToolRequest("web_fetch", {"url": "https://other.example/article"}, "web")))
        self.assertFalse(scenario.approval(ToolRequest("web_fetch", {"url": "http://news.example/article"}, "web")))
        self.assertFalse(scenario.approval(ToolRequest("web_fetch", {"url": "https://news.example:8443/article"}, "web")))
        self.assertTrue(scenario.approval(ToolRequest("web_search", {"query": "news"}, "search")))
        self.assertFalse(scenario.approval(ToolRequest("read_file", {"path": "news"}, "read")))
        self.assertEqual(scenario.hosts_path, self.path.parent / "hosts.txt")

    def test_nonnetwork_autoapproval_is_rejected_even_with_enabled_capabilities(self):
        for name in ("list_files", "read_file", "write_file", "create_directory", "search_files", "powershell_run"):
            with self.subTest(name=name), self.assertRaises(ScenarioError):
                self.load({**self.definition, "harness": {"code_enabled": True, "powershell_enabled": True, "repository": "repo", "auto_approve": [name]}})

    def test_schema_rejects_invalid_dialect_definitions_and_remote_refs(self):
        invalid = [
            {"type": "not-a-type"}, {"required": "summary"}, {"$schema": "http://json-schema.org/draft-07/schema#"},
            {"$ref": "https://example.com/schema"}, {"$dynamicRef": "file:///tmp/schema"},
            {"$defs": {"x": {"$ref": "other.json"}}},
        ]
        for schema in invalid:
            with self.subTest(schema=schema), self.assertRaises(ScenarioError):
                self.load({**self.definition, "response_schema": schema})

    def test_response_validation_enforces_required_enum_and_extra_fields(self):
        scenario = self.load()
        response = {"summary": "Новая библиотека", "sentiment": "positive"}
        self.assertEqual(scenario.validate_response(json.dumps(response, ensure_ascii=False)), response)
        for invalid in ({"summary": "News"}, {"summary": "News", "sentiment": "mixed"},
                        {"summary": "News", "sentiment": "neutral", "url": "http://override"}, [], None):
            with self.subTest(invalid=invalid), self.assertRaises(ScenarioError):
                scenario.validate_response(json.dumps(invalid))

    def test_response_schema_local_refs_and_formats(self):
        schema = {"$defs": {"value": {"type": "string", "format": "ipv4"}}, "$ref": "#/$defs/value"}
        scenario = self.load({**self.definition, "response_schema": schema})
        self.assertEqual(scenario.validate_response('"127.0.0.1"'), "127.0.0.1")
        with self.assertRaises(ScenarioError):
            scenario.validate_response('"invalid-ip"')
        schema["$ref"] = "#/$defs/missing"
        scenario = self.load({**self.definition, "response_schema": schema})
        with self.assertRaises(ScenarioError):
            scenario.validate_response('"date"')

    def test_validation_does_not_include_generated_secrets_in_error(self):
        scenario = self.load()
        with self.assertRaises(ScenarioError) as caught:
            scenario.validate_response('{"summary":"NEWS_SECRET_TOKEN","sentiment":"SECRET_TOKEN"}')
        self.assertNotIn("SECRET_TOKEN", str(caught.exception))


class StrictJsonTests(unittest.TestCase):
    def test_unicode_json_is_preserved(self):
        self.assertEqual(parse_json_document('{"text":"Новость 📰"}'), {"text": "Новость 📰"})

    def test_duplicate_nonfinite_surrogate_and_markdown_are_rejected(self):
        for text in ('{"x":1,"x":2}', '{"a":{"x":1,"x":2}}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e10000}', '{"x":"\\ud800"}', '```json\n{}\n```', '{} trailing'):
            with self.subTest(text=text), self.assertRaises(ScenarioError):
                parse_json_document(text)

    def test_limit_counts_utf8_bytes(self):
        self.assertEqual(parse_json_document('"я"', max_bytes=4), "я")
        with self.assertRaises(ScenarioError):
            parse_json_document('"я"', max_bytes=3)


class HostPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "hosts.txt"

    def load(self, text):
        self.path.write_text(text, encoding="utf-8")
        return load_hosts(self.path)

    def test_exact_origins_default_ports_idna_and_ipv6_normalization(self):
        policy = self.load("# service destinations\n HTTPS://NEWS.EXAMPLE/ \n\nhttps://пример.рф\nhttp://[::1]:9000\n")
        self.assertEqual(policy.origins, frozenset({"https://news.example:443", "https://xn--e1afmkfd.xn--p1ai:443", "http://[::1]:9000"}))
        for url in ("https://news.example:443/article?q=1", "https://news.example./article", "https://пример.рф/article", "http://[0:0:0:0:0:0:0:1]:9000/result"):
            with self.subTest(url=url):
                self.assertTrue(policy.allows(url))
        for url in ("http://news.example", "https://news.example:8443", "https://sub.news.example", "https://news.example.evil", "https://news.example@evil", "https://u:p@news.example", "file:///news.example", None, "https://news.example/path#fragment"):
            with self.subTest(url=url):
                self.assertFalse(policy.allows(url))

    def test_hosts_syntax_rejects_wildcards_paths_and_ambiguous_hosts(self):
        for entry in ("*.example.com", "example.com", "https://*.example.com", "http://localhost:0", "https://news.example/path", "https://news.example?x=1", "https://news.example?", "https://user@news.example", "https://news.example#", "https://news.example:", "https://bad..example", "https://bad_host.example", "https://news.example # comment"):
            with self.subTest(entry=entry), self.assertRaises(ScenarioError):
                self.load(entry)

    def test_empty_allowlist_denies_all_and_snapshot_does_not_change(self):
        policy = self.load("# empty\n\n")
        self.assertFalse(policy.allows("https://news.example"))
        policy = self.load("https://news.example\n")
        self.path.write_text("https://evil.example\n", encoding="utf-8")
        self.assertTrue(policy.allows("https://news.example/article"))
        self.assertFalse(policy.allows("https://evil.example"))

    def test_symlink_hardlink_nonregular_and_oversize_hosts_are_rejected(self):
        target = self.path.with_name("target.txt")
        target.write_text("https://news.example\n", encoding="utf-8")
        try:
            self.path.symlink_to(target)
        except (OSError, NotImplementedError):
            pass
        else:
            with self.assertRaises(ScenarioError):
                load_hosts(self.path)
            self.path.unlink()
        try:
            self.path.hardlink_to(target)
        except (OSError, NotImplementedError):
            pass
        else:
            with self.assertRaises(ScenarioError):
                load_hosts(self.path)
            self.path.unlink()
        self.path.mkdir()
        with self.assertRaises(ScenarioError):
            load_hosts(self.path)
        self.path.rmdir()
        self.path.write_bytes(b" " * (MAX_HOSTS_BYTES + 1))
        with self.assertRaises(ScenarioError):
            load_hosts(self.path)

    def test_reparse_point_is_rejected_before_opening(self):
        self.load("https://news.example")
        original = self.path.lstat()
        class Reparse:
            st_mode = original.st_mode
            st_nlink = 1
            st_file_attributes = 0x400
        with patch.object(Path, "lstat", return_value=Reparse()), self.assertRaises(ScenarioError):
            load_hosts(self.path)

    @unittest.skipUnless(supported(), "Native handle boundary is Windows-only")
    def test_junction_ancestor_is_rejected_before_read(self):
        target = self.path.parent / "real"
        target.mkdir()
        target.joinpath("nested").mkdir()
        target.joinpath("nested", "hosts.txt").write_text("https://news.example\n", encoding="utf-8")
        link = self.path.parent / "junction"
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if completed.returncode:
            self.skipTest("Could not create junction")
        try:
            with patch.object(LockedHandle, "read") as read, self.assertRaises(ScenarioError):
                load_hosts(link / "nested" / "hosts.txt")
            read.assert_not_called()
        finally:
            link.rmdir()

    @unittest.skipUnless(supported(), "Native handle boundary is Windows-only")
    def test_policy_read_locks_file_and_ancestors_against_changes(self):
        self.load("https://news.example")
        original_read = LockedHandle.read
        def read(handle, limit):
            with self.assertRaises(OSError):
                self.path.parent.rename(self.path.parent.with_name(self.path.parent.name + "-moved"))
            with self.assertRaises(OSError):
                with self.path.open("wb"):
                    pass
            return original_read(handle, limit)
        with patch.object(LockedHandle, "read", new=read):
            self.assertTrue(load_hosts(self.path).allows("https://news.example/article"))

    @unittest.skipUnless(supported(), "Native handle boundary is Windows-only")
    def test_existing_writer_prevents_policy_read(self):
        self.load("https://news.example")
        with self.path.open("ab"), self.assertRaises(ScenarioError):
            load_hosts(self.path)


if __name__ == "__main__":
    unittest.main()
