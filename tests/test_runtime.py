"""Exercise ownership and cleanup against real local HTTP servers/processes."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ctypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import queue
import unittest
from unittest.mock import patch

from llmopenchat.runtime import ManagedServer, RuntimeErrorDetail


@contextmanager
def api_server(alias="test-model", api_key=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if api_key is not None and self.headers.get("Authorization") != f"Bearer {api_key}":
                self.send_error(401)
                return
            data = json.dumps({"data": [{"id": alias}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


SERVER_SCRIPT = """
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, sys
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({'data': [{'id': 'test-model'}]}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args):
        pass
print('fake inference server starting', flush=True)
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""


class ScriptRuntime(ManagedServer):
    def __init__(self, config, root, script=SERVER_SCRIPT):
        super().__init__(config, root, emit=lambda message: None)
        self.script = script
        self.created_process = None

    def _build_command(self):
        return [sys.executable, "-u", "-c", self.script, str(self._port)]

    def _wait_until_ready(self, timeout):
        self.created_process = self._process
        return super()._wait_until_ready(timeout)


class RuntimeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        # A terminated Windows process becomes signaled slightly before the OS
        # completes file-handle teardown. Keep retries bounded and do not ignore
        # a real surviving process or a persistent leaked handle.
        deadline = time.monotonic() + 3
        while True:
            try:
                self.folder.cleanup()
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) != 32 or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def config(self, url, **server):
        return {"backend": "llama_cpp", "base_url": url, "model": "test-model", "server": server}

    def test_existing_matching_server_is_reused_and_kept_alive(self):
        with api_server() as url, patch("llmopenchat.runtime.subprocess.Popen") as popen:
            runtime = ManagedServer(self.config(url), self.root, emit=lambda _: None)
            with runtime:
                self.assertEqual(runtime._read_models(), ["test-model"])
            popen.assert_not_called()
            self.assertEqual(runtime._read_models(), ["test-model"])

    def test_existing_wrong_server_is_not_started_over_or_stopped(self):
        with api_server("other-model") as url, patch("llmopenchat.runtime.subprocess.Popen") as popen:
            runtime = ManagedServer(self.config(url), self.root, emit=lambda _: None)
            with self.assertRaisesRegex(RuntimeErrorDetail, "other-model"):
                with runtime:
                    self.fail("Wrong model must not enter the context")
            popen.assert_not_called()
            self.assertEqual(runtime._read_models(), ["other-model"])

    def test_external_never_launches_and_survives_exit(self):
        with api_server() as url, patch("llmopenchat.runtime.subprocess.Popen") as popen:
            config = self.config(url)
            config["backend"] = "external"
            runtime = ManagedServer(config, self.root)
            with runtime:
                pass
            popen.assert_not_called()
            self.assertEqual(runtime._read_models(), ["test-model"])

    def test_owned_process_becomes_ready_and_is_stopped_on_exit(self):
        runtime = ScriptRuntime(self.config(f"http://127.0.0.1:{unused_port()}/v1", startup_timeout=5), self.root)
        with runtime:
            self.assertIsNone(runtime.created_process.poll())
            self.assertEqual(runtime._read_models(), ["test-model"])
        self.assertIsNotNone(runtime.created_process.poll())
        # Windows can briefly retain a TCP listener after process termination.
        deadline = time.monotonic() + 2
        while runtime._port_is_open() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(runtime._port_is_open())
        self.assertIsNone(runtime._log_handle)
        self.assertIn("fake inference server", runtime.log_path.read_text())
        runtime.stop()  # Cleanup is idempotent.

    def test_process_failure_is_reported_and_log_closed(self):
        runtime = ScriptRuntime(
            self.config(f"http://127.0.0.1:{unused_port()}/v1", startup_timeout=5),
            self.root,
            "import sys; print('model load failed', flush=True); sys.exit(3)",
        )
        with self.assertRaisesRegex(RuntimeErrorDetail, "model load failed"):
            with runtime:
                self.fail("Failed process must not enter the context")
        self.assertEqual(runtime.created_process.returncode, 3)
        self.assertIsNone(runtime._process)
        self.assertIsNone(runtime._log_handle)

    def test_startup_timeout_stops_owned_process(self):
        runtime = ScriptRuntime(
            self.config(f"http://127.0.0.1:{unused_port()}/v1", startup_timeout=0.1),
            self.root,
            "import time; time.sleep(60)",
        )
        with self.assertRaisesRegex(RuntimeErrorDetail, "startup_timeout"):
            with runtime:
                self.fail("Timed out process must not enter the context")
        self.assertIsNotNone(runtime.created_process.poll())
        self.assertIsNone(runtime._log_handle)

    def test_keyboard_interrupt_during_startup_cleans_up(self):
        class InterruptedRuntime(ScriptRuntime):
            def _wait_until_ready(self, timeout):
                self.created_process = self._process
                raise KeyboardInterrupt

        runtime = InterruptedRuntime(self.config(f"http://127.0.0.1:{unused_port()}/v1"), self.root)
        with self.assertRaises(KeyboardInterrupt):
            with runtime:
                self.fail("Interrupted startup must not enter the context")
        self.assertIsNotNone(runtime.created_process.poll())
        self.assertIsNone(runtime._log_handle)

    def test_public_interface_cannot_be_bound_by_managed_config(self):
        runtime = ManagedServer(self.config("http://0.0.0.0:8081/v1"), self.root)
        with patch("llmopenchat.runtime.subprocess.Popen") as popen:
            with self.assertRaisesRegex(RuntimeErrorDetail, "127.0.0.1"):
                with runtime:
                    pass
            popen.assert_not_called()

    def test_extra_arguments_cannot_override_binding(self):
        runtime = ManagedServer(self.config("http://127.0.0.1:8081/v1", extra_args=["--host=0.0.0.0"]), self.root)
        with self.assertRaisesRegex(RuntimeErrorDetail, "extra_args"):
            runtime._extra_args()

    def test_native_tool_execution_controls_cannot_bypass_client_approvals(self):
        for option in (
            "--tools", "--tools-runtime", "--mcp-servers", "--mcp-servers-config",
            "--mcp-servers-json", "--agent", "-ag", "--ui-mcp-proxy",
            "--webui-mcp-proxy", "--models-preset",
        ):
            underscore = "--" + option[2:].replace("-", "_") if option.startswith("--") else option
            for spelling in (option, option + "=all", underscore):
                with self.subTest(option=spelling):
                    runtime = ManagedServer(self.config("http://127.0.0.1:8081/v1", extra_args=[spelling]), self.root)
                    with self.assertRaisesRegex(RuntimeErrorDetail, "подтверждений"):
                        runtime._extra_args()

    def test_child_environment_removes_native_tool_controls_preserves_gpu_settings(self):
        execution_settings = {
            "LLAMA_ARG_TOOLS": "all", "LLAMA_ARG_TOOLS_RUNTIME": "docker:tools",
            "LLAMA_ARG_MCP_SERVERS_CONFIG": "mcp.json", "LLAMA_ARG_MCP_SERVERS_JSON": "{}",
            "LLAMA_ARG_AGENT": "1", "LLAMA_ARG_UI_MCP_PROXY": "1",
            "LLAMA_ARG_MODELS_PRESET": "preset.ini",
        }
        runtime = ManagedServer(self.config("http://127.0.0.1:8081/v1"), self.root)
        with patch.dict(os.environ, {**execution_settings, "LLAMA_ARG_DEVICE": "Vulkan0", "TASK_TEST_VALUE": "kept"}, clear=True):
            child_environment = runtime._child_environment()
            self.assertFalse(execution_settings.keys() & child_environment.keys())
            self.assertEqual(child_environment["LLAMA_ARG_DEVICE"], "Vulkan0")
            self.assertEqual(child_environment["TASK_TEST_VALUE"], "kept")
            self.assertEqual(os.environ["LLAMA_ARG_TOOLS"], "all")

    def test_llama_command_explicitly_disables_native_tools_from_system_config(self):
        executable, model = self.root / "llama-server.exe", self.root / "model.gguf"
        executable.write_bytes(b"runtime")
        model.write_bytes(b"weights")
        runtime = ManagedServer(self.config(
            "http://127.0.0.1:8081/v1", executable=str(executable), model_path=str(model),
            extra_args=["--fit", "on", "--chat-template-file", "template.jinja"],
        ), self.root)
        runtime._validate_endpoint()
        command = runtime._build_command()
        self.assertEqual(command[-10:], [
            "--tools", "", "--no-agent", "--no-ui-mcp-proxy",
            "--tools-runtime", "", "--mcp-servers-config", "",
            "--mcp-servers-json", "",
        ])
        self.assertIn("--jinja", command)
        self.assertIn("template.jinja", command)

    def test_spawn_passes_sanitized_environment(self):
        runtime = ScriptRuntime(
            self.config(f"http://127.0.0.1:{unused_port()}/v1", startup_timeout=5), self.root,
        )
        with patch.dict(os.environ, {"LLAMA_ARG_TOOLS": "all", "LLAMA_ARG_AGENT": "1"}):
            with patch("llmopenchat.runtime.subprocess.Popen", wraps=subprocess.Popen) as popen:
                with runtime:
                    child_environment = popen.call_args.kwargs["env"]
                    self.assertNotIn("LLAMA_ARG_TOOLS", child_environment)
                    self.assertNotIn("LLAMA_ARG_AGENT", child_environment)

    def test_bundled_llama_parses_safety_overrides_without_enabling_tools(self):
        executable = Path(__file__).resolve().parents[1] / ".local/runtime/llama-b11445-vulkan/llama-server.exe"
        if os.name != "nt" or not executable.is_file():
            self.skipTest("Bundled Windows llama.cpp runtime unavailable")
        model = self.root / "model.gguf"
        model.write_bytes(b"help does not load these placeholder weights")
        runtime = ManagedServer(self.config(
            "http://127.0.0.1:8081/v1", executable=str(executable), model_path=str(model),
        ), self.root)
        runtime._validate_endpoint()
        with patch.dict(os.environ, {"LLAMA_ARG_TOOLS": "all", "LLAMA_ARG_AGENT": "1"}):
            result = subprocess.run(
                runtime._build_command() + ["--help"], env=runtime._child_environment(),
                stdin=subprocess.DEVNULL, capture_output=True, timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertNotIn(b"server tools or MCP servers are enabled", result.stderr)

    def test_unreachable_external_api_is_actionable(self):
        config = self.config(f"http://127.0.0.1:{unused_port()}/v1")
        config["backend"] = "external"
        with self.assertRaisesRegex(RuntimeErrorDetail, "API"):
            with ManagedServer(config, self.root):
                pass

    def test_external_readiness_uses_environment_api_key(self):
        with api_server(api_key="test-key") as url, patch.dict(os.environ, {"LLMOPENCHAT_API_KEY": "test-key"}):
            config = self.config(url)
            config["backend"] = "external"
            with ManagedServer(config, self.root):
                pass

    @unittest.skipUnless(os.name == "nt", "Windows Job Object integration")
    def test_hard_parent_termination_kills_its_owned_server(self):
        port = unused_port()
        parent_script = f"""
import json, sys, time
from pathlib import Path
from llmopenchat.runtime import ManagedServer
class Runtime(ManagedServer):
    def _build_command(self):
        return [sys.executable, '-u', '-c', {SERVER_SCRIPT!r}, str(self._port)]
config = {{'backend': 'llama_cpp', 'model': 'test-model',
          'base_url': 'http://127.0.0.1:{port}/v1', 'server': {{'startup_timeout': 5}}}}
with Runtime(config, Path(sys.argv[1]), emit=lambda message: None) as runtime:
    print(json.dumps({{'child_pid': runtime._process.pid}}), flush=True)
    time.sleep(60)
"""
        parent = subprocess.Popen(
            [sys.executable, "-u", "-c", parent_script, str(self.root)],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        line_queue = queue.Queue()
        reader = threading.Thread(target=lambda: line_queue.put(parent.stdout.readline()), daemon=True)
        reader.start()
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.TerminateProcess.restype = ctypes.c_int
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        child_handle = None
        try:
            line = line_queue.get(timeout=10)
            if not line:
                self.fail("Parent failed before readiness: " + parent.stderr.read().decode(errors="replace"))
            child_pid = json.loads(line)["child_pid"]
            child_handle = kernel.OpenProcess(0x00100001, False, child_pid)
            self.assertTrue(child_handle, "Inference child must be alive before its parent is killed")
            self.assertEqual(kernel.WaitForSingleObject(child_handle, 0), 258)  # WAIT_TIMEOUT
            parent.terminate()  # No __exit__, finally or atexit handler runs.
            parent.wait(timeout=5)
            self.assertEqual(kernel.WaitForSingleObject(child_handle, 5000), 0, "Owned server survived hard parent termination")
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if child_handle:
                if kernel.WaitForSingleObject(child_handle, 0) == 258:
                    kernel.TerminateProcess(child_handle, 1)
                    kernel.WaitForSingleObject(child_handle, 5000)
                kernel.CloseHandle(child_handle)
            parent.stdout.close()
            parent.stderr.close()
            reader.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
