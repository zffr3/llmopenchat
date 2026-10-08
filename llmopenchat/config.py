"""Configuration shared by the console, installer, and server manager."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "mradermacher/Huihui-GLM-4.7-Flash-abliterated-GGUF"
DEFAULT_FILE = "Huihui-GLM-4.7-Flash-abliterated.Q4_K_M.gguf"
DEFAULT_CONFIG = {
    "backend": "llama_cpp",
    "base_url": "http://127.0.0.1:8081/v1",
    "model": "glm-local",
    "target_model": "huihui_ai/glm-4.7-flash-abliterated",
    "temperature": 0.7,
    "max_tokens": 4096,
    "system_prompt": "Ты полезный собеседник. Отвечай на языке пользователя.",
    "show_reasoning": False,
    "request_timeout": 600,
    "request_extra": {"chat_template_kwargs": {"enable_thinking": False}},
    "rag": {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 4,
        "max_context_chars": 5000,
    },
    "server": {
        "executable": ".local/runtime/llama-b11445-vulkan/llama-server.exe",
        "model_path": "models/huihui_ai--glm-4.7-flash-abliterated/Huihui-GLM-4.7-Flash-abliterated-ollama.Q4_K_M.llamacpp.gguf",
        "context_size": 8192,
        "gpu_layers": "auto",
        "threads": 16,
        "startup_timeout": 300,
        "extra_args": ["--fit", "on", "--fit-target", "1024", "--chat-template-file", "llmopenchat/templates/GLM-4.7-Flash.jinja"],
    },
}


class ConfigError(ValueError):
    pass


def _merge(default: dict, values: dict) -> dict:
    merged = copy.deepcopy(default)
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict:
    if path.exists():
        try:
            values = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            raise ConfigError(f"Не удалось прочитать {path}: {error}") from error
        if not isinstance(values, dict):
            raise ConfigError("Конфигурация должна быть JSON-объектом.")
        # Request settings must be replaceable as a whole, including by {}.
        config = _merge(DEFAULT_CONFIG, values)
        if "request_extra" in values:
            config["request_extra"] = values["request_extra"]
    else:
        config = copy.deepcopy(DEFAULT_CONFIG)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("backend") not in {"llama_cpp", "vllm", "external"}:
        raise ConfigError("backend должен быть llama_cpp, vllm или external.")
    url = urlparse(str(config.get("base_url", "")))
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ConfigError("base_url должен быть HTTP(S)-адресом без логина и пароля.")
    if not isinstance(config.get("model"), str) or not config["model"].strip():
        raise ConfigError("Укажите непустое имя model.")
    for key in ("max_tokens", "request_timeout"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{key} должен быть положительным числом.")
    if not isinstance(config["max_tokens"], int):
        raise ConfigError("max_tokens должен быть целым числом.")
    if isinstance(config.get("temperature"), bool) or not isinstance(config.get("temperature"), (int, float)) or not 0 <= config["temperature"] <= 2:
        raise ConfigError("temperature должен быть числом от 0 до 2.")
    if not isinstance(config.get("system_prompt"), str):
        raise ConfigError("system_prompt должен быть строкой.")
    if not isinstance(config.get("request_extra"), dict):
        raise ConfigError("request_extra должен быть JSON-объектом.")
    reserved = {"model", "messages", "stream", "stream_options", "max_tokens", "temperature", "tools", "tool_choice", "parallel_tool_calls"}
    if reserved & config["request_extra"].keys():
        raise ConfigError("request_extra не должен переопределять основные поля запроса.")
    rag = config.get("rag", DEFAULT_CONFIG["rag"])
    if not isinstance(rag, dict):
        raise ConfigError("rag должен быть JSON-объектом.")
    ranges = {"chunk_size": (200, 8000), "chunk_overlap": (0, 2000),
              "top_k": (1, 20), "max_context_chars": (500, 32000)}
    for key, (minimum, maximum) in ranges.items():
        value = rag.get(key, DEFAULT_CONFIG["rag"][key])
        if type(value) is not int or not minimum <= value <= maximum:
            raise ConfigError(f"rag.{key} должен быть целым числом от {minimum} до {maximum}.")
    if rag.get("chunk_overlap", 150) >= rag.get("chunk_size", 1000):
        raise ConfigError("rag.chunk_overlap должен быть меньше rag.chunk_size.")
    server = config.get("server")
    if not isinstance(server, dict):
        raise ConfigError("server должен быть JSON-объектом.")
    for key in ("context_size", "threads", "startup_timeout"):
        value = server.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"server.{key} должен быть положительным целым числом.")
    if not isinstance(server.get("extra_args"), list) or not all(isinstance(arg, str) for arg in server["extra_args"]):
        raise ConfigError("server.extra_args должен быть списком строк.")


def write_config(path: Path, config: dict) -> None:
    validate_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
