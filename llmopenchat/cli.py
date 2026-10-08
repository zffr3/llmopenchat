"""Russian interactive console and one-command installation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .api import ApiClient, ApiError
from .config import DEFAULT_FILE, DEFAULT_REPO, ROOT, ConfigError, load_config, validate_config, write_config
from .conversation import MAX_TOOL_CALLS, MAX_TOOL_ROUNDS, TOOL_LIMIT_MESSAGE, generate_response, initial_messages
from .hotkeys import PromptAction, create_key_bindings, toolbar
from .history import load_session, new_session_path, save_session
from .harness import HarnessError, HarnessSettings, ToolHarness, ToolRequest
from .harness_status import HarnessStatus
from .menu import BACK_TO_HISTORY, BACK_TO_MENU, BACK_TO_MODELS, main_menu, print_catalog, print_installed
from .menu import tools_menu
from .models import ModelError, ModelManager
from .rag import RagError, RagSession
from .sft_cli import add_sft_parser, dispatch_sft
from .scenario_cli import add_scenario_parser, dispatch_scenario, scenario_menu
from .runtime import ManagedServer, RuntimeErrorDetail
from .terminal import terminal_text
from .telegram import run_telegram

console = Console(highlight=False)


def say(text: str) -> None:
    console.print(terminal_text(text), markup=False)


def approve_tool(request: ToolRequest) -> bool:
    """Only terminal input can approve one exact action. No remembered grants."""
    detail = request.summary + "\n\nАргументы модели:\n" + json.dumps(
        request.arguments, ensure_ascii=False, indent=2,
    )
    if request.preview:
        detail += "\n\nПредлагаемое содержимое:\n" + request.preview
    console.print()
    console.print(Panel(Text(terminal_text(detail)), title="Харнес · ожидание вашего подтверждения"))
    say("Разрешить только этот вызов: введите «разрешить». Enter, Ctrl+C или конец ввода — отказ.")
    try:
        allowed = input("Подтверждение инструмента > ").strip() == "разрешить"
    except (EOFError, KeyboardInterrupt):
        allowed = False
    say("Вызов разрешён." if allowed else "Вызов отклонён.")
    return allowed


def generate(client: ApiClient, messages: list[dict], show_reasoning: bool,
             harness: ToolHarness | None = None, status: HarnessStatus | None = None,
             *, rag: RagSession | None = None) -> str:
    reasoning_visible = False
    final_rendered: list[str] = []
    status = status or HarnessStatus(getattr(harness, "settings", HarnessSettings()))

    def render(event) -> None:
        nonlocal reasoning_visible
        if event.kind == "rag_sources":
            sources = event.value.get("sources", [])
            say(f"\n[RAG] Найдено фрагментов: {len(sources)}" if sources else "\n[RAG] Подходящие фрагменты не найдены.")
            console.print("GLM > ", style="bold green", end="")
            reasoning_visible = False
        elif event.kind in {"harness_status", "tool_start", "tool_result", "harness_summary"}:
            status.update(event.kind, event.value)
            if not status.enabled:
                return
            if event.kind == "harness_status" and event.value.get("phase") in {"requesting", "retrying"}:
                if event.value.get("phase") == "retrying":
                    final_rendered.clear()
                say("\n[Харнес] " + status.activity())
            elif event.kind == "tool_start":
                say("\n[Харнес] " + status.activity())
            elif event.kind == "tool_result":
                say("\n[Харнес] " + status.activity())
                result = event.value.get("result", {})
                if event.value.get("status") == "ok" and event.value.get("name") == "write_file":
                    say("Записан файл: " + str(result.get("path", "")))
                elif event.value.get("name") == "powershell_run":
                    if result.get("stdout"):
                        say("Вывод PowerShell:\n" + str(result["stdout"]))
                    if result.get("stderr"):
                        say("Ошибка PowerShell:\n" + str(result["stderr"]))
                    if event.value.get("status") == "error":
                        say(str(result.get("message") or result.get("error") or "Запуск не выполнен."))
                elif event.value.get("status") in {"denied", "error"}:
                    say(str(result.get("message") or result.get("error") or "Вызов не выполнен."))
            elif event.kind == "harness_summary":
                say("\n[Харнес] " + status.activity())
            reasoning_visible = False
        elif event.kind == "content":
            final_rendered.append(event.value)
            if reasoning_visible:
                console.print("\n", end="")
                reasoning_visible = False
            console.print(terminal_text(event.value), end="", markup=False, highlight=False, soft_wrap=True)
        elif event.kind == "reasoning" and show_reasoning:
            if not reasoning_visible:
                console.print("[Рассуждение] ", style="dim", end="")
                reasoning_visible = True
            console.print(terminal_text(event.value), style="dim", end="", markup=False, highlight=False, soft_wrap=True)
        elif event.kind == "tool_round":
            final_rendered.clear()
            console.print("\nGLM > ", style="bold green", end="")

    try:
        options = {"rag": rag} if rag is not None else {}
        result = generate_response(client, messages, harness, on_event=render, **options)
        if result.answer != "".join(final_rendered):
            say("\n" + result.answer)
    finally:
        console.print()
    elapsed = result.elapsed
    stats = f"{elapsed:.1f} с"
    count = result.usage.get("completion_tokens") if isinstance(result.usage, dict) else None
    if type(count) is int and 0 < count <= 1_000_000_000:
        stats += f" · {count} токенов · {count / max(elapsed, 0.001):.1f} токенов/с"
    console.print(stats, style="dim")
    if result.finish == "length":
        say("Достигнут лимит ответа. Можно увеличить max_tokens в config.json.")
    return result.answer


HELP = """/help              команды
/clear             начать новый диалог
/thinking          показать/скрыть рассуждения
/system ТЕКСТ      изменить системную инструкцию и начать новый диалог
/save [ПУТЬ]       сохранить диалог (по умолчанию текущий файл)
/load ПУТЬ         продолжить сохраненный диалог
/menu              главное меню (сохранить чат и освободить память своей модели)
/models            загрузка и выбор моделей
/history           история чатов
/tools             меню веба, кода и PowerShell; каждый вызов требует подтверждения
/rag               команды базы знаний: on/off, add ПУТЬ, list, search ЗАПРОС, remove ID, clear
/status            состояние RAG, харнеса, репозиторий и результаты последних вызовов
/quit              выйти и остановить запущенный этим клиентом сервер
Alt+Enter          новая строка; Enter отправляет сообщение
Ctrl+C             отменить генерацию или текущий ввод
F1                 справка
F2                 главное меню (черновик не отправляется)
F3                 меню инструментов текущего чата (черновик не отправляется)
Ctrl+N             новый диалог (очистить историю и текущий ввод)
Ctrl+S             сохранить диалог, сохранив текущий ввод
Ctrl+L             очистить экран, сохранив диалог и текущий ввод
Ctrl+T             показать/скрыть рассуждения, сохранив текущий ввод
Ctrl+Q             выйти и остановить свой сервер
Горячие клавиши работают при вводе сообщения; во время ответа — Ctrl+C.
Диалоги автоматически сохраняются в .local/sessions/."""


def chat(config: dict, root: Path, resumed_messages: list[dict] | None = None,
         *, rag_enabled: bool = False) -> int:
    console.print("\nllmopenchat", style="bold cyan")
    say(f"Модель: {config['target_model']}\nAPI: {config['base_url']}")
    console.print("F1 — справка · F2 — главное меню · F3 или /tools — инструменты · Ctrl+Q — выход\n", style="dim")
    say("Веб, код и PowerShell выключены. Подключение — в /tools; каждый вызов требует вашего подтверждения.")
    state_dir = root / ".local"
    state_dir.mkdir(exist_ok=True)
    session_path = new_session_path(root)
    session = None
    if sys.stdin.isatty():
        session = PromptSession(history=FileHistory(str(state_dir / "input-history")), auto_suggest=AutoSuggestFromHistory(), key_bindings=create_key_bindings())
    messages = copy.deepcopy(resumed_messages) if resumed_messages is not None else initial_messages(config)
    if resumed_messages is not None:
        say(f"Загружено сообщений: {len(messages)}. Продолжение сохранится отдельным диалогом.")
        for message in [m for m in messages if m["role"] != "system"][-6:]:
            label = {"user": "Вы", "tool": "Инструмент"}.get(message["role"], "GLM")
            content = message["content"]
            say(f"{label} > {content[:600]}" + ("…" if len(content) > 600 else ""))
    client = ApiClient(config)
    show_reasoning = bool(config.get("show_reasoning"))
    settings = HarnessSettings()
    harness_status = HarnessStatus(settings)
    rag = RagSession(root, config)
    if rag_enabled:
        say(rag.command("on"))
    say("RAG: /rag add ПУТЬ — добавить документы; /rag on — отвечать по базе знаний; /rag — справка.")
    draft = Document("")
    def persist(path: Path) -> None:
        save_session(path, messages, config["model"], target_model=config["target_model"],
                     model_id=config.get("model_package"))

    while True:
        try:
            if session:
                result = session.prompt("Вы > ", default=draft,
                                        bottom_toolbar=toolbar(show_reasoning, terminal_text(
                                            harness_status.compact() + (" · RAG: включён" if rag.enabled else ""))))
            else:
                say(harness_status.compact())
                result = input("Вы > ")
        except KeyboardInterrupt:
            draft = Document("")
            say("Ввод отменен. /quit — выход.")
            continue
        except EOFError:
            break
        if isinstance(result, PromptAction):
            prompt, draft = result.command, result.draft
        else:
            prompt, draft = result, Document("")
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.startswith("/"):
            command, _, argument = prompt.partition(" ")
            argument = argument.strip()
            if command != "/rag":
                argument = argument.strip('"')
            try:
                if command in {"/quit", "/exit"}:
                    break
                if command in {"/menu", "/models", "/history"}:
                    if any(message["role"] == "user" for message in messages):
                        persist(session_path)
                    return {"/menu": BACK_TO_MENU, "/models": BACK_TO_MODELS,
                            "/history": BACK_TO_HISTORY}[command]
                if command == "/help":
                    say(HELP)
                elif command == "/clear":
                    messages = initial_messages(config)
                    draft = Document("")
                    session_path = new_session_path(root)
                    settings = HarnessSettings()
                    harness_status = HarnessStatus(settings)
                    rag.enabled = False
                    say("Начат новый диалог.")
                    say("Доступы инструментов сброшены. Подключение — в /tools.")
                    say("RAG выключен. База знаний сохранена; /rag on — включить.")
                elif command == "/tools":
                    settings = tools_menu(settings, root, console)
                    harness_status.configure(settings)
                    say(harness_status.compact())
                elif command == "/status":
                    say(rag.status())
                    say(harness_status.details())
                    if harness_status.enabled:
                        try:
                            checked = ToolHarness(settings, lambda request: False)
                            settings = checked.settings
                            harness_status.settings = settings
                            names = [schema["function"]["name"] for schema in checked.schemas()]
                            say("Проверка подключения: готов · инструменты: " + ", ".join(names))
                            say("Работу модели подтверждают только результаты вызовов выше.")
                        except HarnessError as error:
                            harness_status.phase = "error"
                            say(f"Проверка подключения: ошибка · {error}")
                elif command == "/rag":
                    say(rag.command(argument))
                elif command == "/thinking":
                    show_reasoning = not show_reasoning
                    say("Рассуждения: " + ("показаны" if show_reasoning else "скрыты"))
                elif command == "/system":
                    config["system_prompt"] = argument
                    messages = initial_messages(config)
                    session_path = new_session_path(root)
                    settings = HarnessSettings()
                    harness_status = HarnessStatus(settings)
                    rag.enabled = False
                    say("Системная инструкция изменена. Начат новый диалог.")
                    say("Доступы инструментов сброшены. Подключение — в /tools.")
                    say("RAG выключен. База знаний сохранена; /rag on — включить.")
                elif command == "/save":
                    path = Path(argument).expanduser() if argument else session_path
                    if not path.is_absolute():
                        path = root / path
                    persist(path)
                    say(f"Сохранено: {path}")
                elif command == "/load":
                    if not argument:
                        raise ValueError("Укажите /load ПУТЬ")
                    path = Path(argument).expanduser()
                    if not path.is_absolute():
                        path = root / path
                    messages = load_session(path)
                    session_path = new_session_path(root)
                    settings = HarnessSettings()
                    harness_status = HarnessStatus(settings)
                    rag.enabled = False
                    say(f"Загружено сообщений: {len(messages)}")
                    say("Доступы инструментов сброшены. Подключение — в /tools.")
                    say("RAG выключен. База знаний сохранена; /rag on — включить.")
                else:
                    say("Неизвестная команда. /help — список команд.")
            except (OSError, ValueError) as error:
                say(f"Ошибка: {error}")
            continue
        console.print("\nGLM > ", style="bold green", end="")
        proposed = messages + [{"role": "user", "content": prompt}]
        def persist_tool_progress() -> None:
            nonlocal messages
            if len(proposed) > len(messages) + 1:
                messages = proposed
                try:
                    persist(session_path)
                    say("Вызовы инструментов сохранены в диалоге.")
                except (OSError, ValueError) as error:
                    say(f"Не удалось сохранить вызовы инструментов: {error}")
        try:
            options = {"rag": rag} if rag.enabled else {}
            if settings.web_enabled or settings.code_enabled or settings.powershell_enabled:
                def approve(request: ToolRequest) -> bool:
                    harness_status.phase = "approval"
                    harness_status.last_tool = request.name
                    allowed = approve_tool(request)
                    harness_status.phase = "executing" if allowed else "result"
                    return allowed
                harness = ToolHarness(settings, approve)
                settings = harness.settings
                harness_status.settings = settings
                answer = generate(client, proposed, show_reasoning, harness, status=harness_status, **options)
            else:
                answer = generate(client, proposed, show_reasoning, **options)
        except KeyboardInterrupt:
            harness_status.phase = "error"
            say("Генерация отменена. Незавершенный ответ не добавлен в диалог.")
            persist_tool_progress()
            continue
        except (ApiError, HarnessError, RagError) as error:
            harness_status.phase = "error"
            say(f"Ошибка: {error}\nПри переполнении контекста используйте /clear.")
            persist_tool_progress()
            continue
        messages = proposed + [{"role": "assistant", "content": answer}]
        try:
            persist(session_path)
        except (OSError, ValueError) as error:
            say(f"Не удалось автоматически сохранить диалог: {error}")
        console.print()
    say("До встречи.")
    return 0


def doctor(config: dict, root: Path) -> int:
    say(f"Python: {sys.version.split()[0]}\nСистема: {platform.platform()}\nПроект: {root}")
    say(f"Свободно на диске: {shutil.disk_usage(root).free / 2**30:.1f} ГиБ")
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        say("GPU: " + (result.stdout.strip() or result.stderr.strip()))
    for name in ("executable", "model_path"):
        path = Path(config["server"][name])
        if not path.is_absolute():
            path = root / path
        say(f"{name}: {path} ({'найден' if path.is_file() else 'не установлен'})")
    say(f"Backend: {config['backend']}\nAPI: {config['base_url']}\nМодель: {config['target_model']}")
    say("Исходная GLM-5.3 FP8: ~756 ГБ весов, пример автора — 8×H200. На RTX 4070 Ti SUPER / 64 ГБ ОЗУ не помещается.")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Локальный консольный llmopenchat с автозапуском модели.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json", help="Путь к JSON-конфигурации")
    commands = parser.add_subparsers(dest="command")
    for name in ("chat", "ask", "telegram"):
        description = {"chat": "Интерактивный чат", "ask": "Один запрос и выход",
                       "telegram": "Telegram-бот с отдельными чатами пользователей"}[name]
        sub = commands.add_parser(name, help=description)
        sub.add_argument("--no-autostart", action="store_true", help="Подключиться к уже запущенному API")
        sub.add_argument("--max-tokens", type=int)
        sub.add_argument("--show-reasoning", action="store_true")
        if name in {"chat", "ask"}:
            sub.add_argument("--rag", action="store_true", help="Отвечать по локальной базе знаний; документы добавляются через rag add или /rag add")
        if name == "ask":
            sub.add_argument("prompt", help="Текст запроса")
        elif name == "chat":
            sub.add_argument("--session", type=Path, help="Продолжить диалог из JSON-файла")
        else:
            sub.add_argument("--whitelist", type=Path, help="Файл разрешённых Telegram username через запятую (по умолчанию whitelist.txt)")
            sub.add_argument("--credits", type=Path, help="Файл токена Telegram-бота (по умолчанию credits.txt)")
    commands.add_parser("menu", help="Главное меню: чат, история, модели, Telegram-бот, RAG, SFT и сценарии")
    add_sft_parser(commands)
    add_scenario_parser(commands)
    commands.add_parser("history", help="Открыть историю чатов")
    rag_parser = commands.add_parser("rag", help="Локальная база знаний (без запуска модели)")
    rag_commands = rag_parser.add_subparsers(dest="rag_command", required=True)
    rag_add = rag_commands.add_parser("add", help="Добавить текстовый файл или каталог документов")
    rag_add.add_argument("path", type=Path)
    rag_commands.add_parser("list", help="Показать документы и их ID")
    rag_search = rag_commands.add_parser("search", help="Найти фрагменты без обращения к модели")
    rag_search.add_argument("query", nargs="+", help="Текст поискового запроса")
    rag_remove = rag_commands.add_parser("remove", help="Удалить документ из индекса; исходный файл сохраняется")
    rag_remove.add_argument("document_id")
    rag_commands.add_parser("clear", help="Очистить индекс базы знаний; исходные файлы сохраняются")
    models = commands.add_parser("models", help="Менеджер моделей (без запуска сервера)")
    model_commands = models.add_subparsers(dest="model_command")
    model_commands.add_parser("catalog", help="Показать каталог моделей до 60B")
    model_commands.add_parser("list", help="Показать установленные пакеты")
    model_install = model_commands.add_parser("install", help="Скачать модель из каталога")
    model_install.add_argument("model_id", help="ID из models catalog")
    model_install.add_argument("--quant", default="Q4_K_M", help="Квантизация GGUF, по умолчанию Q4_K_M")
    for action in ("use", "remove"):
        model_sub = model_commands.add_parser(action, help="Выбрать пакет" if action == "use" else "Удалить неактивный пакет")
        model_sub.add_argument("package_id", help="ID из models list")
    install = commands.add_parser("install", help="Установить GPU-сервер и скачать выбранную GGUF-модель")
    install.add_argument("--model", help="ID из models catalog: скачать, установить и выбрать модель")
    install.add_argument("--quant", help="Квантизация для --model, по умолчанию Q4_K_M")
    install.add_argument("--source", choices=("ollama", "huggingface"), default="ollama", help="Источник весов (по умолчанию репозиторий автора в Ollama)")
    install.add_argument("--repo", default=DEFAULT_REPO)
    install.add_argument("--file", default=DEFAULT_FILE)
    commands.add_parser("doctor", help="Проверить оборудование и установку")
    args = parser.parse_args(argv)
    try:
        if args.command == "scenario":
            return dispatch_scenario(args, ROOT)
        if args.command == "sft":
            return dispatch_sft(args, ROOT, emit=say, console=console)
        config = load_config(args.config)
        if args.command == "rag":
            argument = args.rag_command
            if args.rag_command == "add":
                argument += " " + str(args.path)
            elif args.rag_command == "search":
                argument += " " + " ".join(args.query)
            elif args.rag_command == "remove":
                argument += " " + args.document_id
            say(RagSession(ROOT, config).command(argument))
            return 0
        if args.command == "install":
            if args.model:
                if args.source != "ollama" or args.repo != DEFAULT_REPO or args.file != DEFAULT_FILE:
                    raise ModelError("Для --model не указывайте --source huggingface, --repo или --file: источник задан в каталоге.")
                manager = ModelManager(ROOT, emit=say)
                record = manager.install(args.model, args.quant or "Q4_K_M")
                config = manager.activate(record["id"], config, args.config)
                say(f"Установлена и выбрана модель: {record['name']}.\nКонфигурация: {args.config}\nЗапуск: Start-Chat.cmd")
                return 0
            if args.quant is not None:
                raise ModelError("Параметр --quant требует --model с ID из каталога.")
            from .setup import SetupError, install_local, install_ollama_local
            try:
                if args.source == "ollama":
                    if args.repo != DEFAULT_REPO or args.file != DEFAULT_FILE:
                        raise SetupError("Для --repo и --file укажите --source huggingface.")
                    installed = install_ollama_local(ROOT, emit=say)
                else:
                    installed = install_local(ROOT, args.repo, args.file, emit=say)
            except SetupError as error:
                say(f"Ошибка установки: {error}")
                return 1
            config["backend"] = "llama_cpp"
            config.pop("model_package", None)
            config["model"] = "glm-local"
            config["target_model"] = installed["repo_id"]
            config["server"]["executable"] = installed["executable"]
            config["server"]["model_path"] = installed["model_path"]
            extra_args = config["server"]["extra_args"]
            if template := installed.get("chat_template_file"):
                if "--chat-template-file" not in extra_args:
                    extra_args.extend(["--chat-template-file", template])
            elif args.repo != DEFAULT_REPO and "--chat-template-file" in extra_args:
                index = extra_args.index("--chat-template-file")
                if extra_args[index:index + 2] == ["--chat-template-file", "llmopenchat/templates/GLM-4.7-Flash.jinja"]:
                    del extra_args[index:index + 2]
            write_config(args.config, config)
            say(f"Установлено. Конфигурация: {args.config}\nЗапуск: Start-Chat.cmd")
            return 0
        if args.command == "doctor":
            return doctor(config, ROOT)
        if args.command == "models" and args.model_command:
            manager = ModelManager(ROOT, emit=say)
            if args.model_command == "catalog":
                print_catalog(manager, console)
            elif args.model_command == "list":
                print_installed(manager, config, console)
            elif args.model_command == "install":
                record = manager.install(args.model_id, args.quant)
                say(f"Установлено: {record['id']}. Выбор: models use {record['id']}")
            elif args.model_command == "use":
                config = manager.activate(args.package_id, config, args.config)
                say(f"Выбрана модель: {config['target_model']}")
            elif args.model_command == "remove":
                manager.remove(args.package_id, config)
                say(f"Удалён пакет: {args.package_id}")
            return 0
        config = copy.deepcopy(config)
        if getattr(args, "no_autostart", False):
            config["backend"] = "external"
        if getattr(args, "max_tokens", None) is not None:
            config["max_tokens"] = args.max_tokens
        if getattr(args, "show_reasoning", False):
            config["show_reasoning"] = True
        validate_config(config)
        if args.command == "telegram":
            return run_telegram(config, ROOT, whitelist_path=args.whitelist,
                                credits_path=args.credits, emit=say)
        if args.command == "ask":
            options = {}
            if args.rag:
                rag = RagSession(ROOT, config)
                rag.command("on")
                options["rag"] = rag
            with ManagedServer(config, ROOT, emit=say):
                generate(ApiClient(config), initial_messages(config) + [{"role": "user", "content": args.prompt}], bool(config["show_reasoning"]), **options)
                return 0
        def run_chat(selected_config: dict, messages: list[dict] | None = None,
                     *, rag_enabled: bool = False) -> int:
            with ManagedServer(selected_config, ROOT, emit=say):
                options = {"rag_enabled": True} if rag_enabled else {}
                return chat(selected_config, ROOT, messages, **options)
        def run_rag_chat(selected_config: dict) -> int:
            return run_chat(selected_config, rag_enabled=True)
        def start_telegram(selected_config: dict) -> int:
            return run_telegram(selected_config, ROOT, emit=say)
        def run_scenarios(selected_config: dict) -> None:
            scenario_menu(selected_config, ROOT, console)

        if args.command == "chat":
            path = args.session
            if path is not None and not path.is_absolute():
                path = ROOT / path
            options = {"rag_enabled": True} if args.rag else {}
            status = run_chat(config, load_session(path) if path else None, **options)
            if status == 0:
                return 0
            return main_menu(config, args.config, ROOT, console, run_chat, initial_view=status,
                             run_telegram=start_telegram, run_rag_chat=run_rag_chat, run_scenarios=run_scenarios)
        view = {"models": BACK_TO_MODELS, "history": BACK_TO_HISTORY}.get(args.command, BACK_TO_MENU)
        return main_menu(config, args.config, ROOT, console, run_chat, initial_view=view,
                         run_telegram=start_telegram, run_rag_chat=run_rag_chat, run_scenarios=run_scenarios)
    except (ConfigError, ModelError, RuntimeErrorDetail, ApiError, OSError, ValueError) as error:
        say(f"Ошибка: {error}")
        return 1
    except KeyboardInterrupt:
        say("Остановлено.")
        return 130
