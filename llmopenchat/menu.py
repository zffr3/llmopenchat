"""Console navigation without starting the inference server on menu screens."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .history import list_sessions, load_session
from .harness import HarnessError, HarnessSettings, ToolHarness
from .models import ModelManager, ModelError
from .runtime import RuntimeErrorDetail
from .terminal import terminal_text


BACK_TO_MENU = 2
BACK_TO_MODELS = 3
BACK_TO_HISTORY = 4


def read_choice(prompt: str = "Выбор > ") -> str:
    """EOF exits a menu and Ctrl+C backs out without a traceback."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "0"


def say(console: Console, value: str) -> None:
    console.print(terminal_text(value), markup=False)


def tools_menu(settings: HarnessSettings, root: Path, console: Console) -> HarnessSettings:
    """Human-only control plane. Settings never come from model messages/files."""
    while True:
        say(console, "\nИнструменты текущего чата\n"
            f"Веб: {'включён' if settings.web_enabled else 'выключен'}\n"
            f"Код: {'включён' if settings.code_enabled else 'выключен'}\n"
            f"PowerShell: {'включён' if settings.powershell_enabled else 'выключен'}\n"
            f"Репозиторий: {settings.repository or 'не выбран'}\n"
            "Каждое чтение, поиск, запись, запуск и веб-запрос требуют отдельного подтверждения.\n"
            "1 — Включить/выключить веб\n2 — Включить/выключить код\n"
            "3 — Выбрать репозиторий и включить код\n4 — Отключить всё\n"
            "5 — Включить/выключить PowerShell (ограниченный сеанс)\n0 — Вернуться в чат")
        choice = read_choice("Инструменты > ")
        if choice == "0":
            return settings
        try:
            if choice == "1":
                settings = replace(settings, web_enabled=not settings.web_enabled)
                say(console, "Веб включён." if settings.web_enabled else "Веб выключен.")
            elif choice == "2" and settings.code_enabled:
                settings = replace(settings, code_enabled=False)
                say(console, "Код выключен.")
            elif choice == "5" and settings.powershell_enabled:
                settings = replace(settings, powershell_enabled=False)
                say(console, "PowerShell выключен.")
            elif choice in {"2", "3", "5"}:
                repository = settings.repository
                identity = settings.repository_identity
                if choice == "3" or repository is None:
                    say(console, "Укажите существующий каталог репозитория. Enter или 0 — отмена.")
                    selection = read_choice("Репозиторий > ").strip('"')
                    if selection in {"", "0"}:
                        continue
                    repository = Path(selection).expanduser()
                    if not repository.is_absolute():
                        repository = root / repository
                    identity = None
                toggles = {"powershell_enabled": True} if choice == "5" else {"code_enabled": True}
                candidate = replace(settings, repository=repository, repository_identity=identity, **toggles)
                # The same security checks used by tools validate this boundary;
                # construction performs no model action and has no approval grant.
                checked = ToolHarness(candidate, lambda request: False)
                settings = checked.settings
                if choice == "5":
                    say(console, f"PowerShell включён. Репозиторий: {settings.repository}")
                    say(console, "Запуск .ps1 после отдельного подтверждения; разрешены вычисления и вывод. "
                        "Внешние программы, .NET, сеть и файловые команды запрещены.")
                else:
                    say(console, f"Код включён. Репозиторий: {settings.repository}")
            elif choice == "4":
                settings = HarnessSettings()
                say(console, "Веб, код и PowerShell выключены.")
            else:
                say(console, "Введите 1, 2, 3, 4, 5 или 0.")
        except (HarnessError, OSError, ValueError) as error:
            say(console, f"Ошибка подключения инструмента: {error}")


def gib(size: int) -> str:
    return f"{size / 2**30:.2f} ГиБ"


def print_catalog(manager: ModelManager, console: Console, query: str = "") -> list:
    query = query.casefold().strip()
    if query in {"маленькие", "компактные", "small"}:
        specs = [spec for spec in manager.catalog() if spec.parameters_b <= 10]
    elif query in {"большие", "large", "20-60b", "20–60b"}:
        specs = [spec for spec in manager.catalog() if 20 <= spec.parameters_b <= 60]
    else:
        specs = [spec for spec in manager.catalog() if not query or query in " ".join(
            (spec.id, spec.name, spec.description, spec.focus, spec.censorship, f"{spec.parameters_b:g}B")
        ).casefold()]
    table = Table(title="Каталог GGUF · до 60B полных параметров", expand=True, show_lines=True)
    for label in ("№", "Модель", "Параметры", "Загрузка", "Описание / назначение"):
        table.add_column(label, no_wrap=label in {"№", "Параметры", "Загрузка"})
    for number, spec in enumerate(specs, 1):
        summary = spec.description.split(". ", 1)[0].rstrip(".") + "."
        table.add_row(str(number), Text(spec.name + "\n" + spec.id),
                      f"{spec.parameters_b:g}B", f"{spec.default_quantization}\n{gib(spec.default_size)}",
                      Text(summary + "\nНазначение: " + spec.focus))
    console.print(table)
    say(console, "GGUF-квантизация уменьшает размер весов. Память нужна также для контекста.\n"
                 "На GPU 16 ГБ веса моделей 20–60B могут частично размещаться в ОЗУ; "
                 "расход памяти зависит от квантизации и контекста.\n"
                 "Метки abliterated/uncensored описывают заявления авторов; отказы всё ещё возможны.")
    return specs


def print_installed(manager: ModelManager, config: dict, console: Console) -> list[dict]:
    records = manager.installed(config)
    table = Table(title="Установленные модели", expand=True)
    for label in ("№", "Модель / пакет", "Квантизация", "Размер", "Состояние"):
        table.add_column(label)
    states = {"ready": "Готова", "missing": "Нет файлов", "corrupt": "Повреждена"}
    for number, record in enumerate(records, 1):
        state = states.get(record["status"], record["status"])
        if record.get("active"):
            state += " · выбрана"
        table.add_row(str(number), Text(record["name"] + "\n" + record["id"]),
                      record.get("quantization", "—"), gib(record.get("total_size", 0)), state)
    console.print(table)
    if not records:
        say(console, "Пока нет установленных моделей. Откройте каталог и выберите модель.")
    return records


def install_card(manager: ModelManager, spec, console: Console) -> None:
    say(console, f"\n{spec.name} · {spec.parameters_b:g}B\n{spec.description}\n"
                 f"Назначение: {spec.focus}\nПоведение отказов: {spec.censorship}\n"
                 f"Карточка автора: {spec.source_url}\nGGUF: https://huggingface.co/{spec.repo_id}")
    say(console, "Проверяю доступные GGUF и размеры на Hugging Face…")
    variants = sorted(manager.available_quantizations(spec.id),
                      key=lambda item: (item["quantization"] != spec.default_quantization, item["quantization"]))
    if not variants:
        raise ModelError("В репозитории нет подходящих GGUF-квантизаций.")
    table = Table(title="Квантизация · точный размер загрузки")
    for label in ("№", "Вариант", "Размер", "Файлов"):
        table.add_column(label)
    for number, variant in enumerate(variants, 1):
        recommended = " · рекомендуется" if variant["quantization"] == spec.default_quantization else ""
        table.add_row(str(number), variant["quantization"] + recommended,
                      gib(variant["size"]), str(len(variant["files"])))
    console.print(table)
    has_recommended = variants[0]["quantization"] == spec.default_quantization
    say(console, "Q4_K_M — разумный первый выбор; Q5/Q6 требуют больше памяти.\n"
                 "0 — назад. Введите номер варианта, чтобы скачать и установить его." +
                 ("\nEnter — рекомендуемый вариант в первой строке." if has_recommended else ""))
    choice = read_choice("Установить вариант > ")
    if not choice and has_recommended:
        choice = "1"
    if choice == "0":
        return
    if not choice.isdigit() or not 1 <= int(choice) <= len(variants):
        say(console, "Неверный номер; установка отменена.")
        return
    variant = variants[int(choice) - 1]
    record = manager.install(spec.id, variant["quantization"])
    say(console, f"Установлено: {record['name']} ({record['id']}).\n"
                 "Чтобы использовать модель, выберите её в разделе «Установленные». ")


def models_menu(config: dict, config_path: Path, manager: ModelManager, console: Console) -> dict:
    while True:
        say(console, "\nМодели\n1 — Каталог и установка\n2 — Установленные: выбор и удаление\n0 — Главное меню")
        choice = read_choice()
        if choice == "0":
            return config
        try:
            if choice == "1":
                query = ""
                while True:
                    specs = print_catalog(manager, console, query)
                    say(console, "Номер — описание и установка; текст — поиск (анализ текста, маленькие, большие, код, 3B); 0 — назад.")
                    selection = read_choice("Каталог > ")
                    if selection == "0":
                        break
                    if selection.isdigit():
                        index = int(selection) - 1
                        if 0 <= index < len(specs):
                            install_card(manager, specs[index], console)
                        else:
                            say(console, "Нет модели с таким номером.")
                    else:
                        query = selection
            elif choice == "2":
                records = print_installed(manager, config, console)
                say(console, "Номер — использовать модель; у НОМЕР — удалить пакет; 0 — назад.")
                selection = read_choice("Установленные > ")
                if selection == "0":
                    continue
                deleting = selection.casefold().startswith("у ")
                number = selection[2:].strip() if deleting else selection
                if not number.isdigit() or not 1 <= int(number) <= len(records):
                    say(console, "Нет пакета с таким номером.")
                    continue
                record = records[int(number) - 1]
                if deleting:
                    if record.get("active"):
                        say(console, "Сначала выберите другую модель; текущий пакет удалять нельзя.")
                    elif not record.get("managed"):
                        say(console, "Эта модель установлена вне менеджера. Её файлы удаляются вручную.")
                    else:
                        say(console, f"Удалить {record['name']} и освободить {gib(record['total_size'])}?\n"
                                     "Введите «удалить» для подтверждения; Enter — отмена.")
                        if read_choice("Подтверждение > ").casefold() == "удалить":
                            manager.remove(record["id"], config)
                            say(console, "Пакет удалён.")
                else:
                    config = manager.activate(record["id"], config, config_path)
                    say(console, f"Выбрана модель: {record['name']}. Она загрузится при входе в чат.")
            else:
                say(console, "Введите 1, 2 или 0.")
        except KeyboardInterrupt:
            say(console, "Загрузка отменена; повторная установка продолжит скачивание.")
        except (ModelError, OSError, ValueError) as error:
            say(console, f"Ошибка: {error}")


def history_menu(root: Path, config: dict, console: Console) -> list[dict] | None:
    entries = list_sessions(root / ".local" / "sessions", emit=lambda text: say(console, text))
    if not entries:
        say(console, "История пока пуста. Диалоги сохраняются после успешного ответа.")
        return None
    table = Table(title="История чатов · сначала новые", expand=True)
    for label in ("№", "Диалог", "Дата", "Сообщений", "Модель"):
        table.add_column(label)
    for number, entry in enumerate(entries, 1):
        table.add_row(str(number), Text(entry.title), entry.saved_at.astimezone().strftime("%d.%m.%Y %H:%M"),
                      str(entry.message_count), Text(entry.display_model))
    console.print(table)
    say(console, "Номер — продолжить чат с выбранной сейчас моделью; 0 — назад.\n"
                 "Продолжение сохранится отдельным диалогом.")
    while True:
        choice = read_choice("История > ")
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(entries):
            entry = entries[int(choice) - 1]
            if entry.target_model and entry.target_model != config["target_model"]:
                say(console, f"В исходном диалоге: {entry.target_model}; продолжение: {config['target_model']}.")
            return load_session(entry.path)
        say(console, "Введите номер диалога или 0.")


def main_menu(config: dict, config_path: Path, root: Path, console: Console,
              run_chat: Callable[[dict, list[dict] | None], int], initial_view: int = BACK_TO_MENU,
              *, run_telegram: Callable[[dict], int] | None = None,
              run_rag_chat: Callable[[dict], int] | None = None,
              run_scenarios: Callable[[dict], None] | None = None) -> int:
    manager = ModelManager(root, emit=lambda text: say(console, text))
    view = initial_view
    while True:
        console.print("\nllmopenchat", style="bold cyan")
        say(console, f"Выбрана модель: {config['target_model']}\n"
                     "1 — Новый чат\n2 — История чатов\n3 — Модели: загрузка и выбор\n" +
                     ("4 — Telegram-бот\n" if run_telegram is not None else "") +
                     ("5 — Чат по базе знаний (RAG)\n" if run_rag_chat is not None else "") +
                     "6 — Дообучение (SFT)\n" +
                     ("7 — Пользовательские сценарии\n" if run_scenarios is not None else "") +
                     "0 — Выход")
        choice = {BACK_TO_MODELS: "3", BACK_TO_HISTORY: "2"}.get(view) or read_choice()
        view = BACK_TO_MENU
        if choice == "0":
            say(console, "До встречи.")
            return 0
        try:
            if choice == "7" and run_scenarios is not None:
                run_scenarios(config)
                continue
            if choice == "6":
                from .sft_cli import sft_menu
                sft_menu(root, console)
                continue
            if choice == "4" and run_telegram is not None:
                run_telegram(config)
                continue
            if choice == "5" and run_rag_chat is not None:
                status = run_rag_chat(config)
                if status == 0:
                    return 0
                view = status
                continue
            if choice == "3":
                config = models_menu(config, config_path, manager, console)
                continue
            if choice == "2":
                messages = history_menu(root, config, console)
                if messages is None:
                    continue
            elif choice == "1":
                messages = None
            else:
                choices = ["1", "2", "3"]
                if run_telegram is not None:
                    choices.append("4")
                if run_rag_chat is not None:
                    choices.append("5")
                choices.append("6")
                if run_scenarios is not None:
                    choices.append("7")
                say(console, "Введите " + ", ".join(choices) + " или 0.")
                continue
            status = run_chat(config, messages)
            if status == 0:
                return 0
            view = status
        except KeyboardInterrupt:
            say(console, "Запуск отменён. Вы вернулись в меню.")
        except (ModelError, RuntimeErrorDetail, OSError, ValueError) as error:
            say(console, f"Ошибка: {error}\nОткройте «Модели» для установки или выбора другой модели.")
