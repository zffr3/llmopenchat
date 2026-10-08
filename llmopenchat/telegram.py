"""Private Telegram conversations backed by the client's approved tool harness."""

from __future__ import annotations

import copy
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from .api import ApiClient, ApiError
from .conversation import GenerationCancelled, generate_response, initial_messages
from .harness import HarnessError, HarnessSettings, ToolHarness, ToolRequest
from .harness_status import HarnessStatus
from .history import list_sessions, load_session, new_session_path, save_session
from .models import ModelError, ModelManager
from .rag import MAX_FILE_BYTES, SUPPORTED_SUFFIXES, RagSession
from .runtime import ManagedServer, RuntimeErrorDetail
from .telegram_api import TelegramAPI, TelegramError


APPROVAL_TIMEOUT = 300.0
BUTTON_TIMEOUT = 900.0
HELP = """/start — открыть свой чат
/id — узнать свой Telegram username и ID
/help — команды
/status — состояние чата, харнесов и RAG
/clear — новый диалог и сброс харнесов
/thinking — показать/скрыть рассуждения
/system ТЕКСТ — системная инструкция и новый диалог
/save [ФАЙЛ.json] — сохранить и получить JSON диалога
/load ФАЙЛ.json — продолжить свой сохранённый диалог
/menu — главное меню
/models — каталог, установка, выбор и удаление моделей
/history [НОМЕР] — свои сохранённые диалоги
/rag — база знаний текущего чата
/rag on|off — отвечать с поиском по своим документам
/rag add ФАЙЛ — добавить файл из своей рабочей директории
/rag list|remove ID|search ЗАПРОС|clear — управление базой знаний
/tools — подключить веб, код и PowerShell кнопками
/tools web on|off — веб
/tools code on|off — код в своей рабочей директории
/tools powershell on|off — ограниченный PowerShell в своей рабочей директории
/tools repo ПАПКА — подпапка своей рабочей директории
/tools off — отключить все харнесы
/cancel — отменить текущий запрос или апрув
/quit или /exit — закрыть свой чат
Каждый вызов харнеса требует отдельного апрува кнопкой.
Документ, отправленный боту, добавляется в вашу базу знаний; /rag on — включить поиск.
Модель общая для бота; контекст, файлы, база знаний и доступы — персональные."""

COMMANDS = [
    {"command": command, "description": description} for command, description in (
        ("start", "Открыть свой чат"), ("id", "Мой Telegram username и ID"),
        ("help", "Все команды"), ("clear", "Новый диалог"),
        ("status", "Состояние чата"), ("rag", "Своя база знаний RAG"),
        ("thinking", "Показать или скрыть рассуждения"),
        ("system", "Изменить системную инструкцию"), ("save", "Сохранить диалог"),
        ("load", "Загрузить свой диалог"), ("menu", "Главное меню"),
        ("models", "Модели"), ("history", "Своя история чатов"),
        ("tools", "Подключить харнесы"), ("cancel", "Отменить запрос"),
        ("quit", "Закрыть свой чат"), ("exit", "Закрыть свой чат"),
    )
]


def _normalize_username(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    username = value.removeprefix("@")
    if (not re.fullmatch(r"[A-Za-z0-9_]{1,32}", username)
            or not re.search(r"[A-Za-z]", username)):
        return None
    return username.casefold()


def read_whitelist(path: Path) -> set[str]:
    """Accept comma-separated usernames with optional @, ignoring letter case."""
    with Path(path).open("r", encoding="utf-8-sig") as source:
        text = source.read(65537).strip()
    if len(text) > 65536:
        raise ValueError("whitelist.txt слишком большой.")
    if not text:
        return set()
    usernames = {_normalize_username(entry.strip()) for entry in text.split(",")}
    if None in usernames:
        raise ValueError("whitelist.txt: нужны Telegram username через запятую, например @alice, bob_user.")
    return usernames


def read_token(path: Path) -> str:
    try:
        with Path(path).open("r", encoding="utf-8-sig") as source:
            token = source.read(1025).strip()
    except OSError:
        raise ValueError("Создайте credits.txt с токеном Telegram-бота от BotFather.") from None
    if not re.fullmatch(r"[1-9][0-9]*:[A-Za-z0-9_-]{20,200}", token):
        raise ValueError("credits.txt должен содержать только токен Telegram-бота от BotFather.")
    return token


def _no_links(path: Path, boundary: Path) -> None:
    path.relative_to(boundary)
    for current in (path, *path.parents):
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("Ссылки и junction в каталогах Telegram запрещены.")
        if current.exists() and getattr(current.stat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("Reparse points в каталогах Telegram запрещены.")
        if current == boundary:
            break


def _make_directory(path: Path, boundary: Path) -> None:
    _no_links(path, boundary)
    path.mkdir(parents=True, exist_ok=True)
    _no_links(path, boundary)


def _relative_path(boundary: Path, value: str, *, directory: bool = False) -> Path:
    value = value.strip().strip('"').replace("\\", "/")
    if directory and value in {"", "."}:
        return boundary
    parts = value.split("/")
    if any(not part or part in {".", ".."} or part.endswith((".", " "))
           or re.search(r'[<>:"|?*~\x00-\x1f]', part)
           or part.split(".", 1)[0].casefold() in {"con", "prn", "aux", "nul", *(
               name + str(index) for name in ("com", "lpt") for index in range(1, 10))}
           for part in parts):
        raise ValueError("Укажите относительный путь внутри своего каталога без .. и ссылок.")
    result = boundary.joinpath(*parts)
    _no_links(result, boundary)
    result.resolve().relative_to(boundary.resolve())
    return result


@dataclass
class UserChat:
    user_id: int
    directory: Path
    workspace: Path
    config: dict
    messages: list[dict]
    session_path: Path
    rag: RagSession
    settings: HarnessSettings = field(default_factory=HarnessSettings)
    harness_status: HarnessStatus | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    close_requested: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    worker: threading.Thread | None = None

    @property
    def sessions(self) -> Path:
        return self.directory / ".local" / "sessions"


@dataclass
class PendingApproval:
    user_id: int
    chat_id: int
    event: threading.Event = field(default_factory=threading.Event)
    message_id: int | None = None
    allowed: bool = False
    expires: float = field(default_factory=lambda: time.monotonic() + APPROVAL_TIMEOUT)


@dataclass
class CommandButton:
    user_id: int
    chat_id: int
    command: str
    message_id: int | None = None
    expires: float = field(default_factory=lambda: time.monotonic() + BUTTON_TIMEOUT)


class BotRuntime:
    """Serialize inference and model changes while polling remains responsive."""

    def __init__(self, config: dict, root: Path, emit: Callable[[str], None]):
        self.config = copy.deepcopy(config)
        self.root = root
        self.emit = emit
        self.lock = threading.RLock()
        self.server: ManagedServer | None = None
        self.closed = threading.Event()

    def start(self) -> None:
        with self.lock:
            if self.closed.is_set():
                raise RuntimeErrorDetail("Telegram-бот останавливается.")
            server = ManagedServer(self.config, self.root, emit=self.emit)
            self.server = server
            try:
                server.__enter__()
                if self.closed.is_set():
                    server.stop()
                    raise RuntimeErrorDetail("Telegram-бот останавливается.")
            except BaseException:
                self.server = None
                raise

    def stop(self) -> None:
        with self.lock:
            if self.server is not None:
                self.server.stop()
                self.server = None

    def select(self, manager: ModelManager, package: str,
               *, cancelled: Callable[[], bool] | None = None) -> None:
        with self.lock:
            if self.closed.is_set():
                raise RuntimeErrorDetail("Telegram-бот останавливается.")
            previous = self.config
            selected = manager.activate(package, previous)
            if self.closed.is_set() or (cancelled is not None and cancelled()):
                raise GenerationCancelled("Переключение модели отменено.")
            self.stop()
            self.config = selected
            try:
                self.start()
            except (RuntimeErrorDetail, OSError, ValueError):
                self.config = previous
                if not self.closed.is_set():
                    self.start()
                raise

    def shutdown(self) -> None:
        # Shutdown must not wait for an inference worker holding the runtime
        # lock while its HTTP peer is silent. No new server may start afterward.
        self.closed.set()
        server = self.server
        if server is not None:
            server.stop()


class TelegramBot:
    def __init__(self, api: TelegramAPI, config: dict, root: Path,
                 whitelist_path: Path, *, runtime: BotRuntime | None = None,
                 emit: Callable[[str], None] = print):
        self.api = api
        self.root = Path(root).resolve()
        self.whitelist_path = Path(whitelist_path)
        self.runtime = runtime or BotRuntime(config, self.root, emit)
        self.emit = emit
        self.users: dict[int, UserChat] = {}
        self.approvals: dict[str, PendingApproval] = {}
        self.buttons: dict[str, CommandButton] = {}
        self.control_lock = threading.RLock()
        self.stopped = threading.Event()
        self.username = ""
        # Access names come only from Telegram's sender object. Persistent
        # conversation and approval ownership remain bound to numeric IDs.
        self.usernames: dict[int, str | None] = {}
        self.clients: dict[int, ApiClient] = {}

    def allowed(self, user_id: int) -> bool:
        try:
            with self.control_lock:
                username = self.usernames.get(user_id)
            return username is not None and username in read_whitelist(self.whitelist_path)
        except (OSError, UnicodeError, ValueError):
            return False

    def _remember_sender(self, sender: dict) -> None:
        """Refresh access on every private message/click, including removals."""
        user_id = sender["id"]
        username = _normalize_username(sender.get("username"))
        with self.control_lock:
            displaced = [other for other, known in self.usernames.items()
                         if username is not None and known == username and other != user_id]
            for other in displaced:
                self.usernames[other] = None
            self.usernames[user_id] = username
            revoked = displaced + ([] if self.allowed(user_id) else [user_id])
            for other in revoked:
                user = self.users.get(other)
                if user is not None:
                    user.cancelled.set()
                client = self.clients.get(other)
                if client is not None:
                    client.cancel()
                for pending in self.approvals.values():
                    if pending.user_id == other:
                        pending.allowed = False
                        pending.event.set()

    def user_chat(self, user_id: int) -> UserChat:
        if type(user_id) is not int or user_id <= 0:
            raise ValueError("Неверный Telegram ID.")
        with self.control_lock:
            if user_id not in self.users:
                directory = self.root / ".local" / "telegram" / "users" / str(user_id)
                workspace = directory / "workspace"
                _make_directory(workspace, self.root)
                _make_directory(directory / ".local" / "sessions", self.root)
                config = copy.deepcopy(self.runtime.config)
                current = directory / "current.json"
                _no_links(current, self.root)
                messages = load_session(current) if current.exists() else initial_messages(config)
                if current.exists():
                    config["system_prompt"] = next((m["content"] for m in messages if m["role"] == "system"), "")
                self.users[user_id] = UserChat(user_id, directory, workspace, config, messages,
                                              new_session_path(directory),
                                              RagSession(directory, config, boundary=workspace))
            return self.users[user_id]

    def _save(self, user: UserChat, path: Path | None = None) -> None:
        config = self.runtime.config
        for target in (path or user.session_path, user.directory / "current.json"):
            _no_links(target, self.root)
            save_session(target, user.messages, config["model"],
                         target_model=config.get("target_model"), model_id=config.get("model_package"))

    def _reset(self, user: UserChat) -> None:
        if any(message["role"] == "user" for message in user.messages):
            self._save(user)
        user.messages = initial_messages(user.config)
        user.session_path = new_session_path(user.directory)
        user.settings = HarnessSettings()
        user.harness_status = None
        user.rag.enabled = False
        with self.control_lock:
            self.buttons = {token: action for token, action in self.buttons.items() if action.user_id != user.user_id}
        _no_links(user.directory / "current.json", self.root)
        save_session(user.directory / "current.json", user.messages, self.runtime.config["model"])

    def _send(self, chat_id: int, text: str, reply_markup: dict | None = None) -> dict:
        return self.api.send_message(chat_id, text, reply_markup=reply_markup)

    def _menu(self, user: UserChat, chat_id: int, text: str,
              rows: list[list[tuple[str, str]]]) -> None:
        tokens = []
        keyboard = []
        with self.control_lock:
            self.buttons = {token: action for token, action in self.buttons.items()
                            if action.user_id != user.user_id and action.expires > time.monotonic()}
            for row in rows:
                buttons = []
                for label, command in row:
                    token = secrets.token_urlsafe(12)
                    tokens.append(token)
                    self.buttons[token] = CommandButton(user.user_id, chat_id, command)
                    buttons.append({"text": label, "callback_data": "cmd:" + token})
                keyboard.append(buttons)
        try:
            message = self._send(chat_id, text, {"inline_keyboard": keyboard})
            with self.control_lock:
                for token in tokens:
                    if token in self.buttons:
                        self.buttons[token].message_id = message["message_id"]
        except Exception:
            with self.control_lock:
                for token in tokens:
                    self.buttons.pop(token, None)
            raise

    def approve(self, user: UserChat, chat_id: int, request: ToolRequest) -> bool:
        if self._cancelled(user):
            return False
        detail = request.summary + "\n\nАргументы модели:\n" + json.dumps(
            request.arguments, ensure_ascii=False, indent=2)
        if request.preview:
            heading = "Полный скрипт PowerShell:" if request.name == "powershell_run" else "Полное предлагаемое содержимое:"
            detail += "\n\n" + heading + "\n" + request.preview
        if request.name == "powershell_run":
            detail += "\n\nРазрешены только вычисления и вывод. Файловые команды, сеть и запуск других программ запрещены."
        token = secrets.token_urlsafe(16)
        pending = PendingApproval(user.user_id, chat_id)
        with self.control_lock:
            self.approvals[token] = pending
        try:
            if len(detail.encode("utf-16-le")) > 20000:
                self.api.send_document(chat_id, "approval.txt", detail.encode("utf-8"),
                                       caption="Полные аргументы и содержимое для этого апрува")
                detail = request.summary + "\n\nПолные детали — в файле approval.txt выше."
            message = self._send(chat_id, "Апрув одного вызова\n\n" + detail
                                 + "\n\nБез ответа за 5 минут вызов будет отклонён.", {
                "inline_keyboard": [[
                    {"text": "Разрешить", "callback_data": f"approve:{token}:yes"},
                    {"text": "Отклонить", "callback_data": f"approve:{token}:no"},
                ]]})
            pending.message_id = message["message_id"]
            while not pending.event.wait(0.25):
                if self._cancelled(user) or time.monotonic() >= pending.expires:
                    break
            return pending.allowed and pending.event.is_set() and not self._cancelled(user)
        except (TelegramError, OSError):
            return False
        finally:
            with self.control_lock:
                self.approvals.pop(token, None)
            if pending.message_id is not None:
                try:
                    self.api.edit_message_reply_markup(chat_id, pending.message_id)
                except TelegramError:
                    pass

    def _cancelled(self, user: UserChat) -> bool:
        return self.stopped.is_set() or user.cancelled.is_set() or not self.allowed(user.user_id)

    def handle_update(self, update: dict) -> None:
        if not isinstance(update, dict):
            return
        if isinstance(update.get("callback_query"), dict):
            self._callback(update["callback_query"])
            return
        message = update.get("message")
        if not isinstance(message, dict):
            return
        sender, chat = message.get("from", {}), message.get("chat", {})
        if not isinstance(sender, dict) or not isinstance(chat, dict):
            return
        user_id, chat_id = sender.get("id"), chat.get("id")
        if (type(user_id) is not int or user_id <= 0 or type(chat_id) is not int
                or chat.get("type") != "private" or chat_id != user_id or sender.get("is_bot")):
            return
        self._remember_sender(sender)
        document = message.get("document")
        if isinstance(document, dict) and self.allowed(user_id):
            self.dispatch(user_id, chat_id, "", document=document)
            return
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            if self.allowed(user_id):
                self._send(chat_id, "Отправьте текст или команду /help.")
            return
        text = text.strip()
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            command, argument = parts[0], parts[1] if len(parts) > 1 else ""
            command, _, mention = command.partition("@")
            if mention and mention.casefold() != self.username.casefold():
                return
            text = command.casefold() + (" " + argument if argument else "")
        if text == "/id":
            username = self.usernames.get(user_id)
            self._send(chat_id, "Ваш Telegram username: " + ("@" + username if username else "не задан")
                       + f"\nTelegram ID: {user_id}")
            return
        if not self.allowed(user_id):
            username = self.usernames.get(user_id)
            self._send(chat_id, f"Доступ запрещён. Ваш Telegram username: @{username}. Добавьте его в whitelist.txt на ПК клиента."
                       if username else "Доступ запрещён. Задайте username в настройках Telegram и добавьте его в whitelist.txt на ПК клиента.")
            return
        self.dispatch(user_id, chat_id, text)

    def dispatch(self, user_id: int, chat_id: int, text: str,
                 *, document: dict | None = None) -> None:
        if self.stopped.is_set() or not self.allowed(user_id):
            return
        user = self.user_chat(user_id)
        if text in {"/cancel", "/quit", "/exit"} and user.lock.locked():
            user.cancelled.set()
            if text != "/cancel":
                user.close_requested.set()
            with self.control_lock:
                client = self.clients.get(user_id)
                if client is not None:
                    client.cancel()
                for pending in self.approvals.values():
                    if pending.user_id == user_id:
                        pending.event.set()
            self._send(chat_id, "Запрос отменяется; ваш чат будет закрыт."
                       if text != "/cancel" else "Запрос отменяется.")
            return
        if not user.lock.acquire(blocking=False):
            self._send(chat_id, "Предыдущий запрос ещё выполняется. Для отмены: /cancel.")
            return
        user.cancelled.clear()
        arguments = (user, chat_id, text) if document is None else (user, chat_id, text, document)
        user.worker = threading.Thread(target=self._work, args=arguments,
                                       name=f"telegram-user-{user_id}", daemon=True)
        user.worker.start()

    def _callback(self, callback: dict) -> None:
        callback_id = callback.get("id")
        if not isinstance(callback_id, str):
            return
        sender, message = callback.get("from", {}), callback.get("message", {})
        if not isinstance(sender, dict) or not isinstance(message, dict):
            try:
                self.api.answer_callback_query(callback_id, text="Кнопка недоступна.")
            except TelegramError:
                pass
            return
        chat = message.get("chat", {})
        if not isinstance(chat, dict):
            return
        user_id, chat_id = sender.get("id"), chat.get("id")
        data, message_id = callback.get("data"), message.get("message_id")
        command = None
        response = "Кнопка устарела или принадлежит другому пользователю."
        valid_sender = (type(user_id) is int and user_id > 0 and chat.get("type") == "private"
                        and chat_id == user_id and not sender.get("is_bot"))
        if valid_sender:
            self._remember_sender(sender)
        if valid_sender and self.allowed(user_id) and isinstance(data, str):
            with self.control_lock:
                if data.startswith("approve:"):
                    parts = data.split(":")
                    pending = self.approvals.get(parts[1]) if len(parts) == 3 else None
                    if (pending and parts[2] in {"yes", "no"} and pending.user_id == user_id
                            and pending.chat_id == chat_id and pending.message_id == message_id
                            and time.monotonic() < pending.expires and not pending.event.is_set()):
                        pending.allowed = parts[2] == "yes"
                        pending.event.set()
                        response = "Вызов разрешён." if pending.allowed else "Вызов отклонён."
                elif data.startswith("cmd:"):
                    token = data[4:]
                    button = self.buttons.get(token)
                    if (button and button.user_id == user_id and button.chat_id == chat_id
                            and button.message_id == message_id and time.monotonic() < button.expires):
                        self.buttons.pop(token)
                        command = button.command
                        response = "Команда выбрана."
        try:
            self.api.answer_callback_query(callback_id, text=response)
        except TelegramError:
            pass
        if command is not None:
            try:
                self.api.edit_message_reply_markup(chat_id, message_id)
            except TelegramError:
                pass
            self.dispatch(user_id, chat_id, command)

    def _work(self, user: UserChat, chat_id: int, text: str,
              document: dict | None = None) -> None:
        user.harness_status = None
        try:
            if not self._cancelled(user):
                if document is not None:
                    self.add_document(user, chat_id, document)
                elif text.startswith("/"):
                    self.command(user, chat_id, text)
                else:
                    self.generate(user, chat_id, text)
        except GenerationCancelled:
            try:
                text = "Генерация отменена. Выполненные вызовы инструментов сохранены."
                if user.harness_status is not None and user.harness_status.enabled:
                    text += "\n\n" + user.harness_status.details()
                self._send(chat_id, text)
            except TelegramError:
                self.emit(f"Telegram: не удалось доставить сообщение пользователю {user.user_id}.")
        except (ApiError, HarnessError, ModelError, RuntimeErrorDetail, OSError, ValueError) as error:
            try:
                text = f"Ошибка: {error}"
                if user.harness_status is not None and user.harness_status.enabled:
                    text += "\n\n" + user.harness_status.details()
                self._send(chat_id, text)
            except TelegramError:
                self.emit(f"Telegram: не удалось доставить сообщение пользователю {user.user_id}.")
        except TelegramError:
            self.emit(f"Telegram: ошибка доставки пользователю {user.user_id}.")
        finally:
            if user.close_requested.is_set():
                try:
                    self._reset(user)
                    self._send(chat_id, "Ваш чат закрыт. /start — открыть новый.")
                except (TelegramError, OSError, ValueError):
                    self.emit(f"Telegram: не удалось завершить закрытие чата {user.user_id}.")
                user.close_requested.clear()
            user.lock.release()

    def command(self, user: UserChat, chat_id: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        command, argument = parts[0], parts[1] if len(parts) > 1 else ""
        argument = argument.strip()
        if command == "/help":
            self._send(chat_id, HELP)
        elif command == "/status":
            harness = user.harness_status or HarnessStatus(settings=user.settings)
            self._send(chat_id, f"Модель: {self.runtime.config['target_model']}\n"
                       f"Рабочая директория: {user.workspace}\n"
                       + harness.details().replace("F3 — подключить", "/tools — подключить")
                       + "\n" + user.rag.status())
        elif command in {"/start", "/menu"}:
            if command == "/menu":
                self._save(user)
            self._menu(user, chat_id, f"llmopenchat\nМодель: {self.runtime.config['target_model']}\n"
                       f"Рабочая директория: {user.workspace}\n/help — все команды", [
                [("Новый диалог", "/clear"), ("История", "/history")],
                [("Харнесы", "/tools"), ("Модели", "/models")],
                [("База знаний RAG", "/rag"), ("Состояние", "/status")],
                [("Рассуждения", "/thinking"), ("Сохранить", "/save")],
            ])
        elif command in {"/clear", "/system", "/quit", "/exit"}:
            if command == "/system":
                user.config["system_prompt"] = argument
            self._reset(user)
            self._send(chat_id, "Ваш чат закрыт. /start — открыть новый." if command in {"/quit", "/exit"}
                       else "Начат новый диалог. Харнесы и RAG отключены; /tools и /rag — подключение.")
        elif command == "/thinking":
            user.config["show_reasoning"] = not user.config.get("show_reasoning", False)
            self._send(chat_id, "Рассуждения: " + ("показаны" if user.config["show_reasoning"] else "скрыты"))
        elif command == "/cancel":
            self._send(chat_id, "Сейчас нет выполняющегося запроса.")
        elif command == "/save":
            path = _relative_path(user.sessions, argument) if argument else user.session_path
            if path.suffix.casefold() != ".json":
                raise ValueError("Укажите имя файла с расширением .json.")
            self._save(user, path)
            self.api.send_document(chat_id, path.name, path.read_bytes(), caption="Сохранённый диалог: " + path.relative_to(user.sessions).as_posix())
        elif command == "/load":
            if not argument:
                raise ValueError("Укажите /load ФАЙЛ.json или выберите /history.")
            path = _relative_path(user.sessions, argument)
            messages = load_session(path)
            self._reset(user)
            user.messages = messages
            user.config["system_prompt"] = next((m["content"] for m in messages if m["role"] == "system"), "")
            self._save(user)
            self._send(chat_id, f"Загружено сообщений: {len(messages)}. Харнесы и RAG отключены; /tools и /rag — подключение.")
        elif command == "/history":
            _no_links(user.sessions, self.root)
            for path in user.sessions.glob("*.json"):
                _no_links(path, self.root)
            entries = list_sessions(user.sessions)
            if argument:
                if not argument.isdecimal() or not 1 <= int(argument) <= len(entries):
                    raise ValueError("Нет диалога с таким номером. /history — список.")
                self.command(user, chat_id, "/load " + entries[int(argument) - 1].path.name)
            elif not entries:
                self._send(chat_id, "Сохранённых диалогов пока нет.")
            else:
                lines = [f"{i}. {entry.title} · {entry.message_count} сообщений\n{entry.path.name}"
                         for i, entry in enumerate(entries[:30], 1)]
                self._menu(user, chat_id, "Ваша история\n\n" + "\n\n".join(lines)
                           + "\n\n/history НОМЕР или /load ФАЙЛ.json — продолжить.", [
                    [(f"{i}. {entry.title[:40]}", "/load " + entry.path.name)]
                    for i, entry in enumerate(entries[:10], 1)])
        elif command == "/tools":
            self.tools(user, chat_id, argument)
        elif command == "/rag":
            self.rag(user, chat_id, argument)
        elif command == "/models":
            self.models(user, chat_id, argument)
        else:
            self._send(chat_id, "Неизвестная команда. /help — список команд.")

    def rag(self, user: UserChat, chat_id: int, argument: str) -> None:
        action, _, value = argument.partition(" ")
        if action == "add":
            path = _relative_path(user.workspace, value)
            argument = "add " + str(path)
        result = user.rag.command(argument)
        if argument:
            self._send(chat_id, result)
            return
        self._menu(user, chat_id, result + "\n\nОтправьте боту документ, чтобы добавить его в свою базу знаний.\n"
                   f"Или положите файл на ПК в {user.workspace}\n"
                   "/rag add ФАЙЛ — импортировать относительный путь из этого каталога.\n"
                   "/rag search ЗАПРОС — проверить поиск; /rag remove ID — удалить документ.\n"
                   "/rag clear — очистить базу знаний; /clear сохраняет документы и отключает RAG.", [
            [("Выключить RAG" if user.rag.enabled else "Включить RAG",
              "/rag " + ("off" if user.rag.enabled else "on"))],
            [("Документы", "/rag list"), ("Состояние", "/status")],
        ])

    def add_document(self, user: UserChat, chat_id: int, document: dict) -> None:
        """Import an explicitly uploaded document into this user's knowledge base."""
        filename, file_id = document.get("file_name"), document.get("file_id")
        if (not isinstance(filename, str) or not filename or len(filename) > 180
                or "/" in filename or "\\" in filename):
            raise ValueError("У документа должно быть обычное имя файла без пути.")
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("Telegram не передал идентификатор документа.")
        if Path(filename).suffix.casefold() not in SUPPORTED_SUFFIXES:
            raise ValueError("Этот формат не поддерживается RAG. Отправьте текст, PDF или DOCX.")
        size = document.get("file_size")
        if size is not None and (type(size) is not int or not 0 <= size <= MAX_FILE_BYTES):
            raise ValueError(f"Документ должен быть не больше {MAX_FILE_BYTES // (1024 * 1024)} МиБ.")
        documents = user.workspace / "documents"
        _relative_path(documents, filename)
        _make_directory(documents, user.workspace)
        path = _relative_path(documents, Path(filename).stem + "-" + secrets.token_hex(6) + Path(filename).suffix)
        self._send(chat_id, "Добавляю документ в вашу базу знаний: " + filename + ". /cancel — отменить.")
        try:
            content = self.api.download_document(file_id, max_bytes=MAX_FILE_BYTES)
        except TelegramError:
            raise ValueError("Не удалось скачать документ из Telegram. Повторите отправку файла.") from None
        if self._cancelled(user):
            raise GenerationCancelled("Импорт документа отменён.")
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("Документ превышает лимит размера RAG.")
        _no_links(path, user.workspace)
        with path.open("xb") as destination:
            destination.write(content)
        try:
            result = user.rag.command("add " + str(path))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self._send(chat_id, result + "\n\nДокумент добавлен в вашу базу знаний. "
                   + ("RAG включён." if user.rag.enabled else "/rag on — включить поиск в ответах."))

    def tools(self, user: UserChat, chat_id: int, argument: str) -> None:
        action, _, value = argument.partition(" ")
        value = value.strip()
        settings = user.settings
        if action in {"web", "code", "powershell"}:
            if value not in {"on", "off"}:
                raise ValueError(f"Укажите /tools {action} on или off.")
            settings = replace(settings, **{action + "_enabled": value == "on"})
            if action in {"code", "powershell"} and settings.repository is None:
                settings = replace(settings, repository=user.workspace)
        elif action == "repo":
            repository = _relative_path(user.workspace, value, directory=True)
            if not repository.is_dir():
                raise ValueError("Папка должна существовать внутри вашей рабочей директории.")
            checked = ToolHarness(replace(settings, code_enabled=True, repository=repository,
                                          repository_identity=None), lambda request: False)
            settings = replace(checked.settings, code_enabled=settings.code_enabled)
        elif action == "off":
            settings = HarnessSettings()
        elif action:
            raise ValueError("/tools: web on|off, code on|off, powershell on|off, repo ПАПКА или off.")
        if settings.code_enabled or settings.powershell_enabled:
            checked = ToolHarness(settings, lambda request: False)
            settings = checked.settings
        user.settings = settings
        self._menu(user, chat_id, "Харнесы текущего чата\n"
                   f"Веб: {'включён' if settings.web_enabled else 'выключен'}\n"
                   f"Код: {'включён' if settings.code_enabled else 'выключен'}\n"
                   f"PowerShell: {'включён' if settings.powershell_enabled else 'выключен'}\n"
                   f"Директория: {settings.repository or user.workspace}\n"
                   "Каждое чтение, поиск, запись, создание каталога, веб-запрос и запуск PowerShell требуют апрува.\n"
                   "PowerShell: только вычисления и вывод; файловые команды, сеть и другие программы запрещены.\n"
                   "/tools repo ПАПКА — выбрать подпапку своей рабочей директории.", [
            [("Выключить веб" if settings.web_enabled else "Включить веб",
              "/tools web " + ("off" if settings.web_enabled else "on"))],
            [("Выключить код" if settings.code_enabled else "Включить код",
              "/tools code " + ("off" if settings.code_enabled else "on"))],
            [("Выключить PowerShell" if settings.powershell_enabled else "Включить PowerShell",
              "/tools powershell " + ("off" if settings.powershell_enabled else "on"))],
            [("Рабочая директория", "/tools repo ."), ("Отключить всё", "/tools off")],
        ])

    def models(self, user: UserChat, chat_id: int, argument: str) -> None:
        def progress(value: str) -> None:
            if self._cancelled(user):
                raise GenerationCancelled("Операция с моделью отменена.")
            self.emit(value)
        manager = ModelManager(self.root, emit=progress)
        parts = argument.split()
        action = parts[0] if parts else ""
        if not action:
            self._menu(user, chat_id, "Модель общая для всех пользователей бота.\n"
                       f"Выбрана: {self.runtime.config['target_model']}\n"
                       "/models install ID [QUANT] — скачать\n/models use ПАКЕТ — выбрать\n"
                       "/models remove ПАКЕТ — удалить неактивный пакет", [
                [("Каталог", "/models catalog"), ("Установленные", "/models list")]])
        elif action == "catalog" and len(parts) == 1:
            self._send(chat_id, "Каталог моделей\n\n" + "\n\n".join(
                f"{spec.id} · {spec.name} · {spec.parameters_b:g}B\n{spec.description}\n"
                f"/models install {spec.id} {spec.default_quantization}" for spec in manager.catalog()))
        elif action == "list" and len(parts) == 1:
            records = manager.installed(self.runtime.config)
            if not records:
                self._send(chat_id, "Установленных моделей нет. /models catalog — каталог.")
            else:
                self._menu(user, chat_id, "Установленные модели\n\n" + "\n\n".join(
                    f"{record['id']} · {record['name']} · {record['status']}"
                    + (" · выбрана" if record.get("active") else "") for record in records), [
                    [("Выбрать " + record["name"][:35], "/models use " + record["id"])]
                    for record in records[:20] if record["status"] == "ready"])
        elif action == "install" and len(parts) in {2, 3}:
            quant = parts[2] if len(parts) == 3 else "Q4_K_M"
            plan = manager.inspect(parts[1], quant)
            request = ToolRequest("install_model", {"model_id": parts[1], "quantization": quant},
                                  f"Скачать модель {plan['name']} ({quant}): {plan['total_size'] / 2**30:.2f} ГиБ на диск клиента.")
            if self.approve(user, chat_id, request):
                self._send(chat_id, "Скачиваю модель. Ход загрузки виден в консоли клиента.")
                record = manager.install(parts[1], quant, emit=progress)
                self._send(chat_id, f"Установлено: {record['id']}. /models use {record['id']} — выбрать.")
            else:
                self._send(chat_id, "Установка отклонена.")
        elif action == "use" and len(parts) == 2:
            request = ToolRequest("select_model", {"package_id": parts[1]},
                                  "Переключить общую модель бота на " + parts[1] + ". Текущий запрос другого пользователя завершится до переключения.")
            if self.approve(user, chat_id, request):
                self._send(chat_id, "Переключаю модель после завершения текущих запросов…")
                while not self.runtime.lock.acquire(timeout=0.25):
                    if self._cancelled(user):
                        raise GenerationCancelled("Переключение модели отменено.")
                try:
                    if self._cancelled(user):
                        raise GenerationCancelled("Переключение модели отменено.")
                    self.runtime.select(manager, parts[1], cancelled=lambda: self._cancelled(user))
                finally:
                    self.runtime.lock.release()
                self._send(chat_id, "Выбрана модель: " + self.runtime.config["target_model"])
            else:
                self._send(chat_id, "Переключение отклонено.")
        elif action == "remove" and len(parts) == 2:
            request = ToolRequest("remove_model", {"package_id": parts[1]}, "Удалить неактивный пакет модели с диска клиента: " + parts[1])
            if self.approve(user, chat_id, request):
                while not self.runtime.lock.acquire(timeout=0.25):
                    if self._cancelled(user):
                        raise GenerationCancelled("Удаление модели отменено.")
                try:
                    if self._cancelled(user):
                        raise GenerationCancelled("Удаление модели отменено.")
                    manager.remove(parts[1], self.runtime.config)
                finally:
                    self.runtime.lock.release()
                self._send(chat_id, "Пакет удалён: " + parts[1])
            else:
                self._send(chat_id, "Удаление отклонено.")
        else:
            raise ValueError("/models: catalog, list, install ID [QUANT], use ПАКЕТ, remove ПАКЕТ.")

    def generate(self, user: UserChat, chat_id: int, prompt: str) -> None:
        harness_status = HarnessStatus(settings=user.settings)
        user.harness_status = harness_status
        accepted = "Запрос принят. /cancel — отменить."
        if harness_status.enabled:
            accepted += "\n" + harness_status.connection()
        if user.rag.enabled:
            accepted += "\n" + user.rag.status()
        self._send(chat_id, accepted)
        while not self.runtime.lock.acquire(timeout=0.25):
            if self._cancelled(user):
                raise GenerationCancelled("Запрос отменён.")
        proposed = copy.deepcopy(user.messages) + [{"role": "user", "content": prompt}]
        reasoning: list[str] = []
        content: list[str] = []
        def on_event(event):
            if event.kind == "rag_sources":
                sources = event.value.get("sources", [])
                self._send(chat_id, f"RAG: найдено фрагментов: {len(sources)}."
                           if sources else "RAG: подходящих фрагментов в вашей базе знаний не найдено.")
            if event.kind in {"harness_status", "tool_start", "tool_result", "harness_summary"}:
                harness_status.update(event.kind, event.value)
                if event.kind == "harness_status" and event.value.get("phase") == "retrying":
                    # The core discards this reply before its one corrective
                    # attempt. It must not reach the next approval preview.
                    content.clear()
                    reasoning.clear()
            if event.kind == "reasoning" and user.config.get("show_reasoning"):
                reasoning.append(event.value)
            elif event.kind == "content":
                content.append(event.value)
        def flush():
            text = ("[Рассуждение]\n" + "".join(reasoning) + "\n\n" if reasoning else "") + "".join(content)
            if text.strip():
                self._send(chat_id, text)
            content.clear()
            reasoning.clear()
        def approve(request):
            flush()
            harness_status.last_tool = request.name
            harness_status.phase = "approval"
            allowed = self.approve(user, chat_id, request)
            harness_status.phase = "executing" if allowed else "result"
            return allowed
        try:
            if self._cancelled(user):
                raise GenerationCancelled("Запрос отменён.")
            config = copy.deepcopy(self.runtime.config)
            config["system_prompt"] = user.config["system_prompt"]
            harness = ToolHarness(user.settings, approve) if (user.settings.web_enabled or user.settings.code_enabled
                                                             or user.settings.powershell_enabled) else None
            if harness is not None:
                user.settings = harness.settings
                harness_status.configure(harness.settings)
            client = ApiClient(config)
            with self.control_lock:
                self.clients[user.user_id] = client
            rag_options = {"rag": user.rag} if user.rag.enabled else {}
            result = generate_response(client, proposed, harness=harness,
                                       on_event=on_event, cancelled=lambda: self._cancelled(user),
                                       **rag_options)
            user.messages = proposed + [{"role": "assistant", "content": result.answer}]
            self._save(user)
            text = ("[Рассуждение]\n" + "".join(reasoning) + "\n\n" if reasoning else "") + result.answer
            if harness_status.enabled:
                text += "\n\n" + harness_status.details()
            self._send(chat_id, text)
            if result.finish == "length":
                self._send(chat_id, "Достигнут лимит ответа max_tokens.")
        except (ApiError, HarnessError, TelegramError):
            if len(proposed) > len(user.messages) + 1:
                user.messages = proposed
                self._save(user)
            if self._cancelled(user):
                raise GenerationCancelled("Запрос отменён.") from None
            raise
        finally:
            with self.control_lock:
                self.clients.pop(user.user_id, None)
            self.runtime.lock.release()

    def poll(self) -> None:
        offset = None
        failures = 0
        while not self.stopped.is_set():
            try:
                updates = self.api.get_updates(offset=offset, timeout=25)
                failures = 0
                for update in updates:
                    update_id = update.get("update_id") if isinstance(update, dict) else None
                    if type(update_id) is not int:
                        continue
                    try:
                        self.handle_update(update)
                    except (TelegramError, OSError, ValueError) as error:
                        self.emit(f"Telegram: обновление {update_id} не обработано: {error}")
                    offset = update_id + 1
            except TelegramError as error:
                if error.error_code in {401, 403, 404, 409}:
                    raise
                failures += 1
                self.emit(f"Telegram: соединение прервано; повтор через {min(2 ** failures, 30)} с.")
                self.stopped.wait(min(2 ** failures, 30))

    def close(self) -> None:
        self.stopped.set()
        with self.control_lock:
            for user in self.users.values():
                user.cancelled.set()
            for pending in self.approvals.values():
                pending.event.set()
            for client in self.clients.values():
                client.cancel()
        deadline = time.monotonic() + 2
        for user in list(self.users.values()):
            if user.worker is not None:
                user.worker.join(timeout=max(0, deadline - time.monotonic()))


def run_telegram(config: dict, root: Path, *, whitelist_path: Path | None = None,
                 credits_path: Path | None = None, emit: Callable[[str], None] = print) -> int:
    root = Path(root).resolve()
    whitelist_path = Path(whitelist_path) if whitelist_path is not None else root / "whitelist.txt"
    credits_path = Path(credits_path) if credits_path is not None else root / "credits.txt"
    if not whitelist_path.is_absolute():
        whitelist_path = root / whitelist_path
    if not credits_path.is_absolute():
        credits_path = root / credits_path
    bot = None
    runtime = BotRuntime(config, root, emit)
    try:
        token = read_token(credits_path)
        allowed = read_whitelist(whitelist_path)
        api = TelegramAPI(token)
        identity = api.get_me()
        api.set_commands(COMMANDS)
        bot = TelegramBot(api, config, root, whitelist_path, runtime=runtime, emit=emit)
        bot.username = identity.get("username", "")
        runtime.start()
        emit(f"Telegram-бот @{bot.username} запущен. Разрешено пользователей: {len(allowed)}. Ctrl+C — остановить.")
        bot.poll()
        return 0
    except KeyboardInterrupt:
        emit("Telegram-бот остановлен.")
        return 0
    except (TelegramError, RuntimeErrorDetail, OSError, ValueError) as error:
        emit(f"Ошибка Telegram-бота: {error}")
        return 1
    finally:
        if bot is not None:
            bot.close()
        runtime.shutdown()
