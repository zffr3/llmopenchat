"""Isolated automatic replies through Telegram Business connections."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .api import ApiClient, ApiError
from .conversation import GenerationCancelled, generate_response, initial_messages
from .history import load_session, new_session_path, save_session
from .telegram_api import TelegramAPI, TelegramError
from .vision import VisionError, VisionSettings, image_message, incoming_image, prune_images


MAX_PENDING_MESSAGES = 100
MAX_TOTAL_PENDING_MESSAGES = 1000
MAX_SEEN_MESSAGES = 1000
_MEDIA = {"photo": "фото", "document": "файл", "video": "видео", "audio": "аудио",
          "voice": "голосовое сообщение", "animation": "анимация",
          "video_note": "видеосообщение", "sticker": "стикер"}


@dataclass
class BusinessChat:
    connection_id: str
    chat_id: int
    directory: Path
    messages: list[dict]
    session_path: Path
    pending: deque[tuple[int, str, dict | None]] = field(default_factory=deque)
    seen: set[int] = field(default_factory=set)
    seen_order: deque[int] = field(default_factory=deque)
    delivered: deque[int] = field(default_factory=deque)
    cancelled: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None
    client: ApiClient | None = None


class BusinessResponder:
    """Keep business contacts outside bot commands, tools and shared context."""

    def __init__(self, api: TelegramAPI, config: dict, root: Path, runtime,
                 emit: Callable[[str], None] = print):
        self.api = api
        self.runtime = runtime
        self.root = Path(root).resolve()
        self.emit = emit
        options = config.get("telegram", {})
        self.enabled = options.get("business_enabled", False)
        self.owner_id = options.get("admin_user_id")
        if type(self.enabled) is not bool:
            raise ValueError("telegram.business_enabled должен быть true или false.")
        if self.enabled and (type(self.owner_id) is not int or self.owner_id <= 0):
            raise ValueError("Для режима Telegram Business укажите telegram.admin_user_id.")
        self.connections: dict[str, dict] = {}
        self.chats: dict[tuple[str, int], BusinessChat] = {}
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self._ready: deque[BusinessChat] = deque()
        self._scheduled: set[tuple[str, int]] = set()
        self._pending_count = 0
        self._active: BusinessChat | None = None
        self._worker: threading.Thread | None = None

    @staticmethod
    def _connection_id(value: object) -> bool:
        return isinstance(value, str) and bool(value) and len(value) <= 1024

    def _authorized(self, connection: dict | None) -> bool:
        if not isinstance(connection, dict):
            return False
        owner = connection.get("user")
        rights = connection.get("rights")
        # Bot API versions before 9.0 exposed can_reply on the connection itself.
        can_reply = rights.get("can_reply") if isinstance(rights, dict) else connection.get("can_reply")
        return (isinstance(owner, dict) and type(owner.get("id")) is int
                and owner["id"] == self.owner_id and connection.get("is_enabled") is True
                and can_reply is True)

    def _remember_connection(self, connection: dict) -> None:
        identifier = connection.get("id")
        if not self._connection_id(identifier):
            return
        clients = []
        with self.lock:
            self.connections[identifier] = copy.deepcopy(connection)
            if not self._authorized(connection):
                for (connection_id, _), chat in self.chats.items():
                    if connection_id == identifier:
                        chat.cancelled.set()
                        self._pending_count -= len(chat.pending)
                        chat.pending.clear()
                        if chat is not self._active:
                            chat.worker = None
                        if chat.client is not None:
                            clients.append(chat.client)
                self._ready = deque(chat for chat in self._ready if chat.connection_id != identifier)
                self._scheduled = {key for key in self._scheduled if key[0] != identifier}
        for client in clients:
            client.cancel()

    def handle_update(self, update: dict) -> None:
        if not self.enabled or self.stopped.is_set() or not isinstance(update, dict):
            return
        connection = update.get("business_connection")
        if isinstance(connection, dict):
            self._remember_connection(connection)
            return
        message = update.get("business_message")
        if not isinstance(message, dict):
            return
        identifier = message.get("business_connection_id")
        sender, destination = message.get("from"), message.get("chat")
        message_id = message.get("message_id")
        if (not self._connection_id(identifier) or not isinstance(sender, dict)
                or not isinstance(destination, dict) or destination.get("type") != "private"
                or type(sender.get("id")) is not int or sender["id"] <= 0
                or type(destination.get("id")) is not int or destination["id"] != sender["id"]
                or sender["id"] == self.owner_id or sender.get("is_bot")
                or message.get("via_business_bot") or message.get("sender_business_bot")
                or type(message_id) is not int or message_id <= 0):
            return
        text = message.get("text")
        prompt = text.strip() if isinstance(text, str) else ""
        image = incoming_image(message)
        media = next((label for kind, label in _MEDIA.items() if message.get(kind)), None)
        if media is not None:
            caption = message.get("caption")
            prompt = caption.strip() if isinstance(caption, str) else ""
            if image is None:
                prompt = self._unavailable_media(prompt, media)
        if not prompt and image is None:
            return
        with self.lock:
            known = identifier in self.connections
        if not known:
            # Connections are not persisted: recover authoritative permissions after restart.
            try:
                connection = self.api.get_business_connection(identifier)
            except TelegramError:
                self.emit("Telegram Business: не удалось проверить права подключения.")
                return
            if connection.get("id") != identifier:
                return
            self._remember_connection(connection)
        with self.lock:
            if self.stopped.is_set() or not self._authorized(self.connections.get(identifier)):
                return
            key = (identifier, destination["id"])
            chat = self.chats.get(key)
            if chat is not None and message_id in chat.seen:
                return
            if self._pending_count >= MAX_TOTAL_PENDING_MESSAGES:
                self.emit("Telegram Business: общая очередь заполнена; сообщение пропущено.")
                return
            chat = self._chat(*key)
            if message_id in chat.seen:
                return
            if len(chat.pending) >= MAX_PENDING_MESSAGES:
                self.emit("Telegram Business: очередь собеседника заполнена; сообщение пропущено.")
                return
            chat.seen.add(message_id)
            chat.seen_order.append(message_id)
            if len(chat.seen_order) > MAX_SEEN_MESSAGES:
                chat.seen.discard(chat.seen_order.popleft())
            # Download in the authorized worker, never in Telegram's polling thread.
            chat.pending.append((message_id, prompt, copy.deepcopy(image)))
            self._pending_count += 1
            if chat is not self._active and key not in self._scheduled:
                self._ready.append(chat)
                self._scheduled.add(key)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work,
                                                name="telegram-business", daemon=True)
                chat.worker = self._worker
                self._worker.start()
            else:
                chat.worker = self._worker

    def _chat(self, connection_id: str, chat_id: int) -> BusinessChat:
        key = (connection_id, chat_id)
        if key not in self.chats:
            # IDs supplied by Telegram must never become path components directly.
            digest = hashlib.sha256(connection_id.encode("utf-8")).hexdigest()
            directory = self.root / ".local" / "telegram" / "business" / digest / str(chat_id)
            self._check_path(directory / ".local" / "sessions")
            (directory / ".local" / "sessions").mkdir(parents=True, exist_ok=True)
            self._check_path(directory / ".local" / "sessions")
            current = directory / "current.json"
            self._check_path(current)
            messages = load_session(current) if current.exists() else initial_messages(self.runtime.config)
            receipts = directory / "delivered.json"
            self._check_path(receipts)
            delivered = []
            if receipts.exists():
                with receipts.open(encoding="utf-8") as source:
                    content = source.read(MAX_SEEN_MESSAGES * 32 + 1)
                if len(content) > MAX_SEEN_MESSAGES * 32:
                    raise ValueError("Список доставленных сообщений Telegram Business слишком большой.")
                try:
                    delivered = json.loads(content)
                except RecursionError:
                    raise ValueError("Неверный список доставленных сообщений Telegram Business.") from None
                if (not isinstance(delivered, list) or len(delivered) > MAX_SEEN_MESSAGES
                        or any(type(item) is not int or item <= 0 for item in delivered)):
                    raise ValueError("Неверный список доставленных сообщений Telegram Business.")
            self.chats[key] = BusinessChat(connection_id, chat_id, directory, messages,
                                           new_session_path(directory), seen=set(delivered),
                                           seen_order=deque(delivered), delivered=deque(delivered))
        return self.chats[key]

    def _check_path(self, path: Path) -> None:
        path.relative_to(self.root)
        for current in (path, *path.parents):
            if (current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction())
                    or (current.exists() and getattr(current.stat(), "st_file_attributes", 0) & 0x400)):
                raise ValueError("Ссылки и junction в каталогах Telegram Business запрещены.")
            if current == self.root:
                break

    def _cancelled(self, chat: BusinessChat) -> bool:
        with self.lock:
            return (self.stopped.is_set() or chat.cancelled.is_set()
                    or not self._authorized(self.connections.get(chat.connection_id)))

    def _work(self) -> None:
        worker = threading.current_thread()
        try:
            while True:
                with self.lock:
                    if self.stopped.is_set() or not self._ready:
                        self._worker = None
                        return
                    chat = self._ready.popleft()
                    key = (chat.connection_id, chat.chat_id)
                    self._scheduled.discard(key)
                    if not chat.pending or not self._authorized(self.connections.get(chat.connection_id)):
                        self._pending_count -= len(chat.pending)
                        chat.pending.clear()
                        chat.worker = None
                        continue
                    message_id, prompt, image = chat.pending.popleft()
                    self._pending_count -= 1
                    self._active = chat
                    chat.cancelled.clear()
                try:
                    self._generate(chat, message_id, prompt, image)
                except GenerationCancelled:
                    pass
                except (ApiError, TelegramError, OSError, ValueError):
                    # Model/server errors may contain local paths, prompts or credentials.
                    self.emit("Telegram Business: не удалось подготовить или доставить ответ.")
                finally:
                    with self.lock:
                        self._active = None
                        if (not self.stopped.is_set() and chat.pending
                                and self._authorized(self.connections.get(chat.connection_id))):
                            # One turn per chat keeps a busy contact from starving others.
                            self._ready.append(chat)
                            self._scheduled.add(key)
                        else:
                            self._pending_count -= len(chat.pending)
                            chat.pending.clear()
                            chat.worker = None
        finally:
            with self.lock:
                # Do not discard a replacement worker started after the normal return.
                if self._worker is worker:
                    for chat in self.chats.values():
                        if chat.worker is worker:
                            chat.pending.clear()
                            chat.worker = None
                    self._ready.clear()
                    self._scheduled.clear()
                    self._pending_count = 0
                    self._active = None
                    self._worker = None

    @staticmethod
    def _unavailable_media(prompt: str, media: str) -> str:
        return prompt + ("\n\n" if prompt else "") + (
            f"[Вложение Telegram: {media}. Содержимое вложения недоступно модели. "
            "Не утверждай, что просмотрел или прослушал его; при необходимости попроси текстовое описание.]"
        )

    def _image_prompt(self, chat: BusinessChat, prompt: str, image: dict, config: dict) -> str | list[dict]:
        settings = VisionSettings.from_config(config)
        if not settings.enabled:
            return self._unavailable_media(prompt, "фото" if image.get("media_type") == "photo" else "файл")
        identifier, size = image.get("file_id"), image.get("file_size")
        if (not isinstance(identifier, str) or not identifier or len(identifier) > 1024
                or (size is not None and (type(size) is not int or not 0 <= size <= settings.max_input_bytes))):
            raise VisionError("Изображение превышает допустимый размер или содержит некорректные сведения.")
        if self._cancelled(chat):
            raise GenerationCancelled("Запрос Telegram Business отменён.")
        try:
            content = self.api.download_file(identifier, max_bytes=settings.max_input_bytes)
        except (TelegramError, OSError, ValueError):
            raise VisionError("Не удалось скачать изображение из Telegram.") from None
        if self._cancelled(chat):
            raise GenerationCancelled("Запрос Telegram Business отменён.")
        return image_message(prompt or "Опиши изображение.", content, config)

    def _generate(self, chat: BusinessChat, message_id: int, prompt: str,
                  image: dict | None = None) -> None:
        while not self.runtime.lock.acquire(timeout=0.25):
            if self._cancelled(chat):
                raise GenerationCancelled("Запрос Telegram Business отменён.")
        try:
            if self._cancelled(chat):
                raise GenerationCancelled("Запрос Telegram Business отменён.")
            config = copy.deepcopy(self.runtime.config)
            try:
                content = self._image_prompt(chat, prompt, image, config) if image is not None else prompt
            except VisionError:
                if self._cancelled(chat):
                    raise GenerationCancelled("Запрос Telegram Business отменён.")
                self.api.send_message(chat.chat_id,
                                      "Не удалось обработать изображение. Пришлите JPEG, PNG или WebP "
                                      "в пределах настроенных ограничений либо опишите изображение текстом.",
                                      business_connection_id=chat.connection_id)
                try:
                    self._save_receipt(chat, message_id)
                except (OSError, ValueError):
                    self.emit("Telegram Business: не удалось сохранить ID доставленного сообщения.")
                return
            if self._cancelled(chat):
                raise GenerationCancelled("Запрос Telegram Business отменён.")
            client = ApiClient(config)
            with self.lock:
                chat.client = client
            proposed = copy.deepcopy(chat.messages) + [{"role": "user", "content": content}]
            prune_images(proposed, config)
            result = generate_response(client, proposed, cancelled=lambda: self._cancelled(chat),
                                       vision_enabled=VisionSettings.from_config(config).enabled)
            if self._cancelled(chat):
                raise GenerationCancelled("Запрос Telegram Business отменён.")
            if not result.answer.strip():
                return
            # Only final answer text is delivered; no commands, tools or reasoning events.
            self.api.send_message(chat.chat_id, result.answer, business_connection_id=chat.connection_id)
            chat.messages = proposed + [{"role": "assistant", "content": result.answer}]
            try:
                self._save_receipt(chat, message_id)
            except (OSError, ValueError):
                self.emit("Telegram Business: не удалось сохранить ID доставленного сообщения.")
            for target in (chat.session_path, chat.directory / "current.json"):
                self._check_path(target)
                save_session(target, chat.messages, self.runtime.config["model"],
                             target_model=self.runtime.config.get("target_model"),
                             model_id=self.runtime.config.get("model_package"))
        finally:
            with self.lock:
                chat.client = None
            self.runtime.lock.release()

    def _save_receipt(self, chat: BusinessChat, message_id: int) -> None:
        """Suppress replayed Telegram updates after a normal client restart."""
        chat.delivered.append(message_id)
        while len(chat.delivered) > MAX_SEEN_MESSAGES:
            chat.delivered.popleft()
        target = chat.directory / "delivered.json"
        self._check_path(target)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=chat.directory,
                                             prefix="delivered.", suffix=".tmp", delete=False) as source:
                temporary = Path(source.name)
                json.dump(list(chat.delivered), source)
                source.flush()
                os.fsync(source.fileno())
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def close(self) -> None:
        self.stopped.set()
        clients = []
        with self.lock:
            worker = self._worker
            for chat in self.chats.values():
                chat.cancelled.set()
                chat.pending.clear()
                if chat is not self._active:
                    chat.worker = None
                if chat.client is not None:
                    clients.append(chat.client)
            self._ready.clear()
            self._scheduled.clear()
            self._pending_count = 0
        for client in clients:
            client.cancel()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2)
