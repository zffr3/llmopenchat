"""Business account routing, isolated histories and quiet automatic replies."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from llmopenchat.api import ApiError, StreamEvent
from llmopenchat.config import DEFAULT_CONFIG
from llmopenchat.history import load_session
from llmopenchat.telegram import BotRuntime, TelegramBot
from llmopenchat.telegram_api import TelegramError
from llmopenchat.telegram_business import BusinessResponder


OWNER = 101
CONTACT = 202
CONNECTION = "business-connection-1"


def connection_update(identifier=CONNECTION, *, owner=OWNER, enabled=True, can_reply=True):
    return {"business_connection": {
        "id": identifier, "user": {"id": owner}, "user_chat_id": owner,
        "is_enabled": enabled, "rights": {"can_reply": can_reply},
    }}


def message_update(text="Привет", *, identifier=CONNECTION, contact=CONTACT, message_id=1):
    return {"business_message": {
        "business_connection_id": identifier, "message_id": message_id,
        "from": {"id": contact, "is_bot": False},
        "chat": {"id": contact, "type": "private"}, "text": text,
    }}


class FakeAPI:
    def __init__(self):
        self.messages = []
        self.get_business_connection = Mock(
            side_effect=lambda identifier: connection_update(identifier)["business_connection"],
        )

    def send_message(self, chat_id, text, reply_markup=None, *, business_connection_id=None):
        message = {
            "chat_id": chat_id, "text": text, "reply_markup": copy.deepcopy(reply_markup),
            "business_connection_id": business_connection_id, "message_id": len(self.messages) + 1,
        }
        self.messages.append(message)
        return message


class RecordingClient:
    def __init__(self, requests, *, started=None, release=None, failure=None):
        self.requests = requests
        self.started = started
        self.release = release
        self.failure = failure
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()
        if self.release is not None:
            self.release.set()

    def stream_chat(self, messages, tools=None, tool_choice=None):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools), tool_choice))
        if self.started is not None:
            self.started.set()
        if self.release is not None and not self.release.wait(3):
            raise AssertionError("Test did not release the blocked generation")
        if self.failure is not None:
            raise self.failure
        prompt = next(item["content"] for item in reversed(messages) if item["role"] == "user")
        yield StreamEvent("reasoning", "PRIVATE REASONING")
        yield StreamEvent("content", "Ответ: " + prompt)
        yield StreamEvent("finish", "stop")


class TelegramBusinessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmopenchat-business-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["backend"] = "external"
        self.config["show_reasoning"] = True
        self.config["telegram"].update(admin_user_id=OWNER, business_enabled=True)
        self.api = FakeAPI()
        self.emitted = []
        self.requests = []
        self.clients = []

        def make_client(config):
            client = RecordingClient(self.requests)
            self.clients.append(client)
            return client

        self.client_patch = patch("llmopenchat.telegram_business.ApiClient", side_effect=make_client)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.responder = self.new_responder()

    def new_responder(self, config=None):
        config = self.config if config is None else config
        runtime = BotRuntime(config, self.root, self.emitted.append)
        responder = BusinessResponder(self.api, config, self.root, runtime, emit=self.emitted.append)
        self.addCleanup(responder.close)
        return responder

    def join_workers(self, responder=None):
        responder = self.responder if responder is None else responder
        for chat in list(responder.chats.values()):
            worker = chat.worker
            if worker is not None:
                worker.join(3)
                self.assertFalse(worker.is_alive(), "Business worker did not finish")

    def send(self, text="Привет", *, responder=None, **kwargs):
        responder = self.responder if responder is None else responder
        responder.handle_update(message_update(text, **kwargs))
        self.join_workers(responder)

    def block_next_generation(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        client = RecordingClient(self.requests, started=started, release=release)
        return client, started, release

    def test_disabled_mode_ignores_all_business_events(self):
        config = copy.deepcopy(self.config)
        config["telegram"]["business_enabled"] = False
        responder = self.new_responder(config)

        responder.handle_update(connection_update())
        self.send(responder=responder)

        self.assertEqual(responder.chats, {})
        self.assertEqual(responder.connections, {})
        self.assertEqual(self.api.messages, [])
        self.api.get_business_connection.assert_not_called()

    def test_contact_without_username_or_whitelist_receives_business_reply(self):
        self.responder.handle_update(connection_update())

        self.send("Здравствуйте")

        self.assertFalse((self.root / "whitelist.txt").exists())
        self.assertEqual(len(self.api.messages), 1)
        self.assertEqual(self.api.messages[0], {
            "chat_id": CONTACT, "text": "Ответ: Здравствуйте", "reply_markup": None,
            "business_connection_id": CONNECTION, "message_id": 1,
        })
        self.assertIn((CONNECTION, CONTACT), self.responder.chats)
        self.api.get_business_connection.assert_not_called()

    def test_unknown_connection_is_fetched_once_after_restart(self):
        self.send("Первое")
        self.send("Второе", message_id=2)

        self.api.get_business_connection.assert_called_once_with(CONNECTION)
        self.assertEqual(len(self.api.messages), 2)

    def test_foreign_disabled_or_read_only_connections_never_generate(self):
        cases = (
            {"owner": OWNER + 1}, {"enabled": False}, {"can_reply": False},
        )
        for index, options in enumerate(cases):
            identifier = f"rejected-{index}"
            with self.subTest(options=options):
                self.responder.handle_update(connection_update(identifier, **options))
                self.send(identifier=identifier)

        self.assertEqual(self.requests, [])
        self.assertEqual(self.api.messages, [])
        self.assertEqual(self.responder.chats, {})

    def test_failed_or_mismatched_connection_lookup_never_replies(self):
        self.api.get_business_connection.side_effect = TelegramError("PRIVATE LOOKUP ERROR")
        self.send(identifier="unavailable")
        self.api.get_business_connection.side_effect = None
        self.api.get_business_connection.return_value = connection_update("wrong-id")["business_connection"]
        self.send(identifier="requested-id")

        self.assertEqual(self.api.messages, [])
        self.assertEqual(self.requests, [])
        self.assertTrue(self.emitted)

    def test_owner_bot_outgoing_and_nonprivate_messages_are_ignored(self):
        self.responder.handle_update(connection_update())
        updates = [message_update(contact=OWNER)]
        bot = message_update()
        bot["business_message"]["from"]["is_bot"] = True
        updates.append(bot)
        outgoing = message_update()
        outgoing["business_message"]["via_business_bot"] = {"id": 999, "is_bot": True}
        updates.append(outgoing)
        sent_by_bot = message_update()
        sent_by_bot["business_message"]["sender_business_bot"] = {"id": 999, "is_bot": True}
        updates.append(sent_by_bot)
        group = message_update()
        group["business_message"]["chat"]["type"] = "group"
        updates.append(group)
        wrong_sender = message_update()
        wrong_sender["business_message"]["from"]["id"] = CONTACT + 1
        updates.append(wrong_sender)

        for update in updates:
            self.responder.handle_update(update)
        self.join_workers()

        self.assertEqual(self.responder.chats, {})
        self.assertEqual(self.requests, [])
        self.assertEqual(self.api.messages, [])

    def test_malformed_business_messages_do_not_create_chats(self):
        self.responder.handle_update(connection_update())
        updates = [{}, {"business_message": None}, {"business_message": []}]
        for field, value in (
            ("business_connection_id", None), ("business_connection_id", ""),
            ("message_id", True), ("message_id", 0), ("message_id", "1"),
            ("chat", None), ("from", None), ("text", None),
        ):
            update = message_update()
            update["business_message"][field] = value
            updates.append(update)
        for update in updates:
            with self.subTest(update=update):
                self.responder.handle_update(update)
        self.join_workers()

        self.assertEqual(self.responder.chats, {})
        self.assertEqual(self.api.messages, [])

    def test_duplicate_message_is_answered_only_once(self):
        update = message_update("Один запрос", message_id=17)
        self.responder.handle_update(update)
        self.responder.handle_update(copy.deepcopy(update))
        self.join_workers()
        self.responder.handle_update(update)
        self.join_workers()

        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.api.messages), 1)
        chat = self.responder.chats[(CONNECTION, CONTACT)]
        self.assertEqual([m["content"] for m in chat.messages if m["role"] == "user"], ["Один запрос"])

    def test_each_connection_and_contact_has_a_separate_history(self):
        self.send("Первый контакт")
        self.send("Другой контакт", contact=303)
        self.send("Другая связь", identifier="second-connection")

        self.assertEqual(len(self.responder.chats), 3)
        directories = {chat.directory for chat in self.responder.chats.values()}
        self.assertEqual(len(directories), 3)
        for messages, tools, choice in self.requests:
            self.assertEqual(len([m for m in messages if m["role"] == "user"]), 1)
        self.assertEqual(
            [(m["business_connection_id"], m["chat_id"]) for m in self.api.messages],
            [(CONNECTION, CONTACT), (CONNECTION, 303), ("second-connection", CONTACT)],
        )

    def test_parent_routes_unlisted_business_contact_without_exposing_shared_session_or_commands(self):
        whitelist = self.root / "whitelist.txt"
        whitelist.write_text("", encoding="utf-8")
        normal = TelegramBot(self.api, self.config, self.root, whitelist, emit=self.emitted.append)
        self.addCleanup(normal.close)
        shared = normal.user_chat(OWNER)
        shared.messages.append({"role": "user", "content": "PRIVATE NORMAL SESSION"})

        normal.handle_update(connection_update())
        normal.handle_update(message_update("/system Бизнес-запрос"))
        self.join_workers(normal.business)

        business = normal.business.chats[(CONNECTION, CONTACT)]
        self.assertNotEqual(business.directory, shared.directory)
        self.assertNotIn("PRIVATE NORMAL SESSION", str(self.requests))
        self.assertNotIn("Бизнес-запрос", str(shared.messages))
        self.assertNotIn(CONTACT, normal.users)
        self.assertEqual(shared.config["system_prompt"], self.config["system_prompt"])
        self.assertEqual([m["text"] for m in self.api.messages], ["Ответ: /system Бизнес-запрос"])
        self.assertEqual(self.api.messages[0]["business_connection_id"], CONNECTION)

    def test_slash_commands_are_plain_input_and_no_tools_reasoning_or_status_is_sent(self):
        self.send("/tools powershell on")

        self.assertEqual(self.requests[0][0][-1], {"role": "user", "content": "/tools powershell on"})
        self.assertIsNone(self.requests[0][1])
        self.assertIsNone(self.requests[0][2])
        self.assertEqual([m["text"] for m in self.api.messages], ["Ответ: /tools powershell on"])
        self.assertIsNone(self.api.messages[0]["reply_markup"])

    def test_requests_in_one_chat_are_processed_in_fifo_order(self):
        client, started, release = self.block_next_generation()
        with patch("llmopenchat.telegram_business.ApiClient", return_value=client):
            self.responder.handle_update(message_update("Первый", message_id=1))
            self.assertTrue(started.wait(2), "First generation did not start")
            self.responder.handle_update(message_update("Второй", message_id=2))
            self.responder.handle_update(message_update("Третий", message_id=3))
            self.assertEqual(len(self.requests), 1)
            release.set()
            self.join_workers()

        self.assertEqual([m["text"] for m in self.api.messages], ["Ответ: Первый", "Ответ: Второй", "Ответ: Третий"])
        prompts = [[m["content"] for m in messages if m["role"] == "user"] for messages, _, _ in self.requests]
        self.assertEqual(prompts, [["Первый"], ["Первый", "Второй"], ["Первый", "Второй", "Третий"]])

    def test_many_contacts_share_bounded_workers_and_close_while_runtime_is_busy(self):
        self.responder.handle_update(connection_update())
        with self.responder.runtime.lock:
            for contact in range(300, 340):
                self.responder.handle_update(message_update("Ждём модель", contact=contact))
            workers = {chat.worker for chat in self.responder.chats.values() if chat.worker is not None}
            started = time.monotonic()
            self.responder.close()
            elapsed = time.monotonic() - started

            self.assertEqual(self.requests, [])
            self.assertEqual(self.api.messages, [])
            self.assertLess(elapsed, 1.5, "Shutdown waited for the unavailable model runtime")
            self.assertTrue(workers, "Pending business requests must have a dispatcher")
            self.assertLessEqual(len(workers), 2, "Every new contact created another waiting thread")
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertTrue(all(not chat.pending for chat in self.responder.chats.values()))

    def test_dispatcher_preserves_contact_order_and_gives_other_contacts_a_turn(self):
        client, started, release = self.block_next_generation()
        with patch("llmopenchat.telegram_business.ApiClient", return_value=client):
            self.responder.handle_update(message_update("A1"))
            self.assertTrue(started.wait(2), "First contact did not start")
            self.responder.handle_update(message_update("A2", message_id=2))
            self.responder.handle_update(message_update("A3", message_id=3))
            self.responder.handle_update(message_update("B1", contact=303))
            self.responder.handle_update(message_update("B2", contact=303, message_id=2))
            release.set()
            self.join_workers()

        first = [m["text"] for m in self.api.messages if m["chat_id"] == CONTACT]
        second = [m["text"] for m in self.api.messages if m["chat_id"] == 303]
        self.assertEqual(first, ["Ответ: A1", "Ответ: A2", "Ответ: A3"])
        self.assertEqual(second, ["Ответ: B1", "Ответ: B2"])
        replies = [m["text"] for m in self.api.messages]
        self.assertLess(replies.index("Ответ: B1"), replies.index("Ответ: A3"))

    def test_global_queue_limit_is_shared_across_contacts_and_revocation_frees_capacity(self):
        self.responder.handle_update(connection_update("overloaded"))
        self.responder.handle_update(connection_update("still-enabled"))
        with patch("llmopenchat.telegram_business.MAX_TOTAL_PENDING_MESSAGES", 3):
            with self.responder.runtime.lock, self.responder.lock:
                # Hold admission lock too so the dispatcher cannot consume a slot yet.
                for contact in range(300, 305):
                    self.responder.handle_update(message_update("Очередь", identifier="overloaded", contact=contact))
                self.assertEqual(sum(len(chat.pending) for chat in self.responder.chats.values()), 3)
                self.assertEqual(len(self.responder.chats), 3)
                self.assertTrue(any("очередь" in text and "заполнена" in text for text in self.emitted))

                self.responder.handle_update(connection_update("overloaded", can_reply=False))
                self.responder.handle_update(message_update("Разрешённый", identifier="still-enabled", contact=400))
                accepted = self.responder.chats[("still-enabled", 400)]
                self.assertEqual(len(accepted.pending), 1)
                self.assertEqual(sum(len(chat.pending) for chat in self.responder.chats.values()), 1)
            self.join_workers()

        self.assertEqual([m["text"] for m in self.api.messages], ["Ответ: Разрешённый"])
        self.assertEqual(self.api.messages[0]["business_connection_id"], "still-enabled")

    def test_revoking_one_connection_keeps_other_contacts_queued_and_running(self):
        blocked, started, release = self.block_next_generation()
        other = RecordingClient(self.requests)
        with patch("llmopenchat.telegram_business.ApiClient", side_effect=[blocked, other, other]):
            self.responder.handle_update(message_update("Отменяемый", identifier="revoked"))
            self.assertTrue(started.wait(2), "First connection did not start")
            self.responder.handle_update(message_update("Не выполнять", identifier="revoked", message_id=2))
            self.responder.handle_update(message_update("Продолжаем 1", identifier="still-enabled", contact=303))
            self.responder.handle_update(message_update("Продолжаем 2", identifier="still-enabled", contact=303, message_id=2))

            self.responder.handle_update(connection_update("revoked", can_reply=False))
            release.set()
            self.join_workers()

        self.assertTrue(blocked.cancelled.is_set())
        self.assertFalse(other.cancelled.is_set())
        self.assertEqual([m["text"] for m in self.api.messages], ["Ответ: Продолжаем 1", "Ответ: Продолжаем 2"])
        self.assertTrue(all(m["business_connection_id"] == "still-enabled" for m in self.api.messages))
        self.assertFalse(self.responder.chats[("still-enabled", 303)].cancelled.is_set())

    def test_disabling_or_revoking_connection_cancels_work_and_discards_pending_replies(self):
        for identifier, options in (("disabled", {"enabled": False}), ("revoked", {"can_reply": False})):
            with self.subTest(options=options):
                client, started, release = self.block_next_generation()
                with patch("llmopenchat.telegram_business.ApiClient", return_value=client):
                    self.responder.handle_update(message_update("В работе", identifier=identifier))
                    self.assertTrue(started.wait(2), "Generation did not start")
                    self.responder.handle_update(message_update("В очереди", identifier=identifier, message_id=2))
                    self.responder.handle_update(connection_update(identifier, **options))
                    chat = self.responder.chats[(identifier, CONTACT)]
                    self.assertTrue(chat.cancelled.is_set())
                    self.assertTrue(client.cancelled.is_set())
                    self.assertFalse(chat.pending)
                    release.set()
                    self.join_workers()

        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.api.messages, [])

    def test_model_errors_are_logged_locally_without_sending_error_details(self):
        client = RecordingClient(self.requests, failure=ApiError("PRIVATE UPSTREAM FAILURE"))
        with patch("llmopenchat.telegram_business.ApiClient", return_value=client):
            self.send("Вопрос")

        self.assertEqual(self.api.messages, [])
        self.assertTrue(self.emitted)
        self.assertFalse(any(m["role"] == "assistant" for m in self.responder.chats[(CONNECTION, CONTACT)].messages))

    def test_media_caption_reaches_model_with_binary_unavailable_note(self):
        update = message_update()
        message = update["business_message"]
        message.pop("text")
        message.update(caption="Что на фото?", photo=[{"file_id": "photo-id", "width": 200, "height": 200}])

        self.responder.handle_update(update)
        self.join_workers()

        prompt = self.requests[0][0][-1]["content"]
        self.assertIn("Что на фото?", prompt)
        self.assertNotEqual(prompt, "Что на фото?")
        self.assertTrue(any(word in prompt.lower() for word in ("недоступ", "не передан", "не загруж", "не включ", "не вид", "binary", "unavailable")), prompt)
        self.assertEqual(len(self.api.messages), 1)

    def test_saved_business_history_is_reloaded_after_responder_restart(self):
        self.send("Запомни это")
        first = self.responder.chats[(CONNECTION, CONTACT)]
        directory = first.directory
        self.responder.close()
        restarted = self.new_responder()

        self.send("Продолжим", responder=restarted, message_id=2)

        second = restarted.chats[(CONNECTION, CONTACT)]
        self.assertEqual(second.directory, directory)
        prompts = [m["content"] for m in self.requests[-1][0] if m["role"] == "user"]
        self.assertEqual(prompts, ["Запомни это", "Продолжим"])
        self.assertIn({"role": "assistant", "content": "Ответ: Запомни это"}, self.requests[-1][0])

    def test_replayed_accepted_message_is_not_answered_again_after_restart(self):
        self.send("Не повторяй", message_id=33)
        saved_messages = copy.deepcopy(self.responder.chats[(CONNECTION, CONTACT)].messages)
        self.responder.close()
        restarted = self.new_responder()

        self.send("Не повторяй", responder=restarted, message_id=33)

        self.assertEqual(len(self.requests), 1)
        self.assertEqual([m["text"] for m in self.api.messages], ["Ответ: Не повторяй"])
        self.assertEqual(restarted.chats[(CONNECTION, CONTACT)].messages, saved_messages)

    def test_receipt_write_failure_keeps_delivered_turn_in_memory_and_history(self):
        with patch.object(self.responder, "_save_receipt", side_effect=OSError("PRIVATE DISK ERROR")):
            self.send("Сохрани контекст", message_id=41)
        chat = self.responder.chats[(CONNECTION, CONTACT)]
        self.assertEqual(load_session(chat.directory / "current.json"), chat.messages)
        self.send("Продолжай", message_id=42)
        self.assertEqual([m["content"] for m in self.requests[-1][0] if m["role"] == "user"],
                         ["Сохрани контекст", "Продолжай"])
        self.assertNotIn("PRIVATE DISK ERROR", str(self.api.messages) + str(self.emitted))

    def test_deeply_nested_receipt_json_is_rejected_without_uncaught_recursion(self):
        self.send("Первый запрос")
        chat = self.responder.chats[(CONNECTION, CONTACT)]
        self.responder.close()
        (chat.directory / "delivered.json").write_text("[" * 5000 + "]" * 5000, encoding="utf-8")
        restarted = self.new_responder()
        with self.assertRaises(ValueError):
            restarted.handle_update(message_update("Следующий запрос", message_id=2))

    def test_close_cancels_active_client_and_suppresses_delivery(self):
        client, started, release = self.block_next_generation()
        with patch("llmopenchat.telegram_business.ApiClient", return_value=client):
            self.responder.handle_update(message_update("Запрос"))
            self.assertTrue(started.wait(2), "Generation did not start")
            self.responder.close()
            release.set()
            self.join_workers()

        self.assertTrue(client.cancelled.is_set())
        self.assertEqual(self.api.messages, [])


if __name__ == "__main__":
    unittest.main()
