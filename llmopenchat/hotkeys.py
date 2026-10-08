"""Console shortcuts that keep unfinished input out of the conversation."""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings


@dataclass(frozen=True)
class PromptAction:
    command: str
    draft: Document


def create_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    def add_command(key: str, command: str) -> None:
        @bindings.add(key)
        def run_command(event):
            # Returning an action instead of accepting the buffer prevents a
            # draft or shortcut command from being sent or added to history.
            event.app.exit(result=PromptAction(command, event.current_buffer.document))

    for key, command in (
        ("f1", "/help"),
        ("f2", "/menu"),
        ("f3", "/tools"),
        ("c-n", "/clear"),
        ("c-s", "/save"),
        ("c-t", "/thinking"),
        ("c-q", "/quit"),
    ):
        add_command(key, command)

    @bindings.add("c-l")
    def clear_screen(event):
        # Redraw only: conversation, draft, selection and cursor stay intact.
        event.app.renderer.clear()

    @bindings.add("escape", "enter")
    def new_line(event):
        event.current_buffer.insert_text("\n")

    return bindings


def toolbar(show_reasoning: bool, harness_status: str = "") -> str:
    visible = "да" if show_reasoning else "нет"
    shortcuts = f" F1 справка · F2 меню · F3 инструменты · ^N новый · ^S сохранить · ^L экран · ^T мысли:{visible} · ^Q выход "
    return shortcuts + ("\n " + harness_status if harness_status else "")
