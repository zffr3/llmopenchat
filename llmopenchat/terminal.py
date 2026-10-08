"""Render untrusted model and tool text without terminal control sequences."""

from __future__ import annotations

import unicodedata


def terminal_text(value: str) -> str:
    return "".join(
        character if character in "\n\t" or not unicodedata.category(character).startswith("C")
        else (f"\\x{ord(character):02x}" if ord(character) < 256 else f"\\u{ord(character):04x}")
        for character in value
    )
