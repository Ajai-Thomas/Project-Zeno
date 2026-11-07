"""The contract between the Voice side (Person A) and the Brain side (Person B).

SHARED FILE. Change it only when both people agree.

Voice implements `VoiceIO`; Brain implements `Brain`. `main.py` joins them.
Neither side imports the other's modules; they only know this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

LANGS = ("en", "ml", "mixed")  # English, Malayalam, Manglish


@dataclass(frozen=True)
class Utterance:
    text: str
    lang: str = "en"  # one of LANGS

    def __post_init__(self) -> None:
        if self.lang not in LANGS:
            raise ValueError(f"lang must be one of {LANGS}, got {self.lang!r}")


@runtime_checkable
class VoiceIO(Protocol):
    """Everything that hears and speaks. Owned by Voice (Person A)."""

    def listen(self) -> Utterance | None:
        """Block until the user says a command (wake word, record, transcribe).

        Return None when input has ended and the program should exit.
        """
        ...

    def speak(self, text: str, lang: str = "en") -> None:
        """Say `text` aloud in `lang`. Blocks until done or stop_speaking()."""
        ...

    def stop_speaking(self) -> None:
        """Cut off speech immediately (for "Zeno, stop"). Safe to call from another thread."""
        ...


@runtime_checkable
class Brain(Protocol):
    """Everything that thinks, acts and learns. Owned by Brain (Person B)."""

    def think(self, u: Utterance) -> str:
        """Reason about `u`, run any skills, learn, and return the spoken reply."""
        ...


# --- status events: Brain/main send, Voice's overlay shows ------------------
# Event names live in core/events.py. `status()` is a tiny publish/subscribe hub
# so the sender never needs to know whether an overlay exists.

_listeners: list[Callable[[str], None]] = []


def subscribe_status(listener: Callable[[str], None]) -> None:
    """Voice's overlay (or a test) calls this to receive status events."""
    _listeners.append(listener)


def clear_status_listeners() -> None:
    _listeners.clear()


def status(event: str) -> None:
    """Announce what Zeno is doing, e.g. status("acting:open_app")."""
    for listener in list(_listeners):
        listener(event)
