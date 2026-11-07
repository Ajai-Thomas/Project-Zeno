"""Typed-console stand-in for the Voice side. Lets Person B test the Brain with no mic.

listen() reads a typed line; speak() prints. Type "exit" or press Ctrl+Z/Ctrl+D to quit.
"""
from __future__ import annotations

from typing import Callable

from core import events
from core.interfaces import Utterance, status


class ConsoleVoice:
    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._output = output_fn

    def listen(self) -> Utterance | None:
        status(events.LISTENING)
        try:
            text = self._input("you> ").strip()
        except EOFError:
            return None
        if text.lower() in {"exit", "quit"}:
            return None
        if not text:
            return self.listen()
        # Malayalam script means Malayalam; otherwise assume English.
        lang = "ml" if any("ഀ" <= ch <= "ൿ" for ch in text) else "en"
        return Utterance(text, lang)

    def speak(self, text: str, lang: str = "en") -> None:
        status(events.SPEAKING)
        self._output(f"zeno> {text}")

    def stop_speaking(self) -> None:
        pass
