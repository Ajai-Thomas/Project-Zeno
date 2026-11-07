"""Echo stand-in for the Brain side. Lets Person A test the Voice with no LLM.

It simply repeats what it heard.
"""
from __future__ import annotations

from core import events
from core.interfaces import Utterance, status


class EchoBrain:
    def think(self, u: Utterance) -> str:
        status(events.THINKING)
        return u.text
