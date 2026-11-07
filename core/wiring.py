"""Chooses real or stub implementations. SHARED FILE.

"real" means the module the owning person builds. Until it exists, we warn and use the stub,
so each side can run `main.py` without the other.
"""
from __future__ import annotations

import importlib
import sys

from core.interfaces import Brain, VoiceIO
from core.stubs.console_voice import ConsoleVoice
from core.stubs.echo_brain import EchoBrain

# (module, class) of the real implementations. Voice builds RealVoice (task A7),
# Brain builds AgentBrain (task B3).
REAL_VOICE = ("core.voice", "RealVoice")
REAL_BRAIN = ("core.brain", "AgentBrain")


def _load(spec: tuple[str, str], label: str):
    module_name, class_name = spec
    try:
        return getattr(importlib.import_module(module_name), class_name)()
    except (ImportError, AttributeError) as exc:
        print(f"[wiring] real {label} not available yet ({exc}); using the stub.", file=sys.stderr)
        return None


def build_voice(kind: str) -> VoiceIO:
    if kind == "real":
        voice = _load(REAL_VOICE, "voice")
        if voice is not None:
            return voice
    return ConsoleVoice()


def build_brain(kind: str) -> Brain:
    if kind == "real":
        brain = _load(REAL_BRAIN, "brain")
        if brain is not None:
            return brain
    return EchoBrain()
