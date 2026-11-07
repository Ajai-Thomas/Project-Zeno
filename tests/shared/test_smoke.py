"""Smoke test for the shared base: the loop runs end to end on the stubs."""
import subprocess
import sys
from pathlib import Path

import pytest

from core import events
from core.interfaces import (
    Brain,
    Utterance,
    VoiceIO,
    clear_status_listeners,
    subscribe_status,
)
from core.stubs.console_voice import ConsoleVoice
from core.stubs.echo_brain import EchoBrain
from main import run

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_leftover_listeners():
    yield
    clear_status_listeners()


def test_stubs_satisfy_the_interfaces():
    assert isinstance(ConsoleVoice(), VoiceIO)
    assert isinstance(EchoBrain(), Brain)


def test_utterance_rejects_unknown_language():
    with pytest.raises(ValueError):
        Utterance("hi", lang="fr")


def test_loop_runs_end_to_end_on_stubs():
    typed = iter(["open notepad", "exit"])
    spoken: list[str] = []
    seen: list[str] = []
    subscribe_status(seen.append)

    voice = ConsoleVoice(input_fn=lambda _prompt: next(typed), output_fn=spoken.append)
    turns = run(voice, EchoBrain())

    assert turns == 1
    assert spoken == ["zeno> open notepad"]
    assert seen[:3] == [events.LISTENING, events.THINKING, events.THINKING]
    assert events.SPEAKING in seen and seen[-2:] == [events.IDLE, events.LISTENING]


def test_main_script_runs_as_a_program():
    result = subprocess.run(
        [sys.executable, "main.py", "--voice", "console", "--brain", "echo"],
        input="hello zeno\nexit\n",
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "zeno> hello zeno" in result.stdout


def test_requirements_never_mention_pytorch():
    for name in ("requirements.txt", "requirements-dev.txt"):
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        active = [ln.split("#")[0].strip().lower() for ln in lines]
        assert not any(ln.startswith(("torch", "torchaudio", "torchvision")) for ln in active)
