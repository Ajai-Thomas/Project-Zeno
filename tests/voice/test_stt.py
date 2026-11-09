"""Tests for core/stt.py (task A3).

The first group uses a fake Whisper model (fast, no download). The second group uses the real
`base` model, which downloads once (about 140 MB) and is skipped if it can't be loaded.
"""
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from core.audio import SAMPLE_RATE
from core.stt import STTError, Transcriber, load_wav

AUDIO = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
Seg = namedtuple("Seg", "text no_speech_prob avg_logprob")


class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(self.segments), None


def loud(seconds=1.0):
    return (np.sin(np.arange(int(seconds * SAMPLE_RATE)) / 20) * 8000).astype(np.int16)


# --- guard logic, fake model --------------------------------------------------
def test_segments_are_joined_with_spaces():
    model = FakeModel([Seg(" open ", 0.0, -0.2), Seg("notepad ", 0.0, -0.2)])
    assert Transcriber(model=model).transcribe(loud()).text == "open notepad"


def test_quiet_noise_never_reaches_the_model():
    model = FakeModel([Seg("Thank you.", 0.0, -0.1)])
    quiet = (np.random.default_rng(0).normal(0, 15, SAMPLE_RATE)).astype(np.int16)
    assert Transcriber(model=model).transcribe(quiet).text == ""
    assert Transcriber(model=model).transcribe(np.empty(0, dtype=np.int16)).text == ""
    assert model.calls == []


def test_segment_that_is_probably_not_speech_and_unsure_is_dropped():
    model = FakeModel([Seg("Thank you.", 0.9, -1.5), Seg("open notepad", 0.1, -0.3)])
    assert Transcriber(model=model).transcribe(loud()).text == "open notepad"


def test_unsure_but_clearly_speech_is_kept():
    model = FakeModel([Seg("volume to thirty", 0.1, -1.4)])
    assert Transcriber(model=model).transcribe(loud()).text == "volume to thirty"


def test_settings_are_tuned_for_short_commands():
    model = FakeModel([Seg("hi", 0.0, -0.1)])
    Transcriber(model=model, initial_prompt="Spotify, Rahul").transcribe(loud())
    audio, kwargs = model.calls[0]
    assert audio.dtype == np.float32 and np.abs(audio).max() <= 1.0
    assert kwargs["beam_size"] == 1 and kwargs["vad_filter"] is False
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["initial_prompt"] == "Spotify, Rahul" and kwargs["language"] == "en"


def test_default_prompt_is_used_and_can_be_turned_off():
    from core.stt import DEFAULT_PROMPT

    model = FakeModel([Seg("hi", 0.0, -0.1)])
    Transcriber(model=model).transcribe(loud())
    Transcriber(model=model, initial_prompt="").transcribe(loud())
    assert model.calls[0][1]["initial_prompt"] == DEFAULT_PROMPT
    assert model.calls[1][1]["initial_prompt"] is None


def test_wav_path_and_array_give_the_same_result():
    model = FakeModel([Seg("open notepad", 0.0, -0.1)])
    stt = Transcriber(model=model)
    wav = AUDIO / "open_notepad.wav"
    assert stt.transcribe(wav) == stt.transcribe(load_wav(wav))
    assert np.array_equal(model.calls[0][0], model.calls[1][0])


def test_wrong_wav_format_is_rejected(tmp_path):
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\0" * 400)
    with pytest.raises(STTError, match="16 kHz"):
        load_wav(path)


# --- real model ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    stt = Transcriber()
    try:
        stt.warm_up()
    except STTError as exc:
        pytest.skip(f"Whisper model unavailable: {exc}")
    return stt


def words(text: str) -> list[str]:
    text = text.lower().replace("-", "").replace("30", "thirty")
    return re.sub(r"[^a-z ]", "", text).split()


def word_errors(expected: list[str], actual: list[str]) -> int:
    """Word-level edit distance."""
    prev = list(range(len(actual) + 1))
    for i, e in enumerate(expected, 1):
        row = [i]
        for j, a in enumerate(actual, 1):
            row.append(min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (e != a)))
        prev = row
    return prev[-1]


COMMANDS = {
    "open_notepad": "open notepad",
    "what_time_is_it": "what time is it",
    "volume_to_thirty": "volume to thirty",
    "search_lofi_on_youtube": "search lofi on youtube",
}


def test_real_model_transcribes_the_fixture_commands(real):
    errors = total = 0
    for name, expected in COMMANDS.items():
        heard = real.transcribe(AUDIO / f"{name}.wav")
        print(f"{name}: {heard.text!r} ({real.last_seconds:.2f} s)")
        errors += word_errors(expected.split(), words(heard.text))
        total += len(expected.split())
    assert errors / total <= 0.25, f"word error rate {errors}/{total}"


def test_real_model_returns_nothing_for_silence(real):
    noise = np.random.default_rng(1).normal(0, 15, SAMPLE_RATE * 2).astype(np.int16)
    assert real.transcribe(noise).text == ""
    assert real.transcribe(np.zeros(SAMPLE_RATE, dtype=np.int16)).text == ""


def test_real_model_is_fast_enough(real):
    timings = []
    for _ in range(4):  # the first run warms up; take the best of the rest so a busy moment doesn't fail the test
        real.transcribe(AUDIO / "open_notepad.wav")
        timings.append(real.last_seconds)
    best = min(timings[1:])
    print(f"latency: best {best:.2f} s for {len(load_wav(AUDIO / 'open_notepad.wav')) / SAMPLE_RATE:.1f} s of audio")
    assert best < 2.0
