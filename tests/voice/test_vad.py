"""Tests for core/vad.py (task A2). No microphone needed."""
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from core.audio import CHUNK_SAMPLES, SAMPLE_RATE
from core.vad import SileroVAD, record_command, save_wav

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "open_notepad.wav"


class ScriptedVAD:
    """Returns pre-set probabilities, one per chunk; keeps the state-machine tests exact."""

    def __init__(self, probs):
        self._probs = iter(probs)

    def reset(self):
        pass

    def feed(self, chunk):
        return next(self._probs)


def tagged_chunks(n):
    """n chunks; chunk i is filled with the value i, so tests can see which ones were kept."""
    consumed = []
    for i in range(n):
        consumed.append(i)
        yield np.full(CHUNK_SAMPLES, i, dtype=np.int16)


def run(probs, **kwargs):
    return record_command(tagged_chunks(len(probs)), ScriptedVAD(probs), **kwargs)


# --- state machine ----------------------------------------------------------
def test_stops_800ms_after_speech_and_keeps_preroll():
    probs = [0.0] * 5 + [0.9] * 6 + [0.0] * 10 + [0.0] * 5
    rec = run(probs)
    assert rec.reason == "silence"
    # 4 pre-roll chunks (1..4) + 6 speech + 10 silent chunks (800 ms)
    assert len(rec.audio) == 20 * CHUNK_SAMPLES
    assert rec.audio[0] == 1


def test_short_pause_inside_a_sentence_does_not_cut():
    probs = [0.9] * 3 + [0.0] * 5 + [0.9] * 3 + [0.0] * 10
    rec = run(probs)
    assert rec.reason == "silence"
    assert len(rec.audio) == 21 * CHUNK_SAMPLES


def test_hysteresis_keeps_recording_between_the_thresholds():
    probs = [0.9] + [0.4] * 20 + [0.0] * 10  # 0.4 is below start (0.5) but above end (0.35)
    rec = run(probs)
    assert rec.reason == "silence"
    assert len(rec.audio) == 31 * CHUNK_SAMPLES


def test_caps_at_ten_seconds():
    rec = run([0.9] * 200)
    assert rec.reason == "max_length"
    assert rec.duration_s == pytest.approx(10.0, abs=0.1)


def test_gives_up_after_five_seconds_without_speech():
    rec = run([0.0] * 200)
    assert rec.reason == "no_speech"
    assert len(rec.audio) == 0


def test_stream_ending_mid_speech_returns_what_was_heard():
    rec = run([0.9] * 4)
    assert rec.reason == "stream_ended"
    assert len(rec.audio) == 4 * CHUNK_SAMPLES


# --- real Silero model ----------------------------------------------------------
@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    with wave.open(str(FIXTURE)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)


def as_chunks(audio: np.ndarray):
    padded = np.pad(audio, (0, -len(audio) % CHUNK_SAMPLES))
    for i in range(0, len(padded), CHUNK_SAMPLES):
        yield padded[i : i + CHUNK_SAMPLES]


def test_model_scores_silence_low():
    vad = SileroVAD()
    for chunk in as_chunks(np.zeros(SAMPLE_RATE, dtype=np.int16)):
        assert vad.feed(chunk) < 0.1


def test_model_carries_leftover_samples_between_chunks():
    vad = SileroVAD()
    vad.feed(np.zeros(CHUNK_SAMPLES, dtype=np.int16))  # 1280 = 2 windows + 256 left over
    assert len(vad._leftover) == 256
    vad.feed(np.zeros(CHUNK_SAMPLES, dtype=np.int16))  # 1536 = 3 windows exactly
    assert len(vad._leftover) == 0


def test_model_detects_speech_in_the_fixture(speech):
    vad = SileroVAD()
    assert max(vad.feed(c) for c in as_chunks(speech)) > 0.5


def test_end_to_end_stops_shortly_after_the_speech_ends(speech):
    rng = np.random.default_rng(0)
    noise = lambda seconds: rng.normal(0, 15, int(seconds * SAMPLE_RATE)).astype(np.int16)
    stream = np.concatenate((noise(1.0), speech, noise(3.0)))

    rec = record_command(as_chunks(stream), SileroVAD())

    # The clip has silence around the words, so measure the audible part (about 0.9 s).
    active = np.flatnonzero(np.abs(speech) > 300)
    spoken_s = (active[-1] - active[0]) / SAMPLE_RATE
    assert rec.reason == "silence"
    # up to 0.32 s pre-roll + the words + 0.8 s of silence, give or take chunk rounding
    assert spoken_s + 0.8 < rec.duration_s < spoken_s + 0.32 + 0.8 + 0.25
    assert np.abs(rec.audio).max() == np.abs(speech).max()  # the loudest word was captured


def test_inference_is_fast_enough_for_real_time(speech):
    vad = SileroVAD()
    chunks = list(as_chunks(speech))
    start = time.perf_counter()
    for chunk in chunks:
        vad.feed(chunk)
    per_chunk_ms = (time.perf_counter() - start) * 1000 / len(chunks)
    print(f"VAD: {per_chunk_ms:.2f} ms per 80 ms chunk")
    assert per_chunk_ms < 40


def test_save_wav_round_trips(tmp_path):
    audio = (np.arange(1000) % 200 - 100).astype(np.int16)
    path = save_wav(tmp_path / "nested" / "clip.wav", audio)
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        assert np.array_equal(np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16), audio)
