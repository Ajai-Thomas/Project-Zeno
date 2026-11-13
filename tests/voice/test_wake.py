"""Tests for core/wake.py (task A5). No microphone needed.

The scoring tests use openWakeWord's pre-trained "hey_jarvis" model as a stand-in for hey_zeno
(it downloads once, 1.3 MB, and is never committed). They are skipped if it can't be fetched.
"""
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from core import wake
from core.audio import CHUNK_SAMPLES, SAMPLE_RATE
from core.wake import WakeError, WakeModel

AUDIO = Path(__file__).resolve().parents[1] / "fixtures" / "audio"


# --- pure evaluation maths --------------------------------------------------------
def test_count_activations_counts_each_phrase_once():
    def quiet(n):
        return [0.0] * n

    loud = [0.9] * 4  # one spoken phrase lights up several chunks in a row
    assert wake.count_activations(quiet(5) + loud + quiet(5), 0.5) == 1
    assert wake.count_activations(loud + quiet(30) + loud, 0.5) == 2  # 2.4 s apart: separate phrases
    assert wake.count_activations(loud + quiet(5) + loud, 0.5) == 1  # 0.4 s apart: same phrase


def test_count_activations_respects_the_threshold():
    assert wake.count_activations([0.4, 0.45, 0.4], 0.5) == 0
    assert wake.count_activations([0.4, 0.55, 0.4], 0.5) == 1


def test_sweep_reports_detection_and_false_accepts_per_threshold():
    rows = wake.sweep([0.95, 0.9, 0.6, 0.2], [0.1, 0.55, 0.0], thresholds=(0.3, 0.5, 0.8))
    assert [r.detection_rate for r in rows] == [0.75, 0.75, 0.5]
    assert [r.false_accept_clips for r in rows] == [1, 1, 0]
    assert all(r.false_accepts_per_hour is None for r in rows)  # no background audio given


def test_sweep_false_accepts_per_hour_from_background_audio():
    one_hour = int(3_600_000 / wake.CHUNK_MS)
    bg = [0.0] * one_hour
    for i in (100, 5000, 20000):  # three separate false triggers in one hour
        bg[i : i + 3] = [0.9] * 3
    (row,) = wake.sweep([0.9], [], [bg], thresholds=(0.5,))
    assert row.false_accepts_per_hour == pytest.approx(3.0)


# --- model lookup ----------------------------------------------------------------------
def test_unknown_model_gives_a_readable_error():
    with pytest.raises(WakeError, match="No wake word model 'nonsense'"):
        wake.resolve_model("nonsense")


def test_hey_zeno_missing_points_to_the_training_instructions(monkeypatch, tmp_path):
    monkeypatch.setattr(wake, "ASSET_DIR", tmp_path)
    with pytest.raises(WakeError, match="assets/wakeword/README.md"):
        wake.resolve_model("hey_zeno")


def test_a_trained_model_in_assets_is_found_by_name(monkeypatch, tmp_path):
    (tmp_path / "hey_zeno.onnx").write_bytes(b"x")
    monkeypatch.setattr(wake, "ASSET_DIR", tmp_path)
    assert wake.resolve_model("hey_zeno") == tmp_path / "hey_zeno.onnx"


# --- real scoring with the stand-in model ---------------------------------------------------
@pytest.fixture(scope="module")
def model():
    try:
        return WakeModel("hey_jarvis", threshold=0.5)
    except WakeError as exc:
        pytest.skip(f"wake word model unavailable: {exc}")


def test_model_fires_on_the_wake_phrase_and_stays_quiet_otherwise(model):
    assert model.detected(model.peak(wake.load_wav(AUDIO / "hey_jarvis.wav")))
    for other in ("open_notepad", "what_time_is_it", "hey_zeno"):  # hey_zeno is a negative for this stand-in
        assert not model.detected(model.peak(wake.load_wav(AUDIO / f"{other}.wav"))), other
    assert model.peak(np.zeros(SAMPLE_RATE, dtype=np.int16)) < 0.1


def test_scoring_chunk_by_chunk_matches_scoring_the_clip(model):
    audio = wake.load_wav(AUDIO / "hey_jarvis.wav")
    expected = model.scores(audio)
    model.reset()
    padded = np.concatenate((audio, np.zeros(int(wake.TAIL_S * SAMPLE_RATE), dtype=np.int16)))
    padded = np.pad(padded, (0, -len(padded) % CHUNK_SAMPLES))
    streamed = [model.score(padded[i : i + CHUNK_SAMPLES]) for i in range(0, len(padded), CHUNK_SAMPLES)]
    assert streamed == pytest.approx(list(expected))
    assert list(model.scores(audio)) == list(expected)  # repeatable: same clip, identical scores


def test_wrong_chunk_size_is_rejected(model):
    with pytest.raises(ValueError, match="1280"):
        model.score(np.zeros(1000, dtype=np.int16))


def test_file_option_prints_the_peak_score(model, capsys):
    assert wake.main(["--file", str(AUDIO / "hey_jarvis.wav"), "--model", "hey_jarvis"]) == 0
    assert "DETECTED" in capsys.readouterr().out
    assert wake.main(["--file", str(AUDIO / "open_notepad.wav"), "--model", "hey_jarvis"]) == 0
    assert "not detected" in capsys.readouterr().out


def test_eval_reports_a_table_for_recorded_clips(model, tmp_path, capsys):
    (tmp_path / "pos").mkdir()
    (tmp_path / "neg").mkdir()
    shutil.copy(AUDIO / "hey_jarvis.wav", tmp_path / "pos" / "pos_01.wav")
    shutil.copy(AUDIO / "open_notepad.wav", tmp_path / "neg" / "neg_01.wav")
    assert wake.main(["--eval", "--dir", str(tmp_path), "--model", "hey_jarvis"]) == 0
    out = capsys.readouterr().out
    assert "1 positive clips, 1 negative clips" in out
    row = next(line for line in out.splitlines() if "<- current" in line)
    assert "100%" in row and "0 of 1" in row


def test_eval_without_recordings_says_how_to_record(tmp_path, capsys):
    (tmp_path / "neg").mkdir()
    assert wake.main(["--eval", "--dir", str(tmp_path), "--model", "hey_jarvis"]) == 1
    assert "--record" in capsys.readouterr().out


# --- guided recording (fake microphone) -------------------------------------------------------
class FakeMic:
    def __init__(self):
        self.drained = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def drain(self):
        self.drained += 1

    def read(self, timeout=1.0):
        return np.full(CHUNK_SAMPLES, 123, dtype=np.int16)


def test_record_saves_numbered_clips_of_the_right_length(tmp_path):
    prompts, said = [], []
    wake.record_clips(tmp_path, count=3, mic_factory=FakeMic, ask=prompts.append, say=said.append)

    assert sorted(p.name for p in (tmp_path / "pos").glob("*.wav")) == ["pos_01.wav", "pos_02.wav", "pos_03.wav"]
    assert sorted(p.name for p in (tmp_path / "neg").glob("*.wav")) == ["neg_01.wav", "neg_02.wav", "neg_03.wav"]
    with wave.open(str(tmp_path / "pos" / "pos_01.wav")) as w:
        assert (w.getframerate(), w.getnchannels()) == (SAMPLE_RATE, 1)
        assert w.getnframes() == int(wake.CLIP_S * 1000 / wake.CHUNK_MS) * CHUNK_SAMPLES
    assert "Hey Zeno" in prompts[0] and "Hey Zoe" in prompts[3]  # negatives start with sound-alikes
    assert "3 positive and 3 negative" in said[-1]


def test_background_recording_is_saved_under_bg(tmp_path):
    path = wake.record_background(tmp_path, 0.05, mic_factory=FakeMic)  # 3 seconds
    assert path.parent == tmp_path / "bg" and path.exists()
    with wave.open(str(path)) as w:
        assert w.getnframes() == int(0.05 * 60_000 / wake.CHUNK_MS) * CHUNK_SAMPLES
