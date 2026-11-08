"""Tests for core/audio.py (task A1). No real microphone needed except the `mic` test."""
import numpy as np
import pytest

from core import audio
from core.audio import CHUNK_SAMPLES, SAMPLE_RATE, AudioError, MicStream, rms_level


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def feed(self, n_samples: int, value: int = 100) -> None:
        self.callback(np.full((n_samples, 1), value, dtype=np.int16), n_samples, None, None)


class FakeSD:
    def __init__(self):
        self.streams: list[FakeStream] = []
        self.fail: Exception | None = None

    def InputStream(self, **kwargs):
        if self.fail:
            raise self.fail
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream

    def query_devices(self, device=None, kind=None):
        if kind or device is not None:
            return {"name": "Fake Mic"}
        return [
            {"name": "Speakers", "max_input_channels": 0},
            {"name": "Fake Mic", "max_input_channels": 1},
        ]


@pytest.fixture
def sd(monkeypatch):
    fake = FakeSD()
    monkeypatch.setattr(audio, "_sd", lambda: fake)
    return fake


def test_rms_level_silence_and_full_scale():
    assert rms_level(np.zeros(CHUNK_SAMPLES, dtype=np.int16)) == 0.0
    assert rms_level(np.full(CHUNK_SAMPLES, 32767, dtype=np.int16)) == pytest.approx(1.0, abs=1e-3)
    assert rms_level(np.full(CHUNK_SAMPLES, 16384, dtype=np.int16)) == pytest.approx(0.5, abs=1e-3)
    assert rms_level(np.empty(0, dtype=np.int16)) == 0.0


def test_stream_opens_at_16khz_mono_int16(sd):
    with MicStream():
        pass
    kwargs = sd.streams[0].kwargs
    assert (kwargs["samplerate"], kwargs["channels"], kwargs["dtype"]) == (SAMPLE_RATE, 1, "int16")
    assert kwargs["blocksize"] == CHUNK_SAMPLES == 1280
    assert sd.streams[0].started and sd.streams[0].closed


def test_chunks_are_exactly_80ms_whatever_size_the_callback_uses(sd):
    with MicStream() as mic:
        stream = sd.streams[0]
        for size in (1000, 1000, 560):  # 2560 samples = two chunks
            stream.feed(size)
        first, second = mic.read(), mic.read()
        with pytest.raises(TimeoutError):
            mic.read(timeout=0.05)
    for chunk in (first, second):
        assert chunk.shape == (CHUNK_SAMPLES,) and chunk.dtype == np.int16


def test_chunks_generator_yields_in_order_then_stops(sd):
    with MicStream() as mic:
        for value in (1, 2, 3):
            sd.streams[0].feed(CHUNK_SAMPLES, value)
    assert [int(c[0]) for c in mic.chunks()] == [1, 2, 3]


def test_full_queue_drops_the_oldest_chunk(sd):
    with MicStream(queue_chunks=2) as mic:
        for value in (1, 2, 3, 4):
            sd.streams[0].feed(CHUNK_SAMPLES, value)
        assert mic.dropped == 2
        assert [int(mic.read()[0]), int(mic.read()[0])] == [3, 4]


def test_drain_empties_buffered_audio(sd):
    with MicStream() as mic:
        sd.streams[0].feed(CHUNK_SAMPLES * 2 + 100)
        mic.drain()
        with pytest.raises(TimeoutError):
            mic.read(timeout=0.05)
        sd.streams[0].feed(CHUNK_SAMPLES - 100)  # the old 100-sample remainder must be gone
        with pytest.raises(TimeoutError):
            mic.read(timeout=0.05)


def test_open_failure_becomes_a_readable_audio_error(sd):
    sd.fail = OSError("Invalid sample rate")
    with pytest.raises(AudioError, match="Privacy"):
        MicStream().start()


def test_list_option_prints_only_input_devices(sd, capsys):
    assert audio.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "Fake Mic" in out and "Speakers" not in out


@pytest.mark.mic
def test_real_microphone_delivers_chunks():
    try:
        devices = audio.list_input_devices()
    except AudioError:
        pytest.skip("sounddevice unavailable")
    if not devices:
        pytest.skip("no input device")
    with MicStream() as mic:
        chunks = [mic.read(timeout=2.0) for _ in range(3)]
    assert all(c.shape == (CHUNK_SAMPLES,) and c.dtype == np.int16 for c in chunks)
