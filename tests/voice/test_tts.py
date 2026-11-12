"""Tests for core/tts.py (task A4).

The first group uses a fake engine and a fake speaker stream that "plays" in real time, so it
needs no speakers or downloads. The second group uses the real Kokoro and Piper voices (downloaded
once) and is skipped if they can't be loaded.
"""
import threading
import time
import wave

import numpy as np
import pytest

from core import tts
from core.tts import Speaker, TTSError, split_sentences

RATE = 24_000


class FakeStream:
    """Plays in simulated real time: write() takes as long as the audio lasts."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.written: list[np.ndarray] = []
        self.write_times: list[float] = []
        self.aborted = threading.Event()
        self.closed = False

    def start(self):
        pass

    def write(self, data):
        self.write_times.append(time.perf_counter())
        self.written.append(data.copy())
        if self.aborted.wait(timeout=len(data) / RATE):  # abort() ends the wait early
            raise OSError("stream aborted")

    def stop(self):
        pass

    def abort(self):
        self.aborted.set()

    def close(self):
        self.closed = True


class FakeSD:
    def __init__(self):
        self.streams: list[FakeStream] = []
        self.fail: Exception | None = None

    def OutputStream(self, **kwargs):
        if self.fail:
            raise self.fail
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


class FakeEngine:
    """Yields one array per sentence; sentence i is filled with (i+1)*10000 and lasts `seconds[i]`."""

    sample_rate = RATE

    def __init__(self, seconds=(0.2, 0.2), synth_delay=0.0):
        self.seconds = seconds
        self.synth_delay = synth_delay
        self.synth_times: list[float] = []

    def synthesize(self, text):
        for i, s in enumerate(self.seconds):
            time.sleep(self.synth_delay)
            self.synth_times.append(time.perf_counter())
            yield np.full(int(RATE * s), (i + 1) * 10000, dtype=np.int16)


@pytest.fixture
def sd(monkeypatch):
    fake = FakeSD()
    monkeypatch.setattr(tts, "_sd", lambda: fake)
    return fake


def make(engine, **kwargs):
    return Speaker(lead_in_ms=kwargs.pop("lead_in_ms", 0), engine=engine, **kwargs)


# --- fake engine + fake speakers -------------------------------------------------
def test_all_audio_is_played_in_order(sd):
    make(FakeEngine((0.2, 0.2))).speak("one. two.")
    stream = sd.streams[0]
    audio = np.concatenate(stream.written)
    assert stream.kwargs["samplerate"] == RATE and stream.kwargs["channels"] == 1
    assert len(audio) == int(RATE * 0.2) * 2
    assert audio[0] == 8000 and audio[-1] == 16000  # sentence values 10000 and 20000, times the 0.8 gain
    assert all(w.shape[1] == 1 for w in stream.written)  # (frames, channels)
    assert stream.closed


def test_first_sentence_plays_before_the_second_is_ready(sd):
    engine = FakeEngine((0.3, 0.3), synth_delay=0.25)
    speaker = make(engine)
    speaker.speak("one. two.")
    assert sd.streams[0].write_times[0] < engine.synth_times[1]  # playing began before sentence 2 existed
    assert speaker.first_audio_s < 0.4


def test_stop_speaking_from_another_thread_ends_playback_quickly(sd):
    speaker = make(FakeEngine((2.0, 2.0)))  # four seconds of speech
    stopped_at = []

    def stopper():
        time.sleep(0.3)
        stopped_at.append(time.perf_counter())
        speaker.stop_speaking()

    threading.Thread(target=stopper).start()
    speaker.speak("long")
    returned_at = time.perf_counter()

    assert returned_at - stopped_at[0] < 0.2
    assert sum(len(w) for w in sd.streams[0].written) < RATE * 1.0  # nowhere near the full 4 s
    assert not speaker.is_speaking and sd.streams[0].closed


def test_stop_is_harmless_when_idle_and_speaking_works_again_after_a_stop(sd):
    speaker = make(FakeEngine((0.5,)))
    speaker.stop_speaking()  # nothing playing: no error
    threading.Timer(0.1, speaker.stop_speaking).start()
    speaker.speak("first")
    speaker.speak("second")
    assert sum(len(w) for w in sd.streams[1].written) == int(RATE * 0.5)  # played in full


def test_stop_works_while_the_engine_is_still_synthesizing(sd):
    speaker = make(FakeEngine((1.0,), synth_delay=1.0))
    threading.Timer(0.1, speaker.stop_speaking).start()
    began = time.perf_counter()
    speaker.speak("slow")
    assert time.perf_counter() - began < 0.5
    assert sd.streams[0].written == []


def test_audible_is_set_once_the_first_word_plays(sd):
    speaker = make(FakeEngine((0.2,)))
    assert not speaker.audible.is_set()
    speaker.speak("x")
    assert speaker.audible.is_set()
    speaker.speak("")  # empty text must not touch it
    assert speaker.audible.is_set()


def test_gain_keeps_full_scale_audio_away_from_clipping(sd):
    class Loud(FakeEngine):
        def synthesize(self, text):
            yield np.full(2000, 32767, dtype=np.int16)

    make(Loud()).speak("x")
    assert np.concatenate(sd.streams[0].written).max() == int(32767 * tts.GAIN) < 32767


def test_silent_lead_in_comes_first_so_sleeping_outputs_can_wake(sd):
    make(FakeEngine((0.1,)), lead_in_ms=250).speak("x")
    audio = np.concatenate(sd.streams[0].written)
    lead = int(RATE * 0.25)
    assert len(audio) == lead + int(RATE * 0.1)
    assert not audio[:lead].any() and audio[lead] == 8000


def test_output_device_is_passed_to_the_stream(sd):
    make(FakeEngine((0.1,)), device=3).speak("x")
    assert sd.streams[0].kwargs["device"] == 3


def test_empty_text_does_nothing(sd):
    make(FakeEngine()).speak("   ")
    assert sd.streams == []


def test_non_english_still_speaks_but_warns_once(sd, capsys):
    speaker = make(FakeEngine((0.1,)))
    speaker.speak("namaskaram", lang="ml")
    speaker.speak("namaskaram", lang="ml")
    assert capsys.readouterr().err.count("no 'ml' voice") == 1
    assert len(sd.streams) == 2


def test_speaker_failure_is_a_readable_error(sd):
    sd.fail = OSError("no device")
    with pytest.raises(TTSError, match="speakers"):
        make(FakeEngine()).speak("hi")


def test_synthesis_failure_is_reported(sd):
    class Broken(FakeEngine):
        def synthesize(self, text):
            raise ValueError("bad phonemes")
            yield

    with pytest.raises(TTSError, match="bad phonemes"):
        make(Broken()).speak("hi")


# --- engine choice and downloads (no real models) ---------------------------------------
def test_defaults_come_from_config_kokoro_with_bella():
    speaker = Speaker()
    assert (speaker.engine_name, speaker.voice_name) == ("kokoro", "af_bella")


def test_a_piper_style_voice_name_selects_the_piper_engine():
    speaker = Speaker("en_US-lessac-medium")
    assert (speaker.engine_name, speaker.voice_name) == ("piper", "en_US-lessac-medium")
    assert Speaker(engine_name="piper").voice_name == "en_US-hfc_female-medium"


def test_unknown_engine_is_rejected():
    with pytest.raises(TTSError, match="Unknown TTS engine"):
        Speaker(engine_name="festival")


def test_split_sentences():
    assert split_sentences("Hello there. Opening Notepad!  Done?") == ["Hello there.", "Opening Notepad!", "Done?"]
    assert split_sentences("  no punctuation  ") == ["no punctuation"]
    assert split_sentences("   ") == []


def test_missing_kokoro_files_without_internet_is_a_readable_error(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("no internet")

    monkeypatch.setattr(tts.urllib.request, "urlretrieve", offline)
    with pytest.raises(TTSError, match="could not be fetched"):
        Speaker(engine_name="kokoro", models_dir=tmp_path).warm_up()
    assert not list(tmp_path.rglob("*.part"))


def test_missing_piper_voice_without_internet_is_a_readable_error(tmp_path, monkeypatch):
    import piper.download_voices as dv

    def offline(*args, **kwargs):
        raise OSError("no internet")

    monkeypatch.setattr(dv, "download_voice", offline)
    with pytest.raises(TTSError, match="could not be fetched"):
        Speaker("en_US-nonexistent-medium", models_dir=tmp_path).warm_up()


# --- real voices ---------------------------------------------------------------------------
def real_speaker(engine_name):
    speaker = Speaker(engine_name=engine_name)
    try:
        speaker.warm_up()
    except TTSError as exc:
        pytest.skip(f"{engine_name} voice unavailable: {exc}")
    list(speaker._engine.synthesize("warm up"))
    return speaker


@pytest.fixture(scope="module", params=["kokoro", "piper"])
def real(request):
    return real_speaker(request.param)


def test_real_voice_renders_a_plausible_command(real, tmp_path):
    began = time.perf_counter()
    seconds = tts.render(real, "Okay, opening Notepad.", tmp_path / "out.wav")
    took = time.perf_counter() - began
    print(f"{real.engine_name}: {seconds:.2f} s of speech rendered in {took:.2f} s (x{took / seconds:.2f})")
    assert 0.8 < seconds < 4.0
    assert took < seconds  # faster than real time, so streaming playback never runs dry


def test_real_voice_is_understood_by_the_real_transcriber(real, tmp_path):
    from core.stt import STTError, Transcriber

    tts.render(real, "Open notepad.", tmp_path / "out.wav")
    stt = Transcriber()
    try:
        stt.warm_up()
    except STTError as exc:
        pytest.skip(f"Whisper unavailable: {exc}")
    with wave.open(str(tmp_path / "out.wav")) as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        rate = wav.getframerate()
    # The voices speak at 22-24 kHz; the transcriber wants 16 kHz (the mic's rate), so resample for this test.
    n_out = int(len(audio) * 16000 / rate)
    audio16 = np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio).astype(np.int16)
    assert "notepad" in stt.transcribe(audio16).text.lower()
