"""Text-to-speech (task A4). Owned by Voice.

Speaker turns text into speech and plays it sentence by sentence: the first sentence starts
playing while the next is still being generated, and stop_speaking() cuts the audio off within
about 100 ms from any thread. No PyTorch.

Two engines sit behind the same interface:
  * Kokoro (English, default): natural voice, 24 kHz, ONNX. Voice from config `tts.kokoro_voice`.
  * Piper (fallback, and Malayalam in Phase 3): lighter and faster, 22 kHz, ONNX.

    python -m core.tts "hello, I am Zeno"
    python -m core.tts "a long sentence ..." --interrupt-after 1.5    # demo the cut-off
    python -m core.tts "hello" --voice en_US-hfc_female-medium         # a Piper voice
    python -m core.tts "hello" --save out.wav                          # render without speakers

Models download to data/models/ the first time (Kokoro about 115 MB), then work offline.
"""
from __future__ import annotations

import argparse
import queue
import re
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from core import config

MODELS_DIR = config.ROOT / "data" / "models"
DEFAULT_ENGINE = "kokoro"
DEFAULT_KOKORO_VOICE = "af_bella"
DEFAULT_PIPER_VOICE = "en_US-hfc_female-medium"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")  # about 88 MB + 27 MB
PIPER_VOICE_PATTERN = re.compile(r"^[a-z]{2}_[A-Z]{2}-")  # e.g. en_US-lessac-medium

BLOCK_MS = 50  # audio is written in blocks this long; stop is checked between blocks
QUEUE_SENTENCES = 2  # how many synthesized sentences may wait ahead of playback
GAIN = 0.8  # voices peak near full scale, which clips on laptop speakers and earbuds (heard as static)
LEAD_IN_MS = 250  # silence played first so Bluetooth/idle outputs wake up before the words start


class TTSError(RuntimeError):
    """The voice or the speakers could not be used."""


def _sd():
    """The only place sounddevice is imported, so tests can replace it."""
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise TTSError(f"sounddevice is not available ({exc}). Run scripts\\setup.ps1.") from exc
    return sounddevice


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


# --- engines: sample_rate + synthesize(text) -> int16 mono arrays, one per sentence ---------------
class PiperEngine:
    def __init__(self, voice_name: str, models_dir: Path) -> None:
        from piper import PiperVoice

        folder = models_dir / "piper"
        model = folder / f"{voice_name}.onnx"
        if not model.exists():
            print(f"[tts] downloading Piper voice '{voice_name}' to {folder} (first run only)...", file=sys.stderr)
            try:
                from piper.download_voices import download_voice

                folder.mkdir(parents=True, exist_ok=True)
                download_voice(voice_name, folder)
            except Exception as exc:
                raise TTSError(
                    f"Piper voice '{voice_name}' is not downloaded and could not be fetched ({exc}). "
                    "Connect to the internet once, or check tts.voice_en in config.yaml."
                ) from exc
        try:
            self._voice = PiperVoice.load(model)
            list(self._voice.synthesize("Ready."))  # the first synthesis is slow; pay for it now
        except Exception as exc:
            raise TTSError(f"Could not load Piper voice {model}: {exc}") from exc
        self.sample_rate: int = self._voice.config.sample_rate

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_array


class KokoroEngine:
    def __init__(self, voice: str, models_dir: Path) -> None:
        folder = models_dir / "kokoro"
        model, voices = (folder / name for name in KOKORO_FILES)
        for path in (model, voices):
            if not path.exists():
                self._download(path)
        try:
            from kokoro_onnx import Kokoro

            self._kokoro = Kokoro(str(model), str(voices))
            self._voice = voice
            self._kokoro.create("Ready.", voice=voice, lang="en-us")  # warm-up: the first call is slow
        except Exception as exc:
            raise TTSError(f"Could not load Kokoro voice '{voice}': {exc}") from exc
        self.sample_rate = 24_000

    @staticmethod
    def _download(path: Path) -> None:
        print(f"[tts] downloading {path.name} to {path.parent} (first run only, about 115 MB in total)...", file=sys.stderr)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(KOKORO_URL + path.name, part)
            part.replace(path)
        except Exception as exc:
            raise TTSError(
                f"Kokoro file {path.name} is not downloaded and could not be fetched ({exc}). "
                "Connect to the internet once, or set tts.engine_en to piper in config.yaml."
            ) from exc

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        for sentence in split_sentences(text):
            samples, _rate = self._kokoro.create(sentence, voice=self._voice, speed=1.0, lang="en-us")
            yield (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)


class Speaker:
    """Speaks with one engine. `engine` can be injected for tests."""

    def __init__(
        self,
        voice_name: str | None = None,
        *,
        engine_name: str | None = None,
        engine: Any = None,
        models_dir: Path | None = None,
        device: int | str | None = None,
        lead_in_ms: int | None = None,
    ) -> None:
        settings = config.load().get("tts", {})
        if voice_name and PIPER_VOICE_PATTERN.match(voice_name):
            engine_name = engine_name or "piper"  # the name looks like a Piper voice
        self.engine_name = engine_name or settings.get("engine_en", DEFAULT_ENGINE)
        if self.engine_name == "kokoro":
            self.voice_name = voice_name or settings.get("kokoro_voice", DEFAULT_KOKORO_VOICE)
        elif self.engine_name == "piper":
            self.voice_name = voice_name or settings.get("voice_en", DEFAULT_PIPER_VOICE)
        else:
            raise TTSError(f"Unknown TTS engine '{self.engine_name}' (use 'kokoro' or 'piper').")
        self.models_dir = models_dir or MODELS_DIR
        self.device = device if device is not None else settings.get("device")  # output device; None = system default
        self.lead_in_ms = LEAD_IN_MS if lead_in_ms is None else lead_in_ms
        self.first_audio_s = 0.0  # time from speak() to the first sound, for the latency budget
        self.audible = threading.Event()  # set when the first word starts playing; cleared by the next speak()
        self._engine = engine
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None  # the running speak() call's stop flag
        self._stream = None
        self._warned_lang = False

    # --- loading ---------------------------------------------------------------
    def warm_up(self) -> None:
        """Load the voice now (call at startup so the first reply isn't slow)."""
        if self._engine is None:
            cls = KokoroEngine if self.engine_name == "kokoro" else PiperEngine
            self._engine = cls(self.voice_name, self.models_dir)

    # --- speaking --------------------------------------------------------------------
    @property
    def is_speaking(self) -> bool:
        return self._stop is not None

    def speak(self, text: str, lang: str = "en") -> None:
        """Say `text` aloud. Blocks until finished or stop_speaking() is called."""
        text = text.strip()
        if not text:
            return
        if lang != "en" and not self._warned_lang:
            self._warned_lang = True
            print(f"[tts] no '{lang}' voice yet (Phase 3); using the English voice.", file=sys.stderr)
        self.warm_up()

        stop = threading.Event()  # one per call, so a stale worker can never be revived by the next call
        self.audible.clear()
        with self._lock:
            self._stop = stop  # from here on, stop_speaking() works
        started = time.perf_counter()
        sentences: queue.Queue = queue.Queue(maxsize=QUEUE_SENTENCES)
        threading.Thread(target=self._synthesize, args=(text, sentences, stop), daemon=True).start()

        stream = None
        interrupted = False
        try:
            rate = self._engine.sample_rate
            block = rate * BLOCK_MS // 1000
            try:
                stream = _sd().OutputStream(samplerate=rate, channels=1, dtype="int16", device=self.device)
                stream.start()
            except Exception as exc:
                raise TTSError(f"Could not open the speakers: {exc}. Check the Windows output device.") from exc
            with self._lock:
                self._stream = stream

            # Silence first: an idle Bluetooth/HDMI output can take a few hundred ms to wake and would eat the first word.
            silence = np.zeros((rate * self.lead_in_ms // 1000, 1), dtype=np.int16)
            for i in range(0, len(silence), block):
                if stop.is_set():
                    return
                stream.write(silence[i : i + block])

            first = True
            while True:
                item = self._next(sentences, stop)
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise TTSError(f"Speech synthesis failed: {item}") from item
                for i in range(0, len(item), block):
                    if stop.is_set():
                        return
                    stream.write(item[i : i + block].reshape(-1, 1))
                    if first:
                        self.first_audio_s, first = time.perf_counter() - started, False
                        self.audible.set()
        except TTSError:
            raise
        except Exception:
            if not stop.is_set():  # a write error is expected when stop_speaking() aborted the stream
                raise
        finally:
            interrupted = stop.is_set()
            stop.set()  # lets the worker thread finish
            with self._lock:
                self._stop = self._stream = None
            if stream is not None:
                for action in (stream.abort if interrupted else stream.stop, stream.close):
                    try:
                        action()
                    except Exception:
                        pass

    def stop_speaking(self) -> None:
        """Cut speech off now. Safe from any thread, and harmless when nothing is playing."""
        with self._lock:
            if self._stop is None:
                return
            self._stop.set()
            if self._stream is not None:
                try:
                    self._stream.abort()  # drops the audio already queued on the device
                except Exception:
                    pass

    # --- internals ----------------------------------------------------------------------
    def _synthesize(self, text: str, out: queue.Queue, stop: threading.Event) -> None:
        """Worker thread: one int16 array per sentence, then None (or an exception)."""
        try:
            for audio in self._engine.synthesize(text):
                audio = (audio.astype(np.float32) * GAIN).astype(np.int16)
                if not self._put(out, audio, stop):
                    return
        except BaseException as exc:  # reported to the playing thread
            self._put(out, exc, stop)
            return
        self._put(out, None, stop)

    @staticmethod
    def _put(out: queue.Queue, item, stop: threading.Event) -> bool:
        while not stop.is_set():
            try:
                out.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _next(sentences: queue.Queue, stop: threading.Event):
        while True:
            try:
                return sentences.get(timeout=0.05)
            except queue.Empty:
                if stop.is_set():
                    return None


def render(speaker: Speaker, text: str, path: Path) -> float:
    """Synthesize `text` into a WAV file without playing it. Returns the seconds of audio."""
    speaker.warm_up()
    audio = np.concatenate(list(speaker._engine.synthesize(text)))
    rate = speaker._engine.sample_rate
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(audio.tobytes())
    return len(audio) / rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Speak text with Zeno's voice")
    parser.add_argument("text", nargs="?")
    parser.add_argument("--interrupt-after", type=float, metavar="SECONDS", help="stop this long after the first sound")
    parser.add_argument("--save", type=Path, help="write a WAV file instead of playing")
    parser.add_argument("--engine", choices=["kokoro", "piper"], help="default: tts.engine_en in config.yaml")
    parser.add_argument("--voice", help="Kokoro voice (af_sarah) or Piper voice (en_US-hfc_female-medium)")
    parser.add_argument("--device", help="output device index or name part (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="list output devices")
    args = parser.parse_args(argv)

    if args.list_devices:
        try:
            for i, d in enumerate(_sd().query_devices()):
                if d["max_output_channels"] > 0:
                    print(f"{i:3d}  {d['name']}")
        except TTSError as exc:
            print(f"FAIL: {exc}")
            return 1
        return 0
    if not args.text:
        parser.error("give some text to speak")

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    try:
        speaker = Speaker(args.voice, engine_name=args.engine, device=device)
        print(f"Loading {speaker.engine_name} voice '{speaker.voice_name}' (takes a few seconds)...")
        if args.save:
            seconds = render(speaker, args.text, args.save)
            print(f"Saved {seconds:.2f} s of speech to {args.save}")
            return 0
        speaker.warm_up()
        if args.interrupt_after is not None:  # counted from the first audible sound, not from the start

            def cut_off() -> None:
                if speaker.audible.wait(timeout=30):
                    time.sleep(args.interrupt_after)
                    speaker.stop_speaking()

            threading.Thread(target=cut_off, daemon=True).start()
        begin = time.perf_counter()
        speaker.speak(args.text)
        print(f"Done in {time.perf_counter() - begin:.2f} s (first sound after {speaker.first_audio_s:.2f} s)")
    except TTSError as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
