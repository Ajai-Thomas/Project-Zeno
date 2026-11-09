"""Speech-to-text (task A3). Owned by Voice.

Transcriber turns a recorded command (int16 audio from core.vad, or a WAV file) into an
Utterance using faster-whisper on the CPU with int8 weights. No PyTorch.

    python -m core.stt clip.wav        # transcribe a file; prints text and latency
    python -m core.stt --mic           # record a command from the mic, then transcribe it

The model downloads to data/models/ the first time it is needed, then works offline.
Settings come from the `stt:` section of config.yaml.
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from core import config
from core.audio import SAMPLE_RATE, rms_level
from core.interfaces import Utterance

MODELS_DIR = config.ROOT / "data" / "models"
DEFAULT_PROMPT = "Open Notepad, Spotify, Chrome. Volume to thirty."
SILENT_CLIP_RMS = 0.003  # a whole clip quieter than this is background noise (speech is about 0.01+)
NO_SPEECH_PROB = 0.6  # a segment is dropped if Whisper thinks it is non-speech...
LOW_CONFIDENCE = -1.0  # ...and is also unsure of its own words (average log-probability)


class STTError(RuntimeError):
    """The speech model could not be loaded."""


def load_wav(path: str | Path) -> np.ndarray:
    """Read a 16 kHz mono 16-bit WAV into an int16 array."""
    with wave.open(str(path)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise STTError(f"{path} must be 16 kHz, mono, 16-bit (got {wav.getframerate()} Hz, "
                           f"{wav.getnchannels()} ch, {wav.getsampwidth() * 8}-bit).")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)


class Transcriber:
    """Loads Whisper once and reuses it. `model` can be injected for tests."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        initial_prompt: str | None = None,
        model: Any = None,
    ) -> None:
        settings = config.load().get("stt", {})
        self.model_name = model_name or settings.get("model", "base")
        self.device = settings.get("device", "cpu")
        self.compute_type = settings.get("compute_type", "int8")
        self.cpu_threads = int(settings.get("cpu_threads", 8))  # 8 was ~20% faster than 4 on the i5-1340P
        # A vocabulary hint: it fixed "open not bad" -> "Open Notepad" and "230" -> "thirty" on
        # real recordings (base alone mis-hears an Indian-English accent). Tuned on 4 clips from one
        # speaker, so treat as a starting point; A7 should pass the app names Zeno actually found.
        # "" turns it off. Keep it short: long prompts slow decoding (small + long prompt took 6 s).
        self.initial_prompt = (
            initial_prompt if initial_prompt is not None else settings.get("prompt", DEFAULT_PROMPT)
        ) or None
        self.last_seconds = 0.0  # how long the latest transcribe() took
        self._model = model

    def warm_up(self) -> None:
        """Load the model now (call at startup so the first command isn't slow)."""
        if self._model is None:
            self._model = self._load()

    def _load(self):
        from faster_whisper import WhisperModel

        kwargs = dict(
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=str(MODELS_DIR),
        )
        try:
            return WhisperModel(self.model_name, local_files_only=True, **kwargs)
        except Exception:
            pass  # not downloaded yet
        print(f"[stt] downloading Whisper '{self.model_name}' to {MODELS_DIR} (first run only)...", file=sys.stderr)
        try:
            return WhisperModel(self.model_name, **kwargs)
        except Exception as exc:
            raise STTError(
                f"Whisper '{self.model_name}' is not downloaded and could not be fetched ({exc}). "
                "Connect to the internet once, or check the model name in config.yaml."
            ) from exc

    def transcribe(self, audio: np.ndarray | str | Path, language: str = "en") -> Utterance:
        """Transcribe int16 audio (or a WAV path). Returns an empty Utterance if nothing was said."""
        samples = load_wav(audio) if isinstance(audio, (str, Path)) else np.asarray(audio)
        start = time.perf_counter()
        text = self._transcribe(samples, language)
        self.last_seconds = time.perf_counter() - start
        return Utterance(text, "en")

    def _transcribe(self, samples: np.ndarray, language: str) -> str:
        if samples.size == 0:
            return ""
        if samples.dtype == np.int16:
            if rms_level(samples) < SILENT_CLIP_RMS:
                return ""  # Whisper invents text ("Thank you.") on silence, so don't ask it
            samples = samples.astype(np.float32) / 32768.0
        self.warm_up()
        segments, _info = self._model.transcribe(
            samples.astype(np.float32),
            language=language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,  # core.vad already trimmed the clip
            without_timestamps=True,
            initial_prompt=self.initial_prompt,
        )
        kept = [
            seg.text.strip()
            for seg in segments
            if not (seg.no_speech_prob > NO_SPEECH_PROB and seg.avg_logprob < LOW_CONFIDENCE)
        ]
        return " ".join(t for t in kept if t)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe a WAV file or a mic command")
    parser.add_argument("wav", nargs="?", type=Path, help="16 kHz mono WAV file")
    parser.add_argument("--mic", action="store_true", help="record one command from the mic")
    parser.add_argument("--save", type=Path, help="with --mic: also keep the recording as this WAV (keep it under data/)")
    args = parser.parse_args(argv)
    if not args.wav and not args.mic:
        parser.error("give a WAV file or --mic")

    try:
        stt = Transcriber()
        if args.mic:
            from core.audio import MicStream
            from core.vad import SileroVAD, record_command

            stt.warm_up()
            vad = SileroVAD()
            with MicStream() as mic:
                print("Listening... say a command, then stop talking.")
                rec = record_command(mic.chunks(timeout=2.0), vad)
            if rec.reason == "no_speech":
                print("No speech heard.")
                return 1
            if args.save:
                from core.vad import save_wav

                print(f"Saved to {save_wav(args.save, rec.audio)}")
            audio: Any = rec.audio
        else:
            audio = args.wav
        stt.warm_up()
        utterance = stt.transcribe(audio)
    except Exception as exc:  # STTError, AudioError, VADError, TimeoutError, bad WAV
        print(f"FAIL: {exc}")
        return 1
    print(f"{utterance.text!r}  [{utterance.lang}, {stt.last_seconds:.2f} s]")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
