"""Voice activity detection and command recording (task A2). Owned by Voice.

SileroVAD scores audio for speech. record_command() uses it to capture one spoken command:
wait for speech, record, and stop after about 0.8 s of silence (or 10 s at most).

The Silero ONNX model is the copy that faster-whisper already ships, run directly through
onnxruntime, so there is no PyTorch and no extra download.

    python -m core.vad                 # speak a command; saves data/last.wav
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from core.audio import CHUNK_MS, SAMPLE_RATE

WINDOW = 512  # samples the model scores at a time (32 ms at 16 kHz)
CONTEXT = 64  # samples of the previous window prepended to each one

START_THRESHOLD = 0.5  # speech starts at or above this probability...
END_THRESHOLD = 0.35  # ...and counts as silence only below this (hysteresis)
SILENCE_MS = 800
PREROLL_MS = 320  # audio kept from before speech started, so the first word isn't clipped
START_TIMEOUT_S = 5.0
MAX_S = 10.0

ROOT = Path(__file__).resolve().parent.parent


class VADError(RuntimeError):
    """The VAD model could not be found or loaded."""


def find_model() -> Path:
    """Locate the Silero ONNX file bundled with faster-whisper (without importing it)."""
    spec = importlib.util.find_spec("faster_whisper")
    if spec is None or not spec.submodule_search_locations:
        raise VADError("faster-whisper is not installed. Run scripts\\setup.ps1.")
    assets = Path(next(iter(spec.submodule_search_locations))) / "assets"
    found = sorted(assets.glob("silero_vad*.onnx"))
    if not found:
        raise VADError(f"No silero_vad*.onnx in {assets}; the faster-whisper version may have moved it.")
    return found[-1]


class SileroVAD:
    """Streaming speech probability. Feed it audio in any chunk size."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        try:
            self._session = ort.InferenceSession(
                str(model_path or find_model()), providers=["CPUExecutionProvider"], sess_options=opts
            )
        except Exception as exc:
            raise VADError(f"Could not load the VAD model: {exc}") from exc
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._leftover = np.empty(0, dtype=np.float32)
        self._last = 0.0

    def feed(self, chunk: np.ndarray) -> float:
        """Add int16 audio; return the highest speech probability among the windows it completed."""
        buf = np.concatenate((self._leftover, chunk.astype(np.float32) / 32768.0))
        probs: list[float] = []
        n_windows = len(buf) // WINDOW
        for i in range(n_windows):
            window = buf[i * WINDOW : (i + 1) * WINDOW]
            model_input = np.concatenate((self._context, window))[None, :]
            out, self._h, self._c = self._session.run(
                None, {"input": model_input, "h": self._h, "c": self._c}
            )
            probs.append(float(np.asarray(out).reshape(-1)[0]))
            self._context = window[-CONTEXT:]
        self._leftover = buf[n_windows * WINDOW :]
        if probs:
            self._last = max(probs)
        return self._last


@dataclass(frozen=True)
class Recording:
    audio: np.ndarray  # int16, mono, 16 kHz
    reason: str  # "silence", "max_length", "no_speech" or "stream_ended"

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def _ms(chunk: np.ndarray) -> float:
    return len(chunk) * 1000 / SAMPLE_RATE


def record_command(
    chunks: Iterable[np.ndarray],
    vad: SileroVAD | None = None,
    *,
    start_threshold: float = START_THRESHOLD,
    end_threshold: float = END_THRESHOLD,
    silence_ms: float = SILENCE_MS,
    preroll_ms: float = PREROLL_MS,
    start_timeout_s: float = START_TIMEOUT_S,
    max_s: float = MAX_S,
) -> Recording:
    """Record one spoken command from an iterable of 80 ms int16 chunks (e.g. MicStream.chunks())."""
    vad = vad or SileroVAD()
    vad.reset()
    preroll: deque[np.ndarray] = deque(maxlen=max(1, math.ceil(preroll_ms / CHUNK_MS)))
    recorded: list[np.ndarray] = []
    speaking = False
    waited_ms = silent_ms = recorded_ms = 0.0

    for chunk in chunks:
        prob = vad.feed(chunk)
        if not speaking:
            if prob >= start_threshold:
                speaking = True
                recorded = [*preroll, chunk]
                recorded_ms = sum(_ms(c) for c in recorded)
                silent_ms = 0.0
            else:
                preroll.append(chunk)
                waited_ms += _ms(chunk)
                if waited_ms >= start_timeout_s * 1000:
                    return Recording(np.empty(0, dtype=np.int16), "no_speech")
            continue

        recorded.append(chunk)
        recorded_ms += _ms(chunk)
        silent_ms = silent_ms + _ms(chunk) if prob < end_threshold else 0.0
        if silent_ms >= silence_ms:
            return Recording(np.concatenate(recorded), "silence")
        if recorded_ms >= max_s * 1000:
            return Recording(np.concatenate(recorded), "max_length")

    if speaking:
        return Recording(np.concatenate(recorded), "stream_ended")
    return Recording(np.empty(0, dtype=np.int16), "no_speech")


def save_wav(path: str | Path, audio: np.ndarray) -> Path:
    """Write int16 mono 16 kHz audio as a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio.astype(np.int16).tobytes())
    return path


def main(argv: list[str] | None = None) -> int:
    from core.audio import AudioError, MicStream

    parser = argparse.ArgumentParser(description="Record one spoken command")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "last.wav")
    parser.add_argument("--device", default=None, help="device index or name part")
    args = parser.parse_args(argv)
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    try:
        vad = SileroVAD()  # load before the mic opens so no audio piles up meanwhile
        with MicStream(device) as mic:
            print("Listening... say a command, then stop talking.")
            rec = record_command(mic.chunks(timeout=2.0), vad)
    except (AudioError, VADError, TimeoutError) as exc:
        print(f"FAIL: {exc}")
        return 1

    if rec.reason == "no_speech":
        print(f"No speech heard within {START_TIMEOUT_S:g} s.")
        return 1
    saved = save_wav(args.out, rec.audio)
    print(f"Recorded {rec.duration_s:.2f} s (stopped on: {rec.reason}) -> {saved}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
