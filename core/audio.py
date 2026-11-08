"""Microphone capture (task A1). Owned by Voice.

Opens the mic as 16 kHz, mono, int16 and hands out exact 80 ms chunks (1280 samples).
These are the units that the wake word (A5/A6), VAD (A2) and Whisper (A3) all expect.

    python -m core.audio --check       # record 3 s, show a live level meter, print PASS/FAIL
    python -m core.audio --list        # list input devices
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
from types import TracebackType
from typing import Iterator

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_MS = 80
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000  # 1280
DTYPE = "int16"
QUEUE_CHUNKS = 50  # about 4 s of audio buffered before the oldest chunk is dropped
SPEECH_PEAK = 0.01  # --check passes only if the loudest chunk is above this (room noise is about 0.002)


class AudioError(RuntimeError):
    """The microphone could not be opened or used."""


def _sd():
    """The only place sounddevice is imported, so tests can replace it."""
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise AudioError(f"sounddevice is not available ({exc}). Run scripts\\setup.ps1.") from exc
    return sounddevice


def list_input_devices() -> list[tuple[int, str]]:
    """(index, name) of every device that can record."""
    sd = _sd()
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def rms_level(chunk: np.ndarray) -> float:
    """Loudness of a chunk from 0.0 (silence) to about 1.0 (full scale)."""
    if chunk.size == 0:
        return 0.0
    x = chunk.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


class MicStream:
    """Context manager that yields 80 ms int16 chunks from the microphone.

    with MicStream() as mic:
        for chunk in mic.chunks():   # np.ndarray, shape (1280,), dtype int16
            ...
    """

    def __init__(self, device: int | str | None = None, queue_chunks: int = QUEUE_CHUNKS) -> None:
        self.device = device
        self.dropped = 0  # chunks discarded because the consumer fell behind
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_chunks)
        self._pending = np.empty(0, dtype=np.int16)
        self._lock = threading.Lock()
        self._stream = None

    # --- audio thread -----------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:
        """Runs on PortAudio's thread: copy, slice into exact chunks, never block."""
        block = np.asarray(indata, dtype=np.int16).reshape(-1).copy()
        with self._lock:
            self._pending = np.concatenate((self._pending, block))
            while len(self._pending) >= CHUNK_SAMPLES:
                chunk, self._pending = self._pending[:CHUNK_SAMPLES], self._pending[CHUNK_SAMPLES:]
                self._put(chunk)

    def _put(self, item: np.ndarray | None) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:  # drop the oldest chunk so the newest audio always gets in
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    # --- control ------------------------------------------------------------
    def start(self) -> "MicStream":
        sd = _sd()
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=DTYPE,
                blocksize=CHUNK_SAMPLES,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioError(
                f"Could not open the microphone ({self.device if self.device is not None else 'default'}) "
                f"at 16 kHz mono: {exc}. Check that a mic is connected and that Windows Settings > "
                "Privacy > Microphone lets desktop apps use it."
            ) from exc
        return self

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
            self._put(None)  # tells chunks() the stream has ended

    def __enter__(self) -> "MicStream":
        return self.start()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.stop()

    def device_name(self) -> str:
        sd = _sd()
        info = sd.query_devices(self.device, "input") if self.device is not None else sd.query_devices(kind="input")
        return str(info["name"])

    # --- consumer side --------------------------------------------------------
    def _get(self, timeout: float) -> np.ndarray | None:
        """Next chunk, or None once the stream has stopped. Raises TimeoutError if nothing arrives."""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no audio received for {timeout} s; is the mic muted or unplugged?") from None
        if item is None:
            self._put(None)  # keep the end marker for any later reader
        return item

    def read(self, timeout: float = 1.0) -> np.ndarray:
        """Block for the next 80 ms chunk."""
        chunk = self._get(timeout)
        if chunk is None:
            raise AudioError("the microphone stream is closed")
        return chunk

    def chunks(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        """Yield chunks until the stream is stopped."""
        while (chunk := self._get(timeout)) is not None:
            yield chunk

    def drain(self) -> None:
        """Throw away buffered audio (used after Zeno speaks, so it doesn't hear itself)."""
        with self._lock:
            self._pending = np.empty(0, dtype=np.int16)
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._queue.put_nowait(None)
                    break


# --- standalone check ---------------------------------------------------------
def check(seconds: float = 3.0, device: int | str | None = None) -> int:
    """Record for `seconds`, show a live meter, return 0 on PASS or 1 on FAIL."""
    total = int(seconds * 1000 / CHUNK_MS)
    peak = 0.0
    try:
        with MicStream(device) as mic:
            print(f"Microphone: {mic.device_name()}")
            print(f"Speak now for {seconds:g} s...")
            for _ in range(total):
                level = rms_level(mic.read(timeout=2.0))
                peak = max(peak, level)
                filled = int(min(1.0, level * 6) * 40)
                print(f"\r[{'#' * filled}{'.' * (40 - filled)}] {level:.3f}", end="", flush=True)
            dropped = mic.dropped
    except (AudioError, TimeoutError) as exc:
        print(f"\nFAIL: {exc}")
        return 1
    print(f"\nPeak level {peak:.3f}, dropped chunks {dropped}")
    if peak < SPEECH_PEAK:
        print(
            "FAIL: only background noise heard. Speak while it records; if the meter still stays flat, "
            "the mic may be muted or another device is needed (try --list and --device N)."
        )
        return 1
    print("PASS")
    return 0


def _device_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # device names can contain odd characters
    parser = argparse.ArgumentParser(description="Zeno microphone check")
    parser.add_argument("--check", action="store_true", help="record and show a level meter (default)")
    parser.add_argument("--list", action="store_true", help="list input devices")
    parser.add_argument("--device", type=_device_arg, default=None, help="device index or name part")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args(argv)

    if args.list:
        try:
            for index, name in list_input_devices():
                print(f"{index:3d}  {name}")
        except AudioError as exc:
            print(f"FAIL: {exc}")
            return 1
        return 0
    return check(args.seconds, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
