"""Wake word scoring, recording and evaluation (task A5). Owned by Voice.

Wraps openWakeWord (ONNX, no PyTorch): feed it 80 ms chunks (the same size core.audio produces)
and it returns how likely the wake word was just spoken. A6 builds the live listener on top.

    python -m core.wake --file clip.wav                 # peak score of one clip
    python -m core.wake --record                        # record your own "Hey Zeno" test clips
    python -m core.wake --record-bg 10                  # record 10 minutes of TV/music/room noise
    python -m core.wake --eval                          # detection rate + false accepts on those clips

Your recordings go in data/wakeword/ (git-ignored; never commit voices).
The trained model lives in assets/wakeword/hey_zeno.onnx; see assets/wakeword/README.md.
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from core import config
from core.audio import CHUNK_MS, CHUNK_SAMPLES, SAMPLE_RATE, AudioError, MicStream

ASSET_DIR = config.ROOT / "assets" / "wakeword"
DATA_DIR = config.ROOT / "data" / "wakeword"
DEFAULT_MODEL = "hey_jarvis"  # placeholder until hey_zeno.onnx is trained; set in config.yaml (wake.model)
DEFAULT_THRESHOLD = 0.5
REFRACTORY_S = 2.0  # after a detection, ignore the same utterance for this long
TAIL_S = 1.0  # silence appended to a clip so the model has "heard" the end of the phrase
CLIP_S = 2.5  # length of each recorded test clip
SWEEP = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

POSITIVE_STYLES = ("normal voice", "quietly", "loudly", "fast", "from about a metre away", "slowly")
NEGATIVE_PHRASES = (  # sound-alikes first: these are the ones most likely to fire by mistake
    "Hey Zoe", "He knows", "See no evil", "Hey Jenny", "Say no more", "Zero", "Hey Siri", "Okay Google",
    "Open Notepad", "What time is it", "Hello there", "Hey, how are you", "Volume to thirty",
    "Search for lo-fi music", "I know", "Hey, listen to this", "Thank you very much", "Good morning",
    "Turn it off", "Call my sister",
)


class WakeError(RuntimeError):
    """The wake word model could not be found or loaded."""


# --- model files -------------------------------------------------------------------
def models_dir() -> Path:
    import openwakeword

    return Path(openwakeword.__file__).parent / "resources" / "models"


def _fetch(url: str, folder: Path) -> None:
    from openwakeword.utils import download_file

    folder.mkdir(parents=True, exist_ok=True)
    try:
        download_file(url, str(folder))
    except Exception as exc:
        raise WakeError(f"Could not download {url} ({exc}). Connect to the internet once.") from exc


def ensure_feature_models() -> None:
    """openWakeWord needs two small shared models (about 2.4 MB) that its package does not include."""
    import openwakeword

    folder = models_dir()
    for entry in openwakeword.FEATURE_MODELS.values():
        url = entry["download_url"].replace(".tflite", ".onnx")
        if not (folder / Path(url).name).exists():
            _fetch(url, folder)


def resolve_model(model: str | Path) -> Path:
    """A path to an .onnx file, 'hey_zeno' (assets/wakeword/), or a pre-trained name like 'hey_jarvis'."""
    import openwakeword

    path = Path(model)
    if path.suffix == ".onnx" and path.exists():
        return path
    asset = ASSET_DIR / f"{model}.onnx"
    if asset.exists():
        return asset
    pretrained = openwakeword.MODELS.get(str(model))
    if pretrained:  # downloaded on demand, never committed (these models are CC BY-NC-SA, personal use only)
        url = pretrained["download_url"].replace(".tflite", ".onnx")
        target = models_dir() / Path(url).name
        if not target.exists():
            _fetch(url, models_dir())
        return target
    hint = " Train it first: see assets/wakeword/README.md." if str(model) == "hey_zeno" else ""
    raise WakeError(f"No wake word model '{model}' (looked for {asset}).{hint}")


# --- scoring ---------------------------------------------------------------------------
class WakeModel:
    """Scores 80 ms chunks of 16 kHz int16 audio."""

    def __init__(self, model: str | Path | None = None, threshold: float | None = None) -> None:
        settings = config.load().get("wake", {})
        self.model_name = str(model or settings.get("model", DEFAULT_MODEL))
        self.threshold = float(threshold if threshold is not None else settings.get("threshold", DEFAULT_THRESHOLD))
        path = resolve_model(self.model_name)
        ensure_feature_models()
        from openwakeword.model import Model

        try:
            self._oww = Model(wakeword_models=[str(path)], inference_framework="onnx")
        except Exception as exc:
            raise WakeError(f"Could not load wake word model {path}: {exc}") from exc

    def reset(self) -> None:
        # openWakeWord primes its feature buffer with random noise on every reset, so scores for the same
        # clip differed slightly between runs. Seed it so evaluations are repeatable, then restore the state.
        state = np.random.get_state()
        np.random.seed(0)
        try:
            self._oww.reset()
        finally:
            np.random.set_state(state)

    def score(self, chunk: np.ndarray) -> float:
        """Probability (0 to 1) that the wake word was just spoken, for one 1280-sample chunk."""
        if chunk.shape != (CHUNK_SAMPLES,) or chunk.dtype != np.int16:
            raise ValueError(f"expected {CHUNK_SAMPLES} int16 samples, got {chunk.shape} {chunk.dtype}")
        return float(max(self._oww.predict(chunk).values()))

    def scores(self, audio: np.ndarray) -> np.ndarray:
        """Score every chunk of a clip (resets first, and appends a second of silence)."""
        self.reset()
        padded = np.concatenate((audio.astype(np.int16), np.zeros(int(TAIL_S * SAMPLE_RATE), dtype=np.int16)))
        padded = np.pad(padded, (0, -len(padded) % CHUNK_SAMPLES))
        return np.array([self.score(padded[i : i + CHUNK_SAMPLES]) for i in range(0, len(padded), CHUNK_SAMPLES)])

    def peak(self, audio: np.ndarray) -> float:
        return float(self.scores(audio).max())

    def detected(self, score: float) -> bool:
        return score >= self.threshold


def load_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path)) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise WakeError(f"{path} must be 16 kHz, mono, 16-bit.")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)


# --- evaluation maths (pure, so they are easy to test) --------------------------------------
def count_activations(scores: Sequence[float], threshold: float, refractory_s: float = REFRACTORY_S) -> int:
    """How many separate times the score crosses `threshold` (one phrase = one activation)."""
    refractory = int(refractory_s * 1000 / CHUNK_MS)
    count, wait = 0, 0
    for s in scores:
        if wait > 0:
            wait -= 1
        elif s >= threshold:
            count += 1
            wait = refractory
    return count


@dataclass(frozen=True)
class SweepRow:
    threshold: float
    detection_rate: float  # share of positive clips that fire
    false_accept_clips: int  # negative clips that fire
    false_accepts_per_hour: float | None  # on long background recordings, if any


def sweep(
    pos_peaks: Sequence[float],
    neg_peaks: Sequence[float],
    bg_scores: Sequence[Sequence[float]] = (),
    thresholds: Iterable[float] = SWEEP,
) -> list[SweepRow]:
    hours = sum(len(s) for s in bg_scores) * CHUNK_MS / 3_600_000
    rows = []
    for t in thresholds:
        rate = sum(p >= t for p in pos_peaks) / len(pos_peaks) if pos_peaks else float("nan")
        per_hour = sum(count_activations(s, t) for s in bg_scores) / hours if hours else None
        rows.append(SweepRow(t, rate, sum(n >= t for n in neg_peaks), per_hour))
    return rows


def evaluate(model: WakeModel, folder: Path) -> list[SweepRow]:
    def clips(name: str) -> list[Path]:
        return sorted((folder / name).glob("*.wav"))

    pos, neg, bg = clips("pos"), clips("neg"), clips("bg")
    if not pos:
        raise WakeError(f"No positive clips in {folder / 'pos'}. Record some with: python -m core.wake --record")
    pos_peaks = [model.peak(load_wav(p)) for p in pos]
    neg_peaks = [model.peak(load_wav(p)) for p in neg]
    bg_scores = [model.scores(load_wav(p)) for p in bg]
    print(f"Model {model.model_name}: {len(pos)} positive clips, {len(neg)} negative clips, "
          f"{sum(len(s) for s in bg_scores) * CHUNK_MS / 60000:.1f} min of background audio\n")
    rows = sweep(pos_peaks, neg_peaks, bg_scores)
    print("threshold  detected   false-accept clips   false accepts/hour (background)")
    for r in rows:
        mark = "  <- current" if abs(r.threshold - model.threshold) < 1e-9 else ""
        per_hour = "n/a" if r.false_accepts_per_hour is None else f"{r.false_accepts_per_hour:.1f}"
        print(f"   {r.threshold:.1f}     {r.detection_rate:6.0%}      {r.false_accept_clips:3d} of {len(neg):<3d}          {per_hour}{mark}")
    return rows


# --- recording your own test clips --------------------------------------------------------------
def _save(path: Path, audio: np.ndarray) -> None:
    from core.vad import save_wav

    save_wav(path, audio)


def record_clips(
    folder: Path,
    count: int = 20,
    *,
    mic_factory: Callable[[], MicStream] = MicStream,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> None:
    """Guided recording: `count` "Hey Zeno" clips (pos/) and `count` other phrases (neg/)."""
    plan = [("pos", "Say: 'Hey Zeno'", POSITIVE_STYLES[i % len(POSITIVE_STYLES)]) for i in range(count)]
    plan += [("neg", f"Say: '{NEGATIVE_PHRASES[i % len(NEGATIVE_PHRASES)]}'", "normal voice") for i in range(count)]
    n_chunks = int(CLIP_S * 1000 / CHUNK_MS)
    done = {"pos": 0, "neg": 0}
    with mic_factory() as mic:
        for kind, prompt, style in plan:
            ask(f"[{kind} {done[kind] + 1}/{count}] {prompt}  ({style})  -- press Enter, then speak ")
            mic.drain()
            audio = np.concatenate([mic.read(timeout=2.0) for _ in range(n_chunks)])
            done[kind] += 1
            _save(folder / kind / f"{kind}_{done[kind]:02d}.wav", audio)
    say(f"Saved {done['pos']} positive and {done['neg']} negative clips in {folder}")


def record_background(folder: Path, minutes: float, *, mic_factory: Callable[[], MicStream] = MicStream) -> Path:
    n_chunks = int(minutes * 60_000 / CHUNK_MS)
    chunks = []
    with mic_factory() as mic:
        for i in range(n_chunks):
            chunks.append(mic.read(timeout=2.0))
            if (i + 1) % (60_000 // CHUNK_MS) == 0:
                print(f"  {(i + 1) * CHUNK_MS // 60_000} of {minutes:g} min recorded", flush=True)
    path = folder / "bg" / f"bg_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    _save(path, np.concatenate(chunks))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wake word tools")
    parser.add_argument("--file", type=Path, help="print the peak score of a 16 kHz mono WAV")
    parser.add_argument("--record", action="store_true", help="record pos/ and neg/ test clips from the mic")
    parser.add_argument("--record-bg", type=float, metavar="MINUTES", help="record background audio (play TV or music)")
    parser.add_argument("--eval", action="store_true", help="score the recorded clips and sweep the threshold")
    parser.add_argument("--dir", type=Path, default=DATA_DIR, help="where test clips live (default data/wakeword)")
    parser.add_argument("--model", help="model name or .onnx path (default: wake.model in config.yaml)")
    parser.add_argument("--threshold", type=float, help="detection threshold (default: wake.threshold in config.yaml)")
    parser.add_argument("--count", type=int, default=20, help="clips per kind for --record")
    args = parser.parse_args(argv)
    if not (args.file or args.record or args.record_bg or args.eval):
        parser.error("choose --file, --record, --record-bg or --eval")

    try:
        if args.record:
            record_clips(args.dir, args.count)
        if args.record_bg:
            print(f"Recording {args.record_bg:g} minutes. Play the TV or music you normally have on, and talk now and then.")
            print(f"Saved {record_background(args.dir, args.record_bg)}")
        if args.file or args.eval:
            model = WakeModel(args.model, args.threshold)
            if args.file:
                peak = model.peak(load_wav(args.file))
                print(f"{args.file.name}: peak score {peak:.3f} -> {'DETECTED' if model.detected(peak) else 'not detected'}"
                      f" (threshold {model.threshold:g}, model {model.model_name})")
            if args.eval:
                evaluate(model, args.dir)
    except (WakeError, AudioError, TimeoutError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
