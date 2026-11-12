"""Checks that this machine is ready to build Zeno. Standard library only, so it runs anywhere.

    python scripts/check_setup.py            # report
    python scripts/check_setup.py --pull     # also download the Qwen3 model if missing
Exit code is 1 if any required check fails.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"  # keep in sync with config.yaml
OLLAMA_URL = "http://localhost:11434"
OK, WARN, FAIL = "OK  ", "WARN", "FAIL"

results: list[tuple[str, str, str]] = []


def report(level: str, name: str, detail: str) -> None:
    results.append((level, name, detail))
    print(f"[{level}] {name}: {detail}")


def check_python() -> None:
    v = sys.version_info
    if (v.major, v.minor) != (3, 11):
        report(FAIL, "Python", f"{v.major}.{v.minor} found; need 3.11. Run scripts\\setup.ps1 (uses `py -3.11`).")
    elif struct.calcsize("P") != 8:
        report(FAIL, "Python", "3.11 is 32-bit; install the 64-bit build.")
    else:
        report(OK, "Python", f"{sys.version.split()[0]} (64-bit)")
    if sys.prefix == sys.base_prefix:
        report(WARN, "Virtual env", "not active; run scripts\\setup.ps1 and use .venv\\Scripts\\python.")
    else:
        report(OK, "Virtual env", sys.prefix)


def _ollama_models() -> list[str] | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]
    except Exception:
        return None


def check_ollama(model: str, pull: bool) -> None:
    exe = shutil.which("ollama")
    if not exe:  # a shell opened before the install has a stale PATH
        default = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        exe = str(default) if default.exists() else None
    if not exe:
        report(FAIL, "Ollama", "not installed. Run: winget install Ollama.Ollama")
        return
    report(OK, "Ollama", exe)
    models = _ollama_models()
    if models is None:
        report(FAIL, "Ollama server", f"not responding at {OLLAMA_URL}. Start the Ollama app.")
        return
    report(OK, "Ollama server", f"{len(models)} model(s) installed")
    if model in models:
        report(OK, "Model", model)
    elif pull:
        print(f"      pulling {model} (about 2.5 GB)...")
        code = subprocess.call([exe, "pull", model])
        report(OK if code == 0 else FAIL, "Model", f"pull {'finished' if code == 0 else 'failed'}: {model}")
    else:
        report(FAIL, "Model", f"{model} missing. Run: ollama pull {model}  (or use --pull)")


def check_mic() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        report(WARN, "Microphone", "sounddevice not installed yet; run scripts\\setup.ps1 first.")
        return
    try:
        inputs = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    except Exception as exc:
        report(FAIL, "Microphone", f"could not list audio devices: {exc}")
        return
    if inputs:
        report(OK, "Microphone", f"{len(inputs)} input device(s); default: {sd.query_devices(kind='input')['name']}")
    else:
        report(FAIL, "Microphone", "no input device found. Plug in a mic or headset and check Windows privacy settings.")


def check_packages() -> None:
    if importlib.util.find_spec("torch") is not None:
        report(FAIL, "No PyTorch", "torch is installed. Remove it: pip uninstall torch (see CLAUDE.md).")
    else:
        report(OK, "No PyTorch", "torch not installed")
    wanted = ["numpy", "sounddevice", "onnxruntime", "faster_whisper", "openwakeword", "piper", "kokoro_onnx", "openai", "yaml", "pytest"]
    missing = [m for m in wanted if importlib.util.find_spec(m) is None]
    if missing:
        report(WARN, "Packages", "missing: " + ", ".join(missing) + ". Run scripts\\setup.ps1.")
    else:
        report(OK, "Packages", "all Phase 1 packages importable")


def check_env() -> None:
    if (ROOT / ".env").exists():
        report(OK, ".env", "found")
    else:
        report(WARN, ".env", "missing; optional (only for the free cloud helper). Copy .env.example to .env.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pull", action="store_true", help="download the Qwen3 model if missing")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    check_python()
    check_ollama(args.model, args.pull)
    check_mic()
    check_packages()
    check_env()

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print(f"\n{len(results) - len(fails) - len(warns)} ok, {len(warns)} warning(s), {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
