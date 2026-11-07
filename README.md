# Project Zeno

A free, voice-activated desktop assistant for Windows: say "Hey Zeno", and it understands
English, Malayalam and Manglish, acts on your computer through allow-listed tools, learns
from use, and answers by voice. Runs offline on an ordinary laptop with no paid services.

The full design is in [PLAN.md](PLAN.md). Current work is in [TASKS.md](TASKS.md).
Contributor rules are in [CLAUDE.md](CLAUDE.md).

## Quick start

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 --pull   # venv, dependencies, checks, model download
.\.venv\Scripts\python.exe -m pytest                                 # smoke test
.\.venv\Scripts\python.exe main.py --voice console --brain echo     # try the loop with typed input
```

Requires Python 3.11 (64-bit), Git and Ollama. Status: early development, nothing usable yet.
