"""Loads config.yaml and .env. SHARED FILE.

API keys are never in config.yaml; they come from environment variables (.env).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path | None = None) -> dict[str, Any]:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # python-dotenv is optional for the stubs
        pass
    try:
        import yaml
    except ImportError:
        return {}
    cfg_path = path or ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def api_key(name: str) -> str | None:
    """Read a key such as GEMINI_API_KEY from the environment."""
    return os.environ.get(name) or None
