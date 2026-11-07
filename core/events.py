"""Names of status events. SHARED FILE: Brain sends them, Voice's overlay shows them."""

IDLE = "idle"
LISTENING = "listening"
TRANSCRIBING = "transcribing"
THINKING = "thinking"
SPEAKING = "speaking"
ERROR = "error"


def acting(skill: str) -> str:
    """Event for a running skill, e.g. acting("open_app") -> "acting:open_app"."""
    return f"acting:{skill}"
