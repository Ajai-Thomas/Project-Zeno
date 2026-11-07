"""Zeno entry point: listen -> think -> speak. SHARED FILE.

    python main.py                                  # real parts where they exist, stubs otherwise
    python main.py --voice console --brain echo     # both stubs
    python main.py --voice console --brain real     # Person B: type commands to the real Brain
    python main.py --voice real --brain echo        # Person A: real mic/speaker, echo brain
"""
from __future__ import annotations

import argparse
import sys

from core import events
from core.interfaces import Brain, VoiceIO, status, subscribe_status
from core.wiring import build_brain, build_voice


def run(voice: VoiceIO, brain: Brain, max_turns: int | None = None) -> int:
    """Run the loop until listen() returns None. Returns the number of turns handled."""
    turns = 0
    while max_turns is None or turns < max_turns:
        utterance = voice.listen()
        if utterance is None:
            break
        status(events.THINKING)
        try:
            reply = brain.think(utterance)
        except Exception as exc:  # a brain bug must not kill the loop
            status(events.ERROR)
            print(f"[main] brain error: {exc!r}", file=sys.stderr)
            reply = "Sorry, something went wrong."
        voice.speak(reply, utterance.lang)
        status(events.IDLE)
        turns += 1
    return turns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zeno voice assistant")
    parser.add_argument("--voice", choices=["real", "console"], default="real")
    parser.add_argument("--brain", choices=["real", "echo"], default="real")
    parser.add_argument("--verbose", action="store_true", help="print status events")
    args = parser.parse_args(argv)

    if args.verbose:
        subscribe_status(lambda event: print(f"[status] {event}", file=sys.stderr))

    voice, brain = build_voice(args.voice), build_brain(args.brain)
    try:
        run(voice, brain)
    except KeyboardInterrupt:
        voice.stop_speaking()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
