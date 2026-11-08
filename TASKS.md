# TASKS.md — Phase 1: core loop

Goal of the phase: say "Hey Zeno, open Notepad" and it happens, with a spoken reply.
One task = one pull request. Tasks are in build order within each person's column;
A and B work in parallel. Tick the box in the PR that finishes a task.

Run commands from the repo root with the `.venv` active.

## Shared (done)

- [x] **S0 · Base.** Folders, `core/interfaces.py`, stubs, `main.py`, `CLAUDE.md`, requirements, setup scripts, smoke test.
      Test: `pytest`

## Person A · Voice (Ajai-Thomas)

- [x] **A1 · Mic capture.** `core/audio.py`: 16 kHz mono stream in 80 ms chunks (`sounddevice`).
      Test: `python -m core.audio --check` records 3 s and shows a live level meter.
- [x] **A2 · Command recorder.** `core/vad.py`: Silero VAD (ONNX file loaded with `onnxruntime`, not the torch package); record until 0.8 s of silence, cap 10 s.
      Test: `python -m core.vad` saves `data/last.wav` and prints its length; unit test on a synthetic clip.
- [ ] **A3 · Speech-to-text.** `core/stt.py`: faster-whisper `base`, int8, CPU; WAV in, `Utterance` out; models download to `data/models/`.
      Test: `python -m core.stt clip.wav` prints text and latency; unit test on a TTS-generated fixture.
- [ ] **A4 · Text-to-speech.** `core/tts.py`: Piper `en_US-lessac`; `speak()` plus `stop_speaking()` that cuts audio mid-sentence.
      Test: `python -m core.tts "hello"`; a second thread calling `stop_speaking()` ends playback in under 200 ms.
- [ ] **A5 · Train the "Hey Zeno" wake word.** Train a custom openWakeWord model in the project's training notebook (off-repo, e.g. Colab; PyTorch is used there only, never here). Commit only the small `assets/wakeword/hey_zeno.onnx`. Record real clips of yourself and your partner for testing; keep them out of git.
      Test: `python -m core.wake --file clip.wav` prints the score; at least 90% of "Hey Zeno" clips trigger, none of 20 other-phrase clips do.
- [ ] **A6 · Wake word listener.** `core/wake.py`: runs on the A1 stream, config threshold (0.5), plays a chime on a hit. Uses the pre-trained `hey_jarvis` model (set in `config.yaml`) until A5 delivers; then switch the config to `hey_zeno`.
      Test: `python -m core.wake` prints "hit" when you say the wake word; count false triggers over 10 minutes of TV or music (goal: under 1 per hour).
- [ ] **A7 · Real `listen()`.** `core/voice.py` `RealVoice`: wake, chime, record, transcribe, plus `speak()`/`stop_speaking()`; drain the mic buffer after speaking so Zeno doesn't hear itself.
      Test: `python main.py --voice real --brain echo` repeats whatever you say.

## Person B · Brain (not yet assigned)

- [ ] **B1 · Skill registry.** `skills/registry.py`: `@skill` decorator with name, description, typed parameter schema and `risk` (`safe`/`confirm`/`blocked`); `blocked` skills are never shown to the LLM.
      Test: unit tests on registration, JSON schema output and the blocked filter.
- [ ] **B2 · Ollama chat.** `core/brain.py` `AgentBrain`: `openai` client to `http://localhost:11434/v1`, model from `config.yaml` (non-thinking Qwen3 4B), `keep_alive`, 10-turn history, system prompt "you are Zeno, reply in under two sentences".
      Test: `python main.py --voice console --brain real` and chat; "close it" resolves using history.
- [ ] **B3 · Tool loop.** In `AgentBrain`: model returns a tool call, the registry runs it, the result goes back, the model writes the reply; stop after 4 steps; errors go back to the model.
      Test: unit tests with a mocked client (success, error, step cap); one live run.
- [ ] **B4 · Apps.** `skills/apps.py`: Start Menu discovery, `open_app`, `close_app` (`psutil`).
      Test: mocked-OS unit tests; type "open notepad" in the console.
- [ ] **B5 · Web and time.** `skills/web.py`, `skills/clock.py`: `get_time`, `open_url`, `web_search` (`ddgs`).
      Test: mocked unit tests; type "what time is it" and "search lo-fi on YouTube".
- [ ] **B6 · System.** `skills/system.py`: `set_volume` (`pycaw`), `lock_pc`, `screenshot` (`pyautogui`).
      Test: mocked unit tests; type "volume to 30" and watch the slider.
- [ ] **B7 · Action log and status.** `core/actionlog.py`: SQLite row for every executed skill; `AgentBrain` emits `acting:<skill>` status events.
      Test: unit test on the log; run with `--verbose` and see the events.

## Together · end of Phase 1

- [ ] **S1 · Integration.** `python main.py --voice real --brain real`, say "Hey Zeno, open Notepad".
      **Gate: opening apps by voice works.** Review the combined result together before starting Phase 2.
