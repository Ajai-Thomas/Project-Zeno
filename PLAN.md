# ZENO — Voice-Activated Desktop Assistant: Project Write-Up

&#32;

## Overview and goals

ZENO is a Python desktop assistant that wakes on a spoken name, understands a request, performs it on the computer, and replies by voice. It targets everyday chores: opening apps, searching and reading the web, controlling the system, and handling phone calls through the desktop.

**Problem.** Small tasks like launching a program, looking something up, or adjusting volume each take a few clicks and break focus. A hands-free assistant removes that friction and is a strong portfolio project because it touches audio, AI, automation, and OS integration.

**Goals**

1. Respond to a wake word ("Hey Zeno") within about 1 second, with no button press.
2. Think dynamically and learn from use: reason about any phrasing, with no hand-written replies or command lists.
3. Understand English, Malayalam, and Manglish ("Spotify open aakku").
4. Act only through allow-listed tools, built on MCP so new abilities plug in without rewriting the core.
5. Feel instant: start speaking within about 1.5 seconds and stop the moment it is interrupted.
6. Stay private by default, with a fully offline mode, run on an ordinary laptop, and cost nothing: every part is free or open source.
7. Prove it works: publish accuracy, speed, and how much it improves after a week of learning.

**What makes it different.** Most "Python Jarvis" projects are English-only command matchers. This one reasons instead of matching, learns measurably, understands Malayalam and Manglish, uses MCP, the standard way AI agents connect to tools today, and costs nothing to build or run.

**Target platform.** Windows 10/11 first (most common desktop), with the code split so macOS and Linux skills can be added later.

## Feature scope

The first release (MVP) covers the tasks that work reliably on any PC; calls and deeper automation come in later phases.

| Feature | Example command | Release | How it works |
| --- | --- | --- | --- |
| Open / close apps | "Open VS Code", "Close Chrome" | MVP | Apps auto-discovered from the Start Menu; `os.startfile`, `psutil` to close |
| Web search and answers | "What's the weather in Kochi?" | MVP | Browser or the free `ddgs` package, then the LLM summarises the result |
| Open websites | "Open YouTube and search lo-fi" | MVP | `webbrowser` with a built query URL |
| System control | "Volume to 30", "Lock the PC", "Take a screenshot" | MVP | `pycaw`, `ctypes`, `pyautogui` |
| Time, timers, reminders | "Remind me in 20 minutes to stretch" | MVP | `APScheduler` + SQLite |
| Small talk and Q&A | "Explain recursion simply" | MVP | Local LLM with conversation memory |
| Works offline | Any command with Wi-Fi off | MVP | Ollama with Qwen3 4B is the default brain from day one |
| Live status overlay | Shows "Listening…", "Opening Spotify…" | MVP | Small always-on-top PySide6 window with the live transcript and current step |
| Self-learning memory and routines | "Get ready for standup" | Phase 2 | See *Self-learning* below |
| Files, email, calendar | "Find my resume PDF", "What's on today?" | Phase 2 | Open-source MCP servers (filesystem, Gmail, Google Calendar) |
| Media control | "Pause", "Next song" | Phase 2 | Media keys via `keyboard` |
| Malayalam and Manglish | "Spotify open aakku" | Phase 3 | Language detection, then Whisper or IndicConformer (see *Core modules*) |
| Fast, interruptible replies | "Zeno, stop" | Phase 3 | Streaming speech-to-text and TTS, echo cancellation |
| Free cloud helper | Long multi-step requests | Phase 3 | Gemini or Groq free tier through the same code (see *Running it for free*) |
| Multi-step browser tasks | "Find the cheapest flight to Delhi on Friday" | Phase 4 | Playwright MCP server driven by the agent loop |
| Screen awareness | "What's this error on my screen?" | Phase 4 | Screenshot sent to a vision-capable model |
| Messaging | "WhatsApp Amma I'm on my way" | Phase 4 | WhatsApp Desktop deep link + confirm step |
| Calls (optional) | "Answer the call" | Phase 4 | See *Call handling*; build last, only if time allows |

**Out of scope for now:** payments, deleting files, or sending anything without a spoken confirmation.

## System architecture

Every request follows one loop: hear the wake word, transcribe the command, let the Brain choose a skill, run it, and speak the result.

&#91;embedded content: ZENO architecture · wake → understand → act → speak\]

The Brain handles simple commands with rules and sends everything else to the LLM, which can only act through the allow-listed skills below it.

## Technology stack

Python is the core language: it has the best libraries for audio, AI, and OS automation. Every component is free; only the brain has an optional free cloud helper.

| Component | Free choice (runs on the laptop) | Optional free cloud helper |
| --- | --- | --- |
| Audio capture | `sounddevice` | — |
| Echo cancellation (for barge-in) | WebRTC audio processing bindings, or a headset | — |
| Wake word | `openWakeWord` (has a pre-trained "hey jarvis" model) | — |
| Voice activity detection | Silero VAD (ONNX) | — |
| English speech-to-text | `faster-whisper` (`base`, or `small` for Manglish) | — |
| Malayalam speech-to-text | AI4Bharat IndicConformer Malayalam, int8 ONNX via `sherpa-onnx` | — |
| Brain (reasoning + tool use) | Ollama: Qwen3 4B Instruct for live replies, Qwen3 8B for nightly reflection | Gemini API or Groq free tier, called through the same `openai` Python package |
| Tools | MCP Python SDK (`mcp`) + open-source MCP servers (filesystem, Playwright, Google) | — |
| Text-to-speech | Piper (English `en_US-lessac`, Malayalam `ml_IN-meera` / `ml_IN-arjun`) | — |
| OS control | `os`, `psutil`, `pyautogui`, `pywinauto`, `pycaw` | — |
| Web search | Browser, or the `ddgs` (DuckDuckGo) package | — |
| Memory | SQLite + `fastembed` embeddings + Chroma | — |
| Scheduling | `APScheduler` | — |
| UI | Tray icon (`pystray`) + PySide6 status overlay | — |
| Packaging | PyInstaller; models downloaded on first run | — |

**Hardware.** The default setup fits in under 1 GB of disk and runs on most 8 GB laptops; see *Feasibility* below for each profile. Every model above has an ONNX or CTranslate2 build, so PyTorch is never needed.

## Feasibility: will it run on any laptop?

Yes. The default setup needs under 1 GB of disk and runs on most laptops with 8 GB RAM; only the fully offline mode needs a stronger machine (16 GB RAM recommended, about 4 GB disk). The trick is to use small int8/ONNX models and avoid PyTorch, which is the usual reason voice projects balloon to many gigabytes.

### On the target laptop: Lenovo Yoga Slim 7 2-in-1 (Core i5-1340P, 16 GB RAM, 92 GB free)

Every profile fits. Run **Offline** with Malayalam as the everyday setup: the local Qwen3 4B brain is free and private, and the free cloud tier steps in only for hard tasks when online.

| Resource | This laptop | ZENO needs (Malayalam / Offline) | Verdict |
| --- | --- | --- | --- |
| Processor | Intel Core i5-1340P (13th gen), 12 cores: 4 performance + 8 efficiency, 16 threads | Any x86 Intel or AMD chip | Works as written; no ARM workarounds needed |
| RAM | 16 GB installed, 5200 MT/s | \~1.5 GB / \~5–6 GB, on top of Windows and a browser using \~6–8 GB | Fits; use Qwen3 4B for live replies and 8B only for nightly reflection |
| Graphics | Intel Iris Xe, integrated (2 GB shared from RAM) | Not required | No NVIDIA GPU, so every model runs on the CPU with int8 (PLAN.md tells Claude Code to use CPU and int8 everywhere) |
| Storage | 92 GB free of 477 GB | \~1 GB / \~4 GB | Plenty; keep at least \~10% of the disk free for Windows |
| Microphone | Built-in array | — | Good enough to start; a headset helps barge-in |

Expected speed on this CPU (approximate): Whisper `base` transcribes a 3–5 second command in well under a second, and `small` in about one to two seconds. Qwen3 4B writes roughly 10–15 tokens per second, so offline replies take a few seconds. Plug in and set Windows power mode to "Best performance" for the fastest response; on battery the efficiency cores slow transcription noticeably.

### Setup profiles

ZENO checks the laptop's RAM on first run, suggests a profile, and downloads only the models that profile needs.

| Profile | What runs locally | Disk (approx.) | RAM needed | Works on | Trade-off |
| --- | --- | --- | --- | --- | --- |
| **Lite** (default) | Wake word, English STT (Whisper base), English voice; the brain uses a free cloud tier (Gemini or Groq) | \~0.7 GB | 4 GB minimum, 8 GB comfortable | Almost any laptop from the last \~8 years | Needs internet; free-tier daily limits apply |
| **Malayalam** | Lite + Malayalam speech model + Malayalam voice | \~1 GB | 8 GB | Most mid-range laptops | Slightly slower when both languages are loaded |
| **Offline** | Everything, including the Qwen3 4B brain through Ollama; the recommended free setup | \~3.5–4 GB | 8 GB minimum, 16 GB recommended | Recent laptops, ideally with a GPU or Apple Silicon | Small local models are weaker at multi-step tool use |

### What each part costs

| Component | Model | Disk | RAM while running |
| --- | --- | --- | --- |
| Wake word | openWakeWord, "hey jarvis" | A few MB (approx.) | Very small |
| English speech-to-text | faster-whisper `base` (\~74M parameters) | \~150 MB (approx.) | \~0.3 GB |
| Better English / Manglish STT | faster-whisper `small` (\~244M parameters) | \~0.5 GB (approx.) | Up to \~1 GB on CPU with int8 ([benchmark](https://github.com/Robitx/faster-whisper)) |
| Malayalam speech-to-text | AI4Bharat IndicConformer Malayalam, int8 ONNX (\~120M parameters) | 188 MB ([source](https://huggingface.co/remiai3/STT_MODELS_FOR_APP)) | Under 200 MB |
| Voice (per language) | Piper medium voice, e.g. `en_US-lessac` or Malayalam `ml_IN-arjun` / `ml_IN-meera` | \~60–64 MB each ([source](https://huggingface.co/4-alokk/piper-voices)) | Under 100 MB |
| Local brain (default) | Qwen3 4B Instruct via Ollama | \~2.5 GB ([source](https://micdrop.dev/docs/ai-integration/local-models)) | \~4 GB together with Whisper `base` and a voice (same source) |
| Stronger local brain (optional) | Qwen3 8B via Ollama | \~5 GB (same source) | \~6 GB (approx.) |
| Python + libraries | onnxruntime, ctranslate2, numpy, sounddevice, openai (no PyTorch) | \~0.4–0.5 GB (approx.) | \~0.2 GB |

Sizes for wake word, Whisper base/small on disk, and the Python install are estimates; the rest come from the linked model pages.

### How to keep it small

- **No PyTorch.** Use ONNX or CTranslate2 builds of every model (openWakeWord, Silero VAD, faster-whisper, IndicConformer via sherpa-onnx, Piper). AI4Bharat's original NeMo setup is heavy; its int8 ONNX export is 188 MB.
- **Download on demand.** The installer ships only code (\~100–200 MB as a PyInstaller build); models download on first run, per profile.
- **One voice per language.** Add more voices only if asked.
- **Local brain by default.** The free local model handles everyday commands; the free cloud tier is used only for harder tasks.

### Honest limits

- **Small local LLMs struggle with tools.** CPU-only laptops with 8–12 GB RAM run a 3B model at roughly 16 tokens per second, and models that size can [invent extra tool calls](https://github.com/Djtony707/TITAN/wiki/Troubleshooting). That is why ZENO hands hard tasks to the free cloud tier when it is online.
- **Old dual-core laptops** can run Lite but transcription will lag; use Whisper `tiny` or a cloud speech API there.
- **The mic matters more than the CPU.** A cheap headset beats a laptop's built-in mic for accuracy and stops ZENO hearing itself.
- **Windows first.** App discovery and call bridging are Windows-specific; macOS and Linux need their own versions of those two skills.

## Running it for free

Every part of ZENO can be built and run at zero cost, with no credit card. The brain runs on the laptop through Ollama by default, and a free cloud tier is an optional helper for harder tasks.

| What | Free choice | The catch |
| --- | --- | --- |
| AI brain (default) | Qwen3 4B Instruct on Ollama, running on your laptop | Weaker than cloud models on long multi-step tasks; among small models it handles tool calls best ([source](https://micdrop.dev/docs/ai-integration/local-models/agent.md)) |
| AI brain (backup for hard tasks) | Google Gemini API free tier (no card needed) or Groq's free tier; both accept the same OpenAI-style code as Ollama | Rate limits are tight and change often; on free tiers the provider may use prompts to improve its models, so treat cloud requests as not private |
| Speech, voice, wake word | faster-whisper, IndicConformer, Piper, openWakeWord, Silero VAD | None: all open source |
| Web search | Open results in the browser, or the `ddgs` (DuckDuckGo) Python package | Unofficial and may break; paid search APIs are dropped |
| Tools | MCP Python SDK and open-source MCP servers | None |
| Memory | SQLite, Chroma, fastembed | None |
| Calls | Phone Link + `pywinauto` | VoIP services such as Twilio are paid, so they are dropped |
| Code hosting and teamwork | GitHub free private repository | None for a two-person team |
| AI coding help | Claude.ai Free plan for questions and reviewing pasted code; GitHub Copilot's free tier inside VS Code (check current limits) | Claude Code is not included in the Free plan; its cheapest option is Pro at $20/month ([source](https://www.superblocks.com/blog/claude-code-pricing)) |

### How the brain chooses local or cloud

1. Every request starts on the local Qwen3 4B model.
2. If a tool fails twice, or the task runs past 4 steps, and a free cloud key is set, the request moves to the cloud model.
3. If the cloud quota is used up or there is no internet, it stays local and does its best.
4. Nightly reflection always runs locally, and can use the larger Qwen3 8B since nobody is waiting for it.

## Core modules

The assistant is six modules connected by a simple event loop; the "brain" picks a skill, and skills do the real work.

### 1. Wake word listener

Runs continuously on a 16 kHz mono stream in 80 ms chunks. Only this tiny model sees audio until it fires, which keeps CPU low and audio private. On a hit, it plays a short chime and hands the stream to the recorder. Tune the detection threshold (start near 0.5) to balance false triggers against missed calls.

### 2. Command recorder and speech-to-text

After the wake word, record until VAD detects about 0.8 s of silence (cap at 10 s). Pass the clip to `faster-whisper` with `language="en"` (or auto-detect for Malayalam/Hindi mixing). Use the `base` model on CPU for speed; move to `small` if accuracy is poor.

**Malayalam and Manglish.** Whisper detects the language of each clip. English and Manglish (Malayalam words in an English sentence) stay with Whisper `small`; pure Malayalam goes to the IndicConformer Malayalam model, which is trained on Indian speech and is only 188 MB as int8 ONNX. The LLM understands both scripts, so the brain needs no changes, and Piper's Malayalam voice speaks the reply in the language you used.

**Streaming.** In Phase 3, transcribe in chunks while you are still talking, so the text is ready the moment you stop.

### 3. Brain: reasoning agent

Two layers keep it fast and cheap:

1. **Fast path.** A cache of shortcuts ZENO learned itself: a phrase that led to the same action 3 times runs instantly with no AI call. Nothing here is hand-written.
2. **LLM path.** Anything else goes to the LLM with a list of *tools* (skills) described as JSON schemas. The model either answers in text or returns a tool call such as `open_app(name="spotify")`. The router runs the tool, sends the result back, and the model writes the spoken reply.

The system prompt tells the model it is ZENO, to keep replies under two sentences for speech, to ask before risky actions, and to save what it learns. The prompt is rebuilt every turn with your stored facts and routines, so its behaviour changes as its memory grows.

### 4. Skills (action executors)

Each skill is a Python function with a name, description, parameter schema, and a `risk` level. Skills register themselves in a registry so new ones are a single file. ZENO adds its own routines on top of these (see Self-learning). Starter skills: `open_app`, `close_app`, `web_search`, `open_url`, `set_volume`, `screenshot`, `set_timer`, `get_time`, `lock_pc`, `answer_call`, `decline_call`.

**Built on MCP.** ZENO is an MCP client: each group of skills is an MCP server (apps, system, timers) and ready-made servers add files, browser, Gmail, and calendar with no new code. The allow-list and risk level live in ZENO's config, not in the servers, so a new server gets no access until you approve its tools.

### 5. Text-to-speech

Piper turns the reply into audio in well under a second on CPU. Speak the first sentence while the rest is generated (stream sentence by sentence) to cut perceived lag. Support "stop" / barge-in by keeping the wake word listener active while speaking.

### 6. Memory and config

- **Short-term:** the last \~10 turns, sent with each LLM request so "close it" knows what "it" is.
- **Long-term:** SQLite table of facts ("my sister is Anu", app aliases, preferred browser).
- **Config:** `config.yaml` for API keys (loaded from environment variables), voice, wake word, and app paths.

## Self-learning and dynamic thinking

ZENO has no scripted replies and no command list: an LLM reasons through every request, and its memory grows from each conversation so it gets better without you feeding it rules.

It learns the way today's AI agents do: by building memory and reusable routines, not by retraining the model. That is fast, cheap, works on a laptop, and you can always see or undo what it learned.

&#91;embedded content: Learning loop · recall, act, learn, reflect\]

### How it thinks: the agent loop

For each request the model plans, calls a tool, reads the result, and decides the next step until the job is done. Nobody writes the flow in advance.

Example: *"Send the PDF I downloaded today to Rahul on WhatsApp."* ZENO searches Downloads for today's PDF, opens WhatsApp, looks up who Rahul is in memory, asks "Send invoice.pdf to Rahul?", and sends it. If a step fails, the error goes back to the model, which tries another way.

To make this work, give it **general tools** instead of one tool per task: open any installed app, search files, browse (Playwright), read the screen (screenshot + vision model), click and type through UI automation, and ask you a question. Combining these, it handles requests you never planned for.

### What it learns on its own

| What it learns | How | Example |
| --- | --- | --- |
| Facts and preferences | The model calls `remember` whenever it hears something lasting; facts are added to its prompt next time | "My sister is Anu" → later "call my sister" just works |
| Your apps | Scans the Start Menu at startup and on a schedule; no alias list to maintain | Install Figma today, "open Figma" works tomorrow |
| New routines | After a multi-step task succeeds, it saves the steps as a named routine with blanks such as {contact} and {file}, then reuses it with new values | "Get ready for standup" learned once, reused daily |
| Corrections | "No, I meant VS Code" is saved as a lesson and checked on similar requests | Never opens Visual Studio by mistake again |
| Shortcuts | A request that leads to the same single action 3 times is cached and skips the LLM; other wordings of it match by meaning (embeddings), not exact text | "Open Spotify" becomes instant |
| Habits | Nightly reflection reads the action log and spots patterns, then suggests automations | "You open Teams at 9:55 on weekdays. Want me to do that?" |

### Memory design

- **Episodic:** every conversation and action, time-stamped in SQLite.
- **Semantic:** facts, preferences, and lessons. Start with a JSON list in the prompt; switch to a vector store (Chroma + embeddings) once it grows past a few hundred items, retrieving only the most relevant ones each turn. New facts are merged with duplicates and replace contradicting older ones, so a changed phone number overwrites the old one.
- **Procedural:** the routine library, saved as steps (tool + inputs) that can be replayed or edited.

**Nightly reflection.** A scheduled job sends the day's log, failures, and corrections to the LLM and asks: what went wrong, what should be remembered, what should become a routine? It writes the answers to memory and removes outdated facts.

### Guardrails on learning

- New routines run with a spoken confirmation the first two times, then become trusted.
- It learns only from you and its own results, never from text on web pages, so a website cannot plant instructions.
- Say "What do you know about me?" to hear its memory and "Forget that" to delete an item; the memory file is also readable and editable.
- Habit-based automations are always suggested first, never switched on silently.

## Call handling

Calls are the hardest feature because a PC has no phone line of its own; the assistant has to work through a bridge app that does. Calls are optional and come last (Phase 4): they are the most fragile feature and add the least to the project, so build them only once everything else is solid.

| Route | What it gives you | How ZENO controls it | Effort |
| --- | --- | --- | --- |
| Microsoft Phone Link (Windows + Android/iPhone) | Your real phone number, ringing on the PC | No public API: watch for the incoming-call toast and click Answer/Decline with `pywinauto` (UI automation) | Medium, can break on app updates |
| KDE Connect (Windows/Linux + Android) | Call notifications, mute media on ring, send SMS | Read events via its D-Bus/CLI; answering itself still happens on the phone | Low, but partial |
| Android companion app (Tasker or custom app + ADB over Wi-Fi) | Full control of your real number | PC sends a command; phone runs `input keyevent KEYCODE_CALL` / `ENDCALL` | Medium to high |
| VoIP number (Twilio, Plivo, Exotel) | A separate cloud number ZENO fully controls | Webhooks + APIs; can answer, screen callers, take messages | Paid per minute, so not used |
| WhatsApp / Teams / Zoom calls | Calls already on the desktop | Hotkeys or `pywinauto` on the app window | Low to medium |

**Recommended path.** Start with Phone Link + `pywinauto`: it uses your real number and needs no phone-side setup. ZENO announces "Call from Rahul — answer or decline?", waits for a spoken reply (no wake word needed in that window), then clicks the button. Move to a companion Android app later if the UI automation proves flaky.

Always announce the caller and wait for an explicit "answer"; never auto-answer.

## Security, privacy and guardrails

An assistant that can click, type, and run programs needs firm limits, because a misheard sentence or a malicious web page could otherwise trigger real actions.

- **Risk levels on every skill.** `safe` runs immediately (open app, volume). `confirm` requires a spoken "yes" (send message, answer call, close unsaved app). `blocked` is never exposed to the LLM (delete files, run arbitrary shell commands, payments).
- **Allow-list, not free shell.** The LLM can only call registered skills with typed parameters. It never gets a raw `cmd` or `exec` tool.
- **Treat web content as data.** Text pulled from websites is summarised, never followed as instructions, to block prompt injection.
- **Local-first audio.** Nothing leaves the machine before the wake word fires. Show a tray indicator whenever the mic is recording.
- **Secrets in environment variables.** Keep API keys out of code and out of Git (`.env` + `.gitignore`).
- **Mute and kill switch.** A hotkey (e.g. `Ctrl+Alt+Z`) mutes listening; saying "Zeno, stop" cancels any running action.
- **Action log.** Every executed skill is written to SQLite with time, command text, and result, so you can audit what happened.

## Implementation roadmap

Build in four phases and only move on when the gate at the end of each one passes; a usable assistant exists after Phase 1.

&#91;embedded content: Roadmap · 4 phases, a gate at the end of each\]

Phase 1 is the highlighted MVP: get the full listen-think-speak loop working before adding intelligence, so every later feature plugs into a loop you already trust.

### Splitting the work between two people

Split by layer, not by phase: **Person A owns the Voice side** (everything that hears, speaks, and shows) and **Person B owns the Brain side** (everything that thinks, acts, and learns). Both work in every phase at the same time and meet at a small interface agreed on day 1.

**Day 1, together: the interface contract.** Claude Code drafts it in the kickoff session (see \*Building it with Claude Code\*), and you both approve it before anyone builds. Afterwards it changes only when both agree.

```python
# core/interfaces.py
from dataclasses import dataclass

@dataclass
class Utterance:
    text: str
    lang: str                                  # "en", "ml" or "mixed"

def listen() -> Utterance: ...                 # Voice: wake word, record, transcribe
def speak(text: str, lang: str = "en") -> None: ...   # Voice
def stop_speaking() -> None: ...               # Voice: for "Zeno, stop"
def think(u: Utterance) -> str: ...            # Brain: reason, act, learn, return the reply
def status(event: str) -> None: ...            # Brain sends, Voice's overlay shows it
```

Neither person waits for the other. Person B tests the Brain by typing commands into the console instead of speaking. Person A tests the Voice with a dummy brain that just repeats what it heard.

| Phase | Person A: Voice | Person B: Brain | Together, at the end of the phase |
| --- | --- | --- | --- |
| **1 · Core loop** (weeks 1–2) | Mic capture and silence detection, wake word, Whisper speech-to-text, text-to-speech; deliver `listen()` and `speak()` | App discovery, open/close apps, time, web search, volume and lock; the Ollama tool loop in `think()` | Join both sides in `main.py`. Gate: open apps by voice |
| **2 · Smart brain** (weeks 3–4) | Status overlay and tray icon, media keys; test harness that plays the 75 recorded commands and scores transcription | MCP client and servers (files, Gmail, calendar), self-learning memory (facts, routines, shortcuts), risk levels and confirmations, timers | Run the full test set. Gate: 90% correct |
| **3 · Malayalam + speed** (weeks 5–6) | Malayalam and Manglish speech (language detection, IndicConformer, Malayalam voice), streaming, interrupting with "stop" | Free cloud helper for hard tasks (Gemini or Groq) and quota handling, nightly reflection, replying in the user's language, day 1/7/30 learning-curve runs | Manglish tests. Gate: first word within 1.5 s |
| **4 · Polish + extras** (week 7 on) | PyInstaller packaging, first-run profile picker and model downloads, demo video | Browser agent (Playwright MCP), screen vision, WhatsApp messages, calls (optional) | README with results. Gate: publish |

Phase 2 is heavier for the Brain and Phase 3 for the Voice, so whoever is ahead in a phase picks up the testing tasks or helps the other.

**How to work together**

- **One GitHub repo where `main` always runs.** Each task gets its own branch and is merged through a pull request that the other person reviews.
- **Separate folders to avoid merge conflicts.** Voice owns `core/wake.py`, `core/stt.py`, `core/tts.py`, and a new `ui/` folder. Brain owns `core/brain.py`, `core/memory.py`, `skills/`, and a new `mcp_servers/` folder (not `mcp/`, which would shadow the `mcp` package). Shared files (`main.py`, `core/interfaces.py`, `config.yaml`) change only after a quick check with each other.
- **The same instructions for both AI coding assistants.** Keep this write-up as `PLAN.md` and put the interface rules and folder ownership in `CLAUDE.md`, so both of you get code that fits the same design.
- **Integrate weekly, not just at phase end.** Merge both sides into `main.py` at least once a week and do a short sync twice a week.
- **One free key each.** Each person creates their own free Gemini or Groq key, so you don't share one daily limit.

## Building it with Claude Code

There is no hand-written starter code. Instead, give this write-up to Claude Code in plan mode: it reads the design, proposes the structure and the split of work between you, and builds the shared base only after you approve. Each of you then builds your own side in your own session.

Claude Code is the one paid tool here: it needs a Claude Pro plan. ZENO itself stays free to run. If only one of you has Pro, that person runs the kickoff, and the other can use the free coding help listed in *Running it for free*.

### Before you start

- Install Python 3.11 (64-bit), Git, Ollama, VS Code, and Claude Code. Create one GitHub repo and both clone it.
- Export this doc as Markdown and save it in the repo's root folder as `PLAN.md`.
- In Claude Code, run `/model opusplan` so Opus does the planning and Sonnet writes the code, then press Shift+Tab until plan mode is on.

### Step 1: Kickoff session (together)

Paste this into Claude Code in plan mode:

```text
Read PLAN.md fully. It is the design for ZENO, a free, voice-activated desktop assistant
for Windows 11 (Intel i5-1340P, 16 GB RAM, no NVIDIA GPU). Two people are building it:
- Person A owns the Voice side: wake word, speech-to-text, text-to-speech, Malayalam speech,
  status overlay, packaging.
- Person B owns the Brain side: agent loop on Ollama with Qwen3 4B, tools via MCP,
  self-learning memory, the free cloud helper.

Do not write feature code yet. Produce a plan with:
1. The folder structure, with every folder marked Voice, Brain, or Shared.
2. core/interfaces.py as described in PLAN.md (listen, speak, stop_speaking, think, status),
   plus stubs so each side runs without the other: a typed-console brain for Person B and
   an echo brain for Person A.
3. A CLAUDE.md for the repo: project goal, interface rules, folder ownership, coding
   conventions, "everything must be free and work offline by default", "no PyTorch",
   and "never commit API keys".
4. TASKS.md: Phase 1 tasks split between Person A and Person B, each small enough for one
   pull request, in build order, each with how to test it on its own.
5. requirements.txt and a setup script that checks Python, Ollama, the Qwen3 model and the mic.
6. Any gaps, risks or open questions you see in PLAN.md.

After we approve, build only the shared base: folders, interfaces with stubs, CLAUDE.md,
TASKS.md, requirements.txt, the setup script, a main.py that runs end to end on the stubs,
and one smoke test. Then commit it to main.
```

### Step 2: Review the plan before approving

Read the plan together and check four things: the interface covers what both sides need, the Phase 1 tasks are roughly balanced, nothing in it costs money, and every task can be tested on its own. Ask Claude Code to revise until you both agree, then approve. Review its base commit together before either of you branches off.

### Step 3: Build your own side

Each person works on their own branch, in their own Claude Code session, one task at a time.

**Person A (Voice):**

```text
Read CLAUDE.md, PLAN.md and TASKS.md. I am Person A (Voice). Take my next Phase 1 task.
Only edit Voice folders. Plan it first, include a standalone test, and stop for my review
before starting the next task.
```

**Person B (Brain):**

```text
Read CLAUDE.md, PLAN.md and TASKS.md. I am Person B (Brain). Take my next Phase 1 task.
Only edit Brain folders and test by typing commands into the console stub. Plan it first
and stop for my review before starting the next task.
```

### Step 4: Start of every later phase

Run a short joint planning session in plan mode:

```text
Read PLAN.md, CLAUDE.md, TASKS.md and the git log. Phase N starts now. Update TASKS.md with
the Phase N tasks for Person A and Person B, mark what is done, and flag anything that needs
a change to core/interfaces.py.
```

**Tips that save time**

- **One task per session.** Run `/clear` between tasks so old context does not confuse new work.
- **Ask for a test with every task**, and read the diff before merging.
- **Treat shared files as a joint decision.** If Claude Code proposes changing `core/interfaces.py`, `main.py`, or `CLAUDE.md`, discuss it with your partner before accepting.
- **Keep `PLAN.md` current.** When you change the design, update the doc and re-export it, so both sessions keep following the same plan.

## Testing, risks and future work

Measure the assistant on a fixed set of spoken commands so every change can be checked against the same baseline.

**Testing**

- **Command test set:** 50 recorded commands in English and 25 in Malayalam/Manglish, in a quiet room and a noisy one. Track word error rate, correct-tool rate, and end-to-end latency.
- **Learning curve:** rerun the test set on day 1, day 7, and day 30 of real use. The improvement (more shortcuts, fewer wrong apps, fewer clarifying questions) is the project's headline result.
- **Unit tests:** each tool tested with mocked OS calls (`pytest` + `unittest.mock`).
- **Wake word:** count false triggers per hour with TV or music playing; target fewer than 1.
- **Latency budget:** first spoken word within 1.5 s for shortcuts and within 3 s for LLM requests.
- **Publish it:** put the numbers, a 2-minute demo video, and the setup profiles in the README.

**Risks and mitigations**

| Risk | Mitigation |
| --- | --- |
| Indian-English accent or Malayalam/English mixing hurts accuracy | Use Whisper `small`, add a custom vocabulary prompt (`initial_prompt`) with your app and contact names |
| Speaker audio triggers the wake word or gets transcribed | Use a headset, drain the buffer after speaking, add echo cancellation later |
| Phone Link UI automation breaks after an update | Keep calls in an isolated skill; fall back to a phone-side companion app |
| LLM picks the wrong tool or invents a parameter | Typed schemas, allow-listed values, confirmation on `confirm`-level skills |
| Free cloud quota used up, or no internet | Fast-path rules handle common commands offline; the local Qwen3 model is the default brain anyway |

**Future enhancements**

- More Indian languages (Tamil, Hindi, Kannada) using the same IndicConformer and Piper pipeline.
- A custom "Zeno" voice trained with Piper on a few hours of recordings.
- Morning briefing: weather, calendar, and email read out on request.
- Smart home control through a Home Assistant MCP server.
- A phone companion app so ZENO works away from the desk.
