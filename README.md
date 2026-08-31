# Voice Agent

A voice-in / voice-out conversational agent with an animated, emotionally
expressive robot face — all in one `main.py`, no extra services to run.

```
🎙️ mic  →  Whisper (local ASR)  →  Gemini / Grok (brain)  →  Maya 2 Native (TTS)  →  🔊 reply
```

## What it does

- **Speech in**: record your voice (browser mic in the web UI, or your
  machine's mic in terminal mode) and it's transcribed locally with
  `openai/whisper-small`.
- **A reply, with feeling**: the transcript goes to Gemini or Grok, which
  returns both a short spoken-style reply *and* its own read of the
  emotional tone behind that reply (`happy`, `sad`, `caring`, `curious`,
  `confused`, `surprised`, `excited`, `thinking`, `neutral`) — in the same
  API call, at no extra cost or latency.
- **Speech out**: the reply is synthesized with Maya Research's Maya 2
  Native API, in your choice of language and voice.
- **A face that reacts in real time**: in the web UI, a self-contained
  SVG robot face goes `idle → listening → thinking →` and then — the
  instant the reply audio actually starts playing — jumps straight into
  the *resolved emotion's own face* (e.g. `caring`) and lip-syncs to the
  reply's real, live volume for the whole clip, rather than talking with
  a generic neutral mouth and only "becoming" caring after the audio
  finishes.

## Requirements

- Python 3.9+
- A GPU is optional (Whisper and inference use it if `torch.cuda` is
  available, otherwise CPU)
- An API key for **either** Gemini or Grok (whichever you set as
  `LLM_PROVIDER`)
- A Maya Research API key for TTS

## Setup

```bash
git clone <this repo>
cd voice-agent
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env
```

Open `.env` and fill in:

```env
LLM_PROVIDER=gemini          # or "grok"
GEMINI_API_KEY=...           # required if LLM_PROVIDER=gemini
GROK_API_KEY=...             # required if LLM_PROVIDER=grok
MAYA_API_KEY=...             # required always (TTS)
```

Maya API keys are issued by Maya Research — reach out to
`charan@mayaresearch.ai`, `bharath@mayaresearch.ai`, or
`dheemanth@mayaresearch.ai`.

## Running it

### Web UI (default, recommended)

```bash
python main.py
```

This launches a Gradio app on `http://0.0.0.0:7860` (or the next free
port after it). Open it in a browser, pick a language and voice, tap the
mic, speak, then hit **Converse**.

Useful flags:

```bash
python main.py --port 8000       # start on a specific port
python main.py --share           # get a public Gradio link
```

### Terminal mode

```bash
python main.py --mode terminal --language "Hindi" --speaker Arjun
```

Terminal mode needs a **real microphone on the machine running the
script** — it will not work inside a headless cloud container (e.g. a
Lightning Studio instance), since there's no sound hardware there. Use
`--mode gradio` in that case; the mic is your browser's, not the
server's. Replies are saved as `.wav` files under `./terminal_replies/`
rather than played through speakers, so you can still inspect them
either way.

## Languages & voices

11 major Indian languages plus Indian-accented English are supported,
via Maya 2 Native's two voices (**Ananya**, **Arjun**), both of which
speak every language listed:

| Display name | Code |
|---|---|
| English (Indian accent) | `en` |
| Hindi | `hi` |
| Hinglish (code-mixed) | auto-detect |
| Telugu | `te` |
| Bengali | `bn` |
| Gujarati | `gu` |
| Kannada | `kn` |
| Malayalam | `ml` |
| Marathi | `mr` |
| Odia | `or` |
| Punjabi | `pa` |
| Tamil | `ta` |

Hinglish/code-mixed replies deliberately omit the language field so the
API auto-detects each script-run and pronounces it with its own
language's rules, instead of forcing one code onto the whole sentence.

Want more voices? Set `MAYA_MODEL=Maya Calyx` in `.env` for a 15-voice
roster (Amit, Gargi, Rahul, Riya, Sagar, Krishna, Seema, Anu, Aarav,
Neha P, Sana, Simran, Tarini, Tripti, Zara) — you'll need to update the
`SPEAKERS` list in `main.py` to match.

## The robot face

Twelve expressions — Idle, Listening, Thinking, Speaking, Happy, Sad,
Excited, Curious, Confused, Surprised, Caring, Neutral — drawn entirely
in inline SVG/CSS/JS with no external assets. State transitions are
tied to real browser events, not timers or guesses:

- `listening` the instant the mic actually starts recording
- `thinking` the instant Converse is clicked
- the resolved emotion's own face (e.g. `caring`) the instant reply
  audio genuinely starts playing, with that face's mouth lip-synced
  live to the audio's real amplitude via a Web Audio `AnalyserNode`
- settles back once playback genuinely ends, errors, or is stopped

If Web Audio is ever unavailable or blocked, a CSS-only fallback mouth
loop keeps the face animated instead of freezing.

## Notes on dependencies

- **`google-generativeai`** is used for the Gemini path but is
  deprecated upstream in favor of `google-genai`. It still works fully
  today; if you see a `FutureWarning` about it, it's informational only
  — nothing is broken. Worth migrating eventually.
- **`librosa`** is optional: it's used to resample mic audio to 16kHz
  for Whisper if your browser/mic doesn't already record at 16kHz. If
  it's not installed, `transcribe()` silently falls back to the
  original sample rate instead of failing.
- **`sounddevice`** is only needed for `--mode terminal`; the default
  `--mode gradio` doesn't use it at all.
- Grok (xAI) is called via plain `requests` — no separate SDK required.

## Troubleshooting

- **"Port unavailable" on launch**: the script automatically tries the
  next 9 ports after `--port` (default 7860). If all are busy, free one
  manually, e.g. `fuser -k 7860/tcp`.
- **Robot face stuck on Idle**: this was a known Gradio pitfall (fixed
  in this version) — a `<script>` tag inside a `gr.HTML()` component
  never executes because Gradio patches it in via `innerHTML`, which
  browsers don't run scripts from. The face-control JS is injected via
  `gr.Blocks(head=...)` instead, which the browser does execute.
- **`sounddevice` errors in terminal mode**: you're likely running
  inside a headless container with no audio hardware. Use
  `--mode gradio` instead, or run terminal mode on your own machine.
- **Maya TTS 403 errors**: check `MAYA_API_KEY` is set correctly in
  `.env`; a missing/invalid key is the most common cause.
