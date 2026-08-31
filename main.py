"""
Voice Agent
===========
A voice-in / voice-out conversational agent built on:
  - ASR   : openai/whisper-small (transformers pipeline)     -> speech to text
  - Brain : Gemini or Grok (xAI) chat completion               -> text reply
  - TTS   : Maya 2 Native, Maya Research's hosted API           -> text to speech
            (https://mayaresearch.ai/llm.txt)

Supports two run modes:
  python main.py --mode gradio     (default) -> web UI, tap-to-talk, robot face
  python main.py --mode terminal              -> mic in, speaker out, all in console

TTS previously ran locally (Veena + Indic Parler-TTS, juggling two model
architectures and a GPU). Maya 2 Native replaces both with one hosted HTTP
endpoint that natively covers 11 major Indian languages plus Indian English,
in two voices (Ananya, Arjun), so there's no local TTS model to load anymore.
ASR (Whisper) is still local, since Maya's API is TTS-only.

The Gradio UI includes an animated robot-face panel (Idle / Listening /
Thinking / Speaking / Happy / Sad / Excited / Curious / Confused /
Surprised / Caring / Neutral), built entirely from inline HTML/CSS/SVG/JS -
no extra files or dependencies. It switches to LISTENING the instant the
browser mic actually starts recording (real `start_recording` event), to
THINKING the instant Converse is pressed, and then - the moment the reply
audio actually starts playing in the browser (real `play`/`playing` audio
events, not a timer) - jumps straight into the resolved emotion's own face
(e.g. "caring") rather than a generic "speaking" face, and drives that
face's own mouth live from the audio's real amplitude for as long as
playback is actually running (see ROBOT_FACE_SCRIPT's toSpeaking()/ampTick()
and the `.audio-live .state-group .mouth` CSS rule below). The emotion tag
itself comes from the same Gemini/Grok call that generates the reply (see
chat_llm(), which asks for a small JSON object with both the reply text and
the model's own contextual emotion read - no extra API round trip). The
original local, multi-signal, rule-based reader (see detect_emotion()) is
kept as the fallback for the rare turn where the model doesn't return valid
JSON.

NOTE ON A COMMON GRADIO PITFALL: the face-control JavaScript (window.
setRobotState) is intentionally NOT defined inside the <script> tag of
the gr.HTML() block below. Gradio (like any DOM innerHTML update) does
not execute <script> tags that arrive as part of an HTML component's
value, so a script embedded there would silently never run - the face
would then look permanently frozen even though every event-wiring call
"succeeds". Instead, that script is injected via gr.Blocks(head=...),
which places it in the real page <head> where the browser executes it
normally on load, exactly once, before any component JS references it.

Put your keys in a .env file next to this script (see .env.example):
  MAYA_API_KEY=maya_sk_live_...
  GEMINI_API_KEY=...            (if LLM_PROVIDER=gemini)
  GROK_API_KEY=...              (if LLM_PROVIDER=grok)
"""

import os
import io
import sys
import json
import wave
import base64
import argparse
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import requests
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()   # "gemini" or "grok"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")

ASR_MODEL_ID = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Maya 2 Native TTS -------------------------------------------------------
MAYA_API_KEY = os.getenv("MAYA_API_KEY", "")
MAYA_BASE_URL = "https://tts.mayaresearch.ai"
# "Maya 2 Native" (2 voices) is the default and what's pre-warmed/region-local.
# "Maya Calyx" (15 voices, still all 11 languages) is available if you want a
# bigger voice roster later - just set MAYA_MODEL=Maya Calyx and swap SPEAKERS
# below for its roster (Amit, Gargi, Rahul, Riya, Sagar, Krishna, Seema, Anu,
# Aarav, Neha P, Sana, Simran, Tarini, Tripti, Zara).
MAYA_MODEL = os.getenv("MAYA_MODEL", "Maya 2 Native")
MAYA_SAMPLE_RATE = 24000  # fixed by the API; response has no header to read it from

SPEAKERS = ["Ananya", "Arjun"]  # both speak all 11 languages

# Display name -> API language code. `None` means "omit the language field",
# which the API treats as auto-detect - this is also the documented way to
# handle Hinglish/code-switched text, since each script-run gets pronounced
# with its own language's rules instead of one code being forced onto all of it.
LANGUAGES: Dict[str, Optional[str]] = {
    "English (Indian accent)": "en",
    "Hindi": "hi",
    "Hinglish (code-mixed)": None,
    "Telugu": "te",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Odia": "or",
    "Punjabi": "pa",
    "Tamil": "ta",
}

# ----------------------------------------------------------------------------
# Lazy-loaded globals
# ----------------------------------------------------------------------------
_asr_pipe: Optional[Any] = None
_maya_session: Optional[requests.Session] = None


def load_asr() -> Any:
    global _asr_pipe
    if _asr_pipe is not None:
        return _asr_pipe
    from transformers import pipeline
    print("[ASR] loading whisper...", flush=True)
    _asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL_ID,
        device=0 if DEVICE == "cuda" else -1,
    )
    print("[ASR] ready.", flush=True)
    return _asr_pipe


def get_maya_session() -> requests.Session:
    """
    A shared requests.Session so the TLS/TCP handshake is paid once, not on
    every turn - the API docs call this out specifically as the single
    biggest lever on latency for the plain HTTP endpoint.
    """
    global _maya_session
    if _maya_session is None:
        _maya_session = requests.Session()
    return _maya_session


# ----------------------------------------------------------------------------
# ASR
# ----------------------------------------------------------------------------
def transcribe(audio_path: str) -> str:
    """
    Speech -> text using Whisper.

    We read the file ourselves with soundfile (libsndfile - no external
    binary needed) and hand the pipeline a raw {"array", "sampling_rate"}
    dict instead of a filepath. Passing a filepath makes transformers shell
    out to the `ffmpeg` binary to decode it, which frequently isn't
    installed on minimal cloud containers. This sidesteps that dependency
    entirely.
    """
    asr = load_asr()

    audio_array, sr = sf.read(audio_path, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # downmix stereo -> mono

    target_sr = 16000
    if sr != target_sr:
        try:
            import librosa  # type: ignore[import-not-found]
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            pass  # fall back to original sample rate if librosa isn't present

    raw_result: Any = asr({"array": audio_array, "sampling_rate": sr})

    # The pipeline can return a dict ({"text": ...}) or, depending on
    # settings, a list of chunk dicts. Handle both explicitly instead of
    # assuming a shape (this is also what confused the type checker).
    if isinstance(raw_result, dict):
        text = raw_result.get("text", "")
    elif isinstance(raw_result, list) and raw_result:
        text = " ".join(chunk.get("text", "") for chunk in raw_result)
    else:
        text = ""

    return str(text).strip()


# ----------------------------------------------------------------------------
# LLM brain
# ----------------------------------------------------------------------------
# The emotion tag now comes from the LLM itself (see chat_llm() below), not
# from local keyword matching - the model actually reads its own reply's
# tone/context (sarcasm, negation like "I'm not happy about this", etc.)
# instead of pattern-matching individual words. This list is what the
# system prompt asks the model to choose from, and is intentionally the
# same set detect_emotion() understands, so a malformed/missing tag can
# fall back to the local detector below with no format mismatch.
_ALLOWED_LLM_EMOTIONS = [
    "happy", "sad", "excited", "curious", "confused",
    "surprised", "caring", "thinking", "neutral",
]


def chat_llm(history, user_text, language_name):
    """
    history: list of {"role": "user"/"assistant", "content": str}
    Returns (reply_text, emotion_tag) - reply_text is the short, spoken-
    friendly reply; emotion_tag is one of _ALLOWED_LLM_EMOTIONS, read
    straight from the LLM's own contextual read of what it just said.

    This piggybacks the emotion tag onto the exact same Gemini/Grok call
    that already generates the reply (via a JSON-structured system prompt
    below), so there is zero extra latency or API cost versus the old
    reply-only call - unlike a second "classify this" round trip.
    """
    spoken_language = language_name.replace(" (code-mixed)", "").replace(" (Indian accent)", "")
    system_prompt = (
        "You are a warm, concise voice assistant speaking out loud, not typing. "
        f"Reply only in {spoken_language}. Keep the reply short (1-3 sentences), "
        "natural, conversational, no markdown, no emojis, no bullet points.\n\n"
        "You must respond with ONLY a single valid JSON object - no markdown "
        "code fences, no text before or after it - with exactly two keys:\n"
        '  "reply": your short spoken-style response, in the target language.\n'
        '  "emotion": a single word describing the genuine emotional tone of '
        "YOUR reply (not the user's message), based on its real meaning and "
        "context (e.g. sarcasm or negation should flip the obvious keyword - "
        '"I\'m not happy about that" is "sad", not "happy"). '
        f"Choose exactly one from: {', '.join(_ALLOWED_LLM_EMOTIONS)}.\n\n"
        'Example: {"reply": "Oh no, I\'m so sorry to hear that - what happened?", '
        '"emotion": "caring"}'
    )

    if LLM_PROVIDER == "gemini":
        raw = _chat_gemini(system_prompt, history, user_text)
    elif LLM_PROVIDER == "grok":
        raw = _chat_grok(system_prompt, history, user_text)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}', use 'gemini' or 'grok'")

    return _parse_llm_reply(raw)


def _parse_llm_reply(raw_text: str) -> Tuple[str, str]:
    """
    Parse the {"reply": ..., "emotion": ...} JSON that chat_llm()'s system
    prompt asks the model for. If the model ever replies with something
    that isn't valid JSON (wrapped in markdown fences, chatty preamble,
    truncated, etc.) this never loses the turn: the raw text is used as
    the reply as-is, and detect_emotion() - the original local, rule-based
    reader - is used as the fallback mood instead of a blind "neutral".
    """
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
        reply_text = str(data.get("reply", "")).strip()
        emotion_tag = str(data.get("emotion", "")).strip().lower()
        if not reply_text:
            raise ValueError("LLM JSON reply had an empty 'reply' field")
        if emotion_tag not in _ALLOWED_LLM_EMOTIONS:
            emotion_tag = detect_emotion(reply_text)
        return reply_text, emotion_tag
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        # Not valid JSON this turn - fall back to the raw text as the
        # reply, and the local keyword/structural detector for the mood,
        # so one malformed model response never breaks the conversation.
        fallback_text = raw_text.strip() if raw_text else ""
        return fallback_text, detect_emotion(fallback_text)


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


def _chat_gemini(system_prompt, history, user_text):
    try:
        import google.generativeai as genai  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "google-generativeai isn't installed. Run "
            "`pip install -r requirements.txt` in this environment."
        ) from e
    genai.configure(api_key=GEMINI_API_KEY)
    # NOTE: "gemini-2.0-flash" was retired by Google; the live API error
    # pointed us to "gemini-3.6-flash" as the replacement. Overridable via
    # GEMINI_MODEL in .env if it changes again in the future.
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)

    gem_history = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        gem_history.append({"role": role, "parts": [turn["content"]]})

    chat = model.start_chat(history=gem_history)
    response = chat.send_message(user_text)
    return response.text.strip()


def _chat_grok(system_prompt, history, user_text):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "grok-4",
            "messages": messages,
            "temperature": 0.7,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ----------------------------------------------------------------------------
# Maya 2 Native TTS
# ----------------------------------------------------------------------------
def synthesize_speech(text: str, language_name: str = "English (Indian accent)", speaker: str = "Ananya") -> str:
    """
    Text -> wav file path, via the Maya 2 Native hosted API.

    The API streams back raw PCM (16-bit signed little-endian, mono, 24kHz)
    with no file header - we wrap it into a proper .wav ourselves. Status is
    checked before anything is treated as audio: on a 4xx/5xx the body is a
    JSON error, and writing that straight to a .wav produces a silent
    0-second clip instead of a visible error, which is exactly the failure
    mode the API docs warn about.
    """
    if not MAYA_API_KEY:
        raise RuntimeError(
            "MAYA_API_KEY is not set. Add it to your .env file "
            "(get one from charan@mayaresearch.ai, bharath@mayaresearch.ai, "
            "or dheemanth@mayaresearch.ai)."
        )
    if language_name not in LANGUAGES:
        raise ValueError(f"Unknown language '{language_name}'")
    if speaker not in SPEAKERS:
        raise ValueError(f"Unknown speaker '{speaker}', choose from {SPEAKERS}")

    payload: Dict[str, Any] = {"text": text, "voice": speaker, "model": MAYA_MODEL}
    lang_code = LANGUAGES[language_name]
    if lang_code is not None:
        payload["language"] = lang_code
    # else: omit "language" entirely -> auto-detect. This is deliberate for
    # Hinglish/code-mixed text: forcing a single language code would apply
    # that code's pronunciation rules to every word, including the ones
    # actually in the other language.

    session = get_maya_session()
    resp = session.post(
        f"{MAYA_BASE_URL}/v1/tts",
        headers={
            "Authorization": f"Bearer {MAYA_API_KEY}",
            "content-type": "application/json",
            # A default User-Agent from some HTTP clients (notably Python's
            # stdlib urllib) can get flagged by an upstream request filter
            # and comes back as a non-JSON 403 that looks like an auth
            # failure but isn't. `requests` isn't affected, but setting this
            # explicitly costs nothing and rules the whole class of bug out.
            "user-agent": "voice-agent/1.0",
        },
        data=json.dumps(payload),
        stream=True,
        timeout=120,
    )

    if not resp.ok:
        try:
            err_body = resp.json()
        except ValueError:
            err_body = resp.text[:500]
        raise RuntimeError(f"Maya TTS API error {resp.status_code}: {err_body}")

    pcm = bytearray()
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            pcm += chunk

    if not pcm:
        raise ValueError("Maya TTS API returned an empty response body")

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(out_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(MAYA_SAMPLE_RATE)
        wav_file.writeframes(bytes(pcm))
    return out_path


# ----------------------------------------------------------------------------
# Emotion detection - local, multi-signal, rule-based FALLBACK
# ----------------------------------------------------------------------------
# The primary source of the emotion tag is now the LLM itself (see
# chat_llm()/_parse_llm_reply() above, which asks Gemini/Grok to return
# its own contextual emotion read alongside the reply, at zero extra API
# cost). detect_emotion() below is kept as the safety net for the rare
# turn where the model's output isn't valid JSON, so a single malformed
# reply never breaks the conversation or forces a blind "neutral".
#
# These state names must match the `data-state` values wired up in
# ROBOT_FACE_HTML / ROBOT_FACE_SCRIPT below.
ROBOT_STATES = [
    "idle", "listening", "thinking", "speaking", "happy", "sad", "excited",
    "curious", "confused", "surprised", "caring", "neutral",
]

# WHY THIS ISN'T "IF SENTENCE X THEN EMOTION Y":
# A single ordered keyword list (first match wins) effectively hardcodes
# specific phrases to specific moods and falls apart the moment the reply
# is phrased differently. Instead, every category below gets a *score*
# from how many of its own cues appear anywhere in the reply, PLUS a few
# structural/tonal signals (question density, exclamation density, hedging
# language, explanatory connectives) that layer on top of *any* wording.
# The category with the highest total score wins; ties are broken by
# PRIORITY_ORDER. A reply with zero signal in every category is NEUTRAL -
# the correct outcome for a plain informational answer, rather than
# forcing it into a false "happy" default.
#
# This is still just word/phrase matching + arithmetic (no model call), but
# because it's additive across many independent cues instead of a single
# early-exit match, it generalizes far better across the many languages
# and phrasings WiseBot can reply in, without hand-pinning any one sentence
# to any one mood.
_EMOTION_LEXICON: Dict[str, List[str]] = {
    "caring": [
        "don't worry", "do not worry", "take care", "i'm here for you",
        "i am here for you", "you can do it", "it's okay", "it is okay",
        "no problem", "i understand how", "you're not alone",
        "you are not alone", "i've got you", "i have got you",
        "it's alright", "it is alright", "you're doing great",
        "you are doing great", "be gentle with yourself", "sending you",
        "चिंता मत करो", "मैं यहाँ हूँ", "कोई बात नहीं", "फ़िक्र मत करो",
        "घबराओ मत", "तुम अकेले नहीं हो",
    ],
    "sad": [
        "sorry to hear", "unfortunately", "sadly", "i'm sorry", "i am sorry",
        "that's sad", "that is sad", "heartbroken", "my condolences",
        "how difficult", "that must be hard", "that sounds hard",
        "i wish things were different", "sorry for your loss",
        "दुख", "अफ़सोस", "माफ़ करना", "खेद है", "बहुत बुरा हुआ",
    ],
    "confused": [
        "i'm not sure", "i am not sure", "i don't understand",
        "i do not understand", "could you repeat", "could you clarify",
        "i'm not certain", "i am not certain", "that's unclear",
        "that is unclear", "hard to say", "i'm a bit lost",
        "not entirely sure", "could go either way", "hmm, tricky",
        "समझ नहीं आया", "क्या मतलब", "ठीक से समझ नहीं पाया",
    ],
    "surprised": [
        "really?", "no way", "seriously?", "unbelievable", "can't believe",
        "cannot believe", "who would have thought", "i did not expect",
        "i didn't expect", "surprisingly", "out of nowhere", "whoa",
        "अरे वाह", "सच में", "अविश्वसनीय", "यह तो हैरान करने वाला है",
    ],
    "excited": [
        "wow", "amazing", "awesome", "fantastic", "great news", "yay",
        "woohoo", "excellent", "brilliant", "so cool", "thrilling",
        "can't wait", "cannot wait", "incredible", "this is huge",
        "वाह", "बहुत बढ़िया", "जबरदस्त", "शानदार", "मज़ा आ गया",
    ],
    "curious": [
        "i wonder", "interesting question", "let's explore", "lets explore",
        "great question", "what if", "have you ever wondered",
        "let's find out", "lets find out", "worth exploring",
        "makes you think", "जानना चाहते", "सोचने वाली बात है",
    ],
    "happy": [
        "glad", "happy to", "great job", "well done", "good job", "congrat",
        "proud of you", "nicely done", "that's wonderful", "that is wonderful",
        "delighted", "pleased to", "खुशी", "बहुत अच्छा", "शाबाश", "बधाई",
    ],
    # Explanatory / reasoning tone: this is the settled mood for a reply
    # that is working through logic, not the "please wait" busy indicator.
    "thinking": [
        "let's break this down", "lets break this down", "step by step",
        "first,", "to begin with", "in other words", "this means",
        "the reason is", "let's think about", "lets think about",
        "so basically", "here's how it works", "here is how it works",
        "on one hand", "on the other hand", "which suggests",
        "for example", "because", "therefore", "यानी", "इसका मतलब है",
    ],
}

# Used only to break ties when two+ categories score equally.
_PRIORITY_ORDER = [
    "caring", "sad", "confused", "surprised", "excited",
    "curious", "happy", "thinking", "neutral",
]


def detect_emotion(reply_text: str) -> str:
    """
    Local, rule-based mood tag for the robot face, derived only from the
    reply text already in hand - no extra API/LLM round trip.

    Every emotion category accumulates a numeric score from (a) how many of
    its own lexicon phrases appear in the reply, and (b) a few
    language-agnostic structural cues (question marks, exclamation marks,
    hedging words) that nudge the relevant category regardless of exact
    wording. The highest-scoring category wins; a completely flat reply
    (score 0 everywhere) settles on NEUTRAL rather than a fake default mood.
    """
    if not reply_text or not reply_text.strip():
        return "idle"
    t = reply_text.lower().strip()

    scores: Dict[str, float] = {emotion: 0.0 for emotion in _EMOTION_LEXICON}

    for emotion, phrases in _EMOTION_LEXICON.items():
        for phrase in phrases:
            if phrase in t:
                scores[emotion] += 1.0

    # Structural/tonal signals layer on top of the lexicon, so tone can
    # shift the outcome even when no exact phrase from the list is used.
    question_count = t.count("?")
    exclaim_count = t.count("!")
    hedge_markers = (" maybe", " perhaps", " i think", " probably", " might be")

    scores["curious"] += min(question_count, 3) * 0.6
    scores["excited"] += min(exclaim_count, 3) * 0.6
    scores["confused"] += sum(0.4 for hm in hedge_markers if hm in t)

    best_emotion = "neutral"
    best_score = 0.0
    for emotion in _PRIORITY_ORDER:
        if emotion == "neutral":
            continue
        score = scores.get(emotion, 0.0)
        if score > best_score:
            best_score = score
            best_emotion = emotion

    if best_score <= 0.0:
        return "neutral"  # plain informational reply - no forced mood
    return best_emotion



# ----------------------------------------------------------------------------
# Terminal mode
# ----------------------------------------------------------------------------
def terminal_chat(language_name="English (Indian accent)", speaker="Ananya", seconds=5, out_dir="terminal_replies"):
    """
    Terminal mode needs a real microphone on whatever machine runs this
    process. A Lightning Studio container is headless (no sound card), so
    this will only work if you run main.py on your own laptop/desktop, not
    inside the cloud Studio. On Lightning, use --mode gradio instead: the
    mic there is your browser's mic, not the server's.

    Replies are saved as .wav files (no attempt to play through hardware
    speakers, since a cloud container has none) so you can still inspect
    the output either way.
    """
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "sounddevice needs a real audio input device. This will not "
            "work inside a headless Lightning Studio container - run this "
            "script locally instead, or use --mode gradio here (the mic "
            "is your browser's, not the server's)."
        ) from e
    import soundfile as sf_

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Terminal voice chat | language={language_name} | speaker={speaker}")
    print(f"Replies will be saved as .wav files in ./{out_dir}/")
    print("Press Ctrl+C to quit.")
    print("=" * 60)

    history = []
    samplerate = 16000
    turn = 0

    try:
        while True:
            input(f"\n[Press Enter, then speak for {seconds}s]")
            print("Listening...")
            recording = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1)
            sd.wait()
            tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf_.write(tmp_in, recording, samplerate)

            user_text = transcribe(tmp_in)
            if not user_text:
                print("(heard nothing, try again)")
                continue
            print(f"You: {user_text}")

            reply, _emotion = chat_llm(history, user_text, language_name)
            print(f"Assistant: {reply}")

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})

            turn += 1
            wav_path = synthesize_speech(reply, language_name=language_name, speaker=speaker)
            saved_path = os.path.join(out_dir, f"reply_{turn:03d}.wav")
            os.replace(wav_path, saved_path)
            print(f"(saved audio: {saved_path})")
    except KeyboardInterrupt:
        print("\nBye!")


# ----------------------------------------------------------------------------
# Gradio UI - robot expression panel
# ----------------------------------------------------------------------------
# Self-contained SVG "chibi" robot face, styled after a cute round-headed
# bot with a glowing dark "screen" for a face - antenna on top, two chunky
# side "ear" bumps, soft blush cheeks, and a glassy screen sheen that are
# always visible regardless of state. Every mood below is drawn with the
# same simple, friendly icon language as the reference sheet (round dot
# eyes, x_x, >_< wink, heart eyes, a floating "?", three thinking dots,
# a big open "o" for surprise, ...): every state is a pre-drawn sibling
# <g>, toggled purely with CSS via the wrapper's `data-state` attribute -
# no path morphing/JS drawing needed, so it's cheap, robust, and has no
# external assets or dependencies. The viewBox has extra headroom above
# and to the sides purely so the antenna and ear-bumps have room to live.
ROBOT_FACE_HTML = """
<div id="robot-panel" data-state="idle">
  <div class="robot-shell">
    <svg class="robot-svg" viewBox="-20 -34 340 260" xmlns="http://www.w3.org/2000/svg">
      <!-- Charm: antenna, always alive with a slow bob -->
      <g class="charm charm-antenna">
        <line x1="150" y1="2" x2="150" y2="-20" class="antenna-stem"></line>
        <circle cx="150" cy="-24" r="9" class="antenna-tip"></circle>
      </g>

      <!-- Charm: chunky rounded "ear" bumps, like the reference bot's head -->
      <rect x="-14" y="66" width="30" height="78" rx="15" class="charm head-ear"></rect>
      <rect x="284" y="66" width="30" height="78" rx="15" class="charm head-ear"></rect>

      <!-- Outer plastic head shell, cream/white, sitting behind the screen -->
      <rect x="-2" y="-6" width="304" height="216" rx="66" class="head-shell"></rect>

      <!-- The glowing "screen" face itself -->
      <rect x="10" y="6" width="280" height="192" rx="54" class="robot-screen"></rect>

      <!-- Charm: soft blush cheeks + a glassy sheen, always present under
           whichever expression is active; opacity is tuned per-state below. -->
      <g class="charm charm-blush">
        <ellipse cx="66" cy="122" rx="18" ry="10" class="blush"></ellipse>
        <ellipse cx="234" cy="122" rx="18" ry="10" class="blush"></ellipse>
      </g>
      <ellipse class="charm charm-shine" cx="88" cy="42" rx="64" ry="20"
                transform="rotate(-16 88 42)"></ellipse>

      <!-- ============================= IDLE =============================
           Calm resting face: gentle line-smile eyes that occasionally
           blink, plus a slow "look around" drift so it never looks frozen
           even when nothing is happening. -->
      <g class="state-group st-idle">
        <g class="eyes-drift">
          <circle class="eye-ring eye-blink" cx="97" cy="90" r="13"></circle>
          <circle class="eye-pupil" cx="97" cy="90" r="6"></circle>
          <circle class="pupil-glint" cx="94.5" cy="87.5" r="2"></circle>
          <circle class="eye-ring eye-blink" cx="203" cy="90" r="13"></circle>
          <circle class="eye-pupil" cx="203" cy="90" r="6"></circle>
          <circle class="pupil-glint" cx="200.5" cy="87.5" r="2"></circle>
        </g>
        <path class="mouth" d="M118,140 Q150,150 182,140"></path>
      </g>

      <!-- ============================ LISTENING ==========================
           Attentive round eyes (slightly raised brows) + audio-level bars,
           with an occasional blink so it still reads as "alive" while it
           waits for you to finish talking. -->
      <g class="state-group st-listening">
        <path class="brow" d="M74,58 Q97,49 120,58"></path>
        <path class="brow" d="M180,58 Q203,49 226,58"></path>
        <circle class="eye-ring eye-blink" cx="97" cy="88" r="14"></circle>
        <circle class="eye-pupil" cx="97" cy="88" r="6"></circle>
        <circle class="pupil-glint" cx="94.5" cy="85.5" r="2"></circle>
        <circle class="eye-ring eye-blink" cx="203" cy="88" r="14"></circle>
        <circle class="eye-pupil" cx="203" cy="88" r="6"></circle>
        <circle class="pupil-glint" cx="200.5" cy="85.5" r="2"></circle>
        <rect class="listen-bar bar-1" x="120" y="130" width="9" height="24" rx="4.5"></rect>
        <rect class="listen-bar bar-2" x="140" y="118" width="9" height="36" rx="4.5"></rect>
        <rect class="listen-bar bar-3" x="160" y="124" width="9" height="30" rx="4.5"></rect>
        <rect class="listen-bar bar-4" x="180" y="132" width="9" height="22" rx="4.5"></rect>
      </g>

      <!-- ============================ THINKING ============================
           Pupils genuinely glance up/sideways on a slow loop (pondering),
           one eyebrow lifts, and a small "..." thinking bubble pulses one
           dot at a time - "working on it", not "busy spinner". -->
      <g class="state-group st-thinking">
        <path class="brow" d="M74,60 L120,60"></path>
        <path class="brow think-brow" d="M180,52 Q203,44 226,54"></path>
        <circle class="eye-ring" cx="97" cy="90" r="11"></circle>
        <circle class="eye-pupil think-pupil-a" cx="97" cy="88" r="5"></circle>
        <circle class="eye-ring" cx="203" cy="90" r="11"></circle>
        <circle class="eye-pupil think-pupil-b" cx="203" cy="88" r="5"></circle>
        <circle class="think-dot think-dot-1" cx="132" cy="144" r="6.5"></circle>
        <circle class="think-dot think-dot-2" cx="150" cy="144" r="6.5"></circle>
        <circle class="think-dot think-dot-3" cx="168" cy="144" r="6.5"></circle>
      </g>

      <!-- ============================ SPEAKING ============================
           A single live "lip" shape whose height is driven straight from
           the *real* TTS audio's own volume (Web Audio AnalyserNode,
           sampled every animation frame while the browser's play/playing
           events say audio is actually running - see ROBOT_FACE_SCRIPT).
           This generic face is now only a FALLBACK, used when no resolved
           emotion is available for some reason - normally playback jumps
           straight into the resolved emotion's own face instead (see
           toSpeaking() in ROBOT_FACE_SCRIPT), whose own mouth gets driven
           live the same way via the `.audio-live .state-group .mouth`
           rule further down. If the browser ever blocks live amplitude
           reading (autoplay/Web Audio policy), a gentle CSS loop below
           takes over automatically so the mouth is never just frozen.
           Eyes blink naturally and the brows add a faint conversational
           bounce so the whole face stays animated, not just the mouth. -->
      <g class="state-group st-speaking">
        <path class="brow speak-brow" d="M74,58 Q97,50 120,58"></path>
        <path class="brow speak-brow speak-brow-b" d="M180,58 Q203,50 226,58"></path>
        <circle class="eye-ring eye-blink" cx="97" cy="88" r="14"></circle>
        <circle class="eye-pupil" cx="97" cy="88" r="6"></circle>
        <circle class="pupil-glint" cx="94.5" cy="85.5" r="2"></circle>
        <circle class="eye-ring eye-blink" cx="203" cy="88" r="14"></circle>
        <circle class="eye-pupil" cx="203" cy="88" r="6"></circle>
        <circle class="pupil-glint" cx="200.5" cy="85.5" r="2"></circle>
        <ellipse class="mouth-live" cx="150" cy="140" rx="26" ry="15"></ellipse>
      </g>

      <!-- ============================= HAPPY ==============================
           Bright, closed-curve ^‿^ smiling eyes, raised/arched brows, and a
           big warm smile, with a gentle whole-face bounce. -->
      <g class="state-group st-happy">
        <path class="brow" d="M72,55 Q97,42 122,55"></path>
        <path class="brow" d="M178,55 Q203,42 228,55"></path>
        <path class="eye eye-blink" d="M62,96 Q97,66 132,96"></path>
        <path class="eye eye-blink" d="M168,96 Q203,66 238,96"></path>
        <path class="mouth" d="M100,130 Q150,178 200,130"></path>
      </g>

      <!-- ============================== SAD ===============================
           Soft downturned x_x-adjacent eyes (closed, drooping outward),
           brows angled down toward the outside, a downward mouth curve,
           and a single slow tear - low-energy droop, never mocking. -->
      <g class="state-group st-sad">
        <path class="brow" d="M74,50 Q98,62 120,68"></path>
        <path class="brow" d="M226,50 Q202,62 180,68"></path>
        <path class="eye" d="M64,86 Q97,102 130,92"></path>
        <path class="eye" d="M170,92 Q203,102 236,86"></path>
        <path class="mouth" d="M112,158 Q150,134 188,158"></path>
        <path class="sad-tear" d="M222,100 Q229,112 222,120 Q215,112 222,100 Z"></path>
      </g>

      <!-- ============================ EXCITED =============================
           Wide sparkly eyes, high arched brows that bounce, an extra-big
           lively smile, quick whole-face bounce, and two little twinkle
           sparkles for that "genuinely thrilled" energy. -->
      <g class="state-group st-excited">
        <path class="brow excited-brow" d="M70,50 Q97,34 124,50"></path>
        <path class="brow excited-brow excited-brow-b" d="M176,50 Q203,34 230,50"></path>
        <circle class="eye-ring" cx="97" cy="88" r="18"></circle>
        <circle class="eye-pupil" cx="97" cy="88" r="7.5"></circle>
        <circle class="pupil-glint" cx="94" cy="84.5" r="2.6"></circle>
        <circle class="eye-ring" cx="203" cy="88" r="18"></circle>
        <circle class="eye-pupil" cx="203" cy="88" r="7.5"></circle>
        <circle class="pupil-glint" cx="200" cy="84.5" r="2.6"></circle>
        <path class="mouth" d="M92,128 Q150,186 208,128"></path>
        <g class="sparkle sparkle-a"><path d="M32,52 L36,60 L44,62 L36,64 L32,72 L28,64 L20,62 L28,60 Z"></path></g>
        <g class="sparkle sparkle-b"><path d="M270,98 L273,104 L279,106 L273,108 L270,114 L267,108 L261,106 L267,104 Z"></path></g>
      </g>

      <!-- ============================ CURIOUS ==============================
           One flat brow + one arched brow (the classic "tilted eyebrow"),
           plain round eyes that lean in, a small quizzical half-smile, and
           a floating "?" that gently bobs above the head. -->
      <g class="state-group st-curious">
        <path class="brow" d="M74,60 L120,60"></path>
        <path class="brow curious-brow" d="M178,50 Q203,36 228,50"></path>
        <circle class="eye-ring eye-blink" cx="97" cy="90" r="13"></circle>
        <circle class="eye-pupil" cx="99" cy="90" r="6"></circle>
        <circle class="pupil-glint" cx="96.5" cy="87.5" r="2"></circle>
        <circle class="eye-ring" cx="203" cy="88" r="15"></circle>
        <circle class="eye-pupil" cx="205" cy="88" r="6"></circle>
        <circle class="pupil-glint" cx="202.5" cy="85.5" r="2"></circle>
        <path class="mouth" d="M114,138 Q150,152 186,132"></path>
        <text class="curious-mark" x="150" y="10" text-anchor="middle">?</text>
      </g>

      <!-- =========================== CONFUSED ==============================
           A genuine "x_x" dazed face - straight over-and-under cross-strokes
           for each eye - plus asymmetric puzzled brows and a small wavy,
           uncertain mouth. -->
      <g class="state-group st-confused">
        <path class="brow confused-brow-l" d="M74,54 Q97,66 120,58"></path>
        <path class="brow confused-brow-r" d="M180,48 Q203,38 226,50"></path>
        <path class="eye eye-x" d="M84,78 L110,100 M110,78 L84,100"></path>
        <path class="eye eye-x" d="M190,78 L216,100 M216,78 L190,100"></path>
        <path class="mouth" d="M104,138 Q120,126 136,138 Q152,150 168,138 Q184,126 196,138"></path>
      </g>

      <!-- =========================== SURPRISED =============================
           Both brows shoot straight up, both eyes go wide with big pupils,
           mouth pops into a round "oh!" - plus a quick pop animation on
           the whole face for that startled-instant feel. -->
      <g class="state-group st-surprised">
        <path class="brow surprised-brow" d="M68,45 Q97,28 126,45"></path>
        <path class="brow surprised-brow" d="M174,45 Q203,28 232,45"></path>
        <circle class="eye-ring" cx="97" cy="88" r="21"></circle>
        <circle class="eye-pupil" cx="97" cy="88" r="8.5"></circle>
        <circle class="pupil-glint" cx="94" cy="84" r="2.8"></circle>
        <circle class="eye-ring" cx="203" cy="88" r="21"></circle>
        <circle class="eye-pupil" cx="203" cy="88" r="8.5"></circle>
        <circle class="pupil-glint" cx="200" cy="84" r="2.8"></circle>
        <ellipse class="mouth-o" cx="150" cy="144" rx="16" ry="22"></ellipse>
      </g>

      <!-- ============================ NEUTRAL ==============================
           Plain calm face for a flat informational reply: flat brows,
           round resting eyes with a natural blink, and a straight mouth,
           plus the same slow idle drift so it still feels alive. -->
      <g class="state-group st-neutral">
        <g class="eyes-drift">
          <path class="brow" d="M78,60 L116,60"></path>
          <path class="brow" d="M184,60 L222,60"></path>
          <circle class="eye-ring" cx="97" cy="88" r="13"></circle>
          <circle class="eye-pupil neutral-pupil" cx="97" cy="88" r="6"></circle>
          <circle class="pupil-glint" cx="94.5" cy="85.5" r="1.8"></circle>
          <circle class="eye-ring" cx="203" cy="88" r="13"></circle>
          <circle class="eye-pupil neutral-pupil" cx="203" cy="88" r="6"></circle>
          <circle class="pupil-glint" cx="200.5" cy="85.5" r="1.8"></circle>
        </g>
        <path class="mouth" d="M114,142 L186,142"></path>
      </g>

      <!-- ============================ CARING ===============================
           Warm closed-curve ^‿^ eyes with two soft glowing hearts in place
           of pupils, gentle downward-curved brows, and a soft warm smile -
           a slow breathing motion keeps it calm and reassuring rather than
           overly excitable. -->
      <g class="state-group st-caring">
        <path class="brow" d="M76,52 Q97,44 118,54"></path>
        <path class="brow" d="M224,52 Q203,44 182,54"></path>
        <path class="heart-eye" d="M97,84 C93,78 82,79 82,88 C82,95 90,100 97,105 C104,100 112,95 112,88 C112,79 101,78 97,84 Z"></path>
        <path class="heart-eye" d="M203,84 C199,78 188,79 188,88 C188,95 196,100 203,105 C210,100 218,95 218,88 C218,79 207,78 203,84 Z"></path>
        <path class="mouth" d="M112,132 Q150,158 188,132"></path>
      </g>
    </svg>
    <div id="robot-status-label" class="robot-label">Idle</div>
  </div>
</div>

<style>
  #robot-panel { display:flex; justify-content:center; padding:6px 0 14px; }
  #robot-panel .robot-shell { display:flex; flex-direction:column; align-items:center; gap:8px; }
  #robot-panel .robot-svg {
    width: 240px; height: 188px;
    filter: drop-shadow(0 0 22px var(--glow, #38bdf8));
    transition: filter .4s ease;
    overflow: visible;
  }
  #robot-panel .head-shell {
    fill: #f3f1fb; stroke: #d9d5f0; stroke-width: 2;
  }
  #robot-panel .head-ear {
    fill: #f3f1fb; stroke: #d9d5f0; stroke-width: 2;
  }
  #robot-panel .robot-screen {
    fill: #0b1120; stroke: var(--glow, #38bdf8); stroke-width: 4;
    transition: stroke .4s ease;
  }
  #robot-panel .eye, #robot-panel .mouth {
    fill: none; stroke: var(--glow, #38bdf8); stroke-width: 9;
    stroke-linecap: round; stroke-linejoin: round;
    transition: stroke .4s ease;
    transform-box: fill-box; transform-origin: center;
  }
  #robot-panel .eye-x {
    fill: none; stroke: var(--glow, #38bdf8); stroke-width: 8;
    stroke-linecap: round; transition: stroke .4s ease;
    transform-box: fill-box; transform-origin: center;
  }
  #robot-panel .brow {
    fill: none; stroke: var(--glow, #38bdf8); stroke-width: 6;
    stroke-linecap: round; opacity: .85;
    transition: stroke .4s ease, opacity .4s ease;
    transform-box: fill-box; transform-origin: center;
  }
  #robot-panel .eye-pupil, #robot-panel .heart-eye {
    fill: var(--glow, #38bdf8); transition: fill .4s ease;
    transform-box: fill-box; transform-origin: center;
  }
  #robot-panel .eye-ring {
    fill: none; stroke: var(--glow, #38bdf8); stroke-width: 6;
    transition: stroke .4s ease;
    transform-box: fill-box; transform-origin: center;
  }
  #robot-panel .mouth-o, #robot-panel .mouth-live {
    fill: var(--glow, #38bdf8); opacity: .92; transition: fill .4s ease;
    transform-box: fill-box; transform-origin: center;
  }
  #robot-panel .mouth-o { fill: none; stroke: var(--glow, #38bdf8); stroke-width: 9; }
  #robot-panel .sad-tear { fill: #7dd3fc; opacity: .85; }
  #robot-panel .think-dot {
    fill: var(--glow, #38bdf8); transform-box: fill-box; transform-origin: center;
  }
  #robot-panel .curious-mark {
    font-family: -apple-system, Segoe UI, sans-serif; font-size: 34px; font-weight: 800;
    fill: var(--glow, #38bdf8); transform-box: fill-box; transform-origin: center;
  }
  #robot-panel .listen-bar {
    fill: var(--glow, #38bdf8); transition: fill .4s ease;
    transform-box: fill-box; transform-origin: center;
  }
  #robot-panel .eyes-drift { transform-box: fill-box; transform-origin: center; }
  #robot-panel .sparkle path { fill: var(--glow, #38bdf8); opacity: 0; }
  #robot-panel .robot-label {
    font-family: -apple-system, Segoe UI, sans-serif; font-size: 13px;
    font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--glow, #38bdf8); opacity: .85; transition: color .4s ease;
  }

  /* --- Charm: always-on decorative touches, independent of state --- */
  #robot-panel .antenna-stem {
    stroke: var(--glow, #38bdf8); stroke-width: 4; stroke-linecap: round;
    transition: stroke .4s ease;
  }
  #robot-panel .antenna-tip {
    fill: var(--glow, #38bdf8); transition: fill .4s ease;
    filter: drop-shadow(0 0 6px var(--glow, #38bdf8));
  }
  #robot-panel .charm-antenna {
    transform-box: view-box; transform-origin: 150px 2px;
    animation: robot-antenna-bob 2.4s ease-in-out infinite;
  }
  #robot-panel .blush {
    fill: #fb7185; opacity: .3; filter: blur(3px);
    transition: opacity .5s ease;
  }
  #robot-panel .charm-shine {
    fill: #ffffff; opacity: .07; pointer-events: none;
  }
  #robot-panel .pupil-glint { fill: #ffffff; opacity: .9; }
  #robot-panel[data-state="happy"]     .blush,
  #robot-panel[data-state="excited"]   .blush,
  #robot-panel[data-state="caring"]    .blush,
  #robot-panel[data-state="surprised"] .blush { opacity: .55; }
  #robot-panel[data-state="sad"]       .blush,
  #robot-panel[data-state="confused"]  .blush,
  #robot-panel[data-state="thinking"]  .blush { opacity: .1; }

  /* All state groups are hidden by default; only the active one shows. */
  #robot-panel .state-group { display: none; }
  #robot-panel[data-state="idle"]      .st-idle,
  #robot-panel[data-state="listening"] .st-listening,
  #robot-panel[data-state="thinking"]  .st-thinking,
  #robot-panel[data-state="speaking"]  .st-speaking,
  #robot-panel[data-state="happy"]     .st-happy,
  #robot-panel[data-state="sad"]       .st-sad,
  #robot-panel[data-state="excited"]   .st-excited,
  #robot-panel[data-state="curious"]   .st-curious,
  #robot-panel[data-state="confused"]  .st-confused,
  #robot-panel[data-state="surprised"] .st-surprised,
  #robot-panel[data-state="neutral"]   .st-neutral,
  #robot-panel[data-state="caring"]    .st-caring { display: block; }

  /* Per-state glow colour + whole-panel motion. */
  #robot-panel[data-state="idle"]      { --glow: #38bdf8; }
  #robot-panel[data-state="listening"] { --glow: #4ade80; }
  #robot-panel[data-state="thinking"]  { --glow: #fbbf24; }
  #robot-panel[data-state="speaking"]  { --glow: #22d3ee; }
  #robot-panel[data-state="happy"]     { --glow: #34d399; }
  #robot-panel[data-state="sad"]       { --glow: #64748b; }
  #robot-panel[data-state="excited"]   { --glow: #fb923c; }
  #robot-panel[data-state="curious"]   { --glow: #06b6d4; }
  #robot-panel[data-state="confused"]  { --glow: #a78bfa; }
  #robot-panel[data-state="surprised"] { --glow: #f8fafc; }
  #robot-panel[data-state="neutral"]   { --glow: #93c5fd; }
  #robot-panel[data-state="caring"]    { --glow: #f472b6; }

  #robot-panel[data-state="idle"] .robot-svg { animation: robot-breathe 3.2s ease-in-out infinite; }
  #robot-panel[data-state="idle"] .eyes-drift { animation: robot-eyes-drift 6s ease-in-out infinite; }
  #robot-panel[data-state="idle"] .eye-blink { animation: robot-blink-soft 4.4s ease-in-out infinite; }

  #robot-panel[data-state="listening"] .robot-svg { animation: robot-breathe 2s ease-in-out infinite; }
  #robot-panel[data-state="listening"] .eye-blink { animation: robot-blink-soft 3.6s ease-in-out infinite; }
  #robot-panel[data-state="listening"] .bar-1 { animation: robot-listen-bar .6s ease-in-out infinite 0s; }
  #robot-panel[data-state="listening"] .bar-2 { animation: robot-listen-bar .6s ease-in-out infinite .12s; }
  #robot-panel[data-state="listening"] .bar-3 { animation: robot-listen-bar .6s ease-in-out infinite .24s; }
  #robot-panel[data-state="listening"] .bar-4 { animation: robot-listen-bar .6s ease-in-out infinite .36s; }

  #robot-panel[data-state="thinking"] .robot-svg { animation: robot-breathe 2.8s ease-in-out infinite; }
  #robot-panel[data-state="thinking"] .think-pupil-a { animation: robot-think-look 3.2s ease-in-out infinite; }
  #robot-panel[data-state="thinking"] .think-pupil-b { animation: robot-think-look 3.2s ease-in-out infinite .12s; }
  #robot-panel[data-state="thinking"] .think-brow { animation: robot-brow-raise 3.2s ease-in-out infinite; }
  #robot-panel[data-state="thinking"] .think-dot-1 { animation: robot-think-dot 1.2s ease-in-out infinite 0s; }
  #robot-panel[data-state="thinking"] .think-dot-2 { animation: robot-think-dot 1.2s ease-in-out infinite .2s; }
  #robot-panel[data-state="thinking"] .think-dot-3 { animation: robot-think-dot 1.2s ease-in-out infinite .4s; }

  /* Speaking: the live mouth's height is normally driven every animation
     frame straight from the *real* TTS audio's own volume via a CSS
     variable (--mouth-scale) that the JS in ROBOT_FACE_SCRIPT updates
     from a Web Audio AnalyserNode while play/playing say audio is
     genuinely running (see the "audio-live" class toggle below). This
     block only supplies a *fallback* idle-mouth loop, used automatically
     if Web Audio is ever unavailable/blocked, so the mouth is never just
     frozen while the state is "speaking". Eyes blink and brows get a
     faint conversational bounce so the whole face stays alive too. */
  #robot-panel[data-state="speaking"] .mouth-live {
    animation: robot-mouth-fallback .38s ease-in-out infinite;
  }
  #robot-panel.audio-live[data-state="speaking"] .mouth-live {
    animation: none;
    transform: scaleY(var(--mouth-scale, .4));
    /* A tiny transition smooths out any residual frame-to-frame jitter
       in the raw amplitude reading without adding perceptible lag -
       kept well under one frame (~16ms) so it never reads as delay. */
    transition: transform .015s linear;
  }
  #robot-panel[data-state="speaking"] .eye-blink { animation: robot-blink-soft 4s ease-in-out infinite; }
  #robot-panel[data-state="speaking"] .speak-brow { animation: robot-brow-talk .42s ease-in-out infinite; }
  #robot-panel[data-state="speaking"] .speak-brow-b { animation: robot-brow-talk .42s ease-in-out infinite .1s; }
  /* The mouth alone used to be the only thing that ever moved while
     talking - the rest of the robot just sat frozen. Give the whole head
     a light conversational bob so it visibly reads as "alive and talking",
     not just a twitching mouth on a static face. */
  #robot-panel[data-state="speaking"] .robot-svg { animation: robot-talk-bob .6s ease-in-out infinite; }

  /* Live lipsync now drives whichever emotion's own mouth is on screen,
     not just the generic "speaking" face's mouth-live ellipse - since
     toSpeaking() in ROBOT_FACE_SCRIPT jumps straight to the resolved
     emotion (e.g. "caring") the moment real playback starts, and plays
     the reply there instead of on a neutral "speaking" face. .mouth and
     .mouth-o already have transform-box:fill-box + transform-origin:
     center set above, so scaling them by the same --mouth-scale variable
     the analyser writes just works, with no per-emotion special-casing. */
  #robot-panel.audio-live .state-group .mouth,
  #robot-panel.audio-live .state-group .mouth-o {
    transform: scaleY(var(--mouth-scale, .4));
    transition: transform .015s linear;
  }

  #robot-panel[data-state="happy"] .robot-svg { animation: robot-bounce 1.1s ease-in-out infinite; }
  #robot-panel[data-state="happy"] .eye-blink { animation: robot-blink-soft 3.8s ease-in-out infinite; }

  #robot-panel[data-state="sad"] .robot-svg { animation: robot-droop 2.4s ease-in-out infinite; }
  #robot-panel[data-state="sad"] .sad-tear { animation: robot-tear-fall 2.6s ease-in infinite; }

  #robot-panel[data-state="excited"] .robot-svg { animation: robot-bounce .5s ease-in-out infinite; }
  #robot-panel[data-state="excited"] .excited-brow { animation: robot-brow-bounce .5s ease-in-out infinite; }
  #robot-panel[data-state="excited"] .excited-brow-b { animation: robot-brow-bounce .5s ease-in-out infinite .08s; }
  #robot-panel[data-state="excited"] .sparkle-a path { animation: robot-sparkle 1.1s ease-in-out infinite; }
  #robot-panel[data-state="excited"] .sparkle-b path { animation: robot-sparkle 1.1s ease-in-out infinite .5s; }

  #robot-panel[data-state="curious"] .robot-svg { animation: robot-tilt 1.6s ease-in-out infinite; }
  #robot-panel[data-state="curious"] .curious-brow { animation: robot-brow-raise 2.4s ease-in-out infinite; }
  #robot-panel[data-state="curious"] .eye-blink { animation: robot-blink-soft 4s ease-in-out infinite; }
  #robot-panel[data-state="curious"] .curious-mark { animation: robot-mark-bob 1.8s ease-in-out infinite; }

  #robot-panel[data-state="confused"] .robot-svg { animation: robot-tilt .9s ease-in-out infinite; }
  #robot-panel[data-state="confused"] .confused-brow-r { animation: robot-brow-raise 1.8s ease-in-out infinite; }

  #robot-panel[data-state="surprised"] .robot-svg { animation: robot-pop .5s ease-out 1; }
  #robot-panel[data-state="surprised"] .surprised-brow { animation: robot-brow-shoot .5s ease-out 1; }

  #robot-panel[data-state="neutral"] .neutral-pupil { animation: robot-natural-blink 3.6s ease-in-out infinite; }
  #robot-panel[data-state="neutral"] .eyes-drift { animation: robot-eyes-drift 7s ease-in-out infinite; }

  #robot-panel[data-state="caring"] .robot-svg { animation: robot-breathe 2.6s ease-in-out infinite; }
  #robot-panel[data-state="caring"] .heart-eye { animation: robot-heart-pulse 1.8s ease-in-out infinite; }

  @keyframes robot-breathe { 0%,100% { transform: scale(1); } 50% { transform: scale(1.02); } }
  @keyframes robot-talk-bob { 0%,100% { transform: translateY(0) scale(1); } 50% { transform: translateY(-4px) scale(1.015); } }
  @keyframes robot-listen-bar { 0%,100% { transform: scaleY(.4); } 50% { transform: scaleY(1); } }
  @keyframes robot-natural-blink {
    0%, 92%, 100% { opacity: 1; r: 6; }
    96% { opacity: .2; r: 2; }
  }
  @keyframes robot-blink-soft {
    0%, 90%, 100% { transform: scaleY(1); }
    95% { transform: scaleY(.12); }
  }
  @keyframes robot-eyes-drift {
    0%, 100% { transform: translate(0, 0); }
    20% { transform: translate(3px, 0); }
    45% { transform: translate(-2px, 1px); }
    70% { transform: translate(2px, -1px); }
    85% { transform: translate(0, 0); }
  }
  @keyframes robot-think-look {
    0%, 100% { transform: translate(0, 0); }
    30% { transform: translate(-4px, -3px); }
    60% { transform: translate(4px, -3px); }
    80% { transform: translate(1px, 0); }
  }
  @keyframes robot-think-dot {
    0%, 100% { opacity: .3; transform: translateY(0); }
    50% { opacity: 1; transform: translateY(-5px); }
  }
  @keyframes robot-brow-raise {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
  }
  @keyframes robot-brow-talk {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-1.5px); }
  }
  @keyframes robot-brow-bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
  }
  @keyframes robot-brow-shoot {
    0% { transform: translateY(4px); }
    60% { transform: translateY(-3px); }
    100% { transform: translateY(0); }
  }
  @keyframes robot-sparkle {
    0%, 100% { opacity: 0; transform: scale(.6); }
    50% { opacity: 1; transform: scale(1.1); }
  }
  @keyframes robot-mark-bob {
    0%, 100% { transform: translateY(0) rotate(-4deg); }
    50% { transform: translateY(-5px) rotate(4deg); }
  }
  @keyframes robot-heart-pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.12); }
  }
  @keyframes robot-tear-fall {
    0% { opacity: 0; transform: translateY(0); }
    20% { opacity: .85; }
    80% { opacity: .85; transform: translateY(14px); }
    100% { opacity: 0; transform: translateY(18px); }
  }
  /* Fallback-only mouth loop, used purely when live audio-amplitude
     control (--mouth-scale, set from ROBOT_FACE_SCRIPT) isn't available;
     see the "audio-live" rule above, which turns this off the moment real
     amplitude data starts flowing. */
  @keyframes robot-mouth-fallback {
    0%, 100% { transform: scaleY(.4); }
    50% { transform: scaleY(1.75); }
  }
  @keyframes robot-antenna-bob {
    0%, 100% { transform: rotate(-4deg); }
    50% { transform: rotate(4deg); }
  }
  @keyframes robot-bounce  { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
  @keyframes robot-droop   { 0%,100% { transform: translateY(0); } 50% { transform: translateY(4px); } }
  @keyframes robot-tilt    { 0%,100% { transform: rotate(-4deg); } 50% { transform: rotate(4deg); } }
  @keyframes robot-pop     { 0% { transform: scale(.9); } 60% { transform: scale(1.08); } 100% { transform: scale(1); } }
</style>
"""


# The face-control script lives here, injected via gr.Blocks(head=...)
# instead of as a <script> tag inside ROBOT_FACE_HTML above.
#
# WHY: gr.HTML() renders its value as an HTML *fragment* that gets patched
# into the page via a DOM update (innerHTML-style). Browsers deliberately
# do not execute <script> tags that arrive that way - it's the same reason
# `element.innerHTML = "<script>...</script>"` is a well-known no-op for
# script execution in plain JS. That meant `window.setRobotState` was
# never actually being defined, so every event handler below that calls
# it (or checks `if (window.setRobotState)`) was silently doing nothing -
# which is exactly why the robot looked permanently stuck on IDLE.
#
# `head=` content, by contrast, is written into the real page <head> that
# the browser parses and executes normally on load, so the function is
# guaranteed to exist before any button/audio event ever tries to call it.
#
# This block also owns the *real* lipsync engine: window.__robotBindAmp
# wires a Web Audio AnalyserNode onto the actual <audio> element the reply
# plays from (once per element), and window.__robotAmpStart/__robotAmpStop
# start/stop a requestAnimationFrame loop that reads the audio's genuine
# live volume every frame and writes it straight into a CSS variable
# (--mouth-scale) the mouth's transform uses. There is no timer or
# sleep() anywhere in this pipeline: the rAF loop itself does nothing but
# mirror whatever the audio is actually doing that instant, and it is only
# ever started by a real 'play'/'playing' event and only ever stopped by a
# real 'pause'/'ended'/'error' event (wired up per-turn below, in
# build_gradio_app). If Web Audio can't be used for any reason (older
# browser, autoplay/security policy, etc.) the whole thing fails silently
# and the CSS-only fallback mouth loop in ROBOT_FACE_HTML's <style> keeps
# the mouth animated instead - the face is never left frozen either way.
#
# toSpeaking() below is what makes the reply play *in* the resolved
# emotion's own face (e.g. "caring") instead of a generic "speaking" face:
# the moment real playback starts, it calls setRobotState() with the
# target emotion computed in Python (carried on audio.dataset.targetEmotion),
# and the `.audio-live .state-group .mouth` CSS rule in ROBOT_FACE_HTML
# drives THAT state's own mouth from the same live amplitude data. The
# generic "speaking" state is kept only as a fallback for the (normally
# unreachable) case where no target emotion was supplied.
ROBOT_FACE_SCRIPT = """
<script>
  (function () {
    // Modern Gradio mounts the whole app inside a <gradio-app> custom
    // element, and on some Gradio versions that element uses a *shadow
    // DOM* - which plain document.getElementById()/querySelector() from
    // a page-level <script> simply cannot see inside of, since shadow
    // roots deliberately wall off their contents from the outer document.
    // On those versions, every lookup below would silently return null
    // and every handler that depends on it would just quietly no-op -
    // which is exactly why the robot could look completely frozen (or
    // the mouth specifically could never move) with no visible error.
    // This helper searches the normal document first, then - only if
    // that fails - recursively descends into any shadow roots it finds,
    // so element lookups work the same regardless of which Gradio
    // version/rendering mode is actually mounting the page.
    function deepQuery(selector, root) {
      root = root || document;
      var found = root.querySelector(selector);
      if (found) { return found; }
      var all = root.querySelectorAll("*");
      for (var i = 0; i < all.length; i++) {
        if (all[i].shadowRoot) {
          found = deepQuery(selector, all[i].shadowRoot);
          if (found) { return found; }
        }
      }
      return null;
    }
    window.__robotDeepQuery = deepQuery;

    // A single, persistent <audio> element that ONLY this script ever
    // creates, sources, or plays - added straight to the real page
    // <body> (not inside Gradio's own component tree), so it can never be
    // torn down/recreated by a Gradio re-render and its events are always
    // unambiguous: nothing else on the page ever calls .play()/.pause() on
    // it, so every event it fires reflects a genuine playback state
    // change, never a competing player's internal buffering.
    window.__robotGetSyncAudio = function () {
      if (window.__robotSyncAudioEl) { return window.__robotSyncAudioEl; }
      var el = document.createElement("audio");
      el.style.display = "none";
      el.setAttribute("playsinline", "");
      document.body.appendChild(el);
      window.__robotSyncAudioEl = el;
      return el;
    };

    window.setRobotState = function (state) {
      var allowed = ["idle","listening","thinking","speaking","happy","sad",
                      "excited","curious","confused","surprised","neutral",
                      "caring"];
      var panel = deepQuery("#robot-panel");
      if (!panel) { return; }
      var s = (state || "idle").toString().toLowerCase().trim();
      if (allowed.indexOf(s) === -1) { s = "idle"; }
      panel.setAttribute("data-state", s);
      if (s !== "speaking") {
        window.__robotAmpStop();
      }
      var label = deepQuery("#robot-status-label");
      if (label) { label.textContent = s.charAt(0).toUpperCase() + s.slice(1); }
    };

    // One shared AnalyserNode per <audio> element (Web Audio only allows
    // a MediaElementSource to be created once per element, ever).
    window.__robotBindAmp = function (audio) {
      if (!audio || audio.dataset.robotAmpBound === "1") { return; }
      audio.dataset.robotAmpBound = "1";
      try {
        var Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) { return; }
        var ctx = new Ctx();
        var source = ctx.createMediaElementSource(audio);
        var analyser = ctx.createAnalyser();
        // A small fftSize keeps this a plain volume/amplitude read (not a
        // detailed spectrum) so it reacts to actual loudness changes
        // quickly, frame by frame. smoothingTimeConstant is kept low
        // (rather than the analyser's own default of .8) because it is
        // the analyser's INTERNAL exponential averaging - stacking that
        // on top of the JS-side smoothing in ampTick() below was what
        // made the mouth feel a beat behind the actual audio. Only one
        // smoothing pass (the JS one) now runs, so the on-screen mouth
        // tracks the true waveform envelope with effectively zero added
        // delay beyond a single animation frame (~16ms).
        analyser.fftSize = 64;
        analyser.smoothingTimeConstant = 0.15;
        // Route audio THROUGH the analyser and on to real speakers, so
        // sound still plays normally - the analyser is just tapping it.
        source.connect(analyser);
        analyser.connect(ctx.destination);
        audio._robotCtx = ctx;
        audio._robotAnalyser = analyser;
        audio._robotFreqData = new Uint8Array(analyser.frequencyBinCount);
      } catch (e) {
        // Web Audio unavailable/blocked - the CSS fallback mouth loop in
        // ROBOT_FACE_HTML's <style> will animate the mouth instead.
      }
    };

    var rafId = null;

    // Reads the *actual current* volume of whatever the audio element is
    // playing this instant and writes it into --mouth-scale on the panel,
    // then schedules itself again via requestAnimationFrame - i.e. it is
    // purely reactive to real playback, frame by frame, not a fixed-length
    // timer standing in for the audio's true duration/loudness. Whichever
    // state-group is currently visible (the resolved emotion's face, once
    // toSpeaking() below has switched to it) is what --mouth-scale ends up
    // driving, via the `.audio-live .state-group .mouth`/`.mouth-o` CSS
    // rule in ROBOT_FACE_HTML.
    function ampTick(audio) {
      var panel = deepQuery("#robot-panel");
      if (!panel || !audio || !audio._robotAnalyser) { return; }
      if (audio.paused || audio.ended) { window.__robotAmpStop(); return; }

      var analyser = audio._robotAnalyser;
      var data = audio._robotFreqData;
      analyser.getByteFrequencyData(data);

      var sum = 0;
      for (var i = 0; i < data.length; i++) { sum += data[i]; }
      var avg = sum / data.length / 255; // 0..1, real current loudness

      // Light smoothing + a floor/ceiling so silence still shows a small
      // closed oval mouth and loud peaks don't clip visually. The .7
      // blend factor (up from .55) means the mouth reaches ~90% of a new
      // target within 2 frames instead of ~5, which is what removes the
      // perceptible lag between the audio's attack and the mouth opening.
      var prev = parseFloat(panel.dataset.robotAmpPrev || "0.4");
      var target = 0.4 + Math.min(avg * 2.5, 1.0) * 1.6; // ~0.4 (closed oval) .. ~2.0 (wide open)
      var smoothed = prev + (target - prev) * 0.7;
      panel.dataset.robotAmpPrev = String(smoothed);
      panel.style.setProperty("--mouth-scale", smoothed.toFixed(3));

      rafId = requestAnimationFrame(function () { ampTick(audio); });
    }

    window.__robotAmpStart = function (audio) {
      var panel = deepQuery("#robot-panel");
      if (!audio || !audio._robotAnalyser) { return; }
      try {
        if (audio._robotCtx && audio._robotCtx.state === "suspended") {
          audio._robotCtx.resume();
        }
      } catch (e) { /* ignore */ }
      if (panel) { panel.classList.add("audio-live"); }
      if (rafId !== null) { cancelAnimationFrame(rafId); }
      ampTick(audio);
    };

    window.__robotAmpStop = function () {
      var panel = deepQuery("#robot-panel");
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      if (panel) {
        panel.classList.remove("audio-live");
        // Snap the mouth back to its closed-oval rest shape immediately
        // rather than leaving it at whatever amplitude it last read -
        // avoids a stray "open mouth" flash for the instant between
        // playback stopping and the face settling onto its resolved
        // emotion.
        panel.style.setProperty("--mouth-scale", "0.4");
        delete panel.dataset.robotAmpPrev;
      }
    };
  })();
</script>
"""


def build_gradio_app():
    try:
        import gradio as gr  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "gradio isn't installed. Run `pip install -r requirements.txt`."
        ) from e

    def wav_to_data_uri(wav_path: str) -> str:
        """
        Reads a wav file straight into a self-contained `data:audio/wav;
        base64,...` URI. This is what lets the client-side lipsync engine
        own a completely independent <audio> element instead of having to
        locate and trust Gradio's own internal reply-audio DOM/player -
        which changes shape release to release (a plain <audio controls>
        tag on some versions, a custom waveform/streaming player on
        others) and was the actual root cause of the face settling to its
        resolved emotion while the clip was still audibly playing: that
        player's internal buffering can fire a `pause` event mid-clip that
        looks identical to the user genuinely pausing. A same-origin data
        URI sidesteps all of that - no DOM lookup, no shadow roots, no
        version-dependent internals, and no ambiguity about which pause
        event is real, because we're the only code touching this element.
        """
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        return f"data:audio/wav;base64,{b64}"

    def converse(audio_path, language_name, speaker, history):
        """
        Returns (history, transcript_markdown, reply_wav_path_or_None,
        target_emotion_string, reply_audio_data_uri_or_empty). target_emotion
        is always one of ROBOT_STATES and is what the face should settle on
        once/if speech playback ends (or immediately, if there's no audio to
        play at all). reply_audio_data_uri is a self-contained `data:` URI
        the client uses to drive its own independent lipsync <audio>
        element (see wav_to_data_uri above for why) - empty string means no
        clip was produced this turn. This function never raises - every
        stage is guarded so the UI, and the robot state machine on the
        client, always get a clean, final answer.
        """
        history = history or []

        if audio_path is None:
            return history, history_to_display(history), None, "idle", ""

        try:
            user_text = transcribe(audio_path)
        except Exception:
            traceback.print_exc()
            return history, history_to_display(history), None, "confused", ""

        if not user_text:
            # Nothing understood - a real, meaningful use of "confused",
            # not a filler state.
            return history, history_to_display(history), None, "confused", ""

        try:
            reply, target_emotion = chat_llm(history, user_text, language_name)
        except Exception as e:
            traceback.print_exc()
            history = history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": f"(Sorry, I couldn't think of a reply: {e})"},
            ]
            return history, history_to_display(history), None, "sad", ""

        history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]

        try:
            wav_path = synthesize_speech(reply, language_name=language_name, speaker=speaker)
        except Exception as e:
            traceback.print_exc()
            # Reply text and mood are still valid even though audio failed -
            # show/settle on them, just with no audio to play.
            history[-1]["content"] += f"  (voice unavailable: {e})"
            return history, history_to_display(history), None, target_emotion, ""

        try:
            data_uri = wav_to_data_uri(wav_path)
        except Exception:
            traceback.print_exc()
            # The wav file itself is still valid for the visible player even
            # if re-reading it for the data URI somehow failed - just skip
            # the lipsync-driving copy rather than losing the whole turn.
            data_uri = ""

        return history, history_to_display(history), wav_path, target_emotion, data_uri

    def history_to_display(history):
        lines = []
        for turn in history or []:
            speaker_label = "You" if turn["role"] == "user" else "Assistant"
            lines.append(f"**{speaker_label}:** {turn['content']}")
        return "\n\n".join(lines) if lines else "_Tap the mic, say something, then hit Converse._"

    def reset():
        return [], history_to_display([]), None

    # head=ROBOT_FACE_SCRIPT is the fix for the "robot never leaves IDLE"
    # bug: it puts window.setRobotState in the page's real <head>, where
    # the browser actually executes it, instead of inside a gr.HTML()
    # fragment where a <script> tag would silently never run. See the
    # comment above ROBOT_FACE_SCRIPT for the full explanation.
    with gr.Blocks(title="Voice Agent", head=ROBOT_FACE_SCRIPT) as demo:
        gr.Markdown("## 🗣️ Voice Agent\nRecord your voice, pick a language and speaker, then tap **Converse**.")

        gr.HTML(ROBOT_FACE_HTML)

        with gr.Row():
            language = gr.Dropdown(choices=list(LANGUAGES.keys()), value="English (Indian accent)", label="Language")
            speaker = gr.Dropdown(choices=SPEAKERS, value="Ananya", label="Voice")

        mic = gr.Audio(sources=["microphone"], type="filepath", label="Your voice")

        # Real browser mic events, not a guess/timer: the face switches to
        # LISTENING the instant recording actually starts, and drops back
        # to IDLE if recording is stopped/cleared without ever hitting
        # Converse (e.g. the user cancels). If Converse *is* pressed, the
        # very next handler below immediately overrides this with THINKING.
        mic.start_recording(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { if (window.setRobotState) { window.setRobotState('listening'); } }",
        )
        mic.stop_recording(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { if (window.setRobotState) { window.setRobotState('listening'); } }",
        )
        mic.clear(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { if (window.setRobotState) { window.setRobotState('idle'); } }",
        )

        with gr.Row():
            converse_btn = gr.Button("🎙️ Converse", variant="primary")
            reset_btn = gr.Button("Reset conversation")

        transcript = gr.Markdown(history_to_display([]))
        # autoplay is OFF here deliberately: the visible player is now just
        # for the user to review, scrub, download, or share the clip after
        # the fact. Actual auto-playback is handled by our own independent
        # <audio> element (see reply_audio_data_box / ROBOT_FACE_SCRIPT)
        # so the robot's lipsync never depends on this component's internal
        # DOM structure, which is exactly what changed between Gradio
        # versions and caused the face to settle mid-playback.
        reply_audio = gr.Audio(label="Assistant reply", autoplay=False, elem_id="reply_audio")

        # Hidden channel that carries the locally-detected target emotion
        # from Python to the client-side audio-sync script below. It's not
        # meant for the user to see or edit.
        target_emotion_box = gr.Textbox(value="idle", visible=False, elem_id="target_emotion_box")

        # Hidden channel carrying a self-contained `data:audio/wav;base64,
        # ...` URI for the reply clip (empty string if none was produced
        # this turn). The client script feeds this directly into an
        # <audio> element it owns completely, instead of hunting through
        # Gradio's own DOM for a player whose internals differ by version.
        reply_audio_data_box = gr.Textbox(value="", visible=False, elem_id="reply_audio_data_box")

        state = gr.State([])

        # 1) Instant client-side feedback the moment the button is pressed -
        #    tied to the real click event, not a timer.
        converse_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { if (window.setRobotState) { window.setRobotState('thinking'); } }",
        ).then(
            # 2) The actual STT -> LLM -> TTS pipeline (unchanged logic).
            fn=converse,
            inputs=[mic, language, speaker, state],
            outputs=[state, transcript, reply_audio, target_emotion_box, reply_audio_data_box],
        ).then(
            # 3) Play the new clip through our own, fully-owned hidden
            #    <audio> element (see ROBOT_FACE_SCRIPT's
            #    window.__robotGetSyncAudio) and drive the face from ITS
            #    play/playing/ended/error events. We deliberately no longer
            #    touch Gradio's own reply_audio component's internal DOM at
            #    all for this - different Gradio versions render that
            #    component differently under the hood (plain <audio>,
            #    custom waveform player, chunked/streaming playback...) and
            #    that player's own internal buffering can fire a `pause`
            #    event mid-clip that's indistinguishable from the user
            #    actually pausing, which is exactly what was making the
            #    face settle onto its resolved emotion while the reply was
            #    still audibly playing. A single element only we ever
            #    touch, fed a self-contained data URI, has no such
            #    ambiguity: every pause it ever fires is real.
            fn=None,
            inputs=[target_emotion_box, reply_audio_data_box],
            outputs=None,
            js="""
            (targetEmotion, audioDataUri) => {
              var emo = (targetEmotion || "idle").toString().toLowerCase();
              if (!window.setRobotState) { return; }

              if (!audioDataUri) {
                // Python has already told us, as a fact, that no clip was
                // produced this turn (STT/LLM/TTS error, or nothing heard) -
                // go straight to the resolved state.
                if (window.__robotAmpStop) { window.__robotAmpStop(); }
                window.setRobotState(emo);
                return;
              }

              var audio = window.__robotGetSyncAudio();
              audio.dataset.targetEmotion = emo;

              // Web Audio analyser is bound once per <audio> element (a
              // MediaElementSource can only ever be created once for a
              // given element) - this is what lets the mouth react to the
              // TTS clip's *real*, live volume instead of just looping a
              // generic shape while "speaking" is set. Safe to call every
              // turn since this is always the SAME persistent element.
              if (window.__robotBindAmp) { window.__robotBindAmp(audio); }

              if (audio.dataset.robotBound !== "1") {
                audio.dataset.robotBound = "1";

                var toSpeaking = function () {
                  // Go straight to the resolved emotion's own face instead
                  // of the generic "speaking" face - the mouth on THAT
                  // state gets driven live by the audio amplitude via the
                  // `.audio-live .state-group .mouth`/`.mouth-o` CSS rule,
                  // so caring/happy/sad/etc. now visibly talk in their own
                  // expression instead of settling into it only after the
                  // clip has already finished playing.
                  window.setRobotState(audio.dataset.targetEmotion || "speaking");
                  if (window.__robotAmpStart) { window.__robotAmpStart(audio); }
                };
                var settle = function () {
                  clearTimeout(audio._robotSafetyTimer);
                  if (window.__robotAmpStop) { window.__robotAmpStop(); }
                  window.setRobotState(audio.dataset.targetEmotion || "idle");
                };

                audio.addEventListener("play", toSpeaking);
                audio.addEventListener("playing", toSpeaking);
                audio.addEventListener("ended", settle);
                audio.addEventListener("error", settle);
                audio.addEventListener("pause", function () {
                  // Since this element is fully ours and only ever plays
                  // one clip start-to-finish per turn, any pause that
                  // isn't the natural end really is the reply stopping
                  // early - unlike Gradio's own player, there's no
                  // internal buffering/streaming here to confuse this
                  // with. Also fires once real playback ever ends.
                  if (!audio.ended) { settle(); }
                });
                audio.addEventListener("play", function () {
                  // Absolute safety net: if no play/ended/error event ever
                  // resolves things (unexpected browser quirk), force a
                  // settle a few seconds past the clip's own length so the
                  // face can never stay stuck in SPEAKING forever.
                  clearTimeout(audio._robotSafetyTimer);
                  var durMs = isFinite(audio.duration) && audio.duration > 0
                    ? audio.duration * 1000 : 30000;
                  audio._robotSafetyTimer = setTimeout(settle, durMs + 5000);
                });
              }

              // Every turn is a brand-new clip on the same element.
              clearTimeout(audio._robotSafetyTimer);
              audio.pause();
              audio.src = audioDataUri;
              audio.currentTime = 0;
              audio.load();

              var playPromise = audio.play();
              if (playPromise && typeof playPromise.catch === "function") {
                playPromise.catch(function () {
                  // Autoplay blocked, or the browser couldn't decode this
                  // instant (rare for a same-origin data URI) - settle
                  // immediately rather than leaving the face stuck on
                  // "thinking" forever waiting for a play event that will
                  // never come.
                  window.setRobotState(emo);
                });
              }
            }
            """,
        )

        reset_btn.click(
            fn=reset,
            outputs=[state, transcript, reply_audio],
        ).then(
            fn=None,
            inputs=None,
            outputs=None,
            js="() => { if (window.setRobotState) { window.setRobotState('idle'); } }",
        )

    return demo


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gradio", "terminal"], default="gradio")
    parser.add_argument("--language", default="English (Indian accent)", choices=list(LANGUAGES.keys()))
    parser.add_argument("--speaker", default="Ananya", choices=SPEAKERS)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Starting port; auto-tries the next few if busy")
    args: Any = parser.parse_args()

    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        print("WARNING: LLM_PROVIDER=gemini but GEMINI_API_KEY is empty. Set it in .env.")
    if LLM_PROVIDER == "grok" and not GROK_API_KEY:
        print("WARNING: LLM_PROVIDER=grok but GROK_API_KEY is empty. Set it in .env.")
    if not MAYA_API_KEY:
        print("WARNING: MAYA_API_KEY is empty. TTS calls will fail until it's set in .env.")

    if args.mode == "terminal":
        terminal_chat(language_name=args.language, speaker=args.speaker)
        return

    app = build_gradio_app()

    # If a stale process is still holding the port (common after killing a
    # terminal tab rather than the process itself), try the next few ports
    # instead of crashing.
    last_err: Optional[Exception] = None
    for port in range(args.port, args.port + 10):
        try:
            app.launch(server_name="0.0.0.0", server_port=port, share=args.share)
            return
        except OSError as e:
            print(f"Port {port} unavailable ({e}); trying {port + 1}...")
            last_err = e
    raise RuntimeError(
        f"Could not find a free port in range {args.port}-{args.port + 9}. "
        "Free one up manually, e.g.: fuser -k 7860/tcp"
    ) from last_err


if __name__ == "__main__":
    main()
