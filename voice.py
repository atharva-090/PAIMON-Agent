"""Local voice I/O for Hermes: STT via whisper.cpp, TTS via piper.

Both are prebuilt CLI binaries (installed by setup_voice.py) invoked as
subprocesses -- no Python audio libraries, no network. The module degrades
gracefully: if the binaries/models aren't installed, ``status()['enabled']`` is
False and the callers (server.py) return 503, so the UI just hides the mic.

  transcribe(wav_bytes) -> str    : 16 kHz mono WAV in, recognised text out.
  synthesize(text)      -> bytes  : text in, WAV bytes out.
  clean_for_speech(text)-> str    : strip the agent's footer/noise + cap length.
  status()              -> dict   : {enabled, accel, model, voice}.

Each binary loads its sibling DLLs from its own directory (Windows adds the
exe's folder to the DLL search path), so we always invoke them by absolute path.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import config

_TMP = os.path.join(config.VOICE_DIR, "tmp")
_MAX_SPEAK_CHARS = int(os.environ.get("VOICE_MAX_SPEAK_CHARS", "1200"))
# whisper prints these markers instead of words when the clip is silence/noise.
_NON_SPEECH = {"[blank_audio]", "[silence]", "(silence)", "[ silence ]",
               "[music]", "[ pause ]", "you", ""}


def status() -> dict:
    return {
        "enabled": config.voice_enabled(),
        "accel": config.WHISPER_ACCEL,
        "model": os.path.basename(config.WHISPER_MODEL),
        "voice": os.path.basename(config.PIPER_VOICE),
        "wake_word": config.WAKE_WORD,
    }


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, timeout=config.VOICE_TIMEOUT,
                          **kw)


def transcribe(wav_bytes: bytes) -> str:
    """Recognise speech in a 16 kHz mono WAV. Returns '' for silence/non-speech."""
    if not (config.WHISPER_BIN and os.path.isfile(config.WHISPER_BIN)):
        raise RuntimeError("whisper-cli is not installed (run setup_voice.py)")
    if not (config.WHISPER_MODEL and os.path.isfile(config.WHISPER_MODEL)):
        raise RuntimeError(f"whisper model missing: {config.WHISPER_MODEL}")
    os.makedirs(_TMP, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_TMP) as d:
        wav = os.path.join(d, "in.wav")
        with open(wav, "wb") as fh:
            fh.write(wav_bytes)
        out_prefix = os.path.join(d, "out")
        cmd = [config.WHISPER_BIN, "-m", config.WHISPER_MODEL, "-f", wav,
               "-l", "en", "-nt", "-np", "-otxt", "-of", out_prefix]
        proc = _run(cmd)
        txt_path = out_prefix + ".txt"
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        else:
            text = proc.stdout.decode("utf-8", "replace")
            if not text.strip():
                err = proc.stderr.decode("utf-8", "replace")[-300:]
                raise RuntimeError(f"whisper produced no transcript: {err}")
        text = " ".join(text.split()).strip()
        return "" if text.lower() in _NON_SPEECH else text


def synthesize(text: str) -> bytes:
    """Render ``text`` to spoken-audio WAV bytes with piper."""
    if not (config.PIPER_BIN and os.path.isfile(config.PIPER_BIN)):
        raise RuntimeError("piper is not installed (run setup_voice.py)")
    if not (config.PIPER_VOICE and os.path.isfile(config.PIPER_VOICE)):
        raise RuntimeError(f"piper voice missing: {config.PIPER_VOICE}")
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to speak")
    os.makedirs(_TMP, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_TMP) as d:
        out = os.path.join(d, "out.wav")
        cmd = [config.PIPER_BIN, "--model", config.PIPER_VOICE,
               "--output_file", out]
        # piper finds <voice>.onnx.json automatically; point it at espeak data if
        # the bundle shipped it next to the exe (avoids a cwd-relative lookup).
        espeak = os.path.join(os.path.dirname(config.PIPER_BIN), "espeak-ng-data")
        if os.path.isdir(espeak):
            cmd += ["--espeak_data", espeak]
        proc = _run(cmd, input=text.encode("utf-8"))
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            err = proc.stderr.decode("utf-8", "replace")[-300:]
            raise RuntimeError(f"piper produced no audio: {err}")
        with open(out, "rb") as fh:
            return fh.read()


def clean_for_speech(text: str) -> str:
    """Trim an agent reply to what's worth speaking: drop the deterministic
    '[verified actions ...]' ground-truth footer, collapse whitespace, and cap
    the length so TTS stays snappy."""
    t = text or ""
    idx = t.find("[verified actions")
    if idx != -1:
        t = t[:idx]
    t = t.strip()
    if len(t) > _MAX_SPEAK_CHARS:
        t = t[:_MAX_SPEAK_CHARS].rsplit(" ", 1)[0] + " …"
    return t
