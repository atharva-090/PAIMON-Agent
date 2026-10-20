"""Round-trip voice smoke test (NO mic): piper speaks a phrase, whisper reads it
back. Verifies BOTH engines + the transcribe/synthesize pipeline end-to-end.

Skips cleanly when the binaries/models aren't installed (run setup_voice.py).

Note: piper renders at the voice's native rate (amy-medium = 22.05 kHz) but
whisper-cli requires 16 kHz mono, so we downsample piper's WAV here. In the real
app the BROWSER already encodes 16 kHz (see app.js encodeWav), so transcribe()
itself assumes 16 kHz input.

Run:  agent/.venv/Scripts/python.exe -m multi_agent.test_voice
"""

from __future__ import annotations

import array
import io
import sys
import wave

import voice


def _to_16k_mono(wav_bytes: bytes) -> bytes:
    """Downsample a 16-bit PCM WAV to 16 kHz mono (nearest-neighbour) so whisper
    accepts piper's higher-rate output."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        ch, sw, fr, nframes = (w.getnchannels(), w.getsampwidth(),
                               w.getframerate(), w.getnframes())
        raw = w.readframes(nframes)
    if sw != 2:
        return wav_bytes  # unexpected; let whisper try as-is
    samples = array.array("h")
    samples.frombytes(raw)
    if ch == 2:
        samples = array.array("h", samples[0::2])
    ratio = fr / 16000.0
    outlen = int(len(samples) / ratio) if ratio else len(samples)
    out = array.array("h", bytes(2 * outlen))
    for i in range(outlen):
        out[i] = samples[int(i * ratio)]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(out.tobytes())
    return buf.getvalue()


def _run() -> int:
    st = voice.status()
    print("voice status:", st)
    if not st["enabled"]:
        print("[skip] voice not installed (run setup_voice.py) — nothing to test.")
        return 0

    ok = True
    # Plain words only -- whisper renders spoken numbers as digits ("one two" ->
    # "12"), so a numeric phrase would round-trip correctly yet fail a word match.
    phrase = "the quick brown fox jumps over the lazy dog"

    # --- TTS -----------------------------------------------------------------
    try:
        wav = voice.synthesize(phrase)
    except Exception as exc:
        print(f"[FAIL] synthesize raised: {type(exc).__name__}: {exc}")
        return 1
    if wav[:4] != b"RIFF" or b"WAVE" not in wav[:16] or len(wav) < 1000:
        print(f"[FAIL] piper WAV looks wrong (len={len(wav)}, head={wav[:4]!r})")
        ok = False
    else:
        print(f"[ok] piper synthesized {len(wav)} bytes of WAV")

    # --- STT round-trip ------------------------------------------------------
    try:
        text = voice.transcribe(_to_16k_mono(wav))
    except Exception as exc:
        print(f"[FAIL] transcribe raised: {type(exc).__name__}: {exc}")
        return 1
    print("[stt] whisper heard:", repr(text))
    low = text.lower()
    expect = ("quick", "brown", "fox", "lazy", "dog")
    hits = sum(w in low for w in expect)
    if hits < 3:
        print(f"[FAIL] round-trip lost the phrase (matched {hits}/{len(expect)} words)")
        ok = False
    else:
        print(f"[ok] round-trip recognised {hits}/{len(expect)} expected words")

    # --- clean_for_speech strips the ground-truth footer ---------------------
    cleaned = voice.clean_for_speech(
        "Done.\n\n[verified actions this turn] scripter: query_data")
    if "[verified" in cleaned or cleaned.strip() != "Done.":
        print(f"[FAIL] clean_for_speech didn't strip the footer: {cleaned!r}")
        ok = False
    else:
        print("[ok] clean_for_speech strips the '[verified actions…]' footer")

    print("\n" + ("[PASS] voice: piper TTS + whisper STT round-trip works."
                  if ok else "[FAIL] see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_run())
