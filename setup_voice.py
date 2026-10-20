"""Download the prebuilt whisper.cpp (STT) + piper (TTS) binaries and the model/
voice for Hermes voice, into ``agent/voice/``. Stdlib only (urllib + zipfile),
idempotent (skips anything already present).

Layout produced:
  agent/voice/bin/whisper-cpu/whisper-cli.exe   (whisper-blas-bin-x64.zip)
  agent/voice/bin/whisper-cuda/whisper-cli.exe  (whisper-cublas-12.4.0-bin-x64.zip)
  agent/voice/bin/piper/piper.exe               (piper_windows_amd64.zip)
  agent/voice/models/ggml-base.en.bin
  agent/voice/voices/en_US-amy-medium.onnx (+ .onnx.json)

Usage:
  python setup_voice.py                # CPU + CUDA whisper, piper, model, voice
  python setup_voice.py --whisper cpu  # skip the large CUDA build
  python setup_voice.py --whisper cuda # only the CUDA whisper build
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE = os.path.join(HERE, "voice")
BIN = os.path.join(VOICE, "bin")
MODELS = os.path.join(VOICE, "models")
VOICES = os.path.join(VOICE, "voices")

WHISPER_VER = "v1.8.6"
WHISPER_BLAS = (f"https://github.com/ggml-org/whisper.cpp/releases/download/"
                f"{WHISPER_VER}/whisper-blas-bin-x64.zip")
WHISPER_CUBLAS = (f"https://github.com/ggml-org/whisper.cpp/releases/download/"
                  f"{WHISPER_VER}/whisper-cublas-12.4.0-bin-x64.zip")
PIPER_ZIP = ("https://github.com/rhasspy/piper/releases/download/"
             "2023.11.14-2/piper_windows_amd64.zip")
GGML_BASE_EN = ("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
                "ggml-base.en.bin")
VOICE_ONNX = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/en/"
              "en_US/amy/medium/en_US-amy-medium.onnx")
VOICE_JSON = VOICE_ONNX + ".json"

_WHISPER_EXE_CANDIDATES = ("whisper-cli.exe", "main.exe")


def _download(url: str, dest: str) -> str:
    """Stream ``url`` to ``dest`` with a simple progress meter. Skips if present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [skip] {os.path.relpath(dest, HERE)} already present")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  [get ] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-voice-setup"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done, last_mark = 0, -1
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                mark = done * 100 // total // 10        # one line per 10%
                if mark != last_mark:
                    last_mark = mark
                    print(f"         {mark * 10:3d}%  ({done >> 20}/{total >> 20} MB)")
            elif done % (8 << 20) < (1 << 16):           # unknown size: ~every 8 MB
                print(f"         {done >> 20} MB")
    os.replace(tmp, dest)
    return dest


def _extract(zip_path: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)


def _find(root: str, names) -> str | None:
    for base, _dirs, files in os.walk(root):
        for n in names:
            if n in files:
                return os.path.join(base, n)
    return None


def _whisper(url: str, subdir: str, label: str) -> None:
    print(f"\n== whisper.cpp ({label}) ==")
    target = os.path.join(BIN, subdir)
    if _find(target, _WHISPER_EXE_CANDIDATES):
        print(f"  [skip] {label} whisper-cli already extracted")
        return
    zip_path = _download(url, os.path.join(VOICE, "_dl", os.path.basename(url)))
    _extract(zip_path, target)
    exe = _find(target, _WHISPER_EXE_CANDIDATES)
    print(f"  [ok  ] {os.path.relpath(exe, HERE) if exe else 'whisper-cli NOT FOUND'}")


def _piper() -> None:
    print("\n== piper (TTS) ==")
    if _find(BIN, ("piper.exe",)):
        print("  [skip] piper.exe already extracted")
        return
    zip_path = _download(PIPER_ZIP, os.path.join(VOICE, "_dl", "piper_windows_amd64.zip"))
    _extract(zip_path, BIN)
    exe = _find(BIN, ("piper.exe",))
    print(f"  [ok  ] {os.path.relpath(exe, HERE) if exe else 'piper.exe NOT FOUND'}")


def _model_and_voice() -> None:
    print("\n== whisper model (ggml-base.en) ==")
    _download(GGML_BASE_EN, os.path.join(MODELS, "ggml-base.en.bin"))
    print("\n== piper voice (en_US-amy-medium) ==")
    _download(VOICE_ONNX, os.path.join(VOICES, "en_US-amy-medium.onnx"))
    _download(VOICE_JSON, os.path.join(VOICES, "en_US-amy-medium.onnx.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Hermes voice binaries/models.")
    ap.add_argument("--whisper", choices=("cpu", "cuda", "both"), default="both",
                    help="which whisper build(s) to fetch (default: both)")
    args = ap.parse_args()

    print(f"Installing Hermes voice into {VOICE}")
    if args.whisper in ("cpu", "both"):
        _whisper(WHISPER_BLAS, "whisper-cpu", "CPU/BLAS")
    if args.whisper in ("cuda", "both"):
        _whisper(WHISPER_CUBLAS, "whisper-cuda", "CUDA/cuBLAS")
    _piper()
    _model_and_voice()

    print("\nDone. Set WHISPER_ACCEL=cpu|cuda to pick the build (default cpu).")
    print("Verify with:  .venv/Scripts/python.exe -m multi_agent.test_voice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
