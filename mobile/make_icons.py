"""Generate the PWA / Android app icons for the Paimon mobile app.

Deterministic, dependency-light (Pillow only) so the icon set can be rebuilt
without any design tooling. Draws a minimalist mark -- a rounded gradient tile
with a white speech bubble + a little "spark" -- at every size the manifest /
Android / iOS need:

  icon-192.png, icon-512.png           -- standard PWA icons
  icon-maskable-512.png                -- Android adaptive (safe-zone padded)
  apple-touch-icon.png (180)           -- iOS home screen
  favicon-32.png, favicon-16.png       -- browser tab

Run:  agent/.venv/Scripts/python.exe agent/mobile/make_icons.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(__file__)
ICONS = os.path.join(HERE, "icons")

# Brand gradient (indigo -> violet), matching the app's --accent.
TOP = (99, 102, 241)      # #6366f1
BOT = (139, 92, 246)      # #8b5cf6
WHITE = (255, 255, 255)


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _gradient(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        c = _lerp(TOP, BOT, y / max(1, size - 1))
        for x in range(size):
            px[x, y] = c
    return img


def _rounded_mask(size: int, radius_frac: float) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    r = int(size * radius_frac)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=255)
    return m


def _speech_bubble(draw: ImageDraw.ImageDraw, cx: int, cy: int, w: int, h: int):
    """A white rounded speech bubble with a tail + three dots."""
    r = h * 0.34
    left, top, right, bottom = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    draw.rounded_rectangle([left, top, right, bottom], radius=r, fill=WHITE)
    # tail (lower-left)
    tx, ty = left + w * 0.22, bottom - 2
    draw.polygon([(tx, ty), (tx + h * 0.42, ty), (tx + h * 0.08, ty + h * 0.42)],
                 fill=WHITE)
    # three "typing" dots
    dot_r = h * 0.085
    for i, fx in enumerate((0.30, 0.5, 0.70)):
        dcx = left + w * fx
        draw.ellipse([dcx - dot_r, cy - dot_r, dcx + dot_r, cy + dot_r], fill=BOT)


def _spark(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float):
    """A little 4-point sparkle (the 'assistant' accent)."""
    pts = []
    for k in range(8):
        ang = math.pi / 4 * k
        rad = r if k % 2 == 0 else r * 0.38
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=WHITE)


def _make(size: int, maskable: bool = False) -> Image.Image:
    base = _gradient(size).convert("RGBA")
    draw = ImageDraw.Draw(base)
    # Maskable icons must keep the mark inside the central 80% safe zone.
    scale = 0.62 if maskable else 0.74
    bw, bh = size * scale, size * scale * 0.74
    _speech_bubble(draw, size / 2, size / 2 * 1.02, bw, bh)
    _spark(draw, size * (0.5 + scale * 0.30), size * (0.5 - scale * 0.34),
           size * 0.07)
    if maskable:
        return base  # full-bleed tile (Android masks it)
    mask = _rounded_mask(size, 0.22)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(base, (0, 0), mask)
    return out


def main() -> None:
    os.makedirs(ICONS, exist_ok=True)
    jobs = [
        ("icon-192.png", 192, False),
        ("icon-512.png", 512, False),
        ("icon-maskable-512.png", 512, True),
        ("apple-touch-icon.png", 180, False),
        ("favicon-32.png", 32, False),
        ("favicon-16.png", 16, False),
    ]
    for name, size, maskable in jobs:
        img = _make(size, maskable)
        img.save(os.path.join(ICONS, name))
        print(f"wrote icons/{name} ({size}px{', maskable' if maskable else ''})")


if __name__ == "__main__":
    main()
