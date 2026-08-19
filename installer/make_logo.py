"""
Draw the application logo, then hand it to make_icon.py.

    py -3.11 installer/make_logo.py

Writes app-master.png (1024 px) beside this script and regenerates app.ico and
app.png from it. Committing the generator rather than only the images means the
logo can be adjusted or re-rendered at any size later, which a flattened PNG
would not allow.

THE MARK
    A magnifying glass held over a person, with a scan line across the lens.
    "Examine this face", rather than the padlock-and-shield vocabulary of
    security software - which would overpromise, given the detector's measured
    limits.

    Everything is built from circles and rounded rectangles so it survives
    being scaled to 16 px, where fine detail disappears completely. Drawn at
    4x and downsampled, because PIL does not anti-alias shape edges.

    Colours are the installer's own palette (installer.py).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

BG     = (0x1e, 0x1e, 0x2e)     # window background
ACCENT = (0x5a, 0x4f, 0xcf)     # buttons, progress bar
LIGHT  = (0xe8, 0xe8, 0xf0)     # body text
AMBER  = (0xED, 0x6C, 0x02)     # the out-of-scope warning banner

SS = 4                          # supersample factor
S  = 512 * SS


def px(f: float) -> int:
    return int(round(f * S))


def draw(background: bool = True):
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    if background:
        d.rounded_rectangle([0, 0, S - 1, S - 1], radius=px(0.225), fill=BG)

    lx, ly = px(0.435), px(0.425)       # lens centre
    lr     = px(0.275)                  # outer radius
    ring   = px(0.070)                  # ring thickness
    inner  = lr - ring                  # glass radius

    # Handle first, so the ring covers where the two meet.
    hw = px(0.098)
    a  = 0.7071                         # cos 45 degrees
    hx0, hy0 = lx + int((lr - ring * 0.4) * a), ly + int((lr - ring * 0.4) * a)
    hx1, hy1 = px(0.865), px(0.865)
    d.line([hx0, hy0, hx1, hy1], fill=LIGHT, width=hw)
    d.ellipse([hx1 - hw // 2, hy1 - hw // 2,
               hx1 + hw // 2, hy1 + hw // 2], fill=LIGHT)   # rounded cap

    # Glass stays dark so the figure inside keeps its contrast.
    d.ellipse([lx - inner, ly - inner, lx + inner, ly + inner], fill=BG)

    # Figure and scan line, clipped to the glass.
    lay = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ld  = ImageDraw.Draw(lay)

    hr  = px(0.081)
    hcy = ly - px(0.078)
    ld.ellipse([lx - hr, hcy - hr, lx + hr, hcy + hr], fill=ACCENT)

    sw, st = px(0.152), hcy + px(0.048)
    ld.rounded_rectangle([lx - sw, st, lx + sw, st + px(0.30)],
                         radius=px(0.070), fill=ACCENT)

    sy = ly + px(0.052)
    ld.rectangle([0, sy, S, sy + px(0.034)], fill=AMBER)

    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse(
        [lx - inner, ly - inner, lx + inner, ly + inner], fill=255)
    im.paste(lay, (0, 0),
             Image.composite(mask, Image.new("L", (S, S), 0), lay.split()[3]))

    # Ring last, over both handle and glass.
    d.ellipse([lx - lr + ring // 2, ly - lr + ring // 2,
               lx + lr - ring // 2, ly + lr - ring // 2],
              outline=LIGHT, width=ring)
    return im


def main() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is needed.\n")
        print(f'  "{sys.executable}" -m pip install pillow')
        return 1

    master = HERE / "app-master.png"
    draw(True).resize((1024, 1024), Image.LANCZOS).save(master)
    print(f"wrote {master.name}  (1024x1024)")

    flat = HERE / "app-transparent.png"
    draw(False).resize((1024, 1024), Image.LANCZOS).save(flat)
    print(f"wrote {flat.name}  (no background, for docs and slides)")

    r = subprocess.run([sys.executable, str(HERE / "make_icon.py"), str(master)])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
