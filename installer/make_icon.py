"""
Turn a single source image into the icon files the project needs.

    py -3.11 -m pip install pillow
    py -3.11 installer/make_icon.py path/to/your-design.png

Writes, next to this script:

    app.ico    multi-resolution Windows icon (16 to 256 px)
    app.png    512 px copy, used for the macOS build and anywhere else
    app.icns   macOS icon, only if this is run on a Mac (needs iconutil)

A .ico is a container holding several bitmaps. Windows picks whichever size it
needs, so a single 256 px image scaled down on the fly looks muddy in the
taskbar. Generating each size explicitly avoids that.

SOURCE IMAGE
    Square, at least 512x512, ideally 1024x1024.
    PNG with transparency, or a solid background if you prefer.
    Keep detail chunky - anything thinner than about 1/30th of the width
    disappears entirely at 16 px.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    src_path = Path(sys.argv[1]).expanduser()
    if not src_path.exists():
        print(f"No such file: {src_path}")
        return 1

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is needed.\n")
        print(f'  "{sys.executable}" -m pip install pillow')
        return 1

    im = Image.open(src_path).convert("RGBA")
    w, h = im.size
    print(f"source: {src_path.name}  {w}x{h}")

    if w != h:
        side = max(w, h)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(im, ((side - w) // 2, (side - h) // 2))
        im = canvas
        print(f"  not square - padded to {side}x{side}")

    if im.size[0] < 256:
        print(f"  WARNING: only {im.size[0]}px. 512 or 1024 gives a much better"
              f" 256px icon.")

    ico = HERE / "app.ico"
    im.save(ico, format="ICO",
            sizes=[(s, s) for s in ICO_SIZES if s <= im.size[0]])
    print(f"wrote {ico.name}  ({ico.stat().st_size/1024:.0f} KB, "
          f"sizes {', '.join(str(s) for s in ICO_SIZES if s <= im.size[0])})")

    png = HERE / "app.png"
    im.resize((512, 512), Image.LANCZOS).save(png)
    print(f"wrote {png.name}")

    if platform.system() == "Darwin" and shutil.which("iconutil"):
        iconset = HERE / "app.iconset"
        if iconset.exists():
            shutil.rmtree(iconset)
        iconset.mkdir()
        for s in (16, 32, 64, 128, 256, 512):
            im.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
            im.resize((s * 2, s * 2), Image.LANCZOS).save(
                iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset),
                        "-o", str(HERE / "app.icns")], check=False)
        shutil.rmtree(iconset)
        if (HERE / "app.icns").exists():
            print("wrote app.icns")
    elif platform.system() != "Darwin":
        print("(app.icns skipped - build that on a Mac if you need one)")

    print("\nNow rebuild:  py -3.11 installer/build.py --console")
    print("The icon is picked up automatically once app.ico exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
