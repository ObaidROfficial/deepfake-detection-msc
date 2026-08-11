"""
Compile the installer and uninstaller into standalone executables.

    pip install pyinstaller
    python installer/build.py

Run this on each platform you want to support - PyInstaller cannot
cross-compile, so a Windows .exe must be built on Windows and a macOS binary
on macOS. Output lands in installer/dist/.

WHY THE BUNDLE IS SMALL
Only the installer is compiled, and it imports nothing beyond tkinter and the
standard library. PyTorch, Gradio and the rest are downloaded at install time
into a virtual environment on the user's machine. Bundling them instead would
push each download past a gigabyte and make the build fragile.

A NOTE ON CODE SIGNING
These executables are unsigned. On Windows, SmartScreen will show "Windows
protected your PC" and the user must click More info -> Run anyway. On macOS,
Gatekeeper refuses to open them at all on a double-click; the user must
right-click and choose Open, then confirm. Signing requires a paid certificate
from Microsoft or Apple. This is documented for the user in
installer/README.md rather than left as a surprise.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_NAME = "OBI-DeepFake-Image-Detector"

TARGETS = [
    ("installer.py",   f"{APP_NAME}-Setup"),
    ("uninstaller.py", f"{APP_NAME}-Uninstall"),
]


def main() -> int:
    if shutil.which("pyinstaller") is None:
        print("PyInstaller is not installed.\n    pip install pyinstaller")
        return 1

    system = platform.system()
    print(f"Building on {system} ({platform.machine()})")
    print(f"Python {sys.version.split()[0]}\n")

    for script, name in TARGETS:
        src = HERE / script
        if not src.exists():
            print(f"  missing: {src}")
            return 1

        cmd = [
            "pyinstaller",
            "--onefile",            # single self-contained executable
            "--noconsole",          # GUI app - no terminal window behind it
            "--clean",
            "--name", name,
            "--distpath", str(HERE / "dist"),
            "--workpath", str(HERE / "build"),
            "--specpath", str(HERE / "build"),
            # tkinter is picked up automatically, but these are sometimes
            # missed on minimal Python installs
            "--hidden-import", "tkinter",
            "--hidden-import", "tkinter.filedialog",
            "--hidden-import", "tkinter.messagebox",
            "--hidden-import", "tkinter.ttk",
            # exclude anything heavy that might be importable in the build env
            "--exclude-module", "torch",
            "--exclude-module", "torchvision",
            "--exclude-module", "numpy",
            "--exclude-module", "pandas",
            "--exclude-module", "matplotlib",
            "--exclude-module", "cv2",
            "--exclude-module", "gradio",
            "--exclude-module", "PIL",
            "--exclude-module", "scipy",
            "--exclude-module", "sklearn",
            str(src),
        ]

        print(f"Building {name}...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  FAILED: {name}")
            return result.returncode

        produced = list((HERE / "dist").glob(f"{name}*"))
        for p in produced:
            size = p.stat().st_size / 1e6 if p.is_file() else 0
            print(f"  built  {p.name}  ({size:.1f} MB)" if size
                  else f"  built  {p.name}")
    print("\nOutput in", HERE / "dist")
    print("\nBefore publishing, test the executable on a machine that has")
    print("never had this project installed. That is the only way to catch")
    print("a dependency the build environment happened to provide.")

    if system == "Darwin":
        print("\nmacOS: users must right-click -> Open the first time.")
        print("Gatekeeper blocks unsigned binaries on double-click.")
    elif system == "Windows":
        print("\nWindows: SmartScreen will warn on first run.")
        print("Users click More info -> Run anyway.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
