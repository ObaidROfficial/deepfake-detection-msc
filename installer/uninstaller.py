"""
Graphical uninstaller for the OBI DeepFake Image Detector.

Installed alongside the application and compiled to a standalone executable by
installer/build.py, so it works even after the virtual environment is removed -
an uninstaller that depends on what it is deleting is not much use.

WHAT IT REMOVES
    The application folder, its virtual environment, the model weights and the
    desktop shortcut.

WHAT IT LEAVES ALONE BY DEFAULT
    The shared PyTorch and Hugging Face caches in the user's home directory.
    These are used by any other Python project on the machine, so removing them
    would force unrelated software to re-download. There is a tick-box for
    users who want them gone, with the consequence spelled out.

    A Kaggle credential, if one exists, is never touched. It belongs to the
    user, not to this application.
"""

from __future__ import annotations

import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

# ---------------------------------------------------------------------------
# TEXT SHOWN TO THE USER
# ---------------------------------------------------------------------------

APP_NAME = "OBI DeepFake Image Detector"

UNINSTALL_TEXT = """\
Thank you for trying the OBI DeepFake Image Detector. I hope you found it
useful.

This will remove the program from your system. The following will be deleted:

  - the application and its Python environment
  - the downloaded model files
  - the desktop shortcut

Nothing else on your computer is affected. Any images you classified were
processed on your own machine and were never uploaded, copied or stored
anywhere.

If something did not work as expected, I would genuinely like to know:
github.com/ObaidROfficial/deepfake-detection-msc/issues"""

DONE_TEXT = """\
The OBI DeepFake Image Detector has been removed from your computer.

Thank you again for trying it. You can close this window."""

# ---------------------------------------------------------------------------

BG = "#1e1e2e"
FG = "#e8e8f0"
ACCENT = "#5a4fcf"
DANGER = "#c0392b"
MUTED = "#9a9ab0"


def app_root() -> Path:
    """
    Where the application lives.

    Frozen by PyInstaller: sys.executable sits inside the install folder.
    Running as a script: this file is in installer/ one level down.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


class Uninstaller(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Uninstall {APP_NAME}")
        self.geometry("580x460")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.root_dir = app_root()
        self.remove_caches = tk.BooleanVar(value=False)
        self.log_queue: queue.Queue = queue.Queue()

        self._style()
        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)
        self.show_confirm()

    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TProgressbar", troughcolor="#2a2a3e",
                    background=ACCENT, borderwidth=0, thickness=18)

    def _clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def _header(self, text, sub=""):
        tk.Label(self.container, text=text, bg=BG, fg=FG,
                 font=("Segoe UI", 17, "bold"), anchor="w").pack(
                 fill="x", padx=28, pady=(26, 2))
        if sub:
            tk.Label(self.container, text=sub, bg=BG, fg=MUTED,
                     font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=28)

    def _button(self, parent, text, cmd, style="normal"):
        colours = {"normal": "#3a3a52", "primary": ACCENT, "danger": DANGER}
        return tk.Button(
            parent, text=text, command=cmd, width=12, relief="flat",
            bg=colours[style], fg="white", activeforeground="white",
            activebackground="#6d61e0" if style != "danger" else "#e05545",
            font=("Segoe UI", 10, "bold" if style != "normal" else "normal"),
            cursor="hand2", borderwidth=0, pady=6,
        )

    # -- screens ------------------------------------------------------------
    def show_confirm(self):
        self._clear()
        self._header(f"Uninstall {APP_NAME}")

        box = tk.Text(self.container, bg="#252539", fg=FG, relief="flat",
                      font=("Segoe UI", 10), wrap="word", padx=14, pady=12,
                      height=12)
        box.pack(fill="both", expand=True, padx=28, pady=(16, 10))
        box.insert("1.0", UNINSTALL_TEXT)
        box.configure(state="disabled")

        size = dir_size(self.root_dir)
        tk.Label(self.container,
                 text=f"Location:  {self.root_dir}\nSpace to reclaim:  {human(size)}",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9), anchor="w",
                 justify="left").pack(fill="x", padx=28)

        tk.Checkbutton(
            self.container,
            text="Also clear shared PyTorch and Hugging Face caches",
            variable=self.remove_caches, bg=BG, fg=FG, selectcolor="#252539",
            activebackground=BG, activeforeground=FG, font=("Segoe UI", 9),
            borderwidth=0, highlightthickness=0, anchor="w").pack(
            fill="x", padx=26, pady=(12, 0))
        tk.Label(self.container,
                 text="      Other Python software on this computer uses these too,\n"
                      "      and would have to download them again.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8), anchor="w",
                 justify="left").pack(fill="x", padx=28)

        nav = tk.Frame(self.container, bg=BG)
        nav.pack(fill="x", padx=28, pady=(16, 22))
        self._button(nav, "Keep it", self.destroy).pack(side="right", padx=(8, 0))
        self._button(nav, "Uninstall", self.start, style="danger").pack(side="right")

    def show_progress(self):
        self._clear()
        self._header("Removing")
        self.bar = ttk.Progressbar(self.container, mode="indeterminate",
                                   style="TProgressbar")
        self.bar.pack(fill="x", padx=28, pady=(18, 10))
        self.bar.start(12)
        self.logbox = tk.Text(self.container, bg="#181825", fg=MUTED,
                              relief="flat", font=("Consolas", 8), wrap="word",
                              padx=10, pady=8)
        self.logbox.pack(fill="both", expand=True, padx=28, pady=(0, 22))
        self.logbox.configure(state="disabled")

    def show_done(self, ok, message=""):
        self._clear()
        self._header("Finished" if ok else "Could not finish")
        box = tk.Text(self.container, bg="#252539", fg=FG, relief="flat",
                      font=("Segoe UI", 10), wrap="word", padx=14, pady=12)
        box.pack(fill="both", expand=True, padx=28, pady=18)
        box.insert("1.0", DONE_TEXT if ok else
                   "Some files could not be removed:\n\n" + message +
                   "\n\nClose any open windows using the application and try "
                   "again, or delete the folder manually:\n\n" +
                   str(self.root_dir))
        box.configure(state="disabled")
        nav = tk.Frame(self.container, bg=BG)
        nav.pack(fill="x", padx=28, pady=(0, 22))
        self._button(nav, "Close", self.destroy, style="primary").pack(side="right")

    # -- worker -------------------------------------------------------------
    def log(self, m):
        self.log_queue.put(("log", m))

    def start(self):
        self.show_progress()
        threading.Thread(target=self._worker, daemon=True).start()
        self.after(80, self._drain)

    def _drain(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self.logbox.configure(state="normal")
                    self.logbox.insert("end", str(payload) + "\n")
                    self.logbox.see("end")
                    self.logbox.configure(state="disabled")
                elif kind == "done":
                    self.bar.stop()
                    self.show_done(True)
                    return
                elif kind == "error":
                    self.bar.stop()
                    self.show_done(False, str(payload))
                    return
        except queue.Empty:
            pass
        self.after(80, self._drain)

    def _rm(self, p: Path, label: str):
        if not p.exists():
            return
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=False)
            else:
                p.unlink()
            self.log(f"removed  {label}")
        except Exception as e:
            self.log(f"FAILED   {label}: {e}")
            self.failures.append(f"{label}: {e}")

    def _worker(self):
        self.failures: list[str] = []
        try:
            root = self.root_dir
            self.log(f"Target: {root}\n")

            # Desktop shortcut first - cheap, and confirms we have the right app
            desktop = Path.home() / "Desktop"
            for name in (f"{APP_NAME}.lnk", f"{APP_NAME}.command"):
                self._rm(desktop / name, f"desktop shortcut ({name})")

            # Application contents. The uninstaller executable may be running
            # from inside this folder, so its own file is skipped and removed
            # by the trailing self-delete below.
            me = Path(sys.executable).resolve() if getattr(sys, "frozen", False) else None

            def is_self(path: Path) -> bool:
                """True if the running executable is this path, or inside it."""
                if me is None:
                    return False
                resolved = path.resolve()
                if resolved == me:
                    return True
                try:
                    return me.is_relative_to(resolved)
                except (ValueError, AttributeError):
                    # is_relative_to needs 3.9+; fall back to a string check
                    return str(me).startswith(str(resolved) + os.sep)

            for item in sorted(root.iterdir()):
                if is_self(item):
                    self.log(f"skipped  {item.name} (contains this uninstaller)")
                    continue
                self._rm(item, item.name)

            if self.remove_caches.get():
                self.log("")
                home = Path.home()
                for p, label in [
                    (home / ".cache" / "torch", "PyTorch cache"),
                    (home / ".cache" / "huggingface", "Hugging Face cache"),
                    (home / "Library" / "Caches" / "torch", "PyTorch cache (macOS)"),
                ]:
                    self._rm(p, label)

            if self.failures:
                self.log_queue.put(("error", "\n".join(self.failures)))
                return

            self.log("\nDone.")
            self._schedule_self_delete(root)
            self.log_queue.put(("done", None))

        except Exception as e:
            self.log_queue.put(("error", e))

    def _schedule_self_delete(self, root: Path):
        """
        A running executable cannot delete itself on Windows, so hand the job
        to a detached shell that waits for this process to exit first.
        """
        if not getattr(sys, "frozen", False):
            return
        try:
            if platform.system() == "Windows":
                subprocess.Popen(
                    ["cmd", "/c", "timeout /t 3 /nobreak >nul && "
                     f'rmdir /s /q "{root}"'],
                    creationflags=0x08000000,  # CREATE_NO_WINDOW
                )
            else:
                subprocess.Popen(
                    ["/bin/sh", "-c", f'sleep 3; rm -rf "{root}"'],
                    start_new_session=True,
                )
            self.log("Remaining files will be removed when this window closes.")
        except Exception as e:
            self.log(f"Could not schedule final cleanup: {e}")


if __name__ == "__main__":
    Uninstaller().mainloop()
