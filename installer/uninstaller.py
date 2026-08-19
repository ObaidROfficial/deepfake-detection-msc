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
APP_VERSION = "3.2"          # keep in step with installer.py
PUBLISHER = "Obaid"

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
HOVER = "#6d61e0"
NEUTRAL = "#3a3a52"
NEUTRAL_HOVER = "#4a4a68"
DANGER_HOVER = "#e05545"

# Tk on macOS renders tk.Button as a native Aqua control and silently ignores
# bg/fg, which is why the buttons came out grey there while Windows showed the
# theme colours. A Frame with a Label inside is drawn by Tk itself, so it looks
# identical on both platforms.
UI_FONT = ("Segoe UI" if platform.system() == "Windows"
           else "Helvetica Neue" if platform.system() == "Darwin"
           else "DejaVu Sans")


class FlatButton(tk.Frame):
    def __init__(self, parent, text, command, bg, hover, fg="white",
                 bold=False, width=14):
        super().__init__(parent, bg=bg, cursor="hand2",
                         highlightthickness=0, bd=0)
        self.command, self.bg, self.hover = command, bg, hover
        self.label = tk.Label(
            self, text=text, bg=bg, fg=fg, padx=18, pady=8, width=width,
            font=(UI_FONT, 10, "bold" if bold else "normal"))
        self.label.pack(fill="both", expand=True)
        for w in (self, self.label):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)

    def _paint(self, colour):
        self.configure(bg=colour)
        self.label.configure(bg=colour)

    def _enter(self, _=None): self._paint(self.hover)
    def _leave(self, _=None): self._paint(self.bg)

    def _click(self, _=None):
        if callable(self.command):
            self.command()



def bundled(name: str) -> Path:
    """Beside this script, or inside the frozen bundle - see installer.py."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def apply_window_icon(window) -> None:
    """Set the title-bar icon, quietly doing nothing if there isn't one yet."""
    try:
        ico = bundled("app.ico")
        if ico.exists():
            window.iconbitmap(default=str(ico))
    except Exception:
        pass


def desktop_dir() -> Path:
    """
    The user's real Desktop - see the matching helper in installer.py.

    Path.home()/"Desktop" is wrong wherever OneDrive or a roaming profile has
    redirected the folder, which would leave the shortcut behind on uninstall.
    """
    if platform.system() == "Windows":
        try:
            import winreg
            key = (r"Software\Microsoft\Windows\CurrentVersion"
                   r"\Explorer\User Shell Folders")
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
                raw, _ = winreg.QueryValueEx(k, "Desktop")
            resolved = Path(os.path.expandvars(raw))
            if resolved.is_dir():
                return resolved
        except Exception:
            pass
    fallback = Path.home() / "Desktop"
    return fallback if fallback.is_dir() else Path.home()


def app_root() -> Path:
    """
    Where the application lives.

    Frozen by PyInstaller: sys.executable sits inside the install folder.
    Running as a script: this file is in installer/ one level down.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


# A working copy has a .git directory. A downloaded archive never does, because
# GitHub strips it - which makes it the one marker that reliably separates a
# source checkout from an installation.
#
# Earlier versions of this guard also looked for docs/, notebooks/ and
# results/. That was wrong: the installer used to extract the entire archive,
# so a legitimate install contained all three and would have refused to
# uninstall itself. The installer now copies only the runtime files, but the
# guard should not depend on that - if the two ever drift apart the failure is
# silent, and it fails in the direction of "cannot uninstall".
SOURCE_MARKER = ".git"

# What an installation looks like from the outside.
INSTALL_MARKERS = ("app.py", "venv", "models")


def refuse_reason(root: Path) -> str | None:
    """
    Why this folder must not be deleted, or None if it is safe to proceed.

    Two separate checks. The first stops the uninstaller eating a clone: run
    this as a script from a checkout - exactly what you would do to test it -
    and app_root() resolves to the repository, source and git history included.
    Parent directories are checked too, because the compiled uninstaller sits
    in installer/dist/ during development.

    The second is the opposite worry: a folder that is not an installation at
    all, where deleting everything would be equally unwelcome.
    """
    for depth, cand in enumerate([root, *root.parents]):
        if depth > 3:
            break
        if (cand / SOURCE_MARKER).exists():
            where = "this folder" if cand == root else f"{cand}"
            return (f"{where} is a git working copy.\n\n"
                    f"That means it holds the project's source, the written "
                    f"report and its version history, not just the "
                    f"application. Uninstalling would delete all of it.")

    present = [m for m in INSTALL_MARKERS if (root / m).exists()]
    if not present:
        return (f"{root}\n\ndoes not look like an installation. None of "
                f"{', '.join(INSTALL_MARKERS)} are here, so there is nothing "
                f"recognisable to remove and deleting the folder could take "
                f"something else with it.")
    return None


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
        apply_window_icon(self)
        self.geometry("600x580")
        self.minsize(560, 520)
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
                 font=(UI_FONT, 17, "bold"), anchor="w").pack(
                 fill="x", padx=28, pady=(26, 2))
        if sub:
            tk.Label(self.container, text=sub, bg=BG, fg=MUTED,
                     font=(UI_FONT, 9), anchor="w").pack(fill="x", padx=28)

    def _button(self, parent, text, cmd, style="normal"):
        base  = {"normal": NEUTRAL, "primary": ACCENT, "danger": DANGER}
        hover = {"normal": NEUTRAL_HOVER, "primary": HOVER, "danger": DANGER_HOVER}
        return FlatButton(parent, text, cmd, bg=base[style],
                          hover=hover[style], bold=style != "normal")

    # -- screens ------------------------------------------------------------
    def show_refuse(self, reason: str):
        """Shown instead of the confirm screen when deleting would be wrong."""
        self._clear()
        self._header("Nothing has been deleted",
                     "This does not look like something I should remove.")
        box = tk.Text(self.container, bg="#252539", fg=FG, relief="flat",
                      font=(UI_FONT, 10), wrap="word", padx=14, pady=12)
        box.pack(fill="both", expand=True, padx=28, pady=18)
        box.insert("1.0",
                   reason
                   + "\n\nIf you meant to remove a source checkout, use "
                     "uninstall.ps1 or uninstall.sh in the project root - they "
                     "are written for that. This uninstaller is for the copy "
                     "placed by the graphical installer.")
        box.configure(state="disabled")
        nav = tk.Frame(self.container, bg=BG)
        nav.pack(side="bottom", fill="x", padx=28, pady=(0, 22))
        self._button(nav, "Close", self.destroy, style="primary").pack(side="right")

    def show_confirm(self):
        reason = refuse_reason(self.root_dir)
        if reason:
            self.show_refuse(reason)
            return

        self._clear()
        self._header(f"Uninstall {APP_NAME}",
                     f"Version {APP_VERSION}  -  By {PUBLISHER}")

        box = tk.Text(self.container, bg="#252539", fg=FG, relief="flat",
                      font=(UI_FONT, 10), wrap="word", padx=14, pady=12,
                      height=12)
        box.pack(fill="both", expand=True, padx=28, pady=(16, 10))
        box.insert("1.0", UNINSTALL_TEXT)
        box.configure(state="disabled")

        size = dir_size(self.root_dir)
        tk.Label(self.container,
                 text=f"Location:  {self.root_dir}\nSpace to reclaim:  {human(size)}",
                 bg=BG, fg=MUTED, font=(UI_FONT, 9), anchor="w",
                 justify="left").pack(fill="x", padx=28)

        tk.Checkbutton(
            self.container,
            text="Also clear shared PyTorch and Hugging Face caches",
            variable=self.remove_caches, bg=BG, fg=FG, selectcolor="#252539",
            activebackground=BG, activeforeground=FG, font=(UI_FONT, 9),
            borderwidth=0, highlightthickness=0, anchor="w").pack(
            fill="x", padx=26, pady=(12, 0))
        tk.Label(self.container,
                 text="      Other Python software on this computer uses these too,\n"
                      "      and would have to download them again.",
                 bg=BG, fg=MUTED, font=(UI_FONT, 8), anchor="w",
                 justify="left").pack(fill="x", padx=28)

        nav = tk.Frame(self.container, bg=BG)
        nav.pack(side="bottom", fill="x", padx=28, pady=(16, 22))
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
                      font=(UI_FONT, 10), wrap="word", padx=14, pady=12)
        box.pack(fill="both", expand=True, padx=28, pady=18)
        box.insert("1.0", DONE_TEXT if ok else
                   "Some files could not be removed:\n\n" + message +
                   "\n\nClose any open windows using the application and try "
                   "again, or delete the folder manually:\n\n" +
                   str(self.root_dir))
        box.configure(state="disabled")
        nav = tk.Frame(self.container, bg=BG)
        nav.pack(side="bottom", fill="x", padx=28, pady=(0, 22))
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

            # Desktop shortcut first - cheap, and confirms we have the right app.
            # Must resolve the Desktop the same way the installer did, or a
            # redirected Desktop leaves the shortcut behind after uninstalling.
            desktop = desktop_dir()
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
