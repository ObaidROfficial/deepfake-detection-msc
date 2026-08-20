"""
OBI DeepFake Image Detector - desktop interface.

    python app.py

A real application window. Earlier versions ran a Gradio web server and opened
a browser tab, which meant importing fastapi and uvicorn and waiting ten to
twenty seconds before anything appeared on screen. This starts in two or three,
because the only things it loads are torch and the model itself.

Inference is imported from src/predict.py rather than reimplemented, so this
window and the command line tool cannot drift apart. A difference in resizing
or normalisation between the two would silently produce different verdicts for
the same image.

The model is loaded on a worker thread. Doing it on the main thread freezes the
window for as long as torch takes to wake up, which looks like a crash.
"""

from __future__ import annotations

import argparse
import platform
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from predict import classify_pil, load_detector, verdict  # noqa: E402
from utils import get_logger, load_config, setup_threads  # noqa: E402

log = get_logger("app")

# Same palette as the installer, so the two look like one product.
BG = "#1e1e2e"
PANEL = "#252539"
FG = "#e8e8f0"
MUTED = "#9a9ab0"
ACCENT = "#5a4fcf"
HOVER = "#6d61e0"
NEUTRAL = "#3a3a52"
NEUTRAL_HOVER = "#4a4a68"
REAL = "#2E7D32"
FAKE = "#C62828"
AMBER = "#ED6C02"

UI_FONT = ("Segoe UI" if platform.system() == "Windows"
           else "Helvetica Neue" if platform.system() == "Darwin"
           else "DejaVu Sans")
MONO = ("Consolas" if platform.system() == "Windows" else "Menlo")

PREVIEW = 320       # side of the image preview box

SCOPE = ("Trained on GAN-generated faces (StyleGAN) against real photographs. "
         "Images from diffusion models - Midjourney, Stable Diffusion, DALL-E "
         "- are outside what this model was tested on, and it will frequently "
         "call them REAL. That is a documented limitation, not a malfunction.")

RELIABILITY = [
    ("Same kind of data as training", "94.3%"),
    ("A different deepfake method", "51.1%  - chance"),
    ("Thumbnail-sized input", "55.8%"),
    ("Heavy JPEG compression", "72.9%"),
    ("Deliberate adversarial attack", "5.2%"),
]


class FlatButton(tk.Frame):
    """A button that honours its colours on macOS, where tk.Button does not."""

    def __init__(self, parent, text, command, bg=NEUTRAL, hover=NEUTRAL_HOVER,
                 fg="white", bold=False, width=14):
        super().__init__(parent, bg=bg, cursor="hand2",
                         highlightthickness=0, bd=0)
        self.command, self.bg, self.hover = command, bg, hover
        self.enabled = True
        self.label = tk.Label(self, text=text, bg=bg, fg=fg, padx=16, pady=8,
                              width=width,
                              font=(UI_FONT, 10, "bold" if bold else "normal"))
        self.label.pack(fill="both", expand=True)
        for w in (self, self.label):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)

    def _paint(self, c):
        self.configure(bg=c)
        self.label.configure(bg=c)

    def _enter(self, _=None):
        if self.enabled:
            self._paint(self.hover)

    def _leave(self, _=None):
        self._paint(self.bg if self.enabled else NEUTRAL)

    def _click(self, _=None):
        if self.enabled and callable(self.command):
            self.command()

    def set_enabled(self, on: bool):
        self.enabled = on
        self._paint(self.bg if on else NEUTRAL)
        self.label.configure(fg="white" if on else MUTED, cursor="hand2" if on else "")


class Bar(tk.Canvas):
    """A two-tone probability bar. Canvas rather than widgets, so it animates."""

    def __init__(self, parent, width=300, height=16):
        super().__init__(parent, width=width, height=height, bg=PANEL,
                         highlightthickness=0, bd=0)
        self.w, self.h = width, height

    def set(self, fraction: float, colour: str):
        self.delete("all")
        self.create_rectangle(0, 0, self.w, self.h, fill="#1a1a28", outline="")
        end = max(2, int(self.w * max(0.0, min(1.0, fraction))))
        self.create_rectangle(0, 0, end, self.h, fill=colour, outline="")


class DetectorApp(tk.Tk):
    def __init__(self, config, model_name: str):
        super().__init__()
        self.title("OBI DeepFake Image Detector")
        self.configure(bg=BG)
        self.geometry("980x700")
        self.minsize(900, 660)
        self._set_icon()

        self.config_data = config
        self.model_name = tk.StringVar(value=model_name)
        self.bundle = None                  # (model, transform, meta)
        self.queue: queue.Queue = queue.Queue()
        self.photo = None                   # keep a reference or Tk drops it
        self.current_path: Path | None = None

        self._build()
        self.after(80, self._drain)
        self._load_model_async()

    # -- chrome -------------------------------------------------------------
    def _set_icon(self):
        """
        Put our icon on the window and the taskbar.

        Windows needs two separate things. iconbitmap with a .ico sets the
        window icon, but the taskbar still shows Python's, because Windows
        groups taskbar buttons by Application User Model ID and a plain python
        process inherits the interpreter's. Setting an explicit ID first makes
        the taskbar use ours.
        """
        here = Path(__file__).resolve().parent
        if platform.system() == "Windows":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "ObaidROfficial.DeepFakeImageDetector.App.3")
            except Exception:
                pass
            for p in (here / "app.ico", here / "installer" / "app.ico"):
                if p.exists():
                    try:
                        self.iconbitmap(default=str(p))
                        return
                    except Exception:
                        pass
        for p in (here / "app.png", here / "installer" / "app.png"):
            if p.exists():
                try:
                    self._icon_img = tk.PhotoImage(file=str(p))
                    self.iconphoto(True, self._icon_img)
                    return
                except Exception:
                    pass

    def _build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=24, pady=(20, 0))
        tk.Label(head, text="OBI DeepFake Image Detector", bg=BG, fg=FG,
                 font=(UI_FONT, 19, "bold"), anchor="w").pack(fill="x")

        banner = tk.Frame(self, bg="#3a2c14", highlightthickness=0)
        banner.pack(fill="x", padx=24, pady=(10, 0))
        tk.Frame(banner, bg=AMBER, width=4).pack(side="left", fill="y")
        tk.Label(banner, text=SCOPE, bg="#3a2c14", fg="#f0d8b8", justify="left",
                 wraplength=880, anchor="w", font=(UI_FONT, 9),
                 padx=12, pady=10).pack(fill="x")

        # The button bar is packed BEFORE the body so it reserves its height
        # first. Packed afterwards, an expanding body can push it off the
        # bottom of the window - the same trap the installer screens hit.
        foot = tk.Frame(self, bg=BG)
        foot.pack(side="bottom", fill="x", padx=24, pady=(0, 20))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=16)

        # left: image
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")

        # A Label sizes itself in characters and lines, not pixels, so
        # width=320 would ask for a box 320 characters wide. Fixing the size on
        # a Frame and switching off geometry propagation is the way to get an
        # actual 320x320 pixel box.
        holder = tk.Frame(left, bg=PANEL, width=PREVIEW, height=PREVIEW)
        holder.pack_propagate(False)
        holder.pack()
        self.preview = tk.Label(holder, bg=PANEL, fg=MUTED,
                                text="No image loaded\n\nChoose an image to begin",
                                font=(UI_FONT, 10), justify="center")
        self.preview.pack(fill="both", expand=True)

        self.crop_note = tk.Label(left, text="", bg=BG, fg=MUTED,
                                  font=(UI_FONT, 8))
        self.crop_note.pack(pady=(6, 0))

        # right: result
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(24, 0))

        self.verdict_label = tk.Label(right, text="", bg=BG, fg=FG,
                                      font=(UI_FONT, 34, "bold"), anchor="w")
        self.verdict_label.pack(fill="x")
        self.confidence_label = tk.Label(right, text="Loading model...", bg=BG,
                                         fg=MUTED, font=(UI_FONT, 12), anchor="w")
        self.confidence_label.pack(fill="x", pady=(0, 14))

        bars = tk.Frame(right, bg=PANEL)
        bars.pack(fill="x")
        inner = tk.Frame(bars, bg=PANEL)
        inner.pack(fill="x", padx=14, pady=12)
        self.bar_fake_text = tk.Label(inner, text="Fake", bg=PANEL, fg=MUTED,
                                      font=(UI_FONT, 9), anchor="w", width=10)
        self.bar_fake_text.grid(row=0, column=0, sticky="w")
        self.bar_fake = Bar(inner)
        self.bar_fake.grid(row=0, column=1, sticky="w", pady=3)
        self.bar_real_text = tk.Label(inner, text="Real", bg=PANEL, fg=MUTED,
                                      font=(UI_FONT, 9), anchor="w", width=10)
        self.bar_real_text.grid(row=1, column=0, sticky="w")
        self.bar_real = Bar(inner)
        self.bar_real.grid(row=1, column=1, sticky="w", pady=3)

        self.detail = tk.Label(right, text="", bg=BG, fg=MUTED, anchor="w",
                               justify="left", font=(MONO, 9), wraplength=520)
        self.detail.pack(fill="x", pady=(12, 0))

        self.warnings = tk.Label(right, text="", bg=BG, fg=AMBER, anchor="w",
                                 justify="left", font=(UI_FONT, 9),
                                 wraplength=520)
        self.warnings.pack(fill="x", pady=(10, 0))

        # bottom bar (the frame itself was packed first, above)
        self.browse_btn = FlatButton(foot, "Choose image...", self.choose,
                                     bg=ACCENT, hover=HOVER, bold=True, width=16)
        self.browse_btn.pack(side="left")
        self.browse_btn.set_enabled(False)

        FlatButton(foot, "Reliability", self.show_reliability,
                   width=12).pack(side="left", padx=(10, 0))

        picker = tk.Frame(foot, bg=BG)
        picker.pack(side="right")
        tk.Label(picker, text="Model", bg=BG, fg=MUTED,
                 font=(UI_FONT, 9)).pack(side="left", padx=(0, 8))
        for name in ("mobilenet_v2", "efficientnet_b0"):
            tk.Radiobutton(
                picker, text=name, value=name, variable=self.model_name,
                command=self._model_changed, bg=BG, fg=FG, selectcolor=PANEL,
                activebackground=BG, activeforeground=FG, font=(UI_FONT, 9),
                borderwidth=0, highlightthickness=0).pack(side="left")

    # -- model --------------------------------------------------------------
    def _load_model_async(self):
        self.browse_btn.set_enabled(False)
        self.confidence_label.configure(
            text=f"Loading {self.model_name.get()}...", fg=MUTED)

        def work():
            try:
                bundle = load_detector(self.model_name.get(), self.config_data)
                self.queue.put(("model", bundle))
            except Exception as e:                      # noqa: BLE001
                self.queue.put(("model_error", e))

        threading.Thread(target=work, daemon=True).start()

    def _model_changed(self):
        self.bundle = None
        self._load_model_async()
        if self.current_path:
            self.after(50, lambda: self.classify(self.current_path))

    # -- actions ------------------------------------------------------------
    def choose(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                       ("All files", "*.*")])
        if path:
            self.classify(Path(path))

    def classify(self, path: Path):
        if self.bundle is None:
            return
        self.current_path = path
        self.browse_btn.set_enabled(False)
        self.verdict_label.configure(text="")
        self.confidence_label.configure(text="Working...", fg=MUTED)
        self.warnings.configure(text="")
        self.detail.configure(text="")

        model, transform, meta = self.bundle
        device = self.config_data["compute"]["device"]

        def work():
            try:
                from PIL import Image
                with Image.open(path) as raw:
                    image = raw.convert("RGB")
                prob, warns, crop = classify_pil(model, transform, image,
                                                 device, source_path=path)
                self.queue.put(("result", (prob, warns, crop, image.size, meta)))
            except Exception as e:                      # noqa: BLE001
                self.queue.put(("error", e))

        threading.Thread(target=work, daemon=True).start()

    def show_reliability(self):
        win = tk.Toplevel(self)
        win.title("Measured reliability")
        win.configure(bg=BG)
        win.geometry("560x420")
        tk.Label(win, text="What this detector actually does", bg=BG, fg=FG,
                 font=(UI_FONT, 14, "bold"), anchor="w").pack(
                 fill="x", padx=22, pady=(20, 4))
        tk.Label(win, text="Measured on held-out data. Accuracy by condition:",
                 bg=BG, fg=MUTED, font=(UI_FONT, 9), anchor="w").pack(
                 fill="x", padx=22)
        table = tk.Frame(win, bg=PANEL)
        table.pack(fill="x", padx=22, pady=14)
        for i, (cond, acc) in enumerate(RELIABILITY):
            tk.Label(table, text=cond, bg=PANEL, fg=FG, anchor="w",
                     font=(UI_FONT, 10)).grid(row=i, column=0, sticky="w",
                                              padx=(14, 20), pady=5)
            tk.Label(table, text=acc, bg=PANEL, fg=FG, anchor="e",
                     font=(MONO, 10)).grid(row=i, column=1, sticky="e", padx=14)
        table.grid_columnconfigure(0, weight=1)
        tk.Label(win, wraplength=500, justify="left", bg=BG, fg=MUTED,
                 font=(UI_FONT, 9), anchor="w",
                 text=("It is reliable only on full-resolution, lightly "
                       "compressed images of the type it was trained on. "
                       "Against a deepfake made by an unfamiliar technique it "
                       "performs no better than a coin toss - and it fails "
                       "toward REAL rather than reporting uncertainty.\n\n"
                       "A REAL verdict is not evidence that an image is "
                       "authentic.")).pack(fill="x", padx=22)
        FlatButton(win, "Close", win.destroy, bg=ACCENT, hover=HOVER,
                   bold=True).pack(pady=18)

    # -- worker results -----------------------------------------------------
    def _drain(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "model":
                    self.bundle = payload
                    self.browse_btn.set_enabled(True)
                    if not self.current_path:
                        self.confidence_label.configure(
                            text="Ready. Choose an image to classify.", fg=MUTED)
                elif kind == "model_error":
                    self.confidence_label.configure(
                        text="Could not load the model", fg=FAKE)
                    messagebox.showerror(
                        "Model not found",
                        f"{payload}\n\nThe weights should be in the models "
                        f"folder. Download them from the Releases page if "
                        f"they are missing.")
                elif kind == "result":
                    self._show_result(*payload)
                elif kind == "error":
                    self.browse_btn.set_enabled(True)
                    self.confidence_label.configure(text="Could not read that "
                                                    "image", fg=FAKE)
                    self.detail.configure(
                        text=f"{type(payload).__name__}: {payload}\n\n"
                             f"The extension may not match the real format - "
                             f"some sites serve WebP or AVIF named .jpg. Try "
                             f"re-saving it as JPEG or PNG.")
        except queue.Empty:
            pass
        self.after(80, self._drain)

    def _show_result(self, prob, warns, crop, original_size, meta):
        from PIL import ImageTk

        self.browse_btn.set_enabled(True)
        label, band = verdict(prob)
        pct = prob * 100 if label == "FAKE" else (1 - prob) * 100
        colour = FAKE if label == "FAKE" else REAL

        self.verdict_label.configure(text=label, fg=colour)
        self.confidence_label.configure(
            text=f"{pct:.1f}% confidence   -   band: {band}", fg=FG)

        self.bar_fake.set(prob, FAKE)
        self.bar_real.set(1 - prob, REAL)
        self.bar_fake_text.configure(text=f"Fake  {prob*100:5.1f}%")
        self.bar_real_text.configure(text=f"Real  {(1-prob)*100:5.1f}%")

        shown = crop.copy()
        shown.thumbnail((PREVIEW - 8, PREVIEW - 8))
        self.photo = ImageTk.PhotoImage(shown)
        self.preview.configure(image=self.photo, text="")

        w, h = original_size
        cw, ch = crop.size
        note = f"input {w}x{h}"
        if (cw, ch) != (w, h):
            note += f"   ->   classified region {cw}x{ch}"
        self.crop_note.configure(text=note)

        detail = f"model {self.model_name.get()}   P(fake) = {prob:.4f}"
        if meta.get("epoch") is not None:
            detail += f"   weights from epoch {meta['epoch']}"
        self.detail.configure(text=detail)

        self.warnings.configure(
            text=("This image falls outside the tested range:\n"
                  + "\n".join(f"  - {w_}" for w_ in warns)) if warns else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="mobilenet_v2 or efficientnet_b0")
    ap.add_argument("--image", default=None,
                    help="classify this image on startup")
    args = ap.parse_args()

    config = load_config()
    setup_threads(config)
    model_name = args.model or config.get("app", {}).get(
        "default_model", "mobilenet_v2")

    app = DetectorApp(config, model_name)
    if args.image:
        app.after(400, lambda: app.classify(Path(args.image)))
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
