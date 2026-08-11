"""
OBI DeepFake Image Detector - graphical interface.

Launches a local web interface for the trained detector. Drag an image in,
get a verdict with confidence.

    python app.py
    python app.py --share          # temporary public link, useful for a demo
    python app.py --port 8080

Then open the printed URL, normally http://127.0.0.1:7860

Inference logic is imported from src/predict.py rather than reimplemented, so
the interface and the command line tool cannot drift apart. A mismatch in
preprocessing between the two would silently produce different results for the
same image.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from predict import classify_pil, load_detector, verdict  # noqa: E402
from utils import get_logger, load_config, setup_threads  # noqa: E402

log = get_logger("app")

# Models are cached after first load - loading takes about a second and would
# otherwise repeat on every single classification.
_CACHE: dict[str, tuple] = {}

CSS = """
.verdict-real { font-size: 2.2em; font-weight: 700; color: #2E7D32; text-align: center; }
.verdict-fake { font-size: 2.2em; font-weight: 700; color: #C62828; text-align: center; }
.subtle { color: #666; font-size: 0.9em; }
footer { visibility: hidden; }
"""


def get_model(name: str, config: dict):
    if name not in _CACHE:
        _CACHE[name] = load_detector(name, config)
    return _CACHE[name]


def _idle(message="Upload an image to begin."):
    """Neutral state. gr.Label rejects None in Gradio 6, so pass an empty dict."""
    return (
        f"<div class='subtle' style='text-align:center;padding-top:40px'>"
        f"{message}</div>",
        {}, "", "", None,
    )


def analyse(image_path, model_name: str, config: dict, progress=None):
    """
    Classify an uploaded image.

    Takes a FILE PATH rather than a decoded image. Gradio's own preprocessing
    decodes uploads with PIL before the handler runs, and when that fails it
    raises inside the framework where this function cannot catch it - the user
    sees only an unexplained "Error" box. Handling the file ourselves keeps
    decoding failures inside our own error handling.

    It also restores a capability lost in the first version: with a path
    available, the JPEG compression heuristic works for GUI uploads too, not
    just for the command line tool.

    The progress steps are real pipeline stages rather than a timer. On CPU at
    224px the whole thing takes a second or two.
    """
    import gradio as gr
    from PIL import Image, UnidentifiedImageError

    if not image_path:
        return _idle()

    path = Path(image_path)
    if not path.exists():
        return _idle("Uploaded file could not be found. Try again.")

    size = path.stat().st_size
    if size == 0:
        return _idle("The uploaded file is empty (0 bytes). The upload may "
                     "have been interrupted - try again.")

    try:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
    except UnidentifiedImageError:
        return (
            "<div class='verdict-fake' style='font-size:1.2em'>"
            "Unreadable image</div>",
            {},
            f"**This file could not be decoded as an image.**\n\n"
            f"- Name: `{path.name}`\n"
            f"- Size: {size:,} bytes\n\n"
            f"The extension may not match the actual format — some sites serve "
            f"WebP or AVIF with a `.jpg` name. Try opening it in an image "
            f"editor and re-saving as JPEG or PNG.",
            "", None,
        )
    except Exception as e:  # noqa: BLE001
        return _idle(f"Could not open the image: {type(e).__name__}")

    if progress is None:
        progress = gr.Progress()

    progress(0.10, desc="Loading model")
    model, transform, meta = get_model(model_name, config)

    progress(0.35, desc="Detecting face")
    device = config["compute"]["device"]
    time.sleep(0.05)  # let the progress update render before the blocking call

    progress(0.65, desc="Running detector")
    prob_fake, warns, crop = classify_pil(model, transform, image, device,
                                          source_path=path)

    progress(0.95, desc="Preparing result")
    label, band = verdict(prob_fake)

    pct = prob_fake * 100 if label == "FAKE" else (1 - prob_fake) * 100
    css_class = "verdict-fake" if label == "FAKE" else "verdict-real"
    headline = (
        f"<div class='{css_class}'>{label}</div>"
        f"<div style='text-align:center;font-size:1.3em'>{pct:.1f}% confidence</div>"
        f"<div class='subtle' style='text-align:center'>"
        f"confidence band: {band}</div>"
    )

    bars = {"Fake": prob_fake, "Real": 1.0 - prob_fake}

    warn_md = ""
    if warns:
        warn_md = "**This image falls outside the tested range:**\n\n"
        warn_md += "\n".join(f"- {w}" for w in warns)

    w, h = image.size
    cw, ch = crop.size
    _ = size  # file size already validated above
    detail = (
        f"Model: `{model_name}`  ·  input {w}×{h}"
        + (f"  ·  classified region {cw}×{ch}" if (cw, ch) != (w, h) else "")
        + f"  ·  P(fake) = {prob_fake:.4f}"
    )
    if meta.get("epoch") is not None:
        detail += f"  ·  weights from epoch {meta['epoch']}"

    progress(1.0, desc="Done")
    return headline, bars, warn_md, detail, crop


# Every element sets its own colour explicitly. Setting it only on the wrapper
# is not enough: Gradio's dark theme colours <b> white, which is invisible
# against this cream background - the emphasised words simply disappeared.
SCOPE_BANNER = """
<div style="background:#FFF4E5;border-left:4px solid #ED6C02;padding:10px 14px;
            border-radius:4px;margin-bottom:8px;font-size:0.92em;
            color:#663C00 !important;line-height:1.5">
<span style="color:#663C00"><b style="color:#663C00">Scope:</b> trained on
<b style="color:#663C00">GAN-generated</b> faces (StyleGAN) versus real
photographs. Images from <b style="color:#663C00">diffusion models</b> —
Midjourney, Stable Diffusion, DALL&middot;E, Firefly — are
<b style="color:#663C00">outside what this model was trained and tested
on</b>, and it will frequently call them REAL. That is a documented
limitation, not a malfunction. See the reliability panel below.</span>
</div>
"""

CAVEATS = """
This model was trained to tell **StyleGAN-generated faces** from **real
photographs**. Measured performance:

| Condition | Accuracy |
|---|---|
| Same kind of data as training | 94.3% |
| A different deepfake method | 51.1% — chance |
| Thumbnail-sized input | 55.8% |
| Heavy JPEG compression | 72.9% |
| Deliberate adversarial attack | 5.2% |

It is reliable only on full-resolution, lightly compressed images of the type
it was trained on. Against a deepfake produced by an unfamiliar technique it
performs no better than a coin toss — and it tends to fail toward **REAL**
rather than reporting uncertainty.

**A REAL verdict is not evidence that an image is authentic.**

### Diffusion-generated images are out of scope

Both training datasets are GAN-era. GANs build an image by repeatedly
upsampling from a low-resolution representation, and that process leaves
characteristic high-frequency artefacts which this detector learns to find
(Frank et al., 2020). Diffusion models generate by iterative denoising instead,
so those particular artefacts are largely absent.

There is therefore no reason to expect this model to detect Midjourney, Stable
Diffusion, DALL·E or Firefly output, and in testing it confidently labels such
images REAL. Extending to diffusion is identified as further work in Chapter 8;
it would require retraining on diffusion-generated data, not a change to this
tool.

If you upload a diffusion-generated image and get REAL, the tool is behaving as
documented.

### On the confidence figure

The percentage comes from a softmax output and is **not calibrated**. Networks
trained with cross-entropy are systematically overconfident, and this one will
report 100% on inputs it has no basis for judging. Treat it as a relative score
rather than a probability of being correct. Calibrated confidence is listed as
further work.

Full analysis in Chapters 6 and 7 of the accompanying report.
"""


def build_interface(config: dict):
    import gradio as gr

    # Gradio 6.0 moved `theme` and `css` from the Blocks constructor to
    # launch(). Passing them here raises a UserWarning and they are ignored.
    with gr.Blocks(title="OBI DeepFake Image Detector") as demo:
        gr.Markdown(
            "# OBI DeepFake Image Detector\n"
            "MSc Advanced Computing Project (COM7014) — Muhammad Obaid Ur Rehman\n"
        )
        # Kept visible rather than collapsed. Users reach for diffusion-model
        # images first, and a REAL verdict on one would otherwise look like a
        # working detector giving a wrong answer rather than a documented
        # out-of-scope input.
        gr.HTML(SCOPE_BANNER)

        with gr.Row():
            with gr.Column(scale=1):
                # type="filepath" rather than "pil". Gradio decodes "pil"
                # uploads internally with PIL, and a decode failure there
                # raises inside the framework where the handler cannot catch
                # it. Taking the path lets us decode with our own error
                # handling, and gives access to the file size for the
                # compression check.
                image_in = gr.Image(
                    type="filepath", label="Image", height=340,
                    sources=["upload", "clipboard"],
                )
                model_choice = gr.Radio(
                    choices=["mobilenet_v2", "efficientnet_b0"],
                    value="mobilenet_v2",
                    label="Model",
                )
                run_btn = gr.Button("Analyse", variant="primary", size="lg")

            with gr.Column(scale=1):
                headline = gr.HTML(
                    "<div class='subtle' style='text-align:center;padding-top:60px'>"
                    "Upload an image to begin.</div>"
                )
                bars = gr.Label(label="Class probabilities", num_top_classes=2)
                warn_box = gr.Markdown("")
                detail = gr.Markdown("", elem_classes="subtle")

        with gr.Accordion("What the model actually saw", open=False):
            gr.Markdown(
                "The detector was trained on tightly cropped faces. If a face "
                "is found in your image it is cropped out first, and this is "
                "the region that was classified."
            )
            crop_view = gr.Image(label="Classified region", height=240)

        with gr.Accordion("How reliable is this? — read before trusting a result",
                          open=False):
            gr.Markdown(CAVEATS)

        def _run(img, name, progress=gr.Progress()):
            # Catch everything. An unhandled exception inside a Gradio callback
            # surfaces as an unexplained "Error" box in the browser with the
            # traceback only in the terminal, which is a poor experience for
            # anyone running this as a demonstration.
            try:
                return analyse(img, name, config, progress)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                return (
                    "<div class='verdict-fake' style='font-size:1.2em'>"
                    "Something went wrong</div>",
                    {},
                    f"**Error:** `{type(e).__name__}: {e}`\n\n"
                    f"Full traceback printed in the terminal.",
                    "",
                    None,
                )

        outputs = [headline, bars, warn_box, detail, crop_view]
        run_btn.click(_run, inputs=[image_in, model_choice], outputs=outputs)
        # Classify immediately on upload as well, so the tool feels responsive
        image_in.change(_run, inputs=[image_in, model_choice], outputs=outputs)

    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Deepfake detector web interface")
    ap.add_argument("--share", action="store_true",
                    help="create a temporary public link (expires after 72h)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    try:
        import gradio  # noqa: F401
    except ImportError:
        log.error(
            "Gradio is not installed. Install it with:\n"
            "    pip install gradio"
        )
        sys.exit(1)

    config = load_config()
    setup_threads(config)

    # Fail early with a clear message rather than on the first upload.
    models_dir = Path(config["paths"]["models"])
    available = sorted(p.name for p in models_dir.glob("*_best.pt"))
    if not available:
        log.error(
            "No trained weights found in %s\n"
            "  Download them from:\n"
            "    https://github.com/ObaidROfficial/deepfake-detection-msc/releases\n"
            "  and place the .pt files in the models/ folder.",
            models_dir.resolve(),
        )
        sys.exit(1)
    log.info("Found weights: %s", ", ".join(available))

    import gradio as gr

    demo = build_interface(config)

    # Gradio 6 takes theme and css at launch time. Older versions took them on
    # the Blocks constructor and reject them here, so try the modern signature
    # first and fall back rather than pinning to one Gradio major version.
    launch_kwargs = dict(
        server_name=args.host, server_port=args.port,
        share=args.share, inbrowser=True,
    )
    try:
        demo.launch(theme=gr.themes.Soft(), css=CSS, **launch_kwargs)
    except TypeError:
        log.info("Gradio version does not accept theme/css at launch; "
                 "starting without custom styling.")
        demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
