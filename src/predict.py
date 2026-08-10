"""
Run the trained detector on your own images.

Command line classifier. For most purposes the graphical interface (app.py)
is easier; this exists for batch work, scripting, and as the shared inference
logic that app.py imports. Needs only the trained weights and an image - no
datasets, no training.

    python src/predict.py --image path/to/photo.jpg
    python src/predict.py --folder path/to/images/
    python src/predict.py --image photo.jpg --model efficientnet_b0
    python src/predict.py --folder images/ --csv predictions.csv

A NOTE ON HOW THIS REPORTS ITSELF
---------------------------------
The evaluation in Chapter 6 found that this detector reaches 94.3% accuracy on
data resembling its training set, and approximately chance (51%) on deepfakes
produced by a manipulation method it has not seen. Chapter 6 also found that it
degrades sharply on heavily compressed or downscaled input - to 55.8% on
thumbnail-sized images.

A tool that printed "FAKE (98% confident)" while concealing that would be
misleading. Chapter 8 argues that a detector which silently fails while
appearing to work is worse than no detector, because it confers unwarranted
confidence. This script therefore checks the input against the conditions the
model was actually validated on, and says so when it is operating outside them.

The caveats are not decoration. They are the finding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model, build_transforms  # noqa: E402
from utils import get_logger, load_config, resolve, setup_threads  # noqa: E402

log = get_logger("predict")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Validated operating range, from the robustness suite in Chapter 6.
MIN_TRUSTED_SIZE = 128      # below this, downscaling costs >14 accuracy points
MIN_TRUSTED_BPP = 0.25      # bytes per pixel; below this suggests heavy JPEG
MAX_TRUSTED_ASPECT = 1.35   # training images are square face crops


def detect_and_crop_face(im: Image.Image, margin: float = 0.35):
    """
    Find the largest face and return a square crop around it.

    WHY THIS EXISTS. The model was trained on the 140k Real and Fake Faces
    dataset, in which every image is a 256x256 crop tightly framed on a single
    face. A full-body or half-length portrait is a completely different input
    distribution: after resizing to 224x224 the face may occupy only a small
    fraction of the frame, and most of what the model sees is clothing and
    background it was never trained on.

    Without this step the tool silently misapplies the model. Cropping to a
    face first is not a cosmetic improvement, it is what makes the input
    comparable to the training data at all.

    Uses OpenCV's Haar cascade - already a dependency, needs no download, and
    is adequate for finding a large frontal face. It will miss profiles and
    heavily occluded faces, in which case we say so rather than pretending.

    Returns (cropped_image, note). `note` is None on success, or a string
    explaining why cropping did not happen.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return im, "OpenCV not available - using the whole image uncropped."

    try:
        arr = np.array(im.convert("RGB"))
        grey = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return im, "Face cascade could not be loaded - using the whole image."

        faces = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(60, 60))
        if len(faces) == 0:
            return im, (
                "No face detected. The model was trained only on cropped face "
                "images, so a verdict on this input is not meaningful."
            )

        # Largest detection, on the assumption it is the subject
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Expand to a square with margin, roughly matching the framing of the
        # training crops, then clamp to the image bounds.
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * (1 + 2 * margin)
        half = side / 2
        left = int(max(0, cx - half))
        top = int(max(0, cy - half))
        right = int(min(im.width, cx + half))
        bottom = int(min(im.height, cy + half))

        cropped = im.crop((left, top, right, bottom))
        note = None
        if len(faces) > 1:
            note = (f"{len(faces)} faces detected; classified the largest. "
                    f"The model handles one face at a time.")
        return cropped, note
    except Exception as e:  # noqa: BLE001
        return im, f"Face detection failed ({e}) - using the whole image."


def inspect_input(path: Path | None, im: Image.Image) -> list[str]:
    """
    Flag conditions the model was not validated under.

    Each warning cites the measured effect rather than being generic, so the
    user can judge how much weight to give the prediction.

    `path` may be None (e.g. an image uploaded through the GUI, which exists
    only in memory). The compression heuristic needs a file size, so it is
    skipped in that case; the resolution checks still apply.
    """
    warnings = []
    w, h = im.size
    smallest = min(w, h)

    if smallest < MIN_TRUSTED_SIZE:
        warnings.append(
            f"Input is {w}x{h}. Measured accuracy on 32px-downscaled images was "
            f"55.8% against 94.3% on full-resolution input. Treat this "
            f"prediction as unreliable."
        )
    elif smallest < 224:
        warnings.append(
            f"Input is {w}x{h}, below the 224px the model was trained on. "
            f"Upscaling loses high-frequency detail, which is the signal this "
            f"detector depends on."
        )

    aspect = max(w, h) / max(1, min(w, h))
    if aspect > MAX_TRUSTED_ASPECT:
        warnings.append(
            f"Aspect ratio {aspect:.2f}:1. Training images are square face "
            f"crops, and preprocessing squashes the image to 224x224, which "
            f"distorts a non-square input."
        )

    # The compression heuristic uses file size against the ORIGINAL pixel
    # count, so it is only meaningful when `im` is the file as stored. When a
    # crop is passed in, the ratio would be inflated and misleading, so this
    # check is skipped unless the dimensions still match the file.
    if path is not None:
        try:
            bpp = path.stat().st_size / (w * h)
            if path.suffix.lower() in {".jpg", ".jpeg"} and bpp < MIN_TRUSTED_BPP:
                warnings.append(
                    f"Heavy JPEG compression likely ({bpp:.3f} bytes/pixel). "
                    f"Accuracy at JPEG quality 10 measured 72.9%."
                )
        except OSError:
            pass

    return warnings


def load_detector(model_name: str, config: dict, weights: str | None = None):
    """
    Load a trained model and its evaluation transform.

    Shared by the command line tool and the Gradio interface so both use
    identical preprocessing - a mismatch there would silently change results.

    Returns (model, transform, checkpoint_metadata).
    """
    device = config["compute"]["device"]
    ckpt_path = Path(weights) if weights else (
        resolve(config["paths"]["models"]) / f"{model_name}_best.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No weights at {ckpt_path}. Download them from the Releases page "
            f"or train with: python src/train.py --model {model_name}"
        )

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    transform = build_transforms(
        config["subset"]["image_size"], config["augmentation"], train=False
    )
    meta = {"path": ckpt_path, "epoch": ck.get("epoch"), "val_loss": ck.get("val_loss")}
    return model, transform, meta


def classify_pil(model, transform, im: Image.Image, device: str,
                 source_path: Path | None = None, crop_face: bool = True):
    """
    Classify an already-loaded PIL image.

    source_path is optional and used only for the compression heuristic, which
    needs the file size on disk. GUI uploads have no such path, so that check
    is skipped.

    crop_face=True detects and crops to a face first, so the input resembles
    the training data. Set False to classify the frame as given, which is what
    happens for the dataset images in the experiments (already cropped).

    Returns (prob_fake, warnings, crop) - crop is the image actually shown to
    the model, which the GUI displays so the user can see what was classified.
    """
    im = im.convert("RGB")

    crop = im
    crop_note = None
    if crop_face:
        crop, crop_note = detect_and_crop_face(im)

    # Inspect the region actually fed to the model, not the original upload.
    # A 4000x3000 photograph passes every resolution check, but if the face
    # within it crops to 149x149 then the model is working from an upscaled
    # thumbnail and the warning must reflect that.
    warnings = inspect_input(source_path, crop)

    if crop_note:
        warnings.insert(0, crop_note)
    elif crop.size != im.size:
        cw, ch = crop.size
        warnings.insert(0, (
            f"Cropped to the detected face ({cw}x{ch}) before classifying, "
            f"to match the framing of the training data."
        ))

    x = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_fake = torch.softmax(model(x), dim=1)[0, 1].item()
    return prob_fake, warnings, crop


def classify(model, transform, path: Path, device: str, crop_face: bool = True):
    """Classify an image on disk. Returns (prob_fake, warnings, crop)."""
    with Image.open(path) as raw:
        return classify_pil(model, transform, raw, device,
                            source_path=path, crop_face=crop_face)


def verdict(prob_fake: float) -> tuple[str, str]:
    """
    Map probability to a label and a confidence band.

    The bands are deliberately conservative. The model is not calibrated -
    softmax outputs from a cross-entropy-trained network are systematically
    overconfident - so a 0.99 output does not mean 99% likelihood of being
    correct. Calibration is listed as further work in Chapter 8.
    """
    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    margin = abs(prob_fake - 0.5) * 2
    if margin >= 0.90:
        band = "high"
    elif margin >= 0.60:
        band = "moderate"
    elif margin >= 0.30:
        band = "low"
    else:
        band = "very low - close to the decision boundary"
    return label, band


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classify an image as real or deepfake",
        epilog="Trained on StyleGAN-generated faces vs real photographs. "
               "See the caveats printed with each result.",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="path to a single image")
    g.add_argument("--folder", type=str, help="path to a folder of images")
    ap.add_argument("--model", default="mobilenet_v2",
                    choices=["mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--weights", default=None, help="override checkpoint path")
    ap.add_argument("--csv", default=None, help="write results to a CSV file")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the interpretation notes")
    ap.add_argument("--no-crop", action="store_true",
                    help="skip face detection and classify the whole frame "
                         "(use for images that are already face crops)")
    args = ap.parse_args()

    config = load_config()
    setup_threads(config)
    device = config["compute"]["device"]

    try:
        model, transform, meta = load_detector(args.model, config, args.weights)
    except FileNotFoundError as e:
        log.error(
            "%s\n"
            "  Download the weights from:\n"
            "    https://github.com/ObaidROfficial/deepfake-detection-msc/releases\n"
            "  and place the .pt file in the models/ folder.", e,
        )
        sys.exit(1)
    ckpt_path, ck = meta["path"], meta

    # Gather inputs
    if args.image:
        paths = [Path(args.image)]
        if not paths[0].exists():
            log.error("File not found: %s", paths[0])
            sys.exit(1)
    else:
        folder = Path(args.folder)
        if not folder.is_dir():
            log.error("Not a folder: %s", folder)
            sys.exit(1)
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            log.error("No images found in %s", folder)
            sys.exit(1)

    print()
    print("=" * 68)
    print(f"  Deepfake detector - {args.model}")
    # meta values are None if the checkpoint predates their being recorded,
    # so check for None rather than key presence.
    detail = ""
    if meta.get("epoch") is not None and meta.get("val_loss") is not None:
        detail = f"  (epoch {meta['epoch']}, val loss {meta['val_loss']:.4f})"
    print(f"  weights: {ckpt_path.name}{detail}")
    print("=" * 68)

    rows = []
    for p in paths:
        try:
            prob, warns, _crop = classify(model, transform, p, device,
                                          crop_face=not args.no_crop)
        except Exception as e:  # noqa: BLE001
            print(f"\n  {p.name}\n    could not read: {e}")
            continue

        label, band = verdict(prob)
        rows.append({"file": str(p), "prediction": label,
                     "p_fake": round(prob, 4), "confidence": band,
                     "warnings": " | ".join(warns)})

        print()
        print(f"  {p.name}")
        print(f"    verdict     : {label}")
        print(f"    P(fake)     : {prob:.4f}")
        print(f"    confidence  : {band}")
        for w in warns:
            print(f"    !! {w}")

    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\n  Written to {args.csv}")

    if not args.quiet:
        print()
        print("-" * 68)
        print("  HOW TO READ THESE RESULTS")
        print("-" * 68)
        print("  This model was trained to distinguish StyleGAN-generated faces")
        print("  from real photographs. Measured performance:")
        print()
        print("    same kind of data as training .... 94.3% accuracy")
        print("    a different deepfake method ...... 51.1%  (chance)")
        print("    thumbnail-sized input ............ 55.8%")
        print("    heavy JPEG compression ........... 72.9%")
        print("    deliberate adversarial attack .....  5.2%")
        print()
        print("  It is therefore reliable only on full-resolution, lightly")
        print("  compressed images of the type it was trained on. On a deepfake")
        print("  produced by an unfamiliar technique it performs no better than")
        print("  a coin toss, and it tends to fail toward 'REAL' rather than")
        print("  reporting uncertainty.")
        print()
        print("  OUT OF SCOPE: diffusion models")
        print("  Both training datasets are GAN-era. GANs upsample repeatedly")
        print("  during generation, leaving high-frequency artefacts this model")
        print("  learns to find. Diffusion models (Midjourney, Stable Diffusion,")
        print("  DALL-E, Firefly) generate by iterative denoising and largely")
        print("  do not leave them. Such images will usually be called REAL.")
        print("  That is a documented limitation, not a malfunction.")
        print()
        print("  Do not treat a REAL verdict as evidence that an image is")
        print("  authentic. See Chapters 6 and 7 of the report.")
        print("-" * 68)


if __name__ == "__main__":
    main()
