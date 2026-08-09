"""
Model construction and the dataset class.

Kept separate from train.py so that evaluate.py and robustness.py can rebuild
the exact same architecture without importing the training loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics. Used because both backbones are ImageNet-pretrained;
# the input distribution must match what the frozen features expect.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DeepfakeDataset(Dataset):
    """
    Reads a split manifest produced by data_prep.py.

    Label convention: 0 = real, 1 = fake. Consistent across both datasets.
    """

    def __init__(self, manifest_csv: Path, root: Path, transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.root = Path(root)
        self.transform = transform
        self.labels = self.df["label"].to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.root / row["path"]
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
        except Exception:
            # A corrupt file should not kill a 3-hour training run. Return a
            # black image; these are counted and reported at the end of the
            # epoch so they cannot pass unnoticed.
            size = 128
            img = torch.zeros(3, size, size)
        return img, int(row["label"])


def build_transforms(image_size: int, aug_cfg: dict, train: bool):
    """
    Training gets augmentation, evaluation does not.

    Augmentation is deliberately mild. Aggressive colour or geometric
    distortion risks destroying exactly the low-level forensic artefacts the
    model needs to detect, which would be self-defeating.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=aug_cfg["horizontal_flip"]),
            transforms.RandomRotation(aug_cfg["rotation_degrees"]),
            transforms.ColorJitter(
                brightness=aug_cfg["colour_jitter"]["brightness"],
                contrast=aug_cfg["colour_jitter"]["contrast"],
                saturation=aug_cfg["colour_jitter"]["saturation"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Build a 2-class classifier from an ImageNet-pretrained backbone."""
    if name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.last_channel, 2)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def set_backbone_frozen(
    model: nn.Module, name: str, frozen: bool, unfreeze_blocks: int = 1
) -> None:
    """
    Stage 1 freezes everything except the classifier head.
    Stage 2 unfreezes the last `unfreeze_blocks` feature blocks.

    Unfreezing only the tail rather than the whole network is a capacity/compute
    trade-off, not a rule. Raise `unfreeze_blocks` if the model underfits -
    validation accuracy sitting at or above training accuracy is the signal to
    watch, since it means the model has not yet absorbed the training set.
    """
    # Freeze the whole backbone in both stages...
    for p in model.features.parameters():
        p.requires_grad = False

    # ...then in stage 2 release the trailing blocks.
    if not frozen:
        n = max(1, int(unfreeze_blocks))
        for block in model.features[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    # The classifier head is always trainable.
    for p in model.classifier.parameters():
        p.requires_grad = True


@torch.no_grad()
def predict_scores(model: nn.Module, loader, device: str = "cpu"):
    """Return (probabilities of class 1, true labels) over a loader."""
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)
