"""
chestxray_model.py
===================
Loads the trained EfficientNet-B1 ChestX-ray14 classifier and runs
predictions on new images. This is the only file app.py depends on for
"the AI part" — everything else in the site is plain Flask/HTML.

Depends on: torch, timm, opencv-python (cv2), numpy.
Install with:  pip install -r ../requirements-model.txt

Drop your trained checkpoint at:  model/chestxray_b1_portable.pt
(the file your Kaggle notebook's packaging cell produces, or a raw
`best.pt` — both are handled, see load_model() below)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn

cv2.setNumThreads(0)

DEFAULT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


# =============================================================================
# Architecture — must match the training definition exactly, or the saved
# weights won't line up with the right layers.
# =============================================================================

class ChestXrayNet(nn.Module):
    """EfficientNet-B1 backbone + GAP + a small classifier head, 14 outputs.

    Emits LOGITS, not probabilities — the predict_* helpers below apply
    torch.sigmoid() for you.
    """

    def __init__(self, backbone: str, head_dim_1: int, head_dim_2: int,
                 dropout: float, n_classes: int, pretrained: bool = False):
        super().__init__()
        # pretrained=False: we load OUR fine-tuned weights next, so no
        # internet download and no dependency on it at inference time.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="")

        n_feat = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_feat, head_dim_1),
            nn.BatchNorm1d(head_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim_1, head_dim_2),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim_2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return self.head(self.pool(f).flatten(1))


# =============================================================================
# Loading
# =============================================================================

def load_model(path: str | Path, device: str | torch.device = "cpu"):
    """Load a checkpoint. Returns (model_in_eval_mode, metadata_dict).

    Works with either:
      - a "portable" checkpoint (weights + full metadata bundled together), or
      - a raw training `best.pt` (metadata gets reconstructed from the
        embedded config; operating thresholds fall back to 0.5).
    """
    device = torch.device(device)
    ck = torch.load(path, map_location=device, weights_only=False)

    if "meta" in ck:
        meta = ck["meta"]
    else:
        cfg = ck["cfg"]
        meta = {
            "labels": DEFAULT_LABELS,
            "backbone": cfg["backbone"],
            "img_size": cfg["img_size"],
            "head_dim_1": cfg["head_dim_1"],
            "head_dim_2": cfg["head_dim_2"],
            "dropout": cfg["dropout"],
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "thresholds": {label: 0.5 for label in DEFAULT_LABELS},
        }

    model = ChestXrayNet(
        backbone=meta["backbone"],
        head_dim_1=meta["head_dim_1"],
        head_dim_2=meta["head_dim_2"],
        dropout=meta["dropout"],
        n_classes=len(meta["labels"]),
        pretrained=False,
    )
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    model._device = device
    return model, meta


# =============================================================================
# Preprocessing — must match the eval-time transform used in training
# =============================================================================

def preprocess(img, meta: dict) -> torch.Tensor:
    """Any chest X-ray (path, grayscale array, or colour array) -> a
    normalised (3, S, S) tensor, ready for the model.
    """
    if isinstance(img, (str, Path)):
        arr = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"could not read image: {img}")
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)

    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    size = meta["img_size"]
    interp = cv2.INTER_AREA if arr.shape[0] > size else cv2.INTER_LINEAR
    arr = cv2.resize(arr, (size, size), interpolation=interp)

    rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(meta["mean"], np.float32)) / np.array(meta["std"], np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1))


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict(model, tensors: torch.Tensor) -> np.ndarray:
    """(B, 3, S, S) tensor -> (B, 14) probability array."""
    if tensors.ndim == 3:
        tensors = tensors.unsqueeze(0)
    tensors = tensors.to(getattr(model, "_device", "cpu"))
    return torch.sigmoid(model(tensors).float()).cpu().numpy()


def predict_all(model, meta: dict, img) -> dict:
    """One image -> {label: probability} for all 14 labels, no thresholding."""
    probs = predict(model, preprocess(img, meta))[0]
    return {label: float(p) for label, p in zip(meta["labels"], probs)}


def predict_labels(model, meta: dict, img, thresholds: dict | None = None) -> dict:
    """One image -> {label: probability}, only for labels above their
    operating threshold. May return zero, one, or all labels."""
    thr = thresholds or meta["thresholds"]
    probs = predict_all(model, meta, img)
    return {label: p for label, p in probs.items() if p >= thr.get(label, 0.5)}
