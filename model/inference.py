"""
inference.py

Model-agnostic inference interface for histopathology patch classification.

This repo intentionally excludes proprietary datasets and trained weights.
To keep the project runnable end-to-end, we provide:

1) A clean inference contract (input patches + metadata -> predictions)
2) A deterministic "demo" backend that produces stable probabilities
   based on patch statistics (NOT a medical model)

This mirrors a real engineering pattern: separate the "plumbing"
(preprocessing, batching, IO, contracts) from model internals.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


# ----------------------------
# Types / Contracts
# ----------------------------

@dataclass(frozen=True)
class Prediction:
    x: int
    y: int
    patch_size: int
    prob_cancer: float
    backend: str


def _load_patch_rgb(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Patch image not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ----------------------------
# Demo backend (deterministic)
# ----------------------------

class DemoPatchClassifier:
    """
    Deterministic, weight-free scoring backend.

    This is NOT a cancer detector. It simply maps patch intensity / texture-like
    signals into a probability for demonstration purposes, so the repo is runnable.

    The goal is to showcase:
    - batch inference
    - stable outputs
    - IO + contracts
    """

    backend_name = "demo-statistical"

    def predict_batch(self, batch: np.ndarray) -> List[float]:
        """
        Args:
            batch: uint8 RGB array of shape (B, H, W, 3)

        Returns:
            list of probabilities in [0, 1]
        """
        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError("Expected batch shape (B, H, W, 3)")

        # Normalize to 0..1
        x = batch.astype(np.float32) / 255.0

        # Heuristic features: mean + contrast proxy
        mean = x.mean(axis=(1, 2, 3))
        std = x.std(axis=(1, 2, 3))

        # A smooth mapping to probability (deterministic)
        # Tuned to create a spread of values without being extreme.
        z = (std * 6.0) - (mean * 2.0) - 1.0
        probs = [_clamp01(float(_sigmoid(v))) for v in z]
        return probs


# ----------------------------
# Inference orchestration
# ----------------------------

def run_inference_from_manifest(
    patch_manifest_path: str,
    out_path: str,
    batch_size: int = 16,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run inference on patches listed in a manifest (from preprocessing).

    Manifest format (expected):
      - patches: list of {x, y, size, path, ...}
    """
    if not os.path.exists(patch_manifest_path):
        raise FileNotFoundError(f"Manifest not found: {patch_manifest_path}")

    with open(patch_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    patches = manifest.get("patches", [])
    if not patches:
        raise ValueError("No patches found in manifest. Did you run patch extraction?")

    # Select backend
    model = DemoPatchClassifier()
    chosen_backend = backend or model.backend_name

    preds: List[Dict[str, Any]] = []

    # Load patches in batches
    batch_imgs: List[np.ndarray] = []
    batch_meta: List[Dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal batch_imgs, batch_meta, preds
        if not batch_imgs:
            return
        batch_arr = np.stack(batch_imgs, axis=0)
        probs = model.predict_batch(batch_arr)
        for meta, p in zip(batch_meta, probs):
            preds.append(
                {
                    "x": meta["x"],
                    "y": meta["y"],
                    "patch_size": meta.get("size"),
                    "prob_cancer": float(p),
                    "backend": chosen_backend,
                    "path": meta.get("path"),
                }
            )
        batch_imgs = []
        batch_meta = []

    for meta in patches:
        path = meta.get("path")
        if not path:
            raise ValueError("Manifest patch is missing 'path'. Save patches to disk in preprocessing step.")

        img = _load_patch_rgb(path)
        batch_imgs.append(img)
        batch_meta.append(meta)

        if len(batch_imgs) >= batch_size:
            flush_batch()

    flush_batch()

    # Aggregate slide-level score (simple top-k mean for decision support)
    probs = np.array([p["prob_cancer"] for p in preds], dtype=np.float32)
    topk = int(min(50, len(probs)))
    topk_mean = float(np.sort(probs)[-topk:].mean()) if topk > 0 else 0.0

    # Top suspicious patches (for review)
    top_patches = sorted(preds, key=lambda d: d["prob_cancer"], reverse=True)[:10]

    output = {
        "source_image": manifest.get("source_image"),
        "patch_count": len(preds),
        "slide_risk_score": round(topk_mean, 4),
        "aggregation": {
            "method": "topk_mean",
            "k": topk,
        },
        "top_patches": [
            {
                "x": p["x"],
                "y": p["y"],
                "patch_size": p["patch_size"],
                "prob_cancer": round(p["prob_cancer"], 4),
            }
            for p in top_patches
        ],
        "predictions": preds,
        "disclaimer": "Demo backend only. Not a diagnostic model.",
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patch inference from a patch manifest.")
    parser.add_argument("--patch_manifest", required=True, help="Path to patch manifest JSON.")
    parser.add_argument("--out", default="examples/sample_output.json", help="Output JSON path.")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    out = run_inference_from_manifest(
        patch_manifest_path=args.patch_manifest,
        out_path=args.out,
        batch_size=args.batch_size,
    )

    print(f"Predicted {out['patch_count']} patches")
    print(f"Slide risk score: {out['slide_risk_score']}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
