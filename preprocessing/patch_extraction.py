"""
patch_extraction.py

Patch extraction for digitized histology images.

Why patches?
Whole-slide images (WSIs) and high-res histology images are too large to process
as a single tensor efficiently. A common production pattern is to:
  1) tile the image into fixed-size patches
  2) skip background/whitespace tiles
  3) preserve (x, y) coordinates for traceability and heatmap reconstruction

This module implements a lightweight, dependency-minimal version of that workflow.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Patch:
    x: int
    y: int
    size: int
    mean_intensity: float
    tissue_fraction: float
    path: Optional[str] = None  # filled if we save patches to disk


def _load_rgb(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img)


def _compute_tissue_mask(rgb: np.ndarray, threshold: int = 220) -> np.ndarray:
    """
    Simple background filter.
    Many H&E slides have bright/white background. We mark pixels as "tissue"
    if they are not too bright across channels.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image array (H, W, 3)")
    # tissue if NOT (all channels are very bright)
    tissue = ~((rgb[:, :, 0] > threshold) & (rgb[:, :, 1] > threshold) & (rgb[:, :, 2] > threshold))
    return tissue.astype(np.uint8)


def iter_patches(
    rgb: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_fraction: float = 0.15,
    bg_threshold: int = 220,
) -> Iterator[Tuple[np.ndarray, Patch]]:
    """
    Yield (patch_array, Patch metadata) for patches that pass the tissue filter.
    """
    h, w, _ = rgb.shape
    tissue_mask = _compute_tissue_mask(rgb, threshold=bg_threshold)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_rgb = rgb[y : y + patch_size, x : x + patch_size, :]
            patch_tissue = tissue_mask[y : y + patch_size, x : x + patch_size]

            tissue_fraction = float(patch_tissue.mean())
            if tissue_fraction < min_tissue_fraction:
                continue

            mean_intensity = float(patch_rgb.mean())
            meta = Patch(
                x=x,
                y=y,
                size=patch_size,
                mean_intensity=mean_intensity,
                tissue_fraction=tissue_fraction,
            )
            yield patch_rgb, meta


def extract_patches(
    image_path: str,
    out_dir: Optional[str],
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_fraction: float = 0.15,
    bg_threshold: int = 220,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    """
    Extract patches and (optionally) save them to disk.
    Returns a JSON-serializable artifact describing what was extracted.
    """
    rgb = _load_rgb(image_path)
    h, w, _ = rgb.shape

    saved: List[Dict[str, object]] = []
    count = 0

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for patch_rgb, meta in iter_patches(
        rgb,
        patch_size=patch_size,
        stride=stride,
        min_tissue_fraction=min_tissue_fraction,
        bg_threshold=bg_threshold,
    ):
        path = None
        if out_dir:
            fname = f"patch_x{meta.x}_y{meta.y}_s{meta.size}.png"
            path = os.path.join(out_dir, fname)
            Image.fromarray(patch_rgb).save(path)

        saved.append({**asdict(meta), "path": path})
        count += 1

        if limit is not None and count >= limit:
            break

    artifact = {
        "source_image": os.path.basename(image_path),
        "image_shape": {"height": h, "width": w},
        "params": {
            "patch_size": patch_size,
            "stride": stride,
            "min_tissue_fraction": min_tissue_fraction,
            "bg_threshold": bg_threshold,
            "limit": limit,
        },
        "patches": saved,
        "patch_count": len(saved),
    }
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract patches from a histology image.")
    parser.add_argument("--image", required=True, help="Path to input histology image (png/jpg).")
    parser.add_argument("--out_dir", default=None, help="Directory to write patch images (optional).")
    parser.add_argument("--out_json", default="examples/patch_manifest.json", help="Where to write metadata JSON.")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min_tissue_fraction", type=float, default=0.15)
    parser.add_argument("--bg_threshold", type=int, default=220)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N patches (debug).")
    args = parser.parse_args()

    artifact = extract_patches(
        image_path=args.image,
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        min_tissue_fraction=args.min_tissue_fraction,
        bg_threshold=args.bg_threshold,
        limit=args.limit,
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Extracted {artifact['patch_count']} patches")
    print(f"Wrote manifest: {args.out_json}")


if __name__ == "__main__":
    main()
