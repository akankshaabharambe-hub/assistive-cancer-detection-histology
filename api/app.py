"""
api/app.py

Minimal REST API for assistive histology screening.

This service exposes a decision-support inference endpoint that:
- accepts an image upload (PNG/JPG)
- extracts patches (patch-based workflow)
- runs deterministic demo inference (no weights shipped)
- returns top suspicious regions + slide-level risk score

NOTE:
This is a demo-grade serving layer to mirror production integration patterns.
It is not a diagnostic tool.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from preprocessing.patch_extraction import PatchSpec, extract_patches
from model.inference import InferenceConfig, PatchClassifier, run_patch_inference

app = FastAPI(
    title="Assistive Cancer Screening API (Demo)",
    description="Decision-support inference API for histology patch scoring (non-diagnostic).",
    version="0.1.0",
)


def _read_image_bytes(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(file_bytes))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file") from e


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_fraction: float = 0.10,
    batch_size: int = 32,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Returns:
      - slide_risk_score: aggregate risk score (top-k mean)
      - top_regions: top suspicious patches with coordinates
      - patch_count: number of patches scored
      - disclaimer: decision-support framing
    """
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only PNG/JPG uploads are supported")

    file_bytes = await file.read()
    img = _read_image_bytes(file_bytes)

    # Convert PIL -> numpy and reuse existing extract_patches() by saving temporary array
    # We keep it simple: write to a BytesIO-backed image and open via PIL inside extractor is fine,
    # but extractor expects a path. So we do in-memory extraction here.
    arr = np.array(img)

    # Inline patch extraction using same logic as extract_patches(), but without filesystem dependency
    spec = PatchSpec(
        patch_size=patch_size,
        stride=stride,
        min_tissue_fraction=min_tissue_fraction,
    )

    # Reuse core logic: temporarily create a "patch list" in the same format expected by inference
    # by adapting extract_patches() implementation expectations.
    # We'll call extract_patches() by saving to a temp file is overkill; we do it directly here.
    patches: List[Dict[str, Any]] = []
    h, w, _ = arr.shape
    ps, st = spec.patch_size, spec.stride

    def tissue_fraction(patch: np.ndarray) -> float:
        gray = patch.mean(axis=2) / 255.0
        return float((gray < 0.90).mean())

    for y in range(0, max(1, h - ps + 1), st):
        for x in range(0, max(1, w - ps + 1), st):
            patch = arr[y : y + ps, x : x + ps]
            if patch.shape[0] != ps or patch.shape[1] != ps:
                continue
            tf = tissue_fraction(patch)
            if tf < spec.min_tissue_fraction:
                continue
            patches.append({"patch": patch, "x": x, "y": y, "w": ps, "h": ps, "tissue_fraction": tf})

    if not patches:
        raise HTTPException(status_code=422, detail="No tissue patches extracted. Try lowering min_tissue_fraction.")

    classifier = PatchClassifier(config=InferenceConfig(batch_size=batch_size))
    scored = run_patch_inference(patches, classifier)

    # Aggregate slide-level risk score: mean of top-k probs (decision-support signal)
    scored_sorted = sorted(scored, key=lambda d: d["prob"], reverse=True)
    top_k = max(1, min(int(top_k), len(scored_sorted)))
    slide_risk_score = float(np.mean([p["prob"] for p in scored_sorted[:top_k]]))

    top_regions = [
        {"x": p["x"], "y": p["y"], "w": p["w"], "h": p["h"], "prob": round(float(p["prob"]), 4)}
        for p in scored_sorted[:top_k]
    ]

    return {
        "patch_count": len(scored),
        "slide_risk_score": round(slide_risk_score, 4),
        "aggregation": {"method": "topk_mean", "k": top_k},
        "top_regions": top_regions,
        "backend": "demo-statistical",
        "disclaimer": "Demo backend only. Not a diagnostic model. Outputs are decision-support signals.",
    }
