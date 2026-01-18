# System Architecture — Assistive Cancer Detection (Histology)

This document describes the high-level architecture of the **assistive cancer screening** system
for **digitized histopathology slides (H&E-stained tissue)**.

The goal is to support **decision support** in clinical workflows by producing **region-level risk signals**
that can guide review — not replace a pathologist.

> **Disclaimer:** This is not a diagnostic tool. Outputs are intended for research/engineering demonstration
> and decision-support framing only.

---

## What Are “Histopathology (H&E) Images”?

Histopathology images are **digitized microscope slides** of tissue samples stained with **Hematoxylin & Eosin (H&E)**:
- **Hematoxylin** stains nuclei (often bluish/purple)
- **Eosin** stains cytoplasm/extracellular components (often pink)

In modern pathology, slides are frequently scanned into **Whole Slide Images (WSI)** which are extremely high resolution
and cannot be processed as a single image in memory efficiently. This drives the need for **patch-based** processing.

---

## Design Goals

- **Patch-based processing** to handle large slides efficiently
- **Model-agnostic inference interface** (swap models without rewriting pipeline)
- **Deterministic preprocessing** (repeatable results across runs)
- **Explainable outputs** (where did risk come from?)
- **Privacy-first** (no PHI / patient-identifying data stored in repo)

---

## High-Level Flow

```text
Whole Slide Image (WSI) / High-Res Histology Image
                 |
                 v
      Tissue Detection / Filtering (optional)
     (remove background / whitespace regions)
                 |
                 v
        Patch Extraction (tiling)
   - fixed patch size (e.g., 256x256)
   - stride / overlap control
   - patch coordinates preserved
                 |
                 v
        Preprocessing / Normalization
   - resizing, dtype normalization
   - optional stain normalization
                 |
                 v
          Model Inference (CNN)
   - patch-level probabilities
   - batch inference for throughput
                 |
                 v
      Aggregation + Heatmap Generation
   - combine patch predictions
   - reconstruct slide-level map
   - identify top regions of interest
                 |
                 v
      Output (Decision Support)
   - JSON summary + patch coordinates
   - optional heatmap artifact
   - review-friendly signals
```

---

Components

1) Preprocessing (preprocessing/patch_extraction.py)

Responsible for:
	•	validating image inputs (format, size)
	•	splitting large images into patches
	•	tracking patch metadata:
	•	(x, y) coordinate
	•	patch size
	•	slide/source id (if available)

Why it matters: without coordinate metadata, you can’t reconstruct where risk signals came from.

---

2) Model Interface (model/inference.py)

Provides a stable interface for inference, independent of model framework.

Responsibilities:
	•	define an inference contract (input patches → output predictions)
	•	support batching
	•	return patch-level scores with coordinates preserved

Training, weights, and proprietary datasets are excluded.

---

3) Evaluation (evaluation/metrics.py)

Provides evaluation utilities for screening systems, focusing on:
	•	sensitivity/recall-friendly metrics
	•	ROC-AUC / PR-AUC
	•	threshold-based decision support (not diagnosis)

This keeps evaluation logic consistent and repeatable.

---

4) Example Artifacts (examples/sample_output.json)

Stores safe example outputs that demonstrate:
	•	what the system returns
	•	how results are structured
	•	how “top suspicious patches” are represented

No PHI. No real patient data.

---

Output Contract (Representative)

The system returns:
	•	slide-level risk score (aggregate)
	•	patch-level probabilities
	•	top-k patches (for review)
	•	coordinates to enable visualization overlays

Example structure (see examples/sample_output.json).

---

Scope & Constraints

Included:
	•	pipeline architecture and interfaces
	•	patch extraction + inference contracts
	•	evaluation utilities and example outputs

Excluded:
	•	datasets and annotations
	•	model weights / training code
	•	deployment secrets and production infra configs

---

Summary

This repo demonstrates production-oriented ML engineering thinking:
patch-based processing, stable interfaces, traceable outputs, and decision-support design.
It is intentionally lightweight while reflecting how clinical ML systems are typically built.
