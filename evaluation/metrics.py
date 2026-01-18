"""
metrics.py

Evaluation utilities for assistive screening.

This module intentionally focuses on:
- sensitivity/recall-friendly metrics (screening use case)
- ROC-AUC and PR-AUC (implemented without heavy ML libraries)
- threshold selection by target sensitivity

NOTE: This is evaluation code for engineering demonstration.
It is not clinical validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def precision(cm: ConfusionMatrix) -> float:
    denom = cm.tp + cm.fp
    return float(cm.tp / denom) if denom > 0 else 0.0


def recall(cm: ConfusionMatrix) -> float:
    denom = cm.tp + cm.fn
    return float(cm.tp / denom) if denom > 0 else 0.0


def specificity(cm: ConfusionMatrix) -> float:
    denom = cm.tn + cm.fp
    return float(cm.tn / denom) if denom > 0 else 0.0


def f1(cm: ConfusionMatrix) -> float:
    p = precision(cm)
    r = recall(cm)
    denom = p + r
    return float(2 * p * r / denom) if denom > 0 else 0.0


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": float(threshold),
        "tp": float(cm.tp),
        "fp": float(cm.fp),
        "tn": float(cm.tn),
        "fn": float(cm.fn),
        "precision": precision(cm),
        "recall": recall(cm),
        "specificity": specificity(cm),
        "f1": f1(cm),
    }


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns fpr, tpr, thresholds.
    Implemented via sorting unique scores descending.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.unique(y_score)[::-1]
    tpr_list = []
    fpr_list = []

    p = (y_true == 1).sum()
    n = (y_true == 0).sum()
    p = int(p)
    n = int(n)

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tpr = (cm.tp / p) if p > 0 else 0.0
        fpr = (cm.fp / n) if n > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # add (0,0) point for completeness
    fpr = np.array([0.0] + fpr_list + [1.0], dtype=float)
    tpr = np.array([0.0] + tpr_list + [1.0], dtype=float)
    thr = np.array([thresholds.max() + 1e-9] + list(thresholds) + [thresholds.min() - 1e-9], dtype=float)
    return fpr, tpr, thr


def pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns precision, recall, thresholds.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.unique(y_score)[::-1]
    prec_list = []
    rec_list = []

    for thr in thresholds:
        metrics = evaluate_at_threshold(y_true, y_score, float(thr))
        prec_list.append(metrics["precision"])
        rec_list.append(metrics["recall"])

    # Add endpoints: recall=0 with precision=1 at very high threshold
    precision_arr = np.array([1.0] + prec_list, dtype=float)
    recall_arr = np.array([0.0] + rec_list, dtype=float)
    thr_arr = np.array([thresholds.max() + 1e-9] + list(thresholds), dtype=float)
    return precision_arr, recall_arr, thr_arr


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC assuming x is sorted ascending."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(np.trapz(y, x))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    prec, rec, _ = pr_curve(y_true, y_score)
    # PR AUC integrates precision over recall (recall ascending)
    return auc(rec, prec)


def choose_threshold_for_target_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.90,
) -> Dict[str, float]:
    """
    Choose the highest threshold that achieves at least target recall.
    (Screening design: prioritize sensitivity / recall)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.unique(y_score)[::-1]
    best = None

    for thr in thresholds:
        metrics = evaluate_at_threshold(y_true, y_score, float(thr))
        if metrics["recall"] >= target_recall:
            best = metrics
            # keep going (descending) to find highest threshold meeting recall
        else:
            # once recall drops below target, earlier thresholds were higher -> stop
            continue

    if best is None:
        # if nothing meets target recall, fall back to lowest threshold
        thr = float(thresholds.min()) if thresholds.size > 0 else 0.5
        best = evaluate_at_threshold(y_true, y_score, thr)

    return best


def summarize(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Convenience summary for reports.
    """
    out = evaluate_at_threshold(y_true, y_score, threshold)
    out["roc_auc"] = roc_auc(y_true, y_score)
    out["pr_auc"] = pr_auc(y_true, y_score)
    return out
