from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def precision_at_k(pred: List[int], gt_rel: Dict[int, float], k: int = 5, threshold: float = 0.5) -> float:
    if not pred:
        return 0.0
    rel = {d for d, s in gt_rel.items() if s >= threshold}
    top = pred[:k]
    return sum(1 for d in top if d in rel) / max(1, len(top))


def recall_at_k(pred: List[int], gt_rel: Dict[int, float], k: int = 5, threshold: float = 0.5) -> float:
    rel = {d for d, s in gt_rel.items() if s >= threshold}
    if not rel:
        return 0.0
    top = pred[:k]
    return sum(1 for d in top if d in rel) / len(rel)


def ndcg_at_k(pred: List[int], gt_rel: Dict[int, float], k: int = 5) -> float:
    gains = [gt_rel.get(d, 0.0) for d in pred[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gt_rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(pred: List[int], gt_rel: Dict[int, float], threshold: float = 0.5) -> float:
    rel = {d for d, s in gt_rel.items() if s >= threshold}
    for i, d in enumerate(pred, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(pred: List[int], gt_rel: Dict[int, float], k: int = 5, threshold: float = 0.5) -> float:
    rel = {d for d, s in gt_rel.items() if s >= threshold}
    return 1.0 if any(d in rel for d in pred[:k]) else 0.0


def average_precision_at_k(pred: List[int], gt_rel: Dict[int, float], k: int = 5, threshold: float = 0.5) -> float:
    rel = {d for d, s in gt_rel.items() if s >= threshold}
    if not rel:
        return 0.0
    score = 0.0
    hits = 0
    for i, did in enumerate(pred[:k], start=1):
        if did in rel:
            hits += 1
            score += hits / i
    denom = min(len(rel), k)
    if denom == 0:
        return 0.0
    return score / denom


def average(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([r[k] for r in rows]))
    return out
