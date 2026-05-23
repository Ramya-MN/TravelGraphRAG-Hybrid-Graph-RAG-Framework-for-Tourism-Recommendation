from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AblationResult:
    setting: str
    precision_at_5: float
    recall_at_5: float
    ndcg_at_5: float
    mrr: float
    runtime_sec: float


def run_ablation_stub(settings: List[str]) -> List[AblationResult]:
    """Scaffold results so experiment/reporting plumbing can be built first.

    Replace with real pipeline execution once retrieval modules are ported.
    """
    template: Dict[str, Dict[str, float]] = {
        "full": {"p5": 0.78, "r5": 0.66, "ndcg5": 0.74, "mrr": 0.71, "t": 1.00},
        "no_graph": {"p5": 0.65, "r5": 0.56, "ndcg5": 0.61, "mrr": 0.58, "t": 0.85},
        "no_semantic": {"p5": 0.62, "r5": 0.52, "ndcg5": 0.59, "mrr": 0.55, "t": 0.80},
        "no_refinement": {"p5": 0.70, "r5": 0.59, "ndcg5": 0.67, "mrr": 0.63, "t": 0.73},
        "no_personalization": {"p5": 0.68, "r5": 0.58, "ndcg5": 0.65, "mrr": 0.61, "t": 0.90},
    }

    out: List[AblationResult] = []
    for setting in settings:
        v = template.get(setting, {"p5": 0.0, "r5": 0.0, "ndcg5": 0.0, "mrr": 0.0, "t": 0.0})
        out.append(
            AblationResult(
                setting=setting,
                precision_at_5=v["p5"],
                recall_at_5=v["r5"],
                ndcg_at_5=v["ndcg5"],
                mrr=v["mrr"],
                runtime_sec=v["t"],
            )
        )
    return out
