from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..data.models import Destination
from .metrics_real import average, average_precision_at_k, hit_rate_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from ..pipeline import HybridGraphRAGPipeline
from ..retrieval.query_parser import parse_constraints


@dataclass
class EvalRow:
    qid: int
    query: str
    category: str
    query_type: str
    setting: str
    precision_at_5: float
    recall_at_5: float
    recall_at_20: float
    recall_at_50: float
    ndcg_at_5: float
    mrr: float
    map_at_5: float
    hit_rate_at_5: float
    coverage_at_5: float
    diversity_at_5: float
    novelty_at_5: float
    intent_consistency_at_5: float
    grounding_score: float
    violation_rate: float
    confidence_mean: float
    confidence_std: float
    runtime_sec: float
    discarded: int


def _relevance(d: Destination, constraints: Dict[str, object]) -> float:
    if not constraints:
        return 0.0

    score = 0.0
    total = 0
    matched = 0
    if "season" in constraints:
        total += 1
        if constraints["season"] in d.best_seasons:
            score += 1.0
    if "trip_type" in constraints:
        total += 1
        if constraints["trip_type"] in d.trip_types:
            score += 1.0
            matched += 1
    elif "trip_types" in constraints:
        total += 1
        req = constraints.get("trip_types")
        req_set = set(req) if isinstance(req, list) else set()
        if req_set:
            overlap = len(req_set & set(d.trip_types))
            score += overlap / max(1, len(req_set))
            if overlap > 0:
                matched += 1
    if "budget_tier" in constraints:
        total += 1
        if constraints["budget_tier"] == d.budget_tier:
            score += 1.0
            matched += 1
    if "region" in constraints:
        total += 1
        if constraints["region"] == d.region:
            score += 1.0
            matched += 1
    if "accessibility" in constraints:
        total += 1
        if constraints["accessibility"] == d.accessibility:
            score += 1.0
            matched += 1
    if "permit" in constraints:
        total += 1
        expected = bool(constraints["permit"])
        if d.permits_required == expected:
            score += 1.0
            matched += 1

    if "location_terms" in constraints:
        total += 1
        terms = constraints.get("location_terms")
        terms = terms if isinstance(terms, list) else []
        name_l = d.name.lower()
        state_l = d.state.lower()
        district_l = d.district.lower()
        if any((t in name_l) or (t in state_l) or (t in district_l) for t in terms):
            score += 1.0
            matched += 1

    if "intent_mode" in constraints:
        total += 1
        mode = str(constraints.get("intent_mode") or "")
        if mode == "attraction_only":
            if d.poi_type not in {"hotel", "guest_house", "restaurant", "fast_food", "cafe"}:
                score += 1.0
                matched += 1
        elif mode == "food_only":
            if d.poi_type in {"restaurant", "fast_food", "cafe"}:
                score += 1.0
                matched += 1
        elif mode == "stay_only":
            if d.poi_type in {"hotel", "guest_house"}:
                score += 1.0
                matched += 1
        else:
            score += 0.5
            matched += 1

    if total == 0:
        return 0.0
    base = score / total

    # Tighten ground-truth: require multiple constraints for broad queries.
    has_loc = "location_terms" in constraints and constraints.get("location_terms")
    has_intent = "intent_mode" in constraints and constraints.get("intent_mode")
    if has_loc and has_intent:
        if matched < 2:
            return 0.0
    elif total >= 3:
        if matched < 2:
            return 0.0

    return base


def _is_broad_query(constraints: Dict[str, object]) -> bool:
    return not constraints.get("location_terms") and not constraints.get("region")


def build_ground_truth(destinations: List[Destination], query: str) -> Dict[int, float]:
    constraints = parse_constraints(query)
    rel = {d.dest_id: _relevance(d, constraints) for d in destinations}
    rel = {k: v for k, v in rel.items() if v >= 0.2}
    if not rel:
        # fallback: at least popularity-driven weak relevance
        rel = {d.dest_id: d.popularity_score / 10.0 for d in destinations}

    # For broad "best X" queries, cap relevance to a top slice that aligns with intent.
    if rel and _is_broad_query(constraints):
        pop_map = {d.dest_id: d.popularity_score for d in destinations}
        ranked = sorted(rel.items(), key=lambda kv: (kv[1], pop_map.get(kv[0], 0.0)), reverse=True)
        cap = min(200, max(50, int(0.1 * len(ranked))))
        rel = dict(ranked[:cap])

    return rel


def evaluate_setting(
    pipeline: HybridGraphRAGPipeline,
    queries: List[Dict[str, str]],
    setting: str,
) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for q in queries:
        use_graph = setting != "no_graph"
        use_sem = setting != "no_semantic"
        use_ref = setting != "no_refinement"
        use_per = setting != "no_personalization"

        t0 = time.time()
        result = pipeline.recommend(
            q["query"],
            top_k=5,
            use_graph=use_graph,
            use_semantic=use_sem,
            use_refinement=use_ref,
            use_personalization=use_per,
        )
        runtime = time.time() - t0

        gt = build_ground_truth(pipeline.destinations, q["query"])
        pred = [c.dest_id for c in result.accepted]
        raw_pred = [c.dest_id for c in result.raw_candidates]
        constraints = result.constraints
        has_loc = "location_terms" in constraints and constraints.get("location_terms")
        query_type = "city" if has_loc else "broad"

        selected = result.accepted[:5]
        regions = set()
        trip_types = set()
        grounding_vals = []
        violations = 0
        novelty_vals = []
        intent_consistency_hits = 0
        has_intent = "intent_mode" in result.constraints
        mode = str(result.constraints.get("intent_mode") or "")
        for c in selected:
            d = pipeline.dest_map.get(c.dest_id)
            if d is not None:
                regions.add(d.region)
                trip_types.update(d.trip_types)
                novelty_vals.append(max(0.0, 1.0 - (d.popularity_score / 10.0)))
                if has_intent:
                    if mode == "attraction_only":
                        if d.poi_type not in {"hotel", "guest_house", "restaurant", "fast_food", "cafe"}:
                            intent_consistency_hits += 1
                    elif mode == "food_only":
                        if d.poi_type in {"restaurant", "fast_food", "cafe"}:
                            intent_consistency_hits += 1
                    elif mode == "stay_only":
                        if d.poi_type in {"hotel", "guest_house"}:
                            intent_consistency_hits += 1
                    else:
                        intent_consistency_hits += 1
            ev = c.evidence or []
            if ev:
                matched = sum(1 for e in ev if e.get("matched"))
                ratio = matched / len(ev)
                grounding_vals.append(ratio)
                if ratio < 0.5:
                    violations += 1

        coverage = len(regions) / max(1, len(selected))
        diversity = len(trip_types) / max(1, len(selected) * 2)
        novelty = float(pd.Series(novelty_vals).mean()) if novelty_vals else 0.0
        intent_consistency = (intent_consistency_hits / len(selected)) if (has_intent and selected) else 1.0
        grounding = sum(grounding_vals) / len(grounding_vals) if grounding_vals else 0.0
        violation_rate = violations / max(1, len(selected))

        conf = [c.confidence for c in selected]
        conf_mean = float(pd.Series(conf).mean()) if conf else 0.0
        conf_std = float(pd.Series(conf).std(ddof=0)) if conf else 0.0

        rows.append(
            EvalRow(
                qid=int(q["qid"]),
                query=q["query"],
                category=q["category"],
                query_type=query_type,
                setting=setting,
                precision_at_5=precision_at_k(pred, gt, 5),
                recall_at_5=recall_at_k(pred, gt, 5),
                recall_at_20=recall_at_k(raw_pred, gt, 20),
                recall_at_50=recall_at_k(raw_pred, gt, 50),
                ndcg_at_5=ndcg_at_k(pred, gt, 5),
                mrr=mrr(pred, gt),
                map_at_5=average_precision_at_k(pred, gt, 5),
                hit_rate_at_5=hit_rate_at_k(pred, gt, 5),
                coverage_at_5=coverage,
                diversity_at_5=diversity,
                novelty_at_5=novelty,
                intent_consistency_at_5=intent_consistency,
                grounding_score=grounding,
                violation_rate=violation_rate,
                confidence_mean=conf_mean,
                confidence_std=conf_std,
                runtime_sec=runtime,
                discarded=result.discarded_count,
            )
        )
    return rows


def run_full_evaluation(pipeline: HybridGraphRAGPipeline, queries: List[Dict[str, str]], settings: List[str], out_dir: str) -> Dict[str, object]:
    all_rows: List[EvalRow] = []
    for setting in settings:
        all_rows.extend(evaluate_setting(pipeline, queries, setting))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    per_query_csv = out_path / "per_query_results.csv"
    df.to_csv(per_query_csv, index=False)

    summary_rows: List[Dict[str, object]] = []
    def _weighted_score(metrics: Dict[str, object]) -> float:
        # Balanced retrieval score: emphasize recall/quality while keeping precision/diversity.
        return (
            0.35 * float(metrics["recall_at_20"])
            + 0.25 * float(metrics["ndcg_at_5"])
            + 0.20 * float(metrics["precision_at_5"])
            + 0.10 * float(metrics["coverage_at_5"])
            + 0.10 * float(metrics["diversity_at_5"])
        )
    for setting, sdf in df.groupby("setting"):
        metrics = average(
            sdf[
                [
                    "precision_at_5",
                    "recall_at_5",
                    "recall_at_20",
                    "recall_at_50",
                    "ndcg_at_5",
                    "mrr",
                    "map_at_5",
                    "hit_rate_at_5",
                    "coverage_at_5",
                    "diversity_at_5",
                    "novelty_at_5",
                    "intent_consistency_at_5",
                    "grounding_score",
                    "violation_rate",
                    "confidence_mean",
                    "confidence_std",
                    "runtime_sec",
                ]
            ].to_dict(orient="records")
        )
        metrics["weighted_score"] = _weighted_score(metrics)
        metrics["setting"] = setting
        metrics["discarded_avg"] = float(sdf["discarded"].mean())
        summary_rows.append(metrics)

    by_query_type: List[Dict[str, object]] = []
    for (setting, qtype), sdf in df.groupby(["setting", "query_type"]):
        metrics = average(
            sdf[
                [
                    "precision_at_5",
                    "recall_at_5",
                    "recall_at_20",
                    "recall_at_50",
                    "ndcg_at_5",
                    "mrr",
                    "map_at_5",
                    "hit_rate_at_5",
                    "coverage_at_5",
                    "diversity_at_5",
                    "novelty_at_5",
                    "intent_consistency_at_5",
                    "grounding_score",
                    "violation_rate",
                    "confidence_mean",
                    "confidence_std",
                    "runtime_sec",
                ]
            ].to_dict(orient="records")
        )
        metrics["weighted_score"] = _weighted_score(metrics)
        metrics["setting"] = setting
        metrics["query_type"] = qtype
        metrics["discarded_avg"] = float(sdf["discarded"].mean())
        by_query_type.append(metrics)

    summary_df = pd.DataFrame(summary_rows).sort_values(by="ndcg_at_5", ascending=False)
    summary_csv = out_path / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = out_path / "evaluation_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "total_queries": len(queries),
                "settings": settings,
                "summary": summary_rows,
                "by_query_type": by_query_type,
                "files": {
                    "per_query_csv": str(per_query_csv),
                    "ablation_csv": str(summary_csv),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "summary_csv": str(summary_csv),
        "per_query_csv": str(per_query_csv),
        "summary_json": str(summary_json),
    }
