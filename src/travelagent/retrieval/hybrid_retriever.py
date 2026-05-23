from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ..data.models import Destination
from .graph_retriever import GraphRetriever
from .semantic_retriever import SemanticRetriever
from .sparse_retriever import SparseRetriever


@dataclass
class Candidate:
    dest_id: int
    name: str
    s_sem: float = 0.0
    s_sparse: float = 0.0
    s_graph: float = 0.0
    confidence: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    refinement_iter: int = 0


class HybridRetriever:
    def __init__(
        self,
        destinations: List[Destination],
        semantic: SemanticRetriever,
        graph: GraphRetriever,
        sparse: SparseRetriever | None = None,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        if abs((alpha + beta) - 1.0) > 1e-9:
            raise ValueError("alpha + beta must be 1.0")
        self.destinations = destinations
        self.dest_map = {d.dest_id: d for d in destinations}
        self.semantic = semantic
        self.graph = graph
        self.sparse = sparse
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return {k: 0.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    @staticmethod
    def _rrf(ranked_items: List[Tuple[int, float]], k: int = 60) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for rank, (did, _) in enumerate(ranked_items, start=1):
            out[did] = out.get(did, 0.0) + 1.0 / (k + rank)
        return out

    def _adaptive_weights(self, constraints: Dict[str, Any]) -> Tuple[float, float]:
        c = 0
        for key in ["trip_type", "trip_types", "season", "budget_tier", "region", "accessibility", "permit", "location_terms"]:
            if key in constraints and constraints.get(key):
                c += 1

        # More explicit constraints -> trust graph side more.
        graph_boost = min(0.25, 0.04 * c)
        sem = max(0.05, self.alpha - graph_boost)
        gra = max(0.05, self.beta + graph_boost)
        z = sem + gra
        return sem / z, gra / z

    def _dest_similarity(self, a: Destination, b: Destination) -> float:
        score = 0.0
        score += 0.3 if a.state == b.state else 0.0
        score += 0.15 if a.district == b.district else 0.0
        score += 0.2 if a.region == b.region else 0.0

        ta = set(a.trip_types)
        tb = set(b.trip_types)
        if ta and tb:
            score += 0.5 * (len(ta & tb) / max(1, len(ta | tb)))
        return min(1.0, score)

    def _mmr_select(self, candidates: List[Candidate], top_k: int, lam: float = 0.70) -> List[Candidate]:
        if len(candidates) <= top_k:
            return candidates

        selected: List[Candidate] = []
        pool = candidates[:]

        while pool and len(selected) < top_k:
            best_i = 0
            best_score = -1e9
            for i, cand in enumerate(pool):
                if not selected:
                    mmr = cand.confidence
                else:
                    d1 = self.dest_map.get(cand.dest_id)
                    max_sim = 0.0
                    if d1 is not None:
                        for prev in selected:
                            d2 = self.dest_map.get(prev.dest_id)
                            if d2 is None:
                                continue
                            sim = self._dest_similarity(d1, d2)
                            if sim > max_sim:
                                max_sim = sim
                    mmr = lam * cand.confidence - (1.0 - lam) * max_sim

                if mmr > best_score:
                    best_score = mmr
                    best_i = i

            selected.append(pool.pop(best_i))

        return selected

    def _graph_coherence_boost(self, ranked: List[Candidate], window: int = 30) -> List[Candidate]:
        if not ranked:
            return ranked

        top = ranked[:window]
        top_ids = {c.dest_id for c in top}
        boosted: List[Candidate] = []
        for c in top:
            neigh = set(self.graph.get_neighbors(c.dest_id, max_n=12))
            support = len(neigh & top_ids)
            delta = min(0.08, 0.01 * support)
            c.confidence = max(0.0, min(1.0, c.confidence + delta))
            boosted.append(c)

        tail = ranked[window:]
        merged = boosted + tail
        merged.sort(key=lambda x: x.confidence, reverse=True)
        return merged

    def retrieve(self, query: str, constraints: Dict[str, Any], top_k: int = 20) -> List[Candidate]:
        pool_k = max(40, top_k * 6)
        sem_ranked = self.semantic.retrieve(query, top_k=pool_k)
        gra_ranked = self.graph.retrieve(constraints, top_k=pool_k)
        spa_ranked = self.sparse.retrieve(query, top_k=pool_k) if self.sparse else []

        sem = dict(sem_ranked)
        gra = dict(gra_ranked)
        spa = dict(spa_ranked)
        ids = set(sem.keys()) | set(gra.keys()) | set(spa.keys())

        sem_n = self._normalize(sem)
        gra_n = self._normalize(gra)
        spa_n = self._normalize(spa)
        sem_rrf = self._rrf(sem_ranked)
        gra_rrf = self._rrf(gra_ranked)
        spa_rrf = self._rrf(spa_ranked)

        wa, wb = self._adaptive_weights(constraints)

        out: List[Candidate] = []
        trip_targets = set(constraints.get("trip_types") or ([] if "trip_type" not in constraints else [constraints["trip_type"]]))
        loc_terms = constraints.get("location_terms") or []
        intent_mode = str(constraints.get("intent_mode") or "")

        for did in ids:
            d = self.dest_map.get(did)
            if d is None:
                continue

            s_sem = float(sem_n.get(did, 0.0))
            s_graph = float(gra_n.get(did, 0.0))
            s_sparse = float(spa_n.get(did, 0.0))

            rrf_parts = [sem_rrf.get(did, 0.0), gra_rrf.get(did, 0.0)]
            if spa_rrf:
                rrf_parts.append(spa_rrf.get(did, 0.0))
            rrf_score = sum(rrf_parts) / max(1, len(rrf_parts))

            pop_prior = max(0.0, min(1.0, d.popularity_score / 10.0))
            long_tail = max(0.0, 1.0 - pop_prior)
            loc_bonus = 0.0
            if isinstance(loc_terms, list) and loc_terms:
                name_l = d.name.lower()
                state_l = d.state.lower()
                district_l = d.district.lower()
                matched_loc = any((t in name_l) or (t in state_l) or (t in district_l) for t in loc_terms)
                if matched_loc:
                    loc_bonus = 0.14
                else:
                    loc_bonus = -0.18

            trip_penalty = 0.0
            if trip_targets:
                overlap = len(set(d.trip_types) & trip_targets)
                if overlap == 0:
                    trip_penalty = -0.14
                else:
                    trip_penalty = min(0.1, 0.05 * overlap)

            poi_bonus = 0.0
            if intent_mode == "attraction_only":
                if d.poi_type in {"hotel", "guest_house", "restaurant", "fast_food", "cafe"}:
                    poi_bonus = -0.2
                elif d.poi_type in {
                    "attraction",
                    "monument",
                    "museum",
                    "fort",
                    "temple",
                    "viewpoint",
                    "peak",
                    "beach",
                    "national_park",
                    "nature_reserve",
                    "archaeological_site",
                }:
                    poi_bonus = 0.14
            elif intent_mode == "food_only":
                if d.poi_type in {"restaurant", "fast_food", "cafe", "food_court"}:
                    poi_bonus = 0.1
                else:
                    poi_bonus = -0.12
            elif intent_mode == "stay_only":
                if d.poi_type in {"hotel", "guest_house", "hostel", "resort"}:
                    poi_bonus = 0.1
                else:
                    poi_bonus = -0.12

            tail_bonus = 0.04 * long_tail
            conf = (
                0.53 * (wa * s_sem + wb * s_graph)
                + 0.10 * s_sparse
                + 0.30 * rrf_score
                + 0.08 * pop_prior
                + loc_bonus
                + trip_penalty
                + poi_bonus
                + tail_bonus
            )
            conf = max(0.0, min(1.0, conf))
            out.append(
                Candidate(
                    dest_id=did,
                    name=d.name,
                    s_sem=s_sem,
                    s_sparse=s_sparse,
                    s_graph=s_graph,
                    confidence=conf,
                    evidence=self.graph.evidence(d, constraints),
                )
            )

        out.sort(key=lambda x: x.confidence, reverse=True)
        out = self._graph_coherence_boost(out, window=min(40, max(12, top_k * 4)))
        # Avoid repeated names in final output list.
        deduped: List[Candidate] = []
        seen_names = set()
        for c in out:
            key = c.name.strip().lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            deduped.append(c)

        # Intent-aware post-filtering to reduce obvious mismatches.
        if intent_mode == "attraction_only":
            attraction_first = [
                c
                for c in deduped
                if (self.dest_map.get(c.dest_id) is not None)
                and self.dest_map[c.dest_id].poi_type
                not in {"hotel", "guest_house", "restaurant", "fast_food", "cafe"}
            ]
            fallback = [c for c in deduped if c not in attraction_first]
            deduped = attraction_first + fallback
            if len(attraction_first) >= top_k:
                deduped = attraction_first
        elif intent_mode == "food_only":
            food_first = [
                c
                for c in deduped
                if (self.dest_map.get(c.dest_id) is not None)
                and self.dest_map[c.dest_id].poi_type in {"restaurant", "fast_food", "cafe"}
            ]
            fallback = [c for c in deduped if c not in food_first]
            deduped = food_first + fallback
            if len(food_first) >= top_k:
                deduped = food_first
        elif intent_mode == "stay_only":
            stay_first = [
                c
                for c in deduped
                if (self.dest_map.get(c.dest_id) is not None)
                and self.dest_map[c.dest_id].poi_type in {"hotel", "guest_house"}
            ]
            fallback = [c for c in deduped if c not in stay_first]
            deduped = stay_first + fallback
            if len(stay_first) >= top_k:
                deduped = stay_first

        # Diversify final ranking to avoid near-duplicates (MMR).
        ranked = self._mmr_select(deduped, top_k=top_k)

        # Inject a controlled long-tail item if it does not hurt confidence too much.
        if ranked and len(deduped) > top_k:
            base_min = ranked[-1].confidence
            tail_candidate = None
            for c in deduped[top_k:top_k + 25]:
                d = self.dest_map.get(c.dest_id)
                if d is None:
                    continue
                if d.popularity_score >= 7.5:
                    continue
                if c.confidence >= base_min - 0.06:
                    tail_candidate = c
                    break
            if tail_candidate is not None and tail_candidate not in ranked:
                ranked = ranked[:-1] + [tail_candidate]

        return ranked
