from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..retrieval.hybrid_retriever import Candidate, HybridRetriever


@dataclass
class RefinementResult:
    accepted: List[Candidate]
    discarded: List[Candidate]
    log: List[Dict[str, object]]


class SelfRefiner:
    def __init__(
        self,
        retriever: HybridRetriever,
        theta_accept: float = 0.55,
        theta_discard: float = 0.20,
        max_iter: int = 2,
        neighbor_bonus: float = 0.12,
    ):
        self.retriever = retriever
        self.theta_accept = theta_accept
        self.theta_discard = theta_discard
        self.max_iter = max_iter
        self.neighbor_bonus = neighbor_bonus

    def _boost_from_neighbors(self, c: Candidate) -> float:
        neighbors = self.retriever.graph.get_neighbors(c.dest_id, max_n=6)
        if not neighbors:
            return 0.0
        vals = []
        for nid in neighbors:
            d = self.retriever.dest_map.get(nid)
            if d is None:
                continue
            vals.append(d.popularity_score / 10.0)
        if not vals:
            return 0.0
        return min(self.neighbor_bonus, sum(vals) / len(vals) * self.neighbor_bonus)

    @staticmethod
    def _evidence_match_ratio(c: Candidate) -> float:
        ev = c.evidence or []
        if not ev:
            return 0.0
        matched = sum(1 for e in ev if e.get("matched"))
        return matched / len(ev)

    def run(self, candidates: List[Candidate]) -> RefinementResult:
        accepted: List[Candidate] = []
        discarded: List[Candidate] = []
        log: List[Dict[str, object]] = []

        for c in candidates:
            original = c.confidence
            match_ratio = self._evidence_match_ratio(c)

            # Evidence-aware calibration prior to iterative refinement.
            if c.evidence:
                c.confidence = max(0.0, min(1.0, c.confidence + 0.12 * (match_ratio - 0.5)))
                if match_ratio >= 0.6:
                    c.confidence = min(1.0, c.confidence + 0.06)

            # Recover semantic-rich candidates that might be under-scored by graph constraints.
            if c.s_sem >= 0.62 and c.confidence < self.theta_accept:
                c.confidence = min(1.0, c.confidence + 0.06)

            if c.confidence >= self.theta_accept:
                accepted.append(c)
                continue

            # Only hard-discard candidates when both retrieval channels and evidence are weak.
            hard_low = c.confidence < self.theta_discard and c.s_sem < 0.2 and c.s_graph < 0.2 and match_ratio < 0.25
            if hard_low:
                discarded.append(c)
                continue

            current = c
            for i in range(1, self.max_iter + 1):
                bonus = self._boost_from_neighbors(current)
                current.confidence = min(1.0, current.confidence + bonus)
                current.refinement_iter = i
                log.append(
                    {
                        "dest_id": current.dest_id,
                        "name": current.name,
                        "iteration": i,
                        "confidence_before": round(original, 4),
                        "confidence_after": round(current.confidence, 4),
                        "bonus": round(bonus, 4),
                        "evidence_match_ratio": round(match_ratio, 4),
                    }
                )
                if current.confidence >= self.theta_accept:
                    break

            if current.confidence >= self.theta_accept:
                accepted.append(current)
            elif current.confidence < self.theta_discard and match_ratio < 0.35 and current.s_sem < 0.3:
                discarded.append(current)
            else:
                accepted.append(current)

        accepted.sort(key=lambda x: x.confidence, reverse=True)
        return RefinementResult(accepted=accepted, discarded=discarded, log=log)
