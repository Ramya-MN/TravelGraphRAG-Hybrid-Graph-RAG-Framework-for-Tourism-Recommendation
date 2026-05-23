from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx

from ..data.models import Destination


class GraphRetriever:
    def __init__(self, graph: nx.DiGraph, destinations: List[Destination]):
        self.graph = graph
        self.destinations = destinations
        self._pagerank = self._compute_dest_pagerank()

    def _compute_dest_pagerank(self) -> Dict[int, float]:
        dest_nodes = [f"dest_{d.dest_id}" for d in self.destinations if self.graph.has_node(f"dest_{d.dest_id}")]
        if not dest_nodes:
            return {}

        sub = self.graph.subgraph(dest_nodes).copy()
        if len(sub) == 0:
            return {}

        try:
            pr = nx.pagerank(sub, alpha=0.85)
        except Exception:
            return {}

        out: Dict[int, float] = {}
        for node, val in pr.items():
            did = self.graph.nodes[node].get("dest_id")
            if did:
                out[int(did)] = float(val)
        return out

    @staticmethod
    def _norm(val: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return (val - lo) / (hi - lo)

    def _score(self, d: Destination, constraints: Dict[str, Any]) -> float:
        pop_prior = max(0.0, min(1.0, d.popularity_score / 10.0))
        pr_prior = max(0.0, min(1.0, self._pagerank.get(d.dest_id, 0.0) * 100.0))

        if not constraints:
            # Stronger prior than flat constant for unconstrained queries.
            return 0.65 * pop_prior + 0.35 * pr_prior

        score = 0.0
        total = 0

        season = constraints.get("season")
        if season:
            total += 1
            if season in d.best_seasons:
                score += 1.0
            elif season not in d.avoid_seasons:
                score += 0.2

        trip = constraints.get("trip_type")
        trip_list = constraints.get("trip_types") or []
        if trip:
            total += 1
            if trip in d.trip_types:
                score += 1.0
        elif isinstance(trip_list, list) and trip_list:
            total += 1
            overlap = len(set(trip_list) & set(d.trip_types))
            if overlap > 0:
                score += min(1.0, overlap / max(1, len(set(trip_list))))

        budget = constraints.get("budget_tier")
        if budget:
            total += 1
            if budget == d.budget_tier:
                score += 1.0

        region = constraints.get("region")
        if region:
            total += 1
            if region == d.region:
                score += 1.0

        access = constraints.get("accessibility")
        if access:
            total += 1
            if access == d.accessibility:
                score += 1.0

        permit = constraints.get("permit")
        if permit is False:
            total += 1
            if not d.permits_required:
                score += 1.0

        loc_terms = constraints.get("location_terms") or []
        if isinstance(loc_terms, list) and loc_terms:
            total += 1
            name_l = d.name.lower()
            state_l = d.state.lower()
            district_l = d.district.lower()
            if any((term in name_l) or (term in state_l) or (term in district_l) for term in loc_terms):
                score += 1.0

        if total == 0:
            return 0.65 * pop_prior + 0.35 * pr_prior

        constraint_score = score / total
        loc_bonus = 0.0
        if isinstance(loc_terms, list) and loc_terms:
            name_l = d.name.lower()
            state_l = d.state.lower()
            district_l = d.district.lower()
            if any((term in name_l) or (term in state_l) or (term in district_l) for term in loc_terms):
                loc_bonus = 0.08
        return 0.72 * constraint_score + 0.2 * pop_prior + 0.05 * pr_prior + loc_bonus

    def retrieve(self, constraints: Dict[str, Any], top_k: int = 20) -> List[Tuple[int, float]]:
        scored = [(d.dest_id, self._score(d, constraints)) for d in self.destinations]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def evidence(self, d: Destination, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        ev: List[Dict[str, Any]] = []
        for k, v in constraints.items():
            matched = False
            actual = None
            if k == "season":
                actual = d.best_seasons
                matched = v in d.best_seasons
            elif k == "trip_type":
                actual = d.trip_types
                matched = v in d.trip_types
            elif k == "trip_types":
                actual = d.trip_types
                requested = v if isinstance(v, list) else []
                matched = bool(set(requested) & set(d.trip_types))
            elif k == "budget_tier":
                actual = d.budget_tier
                matched = v == d.budget_tier
            elif k == "region":
                actual = d.region
                matched = v == d.region
            elif k == "accessibility":
                actual = d.accessibility
                matched = v == d.accessibility
            elif k == "permit":
                actual = d.permits_required
                matched = (v is False and not d.permits_required) or (v is True and d.permits_required)
            elif k == "location_terms":
                actual = [d.name, d.state, d.district]
                terms = v if isinstance(v, list) else []
                name_l = d.name.lower()
                state_l = d.state.lower()
                district_l = d.district.lower()
                matched = any((t in name_l) or (t in state_l) or (t in district_l) for t in terms)
            elif k == "intent_mode":
                actual = d.poi_type
                mode = str(v or "")
                if mode == "attraction_only":
                    matched = d.poi_type not in {"hotel", "guest_house", "restaurant", "fast_food", "cafe"}
                elif mode == "food_only":
                    matched = d.poi_type in {"restaurant", "fast_food", "cafe"}
                elif mode == "stay_only":
                    matched = d.poi_type in {"hotel", "guest_house"}
                else:
                    matched = True
            else:
                # Skip unsupported bookkeeping constraints instead of marking them as violations.
                continue

            ev.append({"constraint": k, "requested": v, "actual": actual, "matched": matched})
        return ev

    def get_neighbors(self, dest_id: int, max_n: int = 8) -> List[int]:
        node = f"dest_{dest_id}"
        if not self.graph.has_node(node):
            return []
        out = []
        for n in self.graph.successors(node):
            if self.graph.nodes[n].get("node_type") == "destination":
                did = self.graph.nodes[n].get("dest_id")
                if did:
                    out.append(int(did))
        return out[:max_n]
