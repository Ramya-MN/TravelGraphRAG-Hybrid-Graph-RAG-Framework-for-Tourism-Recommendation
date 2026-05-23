"""
retriever.py
════════════
GraphRAG retriever for the India Travel Knowledge Graph.
Implements:
  - Dual-Evidence Retrieval (vector + graph PPR)
  - Evidence-Aware Confidence Scoring
  - Self-Refining Retrieval Loop
  - Hard Constraint Filtering
"""

import numpy as np
import networkx as nx
from collections import defaultdict

# Retrieval config
PPR_ALPHA           = 0.88
PPR_MAX_ITER        = 200
PPR_TOL             = 1e-6
PPR_SEED_THRESHOLD  = 0.40
HARD_FILTER_THRESHOLD = 0.45
DEFAULT_VECTOR_WEIGHT = 0.35
DEFAULT_GRAPH_WEIGHT  = 0.65
CONFIDENCE_THRESHOLD  = 0.45   # Self-refining: re-retrieve below this
MAX_REFINEMENT_ITERS  = 2
RANKING_FINAL_WEIGHT   = 0.70
RANKING_CONF_WEIGHT    = 0.30

def _normalize(scores: dict) -> dict:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    rng = mx - mn
    if rng == 0:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / rng for k, v in scores.items()}

class HardConstraintFilter:

    def __init__(self, G: nx.DiGraph, dest_names: list):
        self.G = G
        self.dest_names = dest_names

    def _get_dest_attr(self, name, attr, default=None):
        return self.G.nodes.get(name, {}).get(attr, default)

    def extract_constraints(self, matched_nodes: list,
                            geo_constraint: dict = None) -> dict:
        constraints = {
            "climates": [],
            "budget_tiers": [],
            "seasons_best": [],
            "accessibility": [],
            "states": [],
            "filter_summary": [],
        }

        for m in matched_nodes:
            node      = m["node"]
            score     = m["score"]
            node_type = m.get("node_type", "")

            if score < HARD_FILTER_THRESHOLD:
                continue

            if node_type == "Climate" or node.startswith("Climate_"):
                val = node.replace("Climate_", "")
                constraints["climates"].append(val)
                constraints["filter_summary"].append(
                    f"Climate must be '{val}' (score {score:.2f})")

            elif node_type == "BudgetTier" or node.startswith("Budget_"):
                val = node.replace("Budget_", "")
                constraints["budget_tiers"].append(val)
                constraints["filter_summary"].append(
                    f"Budget tier must be '{val}' (score {score:.2f})")

            elif node_type == "Season" or node.startswith("Season_"):
                val = node.replace("Season_", "")
                constraints["seasons_best"].append(val)
                constraints["filter_summary"].append(
                    f"Best season must include '{val}' (score {score:.2f})")

            elif node_type == "Accessibility" or node.startswith("Accessibility_"):
                val = node.replace("Accessibility_", "")
                constraints["accessibility"].append(val)
                constraints["filter_summary"].append(
                    f"Accessibility must be '{val}' (score {score:.2f})")

            elif node_type == "State" or node.startswith("State_"):
                val = node.replace("State_", "")
                constraints["states"].append(val)
                constraints["filter_summary"].append(
                    f"State must be '{val}' (score {score:.2f})")

        if geo_constraint and geo_constraint.get("detected"):
            for s in geo_constraint["states"]:
                if s not in constraints["states"]:
                    constraints["states"].append(s)
            constraints["filter_summary"].append(
                f"Geographic filter: {geo_constraint['states']}")

        return constraints

    def filter_cities(self, candidates: list,
                      constraints: dict) -> tuple[list, dict]:
        has_constraint = any([
            constraints["climates"],
            constraints["budget_tiers"],
            constraints["seasons_best"],
            constraints["accessibility"],
            constraints["states"],
        ])

        if not has_constraint:
            return candidates, {
                "total_before": len(candidates),
                "total_after": len(candidates),
                "filtered_out": 0,
                "constraints_applied": 0,
                "reason": "No hard constraints",
                "filter_summary": [],
                "rejection_reasons": {},
            }

        filtered = []
        rejection_reasons = {}
        for name in candidates:
            node = self.G.nodes.get(name, {})
            reasons = []

            if constraints["climates"]:
                city_climate = node.get("climate_category", "")
                if city_climate not in constraints["climates"]:
                    reasons.append(
                        f"Climate '{city_climate}' not in "
                        f"{constraints['climates']}"
                    )

            if constraints["budget_tiers"]:
                city_budget = node.get("budget_tier", "")
                if city_budget not in constraints["budget_tiers"]:
                    reasons.append(
                        f"Budget '{city_budget}' not in "
                        f"{constraints['budget_tiers']}"
                    )

            if constraints["accessibility"]:
                city_acc = node.get("accessibility", "")
                if city_acc not in constraints["accessibility"]:
                    reasons.append(
                        f"Accessibility '{city_acc}' not in "
                        f"{constraints['accessibility']}"
                    )

            if constraints["states"]:
                city_state = node.get("state", "")
                if not any(s.lower() in city_state.lower()
                           for s in constraints["states"]):
                    reasons.append(
                        f"State '{city_state}' does not match "
                        f"{constraints['states']}"
                    )

            if constraints["seasons_best"]:
                # Check BEST_IN edges
                city_seasons = [
                    self.G.nodes[nbr].get("season", "")
                    for _, nbr, d in self.G.out_edges(name, data=True)
                    if d.get("relation") == "BEST_IN"
                ]
                if not any(s in city_seasons
                           for s in constraints["seasons_best"]):
                    reasons.append(
                        f"Best season {city_seasons or 'N/A'} does not include "
                        f"{constraints['seasons_best']}"
                    )

            if reasons:
                rejection_reasons[name] = reasons
            else:
                filtered.append(name)

        # Fallback: relax constraints if nothing passes
        if not filtered:
            filtered = candidates

        report = {
            "total_before":        len(candidates),
            "total_after":         len(filtered),
            "filtered_out":        len(candidates) - len(filtered),
            "constraints_applied": sum(1 for k in ["climates","budget_tiers",
                                                    "seasons_best","accessibility",
                                                    "states"]
                                       if constraints[k]),
            "filter_summary":      constraints["filter_summary"],
            "fallback":            (filtered == candidates
                                    and len(candidates) > 0),
            "sample_rejections":   dict(list(rejection_reasons.items())[:5]),
            "rejection_reasons":   rejection_reasons,
        }
        return filtered, report


class GraphRAGRetriever:

    def __init__(self, G: nx.DiGraph, dest_names: list,
                 query_engine, metadata: dict = None):
        self.G          = G
        self.dest_names = dest_names
        self.dest_set   = set(dest_names)
        self.qe         = query_engine
        self.metadata   = metadata or {}
        self.filter     = HardConstraintFilter(G, dest_names)

    # ── Public retrieval methods ──────────────────────────────────────────────

    def retrieve_vector_only(self, query: str, top_k: int = 5) -> list:
        results = self.qe.vector_search(query, top_k=top_k)
        return [{"city": r["city"], "score": r["score"],
                 "method": "vector"} for r in results]

    def retrieve_graph_only(self, query: str,
                            top_k: int = 5) -> tuple[list, list]:
        matched = self.qe.parse_query_to_graph_nodes(query)
        if not matched:
            return [], matched

        personalization = {
            m["node"]: m["score"] ** 2
            for m in matched
            if m["node"] in self.G and m["score"] >= PPR_SEED_THRESHOLD
        }
        if not personalization:
            return [], matched

        ppr = nx.pagerank(
            self.G, alpha=PPR_ALPHA,
            personalization=personalization,
            max_iter=PPR_MAX_ITER, tol=PPR_TOL)

        city_scores = sorted(
            [(n, s) for n, s in ppr.items() if n in self.dest_set],
            key=lambda x: x[1], reverse=True)

        return ([{"city": n, "score": s, "method": "graph_ppr"}
                 for n, s in city_scores[:top_k]], matched)

    def retrieve_graph_rag(self, query: str, top_k: int = 5,
                           vector_weight: float = None,
                           graph_weight: float = None) -> tuple[list, list]:
        if vector_weight is None: vector_weight = DEFAULT_VECTOR_WEIGHT
        if graph_weight  is None: graph_weight  = DEFAULT_GRAPH_WEIGHT

        refinement_report = {
            "iterations": [],
            "enabled": True,
            "threshold": CONFIDENCE_THRESHOLD,
        }

        results, matched, retrieval_report = self._retrieve_once(
            query, top_k, vector_weight, graph_weight)

        # ── Self-Refining Loop ────────────────────────────────────────────────
        for iteration in range(MAX_REFINEMENT_ITERS):
            low_conf = [r for r in results
                        if r["confidence_score"] < CONFIDENCE_THRESHOLD]
            if not low_conf:
                break

            print(f"  🔄 Refinement iter {iteration+1}: "
                  f"{len(low_conf)} low-confidence results. Re-retrieving...")

            iteration_report = {
                "iteration": iteration + 1,
                "removed": [
                    {
                        "city": r["city"],
                        "confidence_score": r["confidence_score"],
                        "final_score": r["final_score"],
                        "reason": (
                            f"confidence < {CONFIDENCE_THRESHOLD}"
                        ),
                    }
                    for r in low_conf
                ],
                "added": [],
            }

            # Broaden search: temporarily lower vector weight to cast wider net
            results_refined, _, retrieval_report = self._retrieve_once(
                query, top_k * 2,
                vector_weight * 0.8,
                graph_weight * 1.2)

            # Replace low-confidence results with better alternatives
            high_conf_cities = {r["city"] for r in results
                                if r["confidence_score"] >= CONFIDENCE_THRESHOLD}
            new_additions = [r for r in results_refined
                             if r["city"] not in high_conf_cities
                             and r["confidence_score"] >= CONFIDENCE_THRESHOLD]

            iteration_report["added"] = [
                {
                    "city": r["city"],
                    "confidence_score": r["confidence_score"],
                    "final_score": r["final_score"],
                    "reason": "higher confidence replacement",
                }
                for r in new_additions
            ]

            # Merge: keep high-confidence originals + add new ones
            final = [r for r in results
                     if r["confidence_score"] >= CONFIDENCE_THRESHOLD]
            final.extend(new_additions)
            final.sort(key=lambda x: x["final_score"], reverse=True)
            results = final[:top_k]

            refinement_report["iterations"].append(iteration_report)

            if len(results) >= top_k:
                break

        for r in results:
            r["refinement_report"] = refinement_report
            r["retrieval_report"] = retrieval_report

        return results, matched

    # ── Internal retrieval ────────────────────────────────────────────────────

    def _retrieve_once(self, query: str, top_k: int,
                       vector_weight: float,
                       graph_weight: float) -> tuple[list, list, dict]:
        # Step 1: Parse query
        matched = self.qe.parse_query_to_graph_nodes(query)
        geo     = self.qe.detect_geographic_constraint(query)

        # Step 2: Hard constraint filtering
        constraints = self.filter.extract_constraints(matched, geo)
        filtered, filter_report = self.filter.filter_cities(
            self.dest_names, constraints)
        filtered_set = set(filtered)

        print(f"\n  🔍 Constraints: {len(constraints['filter_summary'])} active")
        for s in constraints["filter_summary"]:
            print(f"     • {s}")
        print(f"     → {filter_report['total_before']} → "
              f"{filter_report['total_after']} destinations")

        # Step 3: Vector search
        vec_results = self.qe.vector_search(query, top_k=top_k * 4)
        vec_scores  = {r["city"]: r["score"]
                      for r in vec_results if r["city"] in filtered_set}

        # Step 4: Graph PPR search
        personalization = {
            m["node"]: m["score"] ** 2
            for m in matched
            if m["node"] in self.G and m["score"] >= PPR_SEED_THRESHOLD
        }

        graph_scores = {}
        if personalization:
            ppr = nx.pagerank(
                self.G, alpha=PPR_ALPHA,
                personalization=personalization,
                max_iter=PPR_MAX_ITER, tol=PPR_TOL)
            graph_scores = {n: s for n, s in ppr.items()
                           if n in filtered_set}

        # Step 5: Normalize both score sets
        vec_norm   = _normalize(vec_scores)
        graph_norm = _normalize(graph_scores)

        # Step 6: Score fusion
        all_cities = filtered_set
        results = []
        for city in all_cities:
            v = vec_norm.get(city, 0.0)
            g = graph_norm.get(city, 0.0)

            final = vector_weight * v + graph_weight * g

            # Evidence extraction
            evidence = self._extract_evidence(city, matched)
            direct_evidence = sum(
                1 for e in evidence if e["evidence_type"] == "direct"
            )
            total_evidence = len(evidence)

            # Confidence score (Evidence-Aware)
            node_attrs = self.G.nodes.get(city, {})
            confidence = self._compute_confidence(
                city, evidence, node_attrs, final)

            ranking_score = (
                RANKING_FINAL_WEIGHT * final
                + RANKING_CONF_WEIGHT * confidence
            )

            results.append({
                "city":             city,
                "final_score":      round(final, 4),
                "vector_score":     round(v, 4),
                "graph_score":      round(g, 4),
                "confidence_score": round(confidence, 4),
                "ranking_score":    round(ranking_score, 4),
                "evidence_paths":   evidence,
                "evidence_stats":   {
                    "direct": direct_evidence,
                    "total": total_evidence,
                },
                "matched_nodes":    [m["node"] for m in matched],
                "city_attrs":       dict(node_attrs),
                "filter_report":    filter_report,
                "geo_constraint":   geo,
                "weights_used":     {"vector": vector_weight,
                                     "graph": graph_weight},
                "validation":       self._validate(city, matched),
                "method":           "graph_rag",
            })

        results.sort(key=lambda x: x["ranking_score"], reverse=True)
        ranked_out = []
        for rank, r in enumerate(results[top_k:top_k + 10], start=top_k + 1):
            reasons = [f"ranked {rank} (top_k={top_k})"]
            if r["confidence_score"] < CONFIDENCE_THRESHOLD:
                reasons.append(
                    f"low confidence < {CONFIDENCE_THRESHOLD}"
                )
            if r["evidence_stats"]["direct"] == 0:
                reasons.append("no direct evidence edges")
            if r["vector_score"] == 0:
                reasons.append("vector similarity low/zero")
            if r["graph_score"] == 0:
                reasons.append("graph score low/zero")

            ranked_out.append({
                "city": r["city"],
                "ranking_score": r["ranking_score"],
                "final_score": r["final_score"],
                "vector_score": r["vector_score"],
                "graph_score": r["graph_score"],
                "confidence_score": r["confidence_score"],
                "evidence_stats": r["evidence_stats"],
                "reasons": reasons,
            })

        retrieval_report = {
            "ranked_out": ranked_out,
            "ranked_out_total": max(len(results) - top_k, 0),
        }

        top_results = results[:top_k]
        for r in top_results:
            r["retrieval_report"] = retrieval_report

        return top_results, matched, retrieval_report

    def _extract_evidence(self, city: str, matched_nodes: list) -> list:
        """Extract evidence paths from city to each matched node."""
        paths = []
        for m in matched_nodes:
            target = m["node"]
            if target not in self.G or city not in self.G:
                continue

            if self.G.has_edge(city, target):
                edge = self.G.edges[city, target]
                paths.append({
                    "target":        target,
                    "hops":          1,
                    "relation":      edge.get("relation", "?"),
                    "source":        edge.get("source", "unknown"),
                    "evidence_type": "direct",
                    "path_str":      f"{city} --[{edge.get('relation','?')}]--> {target}",
                    "confidence":    1.0,
                })
                continue

            try:
                path = nx.shortest_path(self.G, city, target)
                if len(path) <= 4:
                    parts = []
                    for i in range(len(path) - 1):
                        e = self.G.edges[path[i], path[i+1]]
                        rel = e.get("relation", "?")
                        src = e.get("source", "")
                        parts.append(
                            f"{path[i]} --[{rel}]--> {path[i+1]}"
                            + (f" [{src}]" if src else ""))
                    hops = len(path) - 1
                    conf = 1.0 / hops
                    paths.append({
                        "target":        target,
                        "hops":          hops,
                        "relation":      "multi-hop",
                        "source":        "graph-traversal",
                        "evidence_type": f"multi_hop_{hops}",
                        "path_str":      " | ".join(parts),
                        "confidence":    round(conf, 3),
                    })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                paths.append({
                    "target":        target,
                    "hops":          -1,
                    "relation":      "none",
                    "source":        "none",
                    "evidence_type": "no_path",
                    "path_str":      f"{city} --[NO PATH]--> {target}",
                    "confidence":    0.0,
                })
        return paths

    def _compute_confidence(self, city: str, evidence: list,
                            node_attrs: dict, final_score: float) -> float:
        """Evidence-Aware Confidence Score."""
        direct_evidence   = sum(1 for e in evidence
                                if e["evidence_type"] == "direct")
        total_evidence    = len(evidence) if evidence else 1
        evidence_strength = direct_evidence / total_evidence

        data_coverage  = (node_attrs.get("coverage_pct", 50)) / 100.0
        safety         = (node_attrs.get("safety_rating") or 5) / 10.0
        node_confidence = node_attrs.get("confidence_score", 0.5)

        return round(
            0.30 * evidence_strength
            + 0.25 * node_confidence
            + 0.20 * data_coverage
            + 0.15 * final_score
            + 0.10 * safety,
            4)

    def _validate(self, city: str, matched_nodes: list) -> dict:
        """Validate city against matched node requirements."""
        checks = []
        for m in matched_nodes:
            node      = m["node"]
            node_type = m.get("node_type", "")
            satisfied = self.G.has_edge(city, node)
            checks.append({
                "requirement":    node,
                "satisfied":      satisfied,
                "score":          1.0 if satisfied else 0.0,
                "constraint_type": ("hard" if node_type in
                                    ["Climate","BudgetTier","State"]
                                    else "soft"),
            })

        if not checks:
            return {"checks": [], "pass_rate": 0.0,
                    "passed": 0, "total": 0,
                    "hard_pass_rate": 1.0, "soft_pass_rate": 1.0,
                    "avg_satisfaction": 0.0}

        passed = sum(1 for c in checks if c["satisfied"])
        total  = len(checks)
        hard   = [c for c in checks if c["constraint_type"] == "hard"]
        soft   = [c for c in checks if c["constraint_type"] == "soft"]

        return {
            "checks":          checks,
            "pass_rate":       round(passed / total, 2),
            "passed":          passed,
            "total":           total,
            "hard_pass_rate":  (round(sum(1 for c in hard if c["satisfied"])
                                / len(hard), 2) if hard else 1.0),
            "soft_pass_rate":  (round(sum(1 for c in soft if c["satisfied"])
                                / len(soft), 2) if soft else 1.0),
            "avg_satisfaction": round(
                sum(c["score"] for c in checks) / total, 3),
        }