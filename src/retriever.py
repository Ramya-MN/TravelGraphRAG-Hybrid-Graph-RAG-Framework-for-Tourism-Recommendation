

import networkx as nx
import numpy as np
import pandas as pd
import json
from collections import defaultdict


HARD_CONSTRAINT_TYPES = {"Climate", "Month"}

HOP_DECAY = {
    0: 1.0,
    1: 0.3,
    2: 0.05,
}

HARD_FILTER_THRESHOLD = 0.40

PREFERENCE_FILTER_MAP = {
    "High": 3,
    "Medium": 2,
    "Low": 0,
}

GROUND_TRUTH_THRESHOLDS = {
    "High": 4,
    "Medium": 2,
    "Low": 0,
}

DEFAULT_VECTOR_WEIGHT = 0.40
DEFAULT_GRAPH_WEIGHT = 0.60

PPR_ALPHA = 0.90
PPR_MAX_ITER = 200
PPR_TOL = 1e-06
PPR_SEED_THRESHOLD = 0.42
PPR_SCORE_AMPLIFICATION = 2.0

NORM_RANK_WEIGHT = 0.4
NORM_RAW_WEIGHT = 0.6

HUB_CENTRALITY_THRESHOLD = 0.05
HUB_PENALTY_FACTOR = 0.5

DIRECT_MATCH_BONUS = 0.20
DIRECT_MATCH_MAX_BONUS = 0.60


def _count_similarity_hops(G, path):
    count = 0
    for i in range(len(path) - 1):
        edge_data = G.edges.get((path[i], path[i + 1]), {})
        if edge_data.get("relation") == "SIMILAR_PROFILE":
            count += 1
    return count


def _path_passes_through_similarity(G, path):
    for i in range(len(path) - 1):
        edge_data = G.edges.get((path[i], path[i + 1]), {})
        if edge_data.get("relation") == "SIMILAR_PROFILE":
            return True
    return False


def _get_node_constraint_type(node_name):
    if node_name in ("Cold", "Moderate", "Hot"):
        return "Climate"
    if node_name.startswith("Month_"):
        return "Month"
    return None


def _build_personalization(matched_nodes, G):
    personalization = {}
    skipped = []

    for m in matched_nodes:
        node = m["node"]
        score = m["score"]

        if node not in G:
            continue

        if score < PPR_SEED_THRESHOLD:
            skipped.append(f"{node}({score:.2f})")
            continue

        amplified_score = score ** PPR_SCORE_AMPLIFICATION
        personalization[node] = amplified_score

    return personalization, skipped


def _hybrid_normalize(scores_dict):
    if not scores_dict:
        return {}

    items = list(scores_dict.items())
    n = len(items)

    if n == 1:
        city, score = items[0]
        return {city: score if score > 0 else 0.0}

    raw_scores = [s for _, s in items]
    min_raw = min(raw_scores)
    max_raw = max(raw_scores)
    raw_range = max_raw - min_raw

    raw_normalized = {}
    for city, score in items:
        if raw_range > 0:
            raw_norm = (score - min_raw) / raw_range
            quality_floor = min(max_raw, 1.0)
            raw_norm = raw_norm * quality_floor
        else:
            raw_norm = score if score > 0 else 0.0
        raw_normalized[city] = raw_norm

    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    rank_normalized = {}
    for rank, (city, _) in enumerate(sorted_items):
        rank_score = (n - rank) / n
        rank_normalized[city] = rank_score

    blended = {}
    for city in scores_dict:
        blended[city] = (
            NORM_RAW_WEIGHT * raw_normalized[city]
            + NORM_RANK_WEIGHT * rank_normalized[city]
        )

    return blended


def _check_intermediary_is_hub(G, node):
    node_data = G.nodes.get(node, {})
    centrality = node_data.get("degree_centrality", 0.0)
    is_hub = centrality > HUB_CENTRALITY_THRESHOLD
    return is_hub, centrality


def _compute_direct_match_scores(G, cities, matched_nodes):
    """
    Compute direct match score for each city.
    """
    scores = {}
    total_possible = len(matched_nodes)

    if total_possible == 0:
        return {city: 0.0 for city in cities}

    for city in cities:
        if city not in G:
            scores[city] = 0.0
            continue

        direct_matches = 0

        for m in matched_nodes:
            node = m["node"]
            if G.has_edge(city, node):
                edge_data = G.edges[city, node]
                relation = edge_data.get("relation", "")
                if relation != "SIMILAR_PROFILE":
                    direct_matches += 1

        # Score is simply the proportion of matches
        scores[city] = direct_matches / total_possible

    return scores


class HardConstraintFilter:

    def __init__(self, df, G):
        self.df = df
        self.G = G

        self.city_climate = {}
        self.city_prefs = {}
        self.city_regions = {}

        for _, row in df.iterrows():
            city = row["city"]
            node_data = G.nodes.get(city, {})
            self.city_climate[city] = node_data.get(
                "climate_category", None
            )
            self.city_regions[city] = node_data.get("region", None)
            self.city_prefs[city] = {
                col: row[col] for col in [
                    "culture", "adventure", "nature", "beaches",
                    "nightlife", "cuisine", "wellness", "urban",
                    "seclusion"
                ]
            }

    def extract_constraints(self, matched_nodes, geo_constraint=None):
        constraints = {
            "required_climates": [],
            "required_prefs": [],
            "required_regions": [],
            "required_months": [],
            "geo_keyword": None,
            "filter_summary": [],
        }

        for m in matched_nodes:
            node = m["node"]
            score = m["score"]

            if score < HARD_FILTER_THRESHOLD:
                continue

            if node in ("Cold", "Moderate", "Hot"):
                constraints["required_climates"].append(node)
                constraints["filter_summary"].append(
                    f"Climate must be '{node}' "
                    f"(matched with score {score:.2f})"
                )

            if node.startswith("Month_"):
                constraints["required_months"].append(node)
                constraints["filter_summary"].append(
                    f"Month must include '{node}' "
                    f"(matched with score {score:.2f})"
                )
                continue

            if "_" in node:
                parts = node.split("_")
                if len(parts) == 2:
                    category = parts[0].lower()
                    level = parts[1]

                    if category in self.city_prefs.get(
                        next(iter(self.city_prefs), ""), {}
                    ):
                        min_score = PREFERENCE_FILTER_MAP.get(level, 0)
                        if min_score > 0:
                            constraints["required_prefs"].append(
                                (category, min_score, level)
                            )
                            constraints["filter_summary"].append(
                                f"'{category}' score must be >= "
                                f"{min_score} "
                                f"(query wants {node}, "
                                f"matched score {score:.2f})"
                            )

        if geo_constraint and geo_constraint.get("detected"):
            constraints["required_regions"] = geo_constraint["regions"]
            constraints["geo_keyword"] = geo_constraint["keyword"]
            constraints["filter_summary"].append(
                f"Region must contain one of "
                f"{geo_constraint['regions']} "
                f"(detected keyword: "
                f"'{geo_constraint['keyword']}')"
            )

        return constraints

    def filter_cities(self, candidate_cities, constraints):
        has_any_constraint = (
            constraints["required_climates"]
            or constraints["required_prefs"]
            or constraints["required_regions"]
            or constraints["required_months"]
        )

        if not has_any_constraint:
            return candidate_cities, {
                "total_before": len(candidate_cities),
                "total_after": len(candidate_cities),
                "filtered_out": 0,
                "constraints_applied": 0,
                "reason": "No hard constraints detected",
                "rejection_reasons": {},
            }

        filtered = []
        rejection_reasons = {}

        for city in candidate_cities:
            rejected = False
            reasons = []

            if constraints["required_climates"]:
                city_climate = self.city_climate.get(city, None)
                if city_climate not in constraints["required_climates"]:
                    rejected = True
                    reasons.append(
                        f"Climate '{city_climate}' not in "
                        f"{constraints['required_climates']}"
                    )

            if not rejected:
                city_pref_data = self.city_prefs.get(city, {})
                for category, min_score, level in (
                    constraints["required_prefs"]
                ):
                    actual_score = city_pref_data.get(category, 0)
                    if actual_score < min_score:
                        rejected = True
                        reasons.append(
                            f"'{category}' score {actual_score} < "
                            f"required {min_score} (for {level})"
                        )

            if not rejected and constraints["required_months"]:
                has_required_month = any(
                    self.G.has_edge(city, month)
                    for month in constraints["required_months"]
                )
                if not has_required_month:
                    rejected = True
                    reasons.append(
                        f"Month data missing for "
                        f"{constraints['required_months']}"
                    )

            if not rejected and constraints["required_regions"]:
                city_region = (
                    self.city_regions.get(city, "") or ""
                )
                region_match = False
                for required_region in constraints["required_regions"]:
                    if required_region.lower() in city_region.lower():
                        region_match = True
                        break
                if not region_match:
                    rejected = True
                    reasons.append(
                        f"Region '{city_region}' does not match "
                        f"required {constraints['required_regions']}"
                    )

            if rejected:
                rejection_reasons[city] = reasons
            else:
                filtered.append(city)

        if len(filtered) == 0:
            relaxed = []
            for city in candidate_cities:
                ok = True

                if constraints["required_climates"]:
                    city_climate = self.city_climate.get(city, None)
                    if city_climate not in (
                        constraints["required_climates"]
                    ):
                        ok = False

                if ok and constraints["required_regions"]:
                    city_region = (
                        self.city_regions.get(city, "") or ""
                    )
                    region_match = False
                    for required_region in (
                        constraints["required_regions"]
                    ):
                        if (required_region.lower()
                                in city_region.lower()):
                            region_match = True
                            break
                    if not region_match:
                        ok = False

                if ok:
                    relaxed.append(city)

            if relaxed:
                filtered = relaxed
            else:
                if constraints["required_regions"]:
                    for city in candidate_cities:
                        city_region = (
                            self.city_regions.get(city, "") or ""
                        )
                        for required_region in (
                            constraints["required_regions"]
                        ):
                            if (required_region.lower()
                                    in city_region.lower()):
                                filtered.append(city)
                                break

            if len(filtered) == 0:
                return candidate_cities, {
                    "total_before": len(candidate_cities),
                    "total_after": len(candidate_cities),
                    "filtered_out": 0,
                    "constraints_applied": (
                        len(constraints["required_climates"])
                        + len(constraints["required_prefs"])
                        + (1 if constraints["required_regions"]
                           else 0)
                        + len(constraints["required_months"])
                    ),
                    "reason": (
                        "WARNING: All cities filtered out, "
                        "returning unfiltered"
                    ),
                    "fallback": True,
                    "rejection_reasons": rejection_reasons,
                }

        filter_report = {
            "total_before": len(candidate_cities),
            "total_after": len(filtered),
            "filtered_out": len(candidate_cities) - len(filtered),
            "constraints_applied": (
                len(constraints["required_climates"])
                + len(constraints["required_prefs"])
                + (1 if constraints["required_regions"] else 0)
                + len(constraints["required_months"])
            ),
            "climate_filter": constraints["required_climates"],
            "pref_filters": [
                f"{cat}>={min_s}"
                for cat, min_s, _ in constraints["required_prefs"]
            ],
            "geo_filter": constraints["required_regions"],
            "geo_keyword": constraints["geo_keyword"],
            "month_filter": constraints["required_months"],
            "sample_rejections": dict(
                list(rejection_reasons.items())[:5]
            ),
            "rejection_reasons": rejection_reasons,
            "filter_summary": constraints["filter_summary"],
        }

        return filtered, filter_report


class GraphRAGRetriever:

    def __init__(self, G, city_names, query_engine, df):
        self.G = G
        self.city_names = set(city_names)
        self.city_list = list(city_names)
        self.qe = query_engine
        self.df = df

        self.constraint_filter = HardConstraintFilter(df, G)

        self._city_ground_truth = {}
        for _, row in df.iterrows():
            city = row["city"]
            node_data = G.nodes.get(city, {})
            self._city_ground_truth[city] = {
                "climate": node_data.get("climate_category", None),
                "yearly_avg_temp": node_data.get(
                    "yearly_avg_temp", None
                ),
                "country": node_data.get("country", None),
                "region": node_data.get("region", None),
                "culture": row.get("culture", 0),
                "adventure": row.get("adventure", 0),
                "nature": row.get("nature", 0),
                "beaches": row.get("beaches", 0),
                "nightlife": row.get("nightlife", 0),
                "cuisine": row.get("cuisine", 0),
                "wellness": row.get("wellness", 0),
                "urban": row.get("urban", 0),
                "seclusion": row.get("seclusion", 0),
            }

    def retrieve_vector_only(self, query, top_k=5):
        results = self.qe.vector_search(query, top_k=top_k)
        return [
            {"city": r["city"], "score": r["score"], "method": "vector"}
            for r in results
        ]

    def retrieve_graph_only(self, query, top_k=5):
        matched_nodes = self.qe.parse_query_to_graph_nodes(query)

        if not matched_nodes:
            return [], matched_nodes

        personalization, skipped_seeds = _build_personalization(
            matched_nodes, self.G
        )

        if not personalization:
            return [], matched_nodes

        ppr = nx.pagerank(
            self.G,
            alpha=PPR_ALPHA,
            personalization=personalization,
            max_iter=PPR_MAX_ITER,
            tol=PPR_TOL,
        )

        city_scores = [
            (node, score)
            for node, score in ppr.items()
            if node in self.city_names
        ]
        city_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for city, score in city_scores[:top_k]:
            results.append({
                "city": city, "score": score, "method": "graph_ppr"
            })

        return results, matched_nodes

    def retrieve_graph_rag(self, query, top_k=5,
                           vector_weight=None, graph_weight=None):
        if vector_weight is None:
            vector_weight = DEFAULT_VECTOR_WEIGHT
        if graph_weight is None:
            graph_weight = DEFAULT_GRAPH_WEIGHT

        # ── Step 1: Parse query into graph nodes ──
        matched_nodes = self.qe.parse_query_to_graph_nodes(query)

        # ── Step 2: Detect geographic constraint ──
        geo_constraint = self.qe.detect_geographic_constraint(query)

        # ── Step 3: Extract and apply ALL constraints ──
        constraints = self.constraint_filter.extract_constraints(
            matched_nodes, geo_constraint=geo_constraint
        )
        all_candidates = list(self.city_names)
        filtered_cities, filter_report = (
            self.constraint_filter.filter_cities(
                all_candidates, constraints
            )
        )
        filtered_set = set(filtered_cities)

        # Print filter report
        print(f"\n  🔍 Hard Constraint Filter:")
        for summary_line in constraints.get("filter_summary", []):
            print(f"     • {summary_line}")
        print(
            f"     → Candidates: {filter_report['total_before']} → "
            f"{filter_report['total_after']} "
            f"({filter_report['filtered_out']} eliminated)"
        )
        if filter_report.get("fallback"):
            print(f"     ⚠️  {filter_report['reason']}")
        if geo_constraint.get("detected"):
            print(
                f"  🌍 Geographic filter active: "
                f"'{geo_constraint['keyword']}' → "
                f"{geo_constraint['regions']}"
            )
        print(
            f"  ⚖️  Fusion weights: vector={vector_weight}, "
            f"graph={graph_weight}"
        )

        # ── Step 4: Vector search (then filter) ──
        vec_results = self.qe.vector_search(query, top_k=top_k * 5)
        vec_results_filtered = [
            r for r in vec_results if r["city"] in filtered_set
        ]

        if len(vec_results_filtered) < top_k:
            vec_results_extended = self.qe.vector_search(
                query, top_k=top_k * 10
            )
            vec_results_filtered = [
                r for r in vec_results_extended
                if r["city"] in filtered_set
            ]

        # ── Step 5: Compute direct match scores ──
        # This IS the graph's primary contribution
        direct_match_scores = _compute_direct_match_scores(
            self.G, filtered_cities, matched_nodes
        )

        direct_cities_with_matches = sum(
            1 for s in direct_match_scores.values() if s > 0
        )
        max_direct = max(direct_match_scores.values()) if direct_match_scores else 0
        print(
            f"  🎯 Direct matches: "
            f"{direct_cities_with_matches}/{len(filtered_cities)} "
            f"cities have direct edges to query nodes "
            f"(max coverage: {max_direct:.0%})"
        )

        # ── Step 6: Score fusion ──
        vec_raw = {}
        for r in vec_results_filtered:
            vec_raw[r["city"]] = r["score"]

        # Normalize vector scores only
        vec_normalized = _hybrid_normalize(vec_raw)

        # Log stats
        if vec_raw:
            raw_vals = list(vec_raw.values())
            print(
                f"  📏 Vector scores: "
                f"raw [{min(raw_vals):.3f}, {max(raw_vals):.3f}] → "
                f"normalized [{min(vec_normalized.values()):.3f}, "
                f"{max(vec_normalized.values()):.3f}]"
            )

        # Distribution of direct match scores
        dm_distribution = {}
        for score in direct_match_scores.values():
            bucket = round(score, 1)
            dm_distribution[bucket] = dm_distribution.get(bucket, 0) + 1
        print(f"  📏 Direct match distribution: {dict(sorted(dm_distribution.items()))}")

        print(
            f"  ⚖️  Final weights: "
            f"vector={vector_weight:.2f}, "
            f"graph(direct)={graph_weight:.2f}"
        )

        # ── Step 7: Combine all candidates ──
        all_cities_in_play = set()
        all_cities_in_play.update(vec_normalized.keys())
        # Add ALL filtered cities that have direct matches
        for city, score in direct_match_scores.items():
            if score > 0:
                all_cities_in_play.add(city)

        final_results = []
        for city in all_cities_in_play:
            if city not in filtered_set:
                continue

            v_score = vec_normalized.get(city, 0.0)
            d_score = direct_match_scores.get(city, 0.0)

            # Uniqueness bonus
            uniqueness = self.G.nodes.get(city, {}).get(
                "uniqueness_score", 0.5
            )
            uniqueness_bonus = 0.02 * (uniqueness - 0.5)

            final = (
                vector_weight * v_score
                + graph_weight * d_score
                + uniqueness_bonus
            )

            evidence = self._extract_evidence(city, matched_nodes)
            validation = self._validate_city_ground_truth(
                city, matched_nodes
            )

            final_results.append({
                "city": city,
                "final_score": round(final, 4),
                "vector_score": round(v_score, 4),
                "graph_score": round(d_score, 4),
                "direct_match_raw": round(d_score, 4),
                "vector_raw": round(vec_raw.get(city, 0.0), 4),
                "uniqueness_bonus": round(uniqueness_bonus, 4),
                "evidence_paths": evidence,
                "validation": validation,
                "matched_nodes": [
                    m["node"] for m in matched_nodes
                ],
                "city_attrs": dict(
                    self.G.nodes.get(city, {})
                ),
                "filter_report": filter_report,
                "geo_constraint": geo_constraint,
                "weights_used": {
                    "vector": vector_weight,
                    "graph": graph_weight,
                },
                "normalization": "hybrid_vector_direct_graph",
            })

        final_results.sort(
            key=lambda x: x["final_score"], reverse=True
        )
        return final_results[:top_k], matched_nodes

    def _extract_evidence(self, city, matched_nodes):
        paths = []
        for m in matched_nodes:
            target = m["node"]
            if target not in self.G or city not in self.G:
                continue

            is_hard_constraint = (
                _get_node_constraint_type(target) is not None
            )

            if self.G.has_edge(city, target):
                edge_data = self.G.edges[city, target]
                relation = edge_data.get("relation", "?")
                paths.append({
                    "target": target,
                    "hops": 1,
                    "similarity_hops": 0,
                    "via_similarity": False,
                    "decay_weight": HOP_DECAY[0],
                    "hub_penalized": False,
                    "path_str": (
                        f"{city} --[{relation}]--> {target}"
                    ),
                    "weight": edge_data.get("weight", None),
                    "is_hard_constraint": is_hard_constraint,
                    "evidence_type": "direct",
                    "intermediaries": [],
                })
                continue

            try:
                path = nx.shortest_path(
                    self.G, source=city, target=target
                )
                if len(path) <= 4:
                    sim_hops = _count_similarity_hops(
                        self.G, path
                    )
                    via_sim = _path_passes_through_similarity(
                        self.G, path
                    )
                    decay = HOP_DECAY.get(sim_hops, 0.01)

                    path_parts = []
                    intermediaries = []
                    hub_penalized = False

                    for i in range(len(path) - 1):
                        src = path[i]
                        dst = path[i + 1]
                        edge_data = self.G.edges[src, dst]
                        rel = edge_data.get("relation", "?")

                        if rel == "SIMILAR_PROFILE":
                            sim_score = edge_data.get(
                                "similarity", 0.0
                            )
                            path_parts.append(
                                f"{src} --[{rel}"
                                f"(sim={sim_score:.3f})]"
                                f"--> {dst}"
                            )
                        else:
                            path_parts.append(
                                f"{src} --[{rel}]--> {dst}"
                            )

                    for mid_idx in range(1, len(path) - 1):
                        mid_node = path[mid_idx]
                        is_hub, centrality = _check_intermediary_is_hub(
                            self.G, mid_node
                        )

                        intermediaries.append({
                            "node": mid_node,
                            "is_hub": is_hub,
                            "degree_centrality": round(
                                centrality, 4
                            ),
                            "node_type": self.G.nodes.get(
                                mid_node, {}
                            ).get("type", "Unknown"),
                        })

                        if is_hub:
                            hub_penalized = True
                            decay *= HUB_PENALTY_FACTOR

                    if not via_sim:
                        evidence_type = "multi_hop_structural"
                    elif sim_hops == 1:
                        evidence_type = "indirect_1hop"
                    else:
                        evidence_type = "indirect_weak"

                    if hub_penalized:
                        evidence_type += "_hub_penalized"

                    paths.append({
                        "target": target,
                        "hops": len(path) - 1,
                        "similarity_hops": sim_hops,
                        "via_similarity": via_sim,
                        "decay_weight": round(decay, 4),
                        "hub_penalized": hub_penalized,
                        "path_str": " | ".join(path_parts),
                        "is_hard_constraint": is_hard_constraint,
                        "evidence_type": evidence_type,
                        "intermediaries": intermediaries,
                    })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                paths.append({
                    "target": target,
                    "hops": -1,
                    "similarity_hops": 0,
                    "via_similarity": False,
                    "decay_weight": 0.0,
                    "hub_penalized": False,
                    "path_str": (
                        f"{city} --[NO PATH]--> {target}"
                    ),
                    "is_hard_constraint": is_hard_constraint,
                    "evidence_type": "no_path",
                    "intermediaries": [],
                })

        return paths

    def _validate_city_ground_truth(self, city, matched_nodes):
        checks = []
        ground_truth = self._city_ground_truth.get(city, {})

        if not ground_truth:
            return {
                "checks": [],
                "pass_rate": 0.0,
                "passed": 0,
                "total": 0,
                "hard_pass_rate": 0.0,
                "soft_pass_rate": 0.0,
                "avg_satisfaction": 0.0,
            }

        for m in matched_nodes:
            node = m["node"]
            constraint_type = _get_node_constraint_type(node)

            if constraint_type == "Climate":
                actual_climate = ground_truth.get("climate", None)
                satisfied = (actual_climate == node)
                checks.append({
                    "requirement": node,
                    "satisfied": satisfied,
                    "satisfaction_score": (
                        1.0 if satisfied else 0.0
                    ),
                    "constraint_type": "hard",
                    "validation_source": "ground_truth",
                    "detail": (
                        f"Ground truth climate: "
                        f"{actual_climate} | "
                        f"Required: {node} | "
                        f"{'✓ MATCH' if satisfied else '✗ MISMATCH'}"
                    ),
                })
                continue

            if constraint_type == "Month":
                has_month = self.G.has_edge(city, node)
                month_detail = ""
                if has_month:
                    edge_data = self.G.edges[city, node]
                    avg_temp = edge_data.get("avg_temp", "?")
                    month_climate = edge_data.get(
                        "climate_in_month", "?"
                    )
                    month_detail = (
                        f"avg_temp={avg_temp}°C, "
                        f"climate_in_month={month_climate}"
                    )
                checks.append({
                    "requirement": node,
                    "satisfied": has_month,
                    "satisfaction_score": (
                        1.0 if has_month else 0.0
                    ),
                    "constraint_type": "hard",
                    "validation_source": "ground_truth",
                    "detail": (
                        f"Month data: "
                        f"{month_detail if has_month else 'NOT FOUND'}"
                    ),
                })
                continue

            if "_" in node:
                parts = node.split("_")
                if len(parts) == 2:
                    category = parts[0].lower()
                    requested_level = parts[1]
                    actual_score = ground_truth.get(
                        category, None
                    )

                    if actual_score is not None:
                        required_threshold = (
                            GROUND_TRUTH_THRESHOLDS.get(
                                requested_level, 0
                            )
                        )

                        if actual_score >= required_threshold:
                            satisfied = True
                            satisfaction_score = 1.0
                        elif (requested_level == "High"
                              and actual_score >= 3):
                            satisfied = False
                            satisfaction_score = 0.5
                        elif (requested_level == "High"
                              and actual_score >= 2):
                            satisfied = False
                            satisfaction_score = 0.3
                        elif (requested_level == "Medium"
                              and actual_score >= 1):
                            satisfied = False
                            satisfaction_score = 0.2
                        else:
                            satisfied = False
                            satisfaction_score = 0.0

                        if actual_score >= 4:
                            actual_level = "High"
                        elif actual_score >= 2:
                            actual_level = "Medium"
                        else:
                            actual_level = "Low"

                        checks.append({
                            "requirement": node,
                            "satisfied": satisfied,
                            "satisfaction_score":
                                satisfaction_score,
                            "constraint_type": "soft",
                            "validation_source": "ground_truth",
                            "detail": (
                                f"Ground truth: "
                                f"{category}={actual_score} "
                                f"(level={actual_level}) | "
                                f"Required: {requested_level} "
                                f"(threshold="
                                f"{required_threshold}) | "
                                f"{'✓' if satisfied else '✗'} "
                                f"satisfaction="
                                f"{satisfaction_score}"
                            ),
                        })
                        continue

            checks.append({
                "requirement": node,
                "satisfied": False,
                "satisfaction_score": 0.0,
                "constraint_type": "soft",
                "validation_source": "unknown",
                "detail": (
                    f"Unknown node type: {node}, "
                    f"cannot validate"
                ),
            })

        if not checks:
            return {
                "checks": checks,
                "pass_rate": 0.0,
                "passed": 0,
                "total": 0,
                "hard_pass_rate": 0.0,
                "soft_pass_rate": 0.0,
                "avg_satisfaction": 0.0,
            }

        passed = sum(1 for c in checks if c["satisfied"])
        total = len(checks)

        hard_checks = [
            c for c in checks if c["constraint_type"] == "hard"
        ]
        soft_checks = [
            c for c in checks if c["constraint_type"] == "soft"
        ]

        hard_passed = sum(
            1 for c in hard_checks if c["satisfied"]
        )
        hard_total = len(hard_checks)

        soft_passed = sum(
            1 for c in soft_checks if c["satisfied"]
        )
        soft_total = len(soft_checks)

        avg_satisfaction = np.mean(
            [c["satisfaction_score"] for c in checks]
        )

        return {
            "checks": checks,
            "pass_rate": round(passed / total, 2),
            "passed": passed,
            "total": total,
            "hard_pass_rate": (
                round(hard_passed / hard_total, 2)
                if hard_total > 0 else 1.0
            ),
            "soft_pass_rate": (
                round(soft_passed / soft_total, 2)
                if soft_total > 0 else 1.0
            ),
            "avg_satisfaction": round(
                float(avg_satisfaction), 3
            ),
        }