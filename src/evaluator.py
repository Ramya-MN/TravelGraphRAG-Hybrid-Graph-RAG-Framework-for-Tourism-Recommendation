import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.graph_builder import (
    categorize_climate, preference_level, PREFERENCE_COLUMNS
)


# ─── Query-to-Preference Keyword Map ───
# Maps user query keywords DIRECTLY to dataset preference columns.
# This is the GROUND TRUTH interpretation of what the user wants,
# independent of what the semantic matcher detected.
# Used for honest evaluation — not circular self-assessment.
QUERY_PREFERENCE_KEYWORDS = {
    # Beaches
    "beach": "beaches",
    "beaches": "beaches",
    "coastal": "beaches",
    "ocean": "beaches",
    "seaside": "beaches",
    "shore": "beaches",
    "island": "beaches",
    "surfing": "beaches",
    "sand": "beaches",
    "waterfront": "beaches",
    # Nightlife
    "nightlife": "nightlife",
    "clubbing": "nightlife",
    "clubs": "nightlife",
    "bars": "nightlife",
    "party": "nightlife",
    "parties": "nightlife",
    "dancing": "nightlife",
    "pub": "nightlife",
    "nightclub": "nightlife",
    # Culture
    "culture": "culture",
    "cultural": "culture",
    "museum": "culture",
    "museums": "culture",
    "heritage": "culture",
    "history": "culture",
    "historical": "culture",
    "ruins": "culture",
    "art": "culture",
    "architecture": "culture",
    # Adventure
    "adventure": "adventure",
    "adventurous": "adventure",
    "hiking": "adventure",
    "trekking": "adventure",
    "climbing": "adventure",
    "extreme": "adventure",
    "outdoor": "adventure",
    "outdoors": "adventure",
    # Nature
    "nature": "nature",
    "natural": "nature",
    "wildlife": "nature",
    "forest": "nature",
    "mountain": "nature",
    "mountains": "nature",
    "waterfall": "nature",
    "scenic": "nature",
    "wilderness": "nature",
    "park": "nature",
    "national park": "nature",
    # Cuisine
    "cuisine": "cuisine",
    "food": "cuisine",
    "culinary": "cuisine",
    "gastronomy": "cuisine",
    "restaurant": "cuisine",
    "restaurants": "cuisine",
    "dining": "cuisine",
    "gourmet": "cuisine",
    # Wellness
    "wellness": "wellness",
    "spa": "wellness",
    "spas": "wellness",
    "yoga": "wellness",
    "meditation": "wellness",
    "retreat": "wellness",
    "relaxation": "wellness",
    "healing": "wellness",
    # Urban
    "urban": "urban",
    "city": "urban",
    "metropolitan": "urban",
    "cosmopolitan": "urban",
    "downtown": "urban",
    "shopping": "urban",
    "skyscraper": "urban",
    # Seclusion
    "seclusion": "seclusion",
    "secluded": "seclusion",
    "isolated": "seclusion",
    "remote": "seclusion",
    "peaceful": "seclusion",
    "quiet": "seclusion",
    "tranquil": "seclusion",
    "solitude": "seclusion",
    "private": "seclusion",
    "getaway": "seclusion",
    "escape": "seclusion",
}

# ─── Query-to-Climate Keyword Map ───
QUERY_CLIMATE_KEYWORDS = {
    "cold": "Cold",
    "cool": "Cold",
    "freezing": "Cold",
    "snow": "Cold",
    "snowy": "Cold",
    "winter": "Cold",
    "arctic": "Cold",
    "moderate": "Moderate",
    "mild": "Moderate",
    "temperate": "Moderate",
    "pleasant": "Moderate",
    "comfortable": "Moderate",
    "hot": "Hot",
    "tropical": "Hot",
    "warm": "Hot",
    "scorching": "Hot",
    "humid": "Hot",
    "desert": "Hot",
}

# Threshold for considering a city's preference score as "satisfying"
PREFERENCE_SATISFACTION_THRESHOLD = 4  # Score >= 4 out of 5


def _parse_query_preferences(query):
    """
    Parse user query text to determine what preferences they ACTUALLY want.
    Uses keyword matching — completely independent of the semantic matcher.

    Returns:
        set of preference column names (e.g., {"beaches", "nightlife"})
    """
    query_lower = query.lower()
    detected = set()

    # Check longer phrases first (e.g., "national park" before "park")
    sorted_keywords = sorted(
        QUERY_PREFERENCE_KEYWORDS.keys(),
        key=len,
        reverse=True,
    )

    for keyword in sorted_keywords:
        if keyword in query_lower:
            detected.add(QUERY_PREFERENCE_KEYWORDS[keyword])

    return detected


def _parse_query_climate(query):
    """
    Parse user query text to determine what climate they want.
    Independent of semantic matcher.

    Returns:
        climate string ("Cold", "Moderate", "Hot") or None
    """
    query_lower = query.lower()

    # Check longer keywords first
    sorted_keywords = sorted(
        QUERY_CLIMATE_KEYWORDS.keys(),
        key=len,
        reverse=True,
    )

    for keyword in sorted_keywords:
        if keyword in query_lower:
            return QUERY_CLIMATE_KEYWORDS[keyword]

    return None


class Evaluator:
    def __init__(self, G, df, retriever):
        self.G = G
        self.df = df
        self.retriever = retriever

    # ──────────────────────────────────
    # 3-Way Comparison
    # ──────────────────────────────────
    def compare_methods(self, query, top_k=5):
        """Run all 3 methods and compare results."""

        # Vector-only
        t0 = time.time()
        vec_results = self.retriever.retrieve_vector_only(
            query, top_k
        )
        vec_time = time.time() - t0

        # Graph-only
        t0 = time.time()
        graph_results, graph_nodes = (
            self.retriever.retrieve_graph_only(query, top_k)
        )
        graph_time = time.time() - t0

        # Graph-RAG (combined)
        t0 = time.time()
        rag_results, rag_nodes = (
            self.retriever.retrieve_graph_rag(query, top_k)
        )
        rag_time = time.time() - t0

        return {
            "query": query,
            "vector_only": {
                "cities": [r["city"] for r in vec_results],
                "scores": [r["score"] for r in vec_results],
                "time": round(vec_time, 3),
            },
            "graph_only": {
                "cities": [r["city"] for r in graph_results],
                "scores": [r["score"] for r in graph_results],
                "matched_nodes": [
                    m["node"] for m in graph_nodes
                ],
                "time": round(graph_time, 3),
            },
            "graph_rag": {
                "cities": [r["city"] for r in rag_results],
                "scores": [
                    r["final_score"] for r in rag_results
                ],
                "validation_rates": [
                    r["validation"]["pass_rate"]
                    for r in rag_results
                ],
                "matched_nodes": [
                    m["node"] for m in rag_nodes
                ],
                "time": round(rag_time, 3),
            },
        }

    # ──────────────────────────────────
    # Metric: Preference Alignment (GROUND TRUTH)
    # ──────────────────────────────────
    def preference_alignment(self, query, results, threshold=None):
        """
        Measures whether recommended cities actually have high scores
        in the preferences the USER asked for.

        KEY CHANGE: Parses preferences directly from query text using
        keyword matching — completely independent of what the semantic
        matcher detected. This prevents circular self-assessment.

        If the semantic matcher wrongly matched Adventure_Medium for
        a "nature" query, this method will still correctly check
        nature scores (not adventure scores).

        Args:
            query: original user query string
            results: list of result dicts from retriever
            threshold: minimum score to count as "aligned"
                       (default: PREFERENCE_SATISFACTION_THRESHOLD)

        Returns:
            dict with alignment score, detected preferences,
            per-city details, and comparison with system interpretation
        """
        if threshold is None:
            threshold = PREFERENCE_SATISFACTION_THRESHOLD

        # ── Parse what user ACTUALLY asked for ──
        user_prefs = _parse_query_preferences(query)
        user_climate = _parse_query_climate(query)

        # ── Also get what the SYSTEM interpreted ──
        system_prefs = set()
        for r in results:
            for node in r.get("matched_nodes", []):
                if "_" in node:
                    category = node.split("_")[0].lower()
                    if category in PREFERENCE_COLUMNS:
                        system_prefs.add(category)

        # ── Check for interpretation mismatch ──
        missed_by_system = user_prefs - system_prefs
        extra_in_system = system_prefs - user_prefs
        interpretation_match = (
            len(missed_by_system) == 0
            and len(extra_in_system) == 0
        )

        if not user_prefs:
            return {
                "alignment": 1.0,
                "user_preferences": [],
                "system_preferences": list(system_prefs),
                "interpretation_match": True,
                "missed_by_system": [],
                "extra_in_system": list(extra_in_system),
                "detail": "No specific preferences detected in query",
            }

        # ── Evaluate each city against user's ACTUAL preferences ──
        city_details = []
        city_alignments = []

        for r in results:
            city = r["city"]
            city_row = self.df[self.df["city"] == city]
            if city_row.empty:
                continue

            hits = 0
            city_pref_detail = {}

            for pref in user_prefs:
                score = city_row[pref].values[0]
                satisfied = score >= threshold
                if satisfied:
                    hits += 1
                city_pref_detail[pref] = {
                    "score": int(score),
                    "satisfied": satisfied,
                    "threshold": threshold,
                }

            alignment = hits / len(user_prefs)
            city_alignments.append(alignment)

            city_details.append({
                "city": city,
                "alignment": round(alignment, 3),
                "hits": hits,
                "total_prefs": len(user_prefs),
                "preferences": city_pref_detail,
            })

        avg_alignment = (
            round(np.mean(city_alignments), 3)
            if city_alignments else 0.0
        )

        return {
            "alignment": avg_alignment,
            "user_preferences": list(user_prefs),
            "system_preferences": list(system_prefs),
            "interpretation_match": interpretation_match,
            "missed_by_system": list(missed_by_system),
            "extra_in_system": list(extra_in_system),
            "city_details": city_details,
            "detail": (
                f"User wants: {sorted(user_prefs)} | "
                f"System detected: {sorted(system_prefs)} | "
                f"{'✓ Match' if interpretation_match else '✗ Mismatch'}"
            ),
        }

    # ──────────────────────────────────
    # Metric: Climate Accuracy (GROUND TRUTH)
    # ──────────────────────────────────
    def climate_accuracy(self, query, results):
        """
        Check if recommended cities match the climate the user
        ACTUALLY asked for.

        KEY CHANGE: Parses climate from query text directly,
        not from matched nodes.

        Args:
            query: original user query string
            results: list of result dicts from retriever

        Returns:
            dict with accuracy, detected climate, per-city details
        """
        user_climate = _parse_query_climate(query)

        # Also get what system interpreted
        system_climate = None
        for r in results:
            for node in r.get("matched_nodes", []):
                if node in ("Cold", "Moderate", "Hot"):
                    system_climate = node
                    break
            if system_climate:
                break

        if user_climate is None:
            return {
                "accuracy": 1.0,
                "user_climate": None,
                "system_climate": system_climate,
                "interpretation_match": (
                    system_climate is None
                ),
                "detail": "No climate constraint in query",
            }

        correct = 0
        total = len(results)
        city_details = []

        for r in results:
            city = r["city"]
            city_data = self.G.nodes.get(city, {})
            actual_climate = city_data.get(
                "climate_category", None
            )
            is_correct = (actual_climate == user_climate)
            if is_correct:
                correct += 1

            city_details.append({
                "city": city,
                "actual_climate": actual_climate,
                "required_climate": user_climate,
                "correct": is_correct,
            })

        accuracy = (
            round(correct / total, 3) if total > 0 else 0.0
        )

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "user_climate": user_climate,
            "system_climate": system_climate,
            "interpretation_match": (
                user_climate == system_climate
            ),
            "city_details": city_details,
            "detail": (
                f"User wants: {user_climate} | "
                f"System detected: {system_climate} | "
                f"{'✓' if user_climate == system_climate else '✗'} | "
                f"{correct}/{total} cities correct"
            ),
        }

    # ──────────────────────────────────
    # Metric: Graph Contribution
    # ──────────────────────────────────
    def graph_contribution(self, results):
        """
        Measure how much the graph improved over pure vector search.
        """
        contributions = []
        for r in results:
            if r["vector_score"] > 0:
                boost = (
                    (r["final_score"] - r["vector_score"])
                    / r["vector_score"]
                )
            else:
                boost = r["final_score"]
            contributions.append(boost)

        return {
            "avg_boost": (
                round(np.mean(contributions), 3)
                if contributions else 0
            ),
            "max_boost": (
                round(max(contributions), 3)
                if contributions else 0
            ),
            "per_city": [
                {
                    "city": r["city"],
                    "boost": round(
                        (
                            (r["final_score"] - r["vector_score"])
                            / r["vector_score"]
                        )
                        if r["vector_score"] > 0
                        else r["final_score"],
                        3,
                    ),
                    "vector": r["vector_score"],
                    "graph": r["graph_score"],
                    "final": r["final_score"],
                }
                for r in results
            ],
        }

    # ──────────────────────────────────
    # Metric: Interpretation Accuracy
    # ──────────────────────────────────
    def interpretation_accuracy(self, query, matched_nodes):
        """
        NEW METRIC: Measures how accurately the semantic matcher
        interpreted the query by comparing system-detected preferences
        and climate against ground truth keyword parsing.

        This directly measures whether the semantic matching pipeline
        correctly understood the user's intent.

        Args:
            query: original user query string
            matched_nodes: list of matched node dicts from retriever

        Returns:
            dict with precision, recall, F1 for interpretation
        """
        # Ground truth from query text
        user_prefs = _parse_query_preferences(query)
        user_climate = _parse_query_climate(query)

        # System interpretation from matched nodes
        system_prefs = set()
        system_climate = None
        for m in matched_nodes:
            node = m["node"] if isinstance(m, dict) else m
            if node in ("Cold", "Moderate", "Hot"):
                system_climate = node
            elif "_" in node:
                category = node.split("_")[0].lower()
                if category in PREFERENCE_COLUMNS:
                    system_prefs.add(category)

        # Preference interpretation metrics
        all_relevant = user_prefs | system_prefs
        if all_relevant:
            true_positives = len(user_prefs & system_prefs)
            precision = (
                true_positives / len(system_prefs)
                if system_prefs else 0.0
            )
            recall = (
                true_positives / len(user_prefs)
                if user_prefs else 1.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
        else:
            precision = 1.0
            recall = 1.0
            f1 = 1.0

        # Climate interpretation
        climate_correct = (
            (user_climate == system_climate)
            or (user_climate is None and system_climate is None)
        )

        return {
            "preference_precision": round(precision, 3),
            "preference_recall": round(recall, 3),
            "preference_f1": round(f1, 3),
            "climate_correct": climate_correct,
            "user_prefs": sorted(user_prefs),
            "system_prefs": sorted(system_prefs),
            "user_climate": user_climate,
            "system_climate": system_climate,
            "missed_prefs": sorted(user_prefs - system_prefs),
            "extra_prefs": sorted(system_prefs - user_prefs),
            "detail": (
                f"Prefs P={precision:.2f} R={recall:.2f} "
                f"F1={f1:.2f} | "
                f"Climate: "
                f"{'✓' if climate_correct else '✗'}"
            ),
        }

    # ──────────────────────────────────
    # Visualization: Comparison Bar Chart
    # ──────────────────────────────────
    def plot_comparison(self, comparison, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (method, title) in enumerate([
            ("vector_only", "Vector-Only"),
            ("graph_only", "Graph-Only (PPR)"),
            ("graph_rag", "Graph-RAG (Ours)"),
        ]):
            data = comparison[method]
            cities = data["cities"][:5]
            scores = data["scores"][:5]

            axes[idx].barh(
                range(len(cities)), scores,
                color=[
                    "#2196F3", "#4CAF50", "#FF9800"
                ][idx],
                alpha=0.8,
            )
            axes[idx].set_yticks(range(len(cities)))
            axes[idx].set_yticklabels(cities, fontsize=9)
            axes[idx].set_xlabel("Score")
            axes[idx].set_title(
                f"{title}\n(Time: {data['time']}s)"
            )
            axes[idx].invert_yaxis()

        plt.suptitle(
            f"Query: \"{comparison['query']}\"",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path, dpi=150, bbox_inches="tight"
            )
        plt.show()

    # ──────────────────────────────────
    # Visualization: Evidence Subgraph
    # ──────────────────────────────────
    def plot_evidence_subgraph(
        self, results, query, save_path=None
    ):
        """Show the subgraph connecting top cities to query nodes."""
        import networkx as nx

        nodes_to_show = set()
        for r in results[:3]:
            nodes_to_show.add(r["city"])
            for ep in r["evidence_paths"]:
                nodes_to_show.add(ep["target"])

        for r in results[:3]:
            for node in r.get("matched_nodes", []):
                if node in self.G:
                    nodes_to_show.add(node)

        subG = self.G.subgraph(nodes_to_show).copy()

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(
            subG, k=1.5, iterations=50, seed=42
        )

        color_map = {
            "City": "#2196F3",
            "TravelPreference": "#9C27B0",
            "Climate": "#F44336",
            "Region": "#4CAF50",
            "Country": "#FF9800",
            "Month": "#00BCD4",
        }

        node_colors = [
            color_map.get(
                subG.nodes[n].get("type", ""), "#999999"
            )
            for n in subG.nodes
        ]

        nx.draw_networkx_nodes(
            subG, pos,
            node_color=node_colors,
            node_size=800, alpha=0.9,
        )
        nx.draw_networkx_labels(
            subG, pos, font_size=8, font_weight="bold"
        )

        edge_labels = {
            (u, v): d.get("relation", "")[:12]
            for u, v, d in subG.edges(data=True)
        }
        nx.draw_networkx_edges(
            subG, pos, alpha=0.4, arrows=True, arrowsize=15
        )
        nx.draw_networkx_edge_labels(
            subG, pos, edge_labels, font_size=6
        )

        legend_elements = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=c, markersize=10, label=t,
            )
            for t, c in color_map.items()
        ]
        plt.legend(
            handles=legend_elements,
            loc="upper left", fontsize=8,
        )
        plt.title(
            f"Evidence Subgraph for: \"{query}\"",
            fontsize=13,
        )
        plt.axis("off")

        if save_path:
            plt.savefig(
                save_path, dpi=150, bbox_inches="tight"
            )
        plt.show()