"""
query_engine.py
════════════════
Parses natural language queries into graph nodes and runs vector search.
Works with the 5-layer India Travel Knowledge Graph.
"""

import re
import json
import numpy as np
import networkx as nx
from pathlib import Path

# ── Optional: sentence-transformers for vector search ────────────────────────
# If not installed: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# ── Keyword → Graph Node Mapping ─────────────────────────────────────────────
# Maps query keywords to actual node types in the 5-layer graph

TRIP_TYPE_KEYWORDS = {
    "beach": "Beach", "beaches": "Beach", "coastal": "Beach",
    "sea": "Beach", "shore": "Beach", "ocean": "Beach",
    "adventure": "Adventure", "trekking": "Adventure", "trek": "Adventure",
    "hiking": "Adventure", "rafting": "Adventure", "skiing": "Adventure",
    "paragliding": "Adventure", "thrilling": "Adventure",
    "culture": "Cultural", "cultural": "Cultural", "heritage": "Cultural",
    "historical": "Cultural", "history": "Cultural", "temple": "Cultural",
    "fort": "Cultural", "palace": "Cultural", "monument": "Cultural",
    "spiritual": "Spiritual", "pilgrimage": "Spiritual", "religious": "Spiritual",
    "holy": "Spiritual", "sacred": "Spiritual", "divine": "Spiritual",
    "wellness": "Wellness", "yoga": "Wellness", "meditation": "Wellness",
    "spa": "Wellness", "ayurveda": "Wellness", "relaxation": "Wellness",
    "wildlife": "Wildlife", "safari": "Wildlife", "jungle": "Wildlife",
    "animals": "Wildlife", "birds": "Wildlife", "forest": "Wildlife",
    "nature": "Nature", "natural": "Nature", "scenic": "Nature",
    "mountains": "Nature", "hills": "Nature", "valley": "Nature",
    "waterfall": "Nature", "lake": "Nature", "river": "Nature",
    "photography": "Photography", "landscape": "Photography",
    "romantic": "Romantic", "honeymoon": "Romantic", "couple": "Romantic",
    "family": "Family", "kids": "Family", "children": "Family",
    "solo": "Solo", "backpacking": "Solo", "solo traveller": "Solo",
    "food": "Food", "cuisine": "Food", "foodie": "Food",
    "nightlife": "Nightlife", "party": "Nightlife", "clubs": "Nightlife",
}

CLIMATE_KEYWORDS = {
    "cold": "Cold", "freezing": "Cold", "snow": "Cold", "snowy": "Cold",
    "icy": "Cold", "winter sports": "Cold",
    "cool": "Cool", "mild": "Cool", "pleasant": "Cool",
    "moderate": "Moderate", "temperate": "Moderate", "comfortable": "Moderate",
    "warm": "Moderate", "tropical": "Tropical", "hot": "Tropical",
    "humid": "Tropical", "beach weather": "Tropical",
}

SEASON_KEYWORDS = {
    "winter": "Winter", "december": "Winter", "january": "Winter",
    "february": "Winter", "cold season": "Winter",
    "summer": "Summer", "april": "Summer", "may": "Summer",
    "march": "Summer", "hot season": "Summer",
    "monsoon": "Monsoon", "rainy": "Monsoon", "rains": "Monsoon",
    "rain": "Monsoon", "july": "Monsoon", "august": "Monsoon",
    "june": "Monsoon", "september": "Monsoon",
    "autumn": "Post-Monsoon", "october": "Post-Monsoon",
    "november": "Post-Monsoon", "post monsoon": "Post-Monsoon",
}

BUDGET_KEYWORDS = {
    "budget": "Budget", "cheap": "Budget", "affordable": "Budget",
    "backpacker": "Budget", "low cost": "Budget", "economical": "Budget",
    "mid range": "Mid-Range", "moderate budget": "Mid-Range",
    "luxury": "Luxury", "premium": "Luxury", "expensive": "Luxury",
    "high end": "Luxury", "5 star": "Luxury",
}

ACCESSIBILITY_KEYWORDS = {
    "easy": "Easy", "accessible": "Easy", "well connected": "Easy",
    "moderate": "Moderate", "offbeat": "Difficult",
    "remote": "Difficult", "difficult": "Difficult",
    "hard to reach": "Difficult", "hidden": "Difficult",
}

STATE_KEYWORDS = {
    "kerala": "Kerala", "rajasthan": "Rajasthan", "goa": "Goa",
    "himachal": "Himachal Pradesh", "uttarakhand": "Uttarakhand",
    "karnataka": "Karnataka", "tamil nadu": "Tamil Nadu",
    "andhra": "Andhra Pradesh", "telangana": "Telangana",
    "maharashtra": "Maharashtra", "west bengal": "West Bengal",
    "sikkim": "Sikkim", "assam": "Assam", "meghalaya": "Meghalaya",
    "arunachal": "Arunachal Pradesh", "uttar pradesh": "Uttar Pradesh",
    "madhya pradesh": "Madhya Pradesh", "odisha": "Odisha",
    "punjab": "Punjab", "jammu": "Jammu and Kashmir",
    "ladakh": "Ladakh", "northeast": ["Assam", "Meghalaya",
                                       "Arunachal Pradesh"],
    "south india": ["Kerala", "Tamil Nadu", "Karnataka",
                    "Andhra Pradesh", "Telangana"],
    "north india": ["Uttarakhand", "Himachal Pradesh",
                    "Uttar Pradesh", "Punjab"],
}

PERMIT_KEYWORDS = {
    "no permit": False, "permit free": False,
    "permit": True, "restricted": True, "inner line": True,
}

CONNECTIVITY_KEYWORDS = {
    "good internet": "Good", "excellent connectivity": "Excellent",
    "wifi": "Good", "digital nomad": "Excellent",
    "no internet": "Poor", "remote": "Poor", "off grid": "Poor",
}


class SemanticQueryEngine:

    def __init__(self, G: nx.DiGraph, dest_names: list,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.G = G
        self.dest_names = dest_names
        self.dest_set = set(dest_names)

        # Build destination descriptions from graph node attributes
        self.dest_descriptions = self._build_descriptions()

        # Load sentence transformer for vector search
        self.model = None
        self.dest_embeddings = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("  Loading sentence transformer...")
            self.model = SentenceTransformer(model_name)
            self._build_embeddings()
        else:
            print("  ⚠ sentence-transformers not found. Vector search disabled.")
            print("    Install: pip install sentence-transformers")

    def _build_descriptions(self) -> dict:
        """Build rich text descriptions from graph node attributes."""
        descriptions = {}
        for name in self.dest_names:
            node = self.G.nodes.get(name, {})
            parts = [f"{name} is a travel destination in India."]

            state = node.get("state", "")
            if state:
                parts.append(f"Located in {state}.")

            climate = node.get("climate_category", "")
            if climate:
                parts.append(f"Has a {climate.lower()} climate.")

            alt = node.get("altitude_m")
            if alt and alt > 500:
                parts.append(f"Situated at {alt}m altitude.")

            budget = node.get("budget_tier", "")
            if budget:
                parts.append(f"Suitable for {budget.lower()} budget travellers.")

            acc = node.get("accessibility", "")
            if acc:
                parts.append(f"Accessibility: {acc}.")

            peak = node.get("peak_tourist_season", "")
            if peak:
                parts.append(f"Best visited during {peak}.")

            unique = node.get("unique_experiences", "")
            if unique:
                parts.append(unique[:200])

            # Add trip types from edges
            trip_types = [
                d.get("relation", "").replace("SUITS_TRIP_TYPE", "")
                for _, nbr, d in self.G.out_edges(name, data=True)
                if d.get("relation") == "SUITS_TRIP_TYPE"
                and self.G.nodes.get(nbr, {}).get("type") == "TripType"
            ]
            trip_type_names = [
                self.G.nodes[nbr].get("trip_type", "")
                for _, nbr, d in self.G.out_edges(name, data=True)
                if d.get("relation") == "SUITS_TRIP_TYPE"
            ]
            if trip_type_names:
                parts.append(f"Good for: {', '.join(trip_type_names)}.")

            # Add activities from edges
            activities = [
                self.G.nodes[nbr].get("activity", "")
                for _, nbr, d in self.G.out_edges(name, data=True)
                if d.get("relation") == "OFFERS_ACTIVITY"
            ]
            if activities:
                parts.append(f"Activities: {', '.join(activities[:4])}.")

            descriptions[name] = " ".join(parts)
        return descriptions

    def _build_embeddings(self):
        """Pre-compute embeddings for all destinations."""
        print(f"  Computing embeddings for {len(self.dest_names)} destinations...")
        texts = [self.dest_descriptions.get(n, n) for n in self.dest_names]
        self.dest_embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True)
        print("  Embeddings ready.")

    def vector_search(self, query: str, top_k: int = 10) -> list:
        """Cosine similarity search against destination embeddings."""
        if self.model is None or self.dest_embeddings is None:
            # Fallback: return all destinations with equal score
            return [{"city": n, "score": 0.5} for n in self.dest_names[:top_k]]

        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.dest_embeddings @ q_emb

        ranked = sorted(
            zip(self.dest_names, scores.tolist()),
            key=lambda x: x[1], reverse=True
        )
        return [{"city": name, "score": float(score)}
                for name, score in ranked[:top_k]]

    def parse_query_to_graph_nodes(self, query: str) -> list:
        """
        Map query keywords to actual nodes in the knowledge graph.
        Returns list of {node, score, node_type} dicts.
        """
        query_lower = query.lower()
        matched = {}

        def add_match(node, score, node_type):
            if node in self.G and node not in matched:
                matched[node] = {"node": node, "score": score,
                                 "node_type": node_type}
            elif node in self.G and score > matched[node]["score"]:
                matched[node]["score"] = score

        # ── Trip types ────────────────────────────────────────────────────────
        for kw, trip_type in TRIP_TYPE_KEYWORDS.items():
            if kw in query_lower:
                node = f"TripType_{trip_type}"
                add_match(node, 0.9, "TripType")

        # ── Climate ───────────────────────────────────────────────────────────
        for kw, climate in CLIMATE_KEYWORDS.items():
            if kw in query_lower:
                node = f"Climate_{climate}"
                add_match(node, 0.95, "Climate")

        # ── Seasons ───────────────────────────────────────────────────────────
        for kw, season in SEASON_KEYWORDS.items():
            if kw in query_lower:
                node = f"Season_{season}"
                add_match(node, 0.85, "Season")

        # ── Budget ────────────────────────────────────────────────────────────
        for kw, tier in BUDGET_KEYWORDS.items():
            if kw in query_lower:
                node = f"Budget_{tier}"
                add_match(node, 0.85, "BudgetTier")

        # ── Accessibility ─────────────────────────────────────────────────────
        for kw, level in ACCESSIBILITY_KEYWORDS.items():
            if kw in query_lower:
                node = f"Accessibility_{level}"
                add_match(node, 0.75, "Accessibility")

        # ── States / Regions ──────────────────────────────────────────────────
        for kw, state_val in STATE_KEYWORDS.items():
            if kw in query_lower:
                if isinstance(state_val, list):
                    for s in state_val:
                        node = f"State_{s}"
                        add_match(node, 0.90, "State")
                else:
                    node = f"State_{state_val}"
                    add_match(node, 0.90, "State")

        # ── Connectivity ──────────────────────────────────────────────────────
        for kw, level in CONNECTIVITY_KEYWORDS.items():
            if kw in query_lower:
                node = f"Connectivity_{level}"
                add_match(node, 0.70, "Connectivity")

        # ── Permit keywords ───────────────────────────────────────────────────
        for kw, val in PERMIT_KEYWORDS.items():
            if kw in query_lower:
                # Tag matched destinations directly
                for name in self.dest_names:
                    permits = self.G.nodes.get(name, {}).get(
                        "permits_required", False)
                    if permits == val:
                        add_match(name, 0.60, "Destination")

        return list(matched.values())

    def detect_geographic_constraint(self, query: str) -> dict:
        """Detect if query mentions a specific Indian state or region."""
        query_lower = query.lower()
        detected_states = []

        for kw, state_val in STATE_KEYWORDS.items():
            if kw in query_lower:
                if isinstance(state_val, list):
                    detected_states.extend(state_val)
                else:
                    detected_states.append(state_val)

        if detected_states:
            return {
                "detected": True,
                "states": list(set(detected_states)),
                "keyword": next(
                    (kw for kw in STATE_KEYWORDS if kw in query_lower), "")
            }
        return {"detected": False, "states": [], "keyword": ""}