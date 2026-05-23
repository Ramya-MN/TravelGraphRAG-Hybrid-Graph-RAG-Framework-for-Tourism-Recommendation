"""
build_graph.py
══════════════════════════════════════════════════════════════════════
Builds a Multi-Layer Knowledge Graph from data/final/_dataset.json

Inspired by TravelRAG (IJGI 2024) — implements 5 graph layers:
  Layer 1 — Geographic      : India → State → District → Destination
  Layer 2 — Destination     : Core destination nodes with all attributes
  Layer 3 — Attribute       : Climate, Budget, Season, Accessibility
  Layer 4 — Experience      : TripType, Activity, Attraction, Festival
  Layer 5 — Similarity      : Cross-destination profile similarity edges

TravelRAG strategies integrated:
  ✓ Multi-layer hierarchical structure
  ✓ Cross-level semantic associations (threshold-based inter-layer links)
  ✓ Community detection (Louvain) for retrieval efficiency
  ✓ Parent-child hierarchy enabling top-down graph traversal

Project-specific additions:
  ✓ Source provenance on every edge (from _sources field)
  ✓ Evidence-Aware Confidence Scoring per node
  ✓ Graph saved as GraphML + metadata JSON for retriever use

Usage:
  python -m data.build_graph
  python build_graph.py
"""

import json
import pickle
import math
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = Path("data/final/_dataset.json")
OUTPUT_DIR   = Path("data/graph")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_PATH    = OUTPUT_DIR / "india_travel_kg.gpickle"
GRAPHML_PATH  = OUTPUT_DIR / "india_travel_kg.graphml"
METADATA_PATH = OUTPUT_DIR / "graph_metadata.json"

# ── Layer Configuration ───────────────────────────────────────────────────────
SIMILARITY_THRESHOLD   = 0.78   # min cosine sim for SIMILAR_TO edge
MAX_SIMILAR_PER_DEST   = 8      # max similarity edges per destination
CROSS_LAYER_THRESHOLD  = 0.60   # min score to create cross-layer semantic link

# Budget tier boundaries (INR/night)
BUDGET_TIERS = {
    "Budget":    (0,    1500),
    "Mid-Range": (1500, 5000),
    "Luxury":    (5000, 999999),
}

# Season → month mapping for graph edges
SEASON_MONTHS = {
    "Winter":      [11, 12, 1, 2],
    "Summer":      [3, 4, 5],
    "Monsoon":     [6, 7, 8, 9],
    "Post-Monsoon":[10, 11],
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# Preference vector fields — maps dataset fields to vector dimensions
# Used for computing destination similarity (Layer 5)
PREF_FIELDS = [
    "trip_types",        # list field — one-hot encoded
    "best_seasons",      # list field — one-hot encoded
    "accessibility",     # categorical → numeric
    "altitude_m",        # numeric (normalized)
    "popularity_score",  # numeric
    "safety_rating",     # numeric
]

TRIP_TYPES_ALL = [
    "Beach", "Adventure", "Cultural", "Spiritual", "Wellness",
    "Wildlife", "Nature", "Photography", "Romantic", "Family",
    "Solo", "Food", "Nightlife"
]

SEASONS_ALL = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]

ACCESSIBILITY_MAP = {"Easy": 3, "Moderate": 2, "Difficult": 1}

STATE_ALIAS_MAP = {
    "agra division": "Uttar Pradesh",
    "lucknow division": "Uttar Pradesh",
    "varanasi division": "Uttar Pradesh",
    "ajmer division": "Rajasthan",
    "bikaner division": "Rajasthan",
    "jaipur division": "Rajasthan",
    "jodhpur division": "Rajasthan",
    "udaipur division": "Rajasthan",
    "garhwal division": "Uttarakhand",
    "kumaon division": "Uttarakhand",
    "jalandhar division": "Punjab",
    "pune division": "Maharashtra",
    "nashik division": "Maharashtra",
    "sambhajinagar division": "Maharashtra",
    "belgaum division": "Karnataka",
    "kalaburagi division": "Karnataka",
    "mysuru division": "Karnataka",
    "indore division": "Madhya Pradesh",
    "sagar division": "Madhya Pradesh",
    "central division": "Madhya Pradesh",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset() -> list[dict]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} destinations from dataset.")
    return data


def get_budget_tier(dest: dict) -> str:
    """Determine budget tier from budget_category accommodation range."""
    try:
        acc_min = dest["budget_category"]["accommodation_range"][0] or 0
        acc_max = dest["mid_range_category"]["accommodation_range"][1] or 0
        midpoint = (acc_min + acc_max) / 2
        if midpoint < 1500:
            return "Budget"
        elif midpoint < 5000:
            return "Mid-Range"
        else:
            return "Luxury"
    except (KeyError, TypeError, IndexError):
        return "Mid-Range"


def get_climate_category(dest: dict) -> str:
    """Infer climate from average temperature data."""
    try:
        temps = dest["average_temperature"]
        # Use winter temp as primary indicator
        winter = temps.get("winter", "15-25°C")
        low = int(winter.split("-")[0].replace("°C","").replace("C","").strip())
        if low < 5:
            return "Cold"
        elif low < 15:
            return "Cool"
        elif low < 25:
            return "Moderate"
        else:
            return "Tropical"
    except (KeyError, TypeError, ValueError, AttributeError):
        return "Moderate"


def build_preference_vector(dest: dict) -> np.ndarray:
    """
    Build a fixed-length numeric preference vector for similarity computation.
    Encodes: trip types (13-dim one-hot) + seasons (4-dim one-hot)
             + accessibility (1) + altitude_norm (1)
             + popularity (1) + safety (1)
    Total: 21 dimensions
    """
    vec = []

    # Trip type one-hot (13 dims)
    trip_types = dest.get("trip_types", [])
    for t in TRIP_TYPES_ALL:
        vec.append(1.0 if t in trip_types else 0.0)

    # Season one-hot (4 dims)
    best_seasons = dest.get("best_seasons", [])
    for s in SEASONS_ALL:
        vec.append(1.0 if s in best_seasons else 0.0)

    # Accessibility (1 dim)
    acc = dest.get("accessibility", "Moderate")
    vec.append(ACCESSIBILITY_MAP.get(acc, 2) / 3.0)

    # Altitude normalized 0–1 (max ~5000m for Indian destinations)
    alt = dest.get("altitude_m", 0) or 0
    vec.append(min(alt / 5000.0, 1.0))

    # Popularity score normalized (1–10 → 0–1)
    pop = dest.get("popularity_score", 5) or 5
    vec.append(pop / 10.0)

    # Safety rating normalized (1–10 → 0–1)
    safe = dest.get("safety_rating", 5) or 5
    vec.append(safe / 10.0)

    return np.array(vec, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 1)

def safe_source(dest: dict, field: str) -> str:
    return dest.get("_sources", {}).get(field, "unknown")

def normalize_state(raw_state: str) -> str:
    if not raw_state:
        return "Unknown"
    state = raw_state.strip()
    alias = STATE_ALIAS_MAP.get(state.lower())
    return alias or state


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(destinations: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()

    dest_vectors    = {}   # name → preference vector
    dest_names      = []   # list of destination names
    dest_map        = {}   # name → full dict

    print("\n── Layer 1 + 2: Geographic Hierarchy & Destination Nodes ──")

    # Root geographic node
    G.add_node("India", type="Country", layer=1)

    for dest in destinations:
        name  = dest["destination_name"]
        raw_state = dest.get("state", "Unknown")
        state = normalize_state(raw_state)
        dist  = dest.get("district", "Unknown")
        coords = dest.get("coordinates", {})
        lat   = coords.get("latitude")
        lon   = coords.get("longitude")

        dest_names.append(name)
        dest_map[name] = dest

        # ── Layer 1: Geographic nodes ──────────────────────────────────────
        state_node    = f"State_{state}"
        district_node = f"District_{dist}"

        G.add_node(state_node,    type="State",    layer=1, name=state)
        G.add_node(district_node, type="District", layer=1, name=dist)

        G.add_edge("India",       state_node,    relation="HAS_STATE",
                   layer=1, source="wikidata")
        G.add_edge(state_node,    district_node, relation="HAS_DISTRICT",
                   layer=1, source=safe_source(dest, "state"))
        G.add_edge(district_node, name,          relation="CONTAINS",
                   layer=1, source=safe_source(dest, "district"))

        # ── Layer 2: Destination node ──────────────────────────────────────
        climate = get_climate_category(dest)
        budget_tier = get_budget_tier(dest)

        G.add_node(
            name,
            type               = "Destination",
            layer              = 2,
            state              = state,
            state_raw          = raw_state,
            district           = dist,
            latitude           = lat,
            longitude          = lon,
            altitude_m         = dest.get("altitude_m"),
            climate_category   = climate,
            budget_tier        = budget_tier,
            accessibility      = dest.get("accessibility", ""),
            road_connectivity  = dest.get("road_connectivity", ""),
            permits_required   = dest.get("permits_required", False),
            permits_details    = dest.get("permits_details", ""),
            minimum_days       = dest.get("minimum_days"),
            ideal_days         = dest.get("ideal_days"),
            maximum_days       = dest.get("maximum_days"),
            popularity_score   = dest.get("popularity_score"),
            safety_rating      = dest.get("safety_rating"),
            internet_connectivity = dest.get("internet_connectivity", ""),
            peak_tourist_season   = dest.get("peak_tourist_season", ""),
            off_season         = dest.get("off_season", ""),
            unique_experiences = dest.get("unique_experiences", ""),
            wikipedia_url      = dest.get("_wikipedia_url", ""),
            wikidata_qid       = dest.get("_wikidata_qid", ""),
            coverage_pct       = dest.get("_coverage_pct", 0),
        )

        # Geographic reverse edges (TravelRAG: bidirectional hierarchy)
        G.add_edge(name, state_node,    relation="LOCATED_IN_STATE",
                   layer=1, source=safe_source(dest, "state"))
        G.add_edge(name, district_node, relation="LOCATED_IN_DISTRICT",
                   layer=1, source=safe_source(dest, "district"))
        G.add_edge(name, "India",       relation="LOCATED_IN_COUNTRY",
                   layer=1, source="wikidata")

        # Nearest city edge
        nearest_city = dest.get("nearest_major_city", {})
        if nearest_city and nearest_city.get("name"):
            city_node = f"City_{nearest_city['name']}"
            G.add_node(city_node, type="MajorCity", layer=1,
                       name=nearest_city["name"])
            G.add_edge(name, city_node,
                       relation="NEAREST_MAJOR_CITY",
                       distance_km=nearest_city.get("distance_km"),
                       layer=1,
                       source=safe_source(dest, "nearest_major_city"))

        # Nearest airport edge
        airport = dest.get("nearest_airport", {})
        if airport and airport.get("name"):
            airport_node = f"Airport_{airport['name']}"
            G.add_node(airport_node, type="Airport", layer=1,
                       name=airport["name"])
            G.add_edge(name, airport_node,
                       relation="NEAREST_AIRPORT",
                       distance_km=airport.get("distance_km"),
                       layer=1,
                       source=safe_source(dest, "nearest_airport"))

        # Nearest railway edge
        station = dest.get("nearest_railway_station", {})
        if station and station.get("name"):
            station_node = f"Station_{station['name']}"
            G.add_node(station_node, type="RailwayStation", layer=1,
                       name=station["name"])
            G.add_edge(name, station_node,
                       relation="NEAREST_RAILWAY",
                       distance_km=station.get("distance_km"),
                       layer=1,
                       source=safe_source(dest, "nearest_railway_station"))

        # Build preference vector for Layer 5
        dest_vectors[name] = build_preference_vector(dest)

    print(f"  Destinations added: {len(dest_names)}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n── Layer 3: Attribute Nodes ──")

    CLIMATE_NODES    = set()
    BUDGET_NODES     = set()
    SEASON_NODES     = set()
    ACCESS_NODES     = set()
    CONNECTIVITY_NODES = set()

    for dest in destinations:
        name        = dest["destination_name"]
        climate     = get_climate_category(dest)
        budget_tier = get_budget_tier(dest)
        src_budget  = safe_source(dest, "budget_category")
        src_weather = safe_source(dest, "average_temperature")

        # ── Climate node ───────────────────────────────────────────────────
        climate_node = f"Climate_{climate}"
        if climate_node not in CLIMATE_NODES:
            G.add_node(climate_node, type="Climate", layer=3, category=climate)
            CLIMATE_NODES.add(climate_node)
        G.add_edge(name, climate_node, relation="HAS_CLIMATE",
                   layer=3, source=src_weather)

        # ── Budget tier node ───────────────────────────────────────────────
        budget_node = f"Budget_{budget_tier}"
        if budget_node not in BUDGET_NODES:
            G.add_node(budget_node, type="BudgetTier", layer=3, tier=budget_tier)
            BUDGET_NODES.add(budget_node)
        G.add_edge(name, budget_node, relation="SUITS_BUDGET",
                   layer=3, source=src_budget)

        # Add detailed budget ranges as node attributes
        try:
            G.nodes[name]["budget_acc_range"] = str(
                dest.get("budget_category", {}).get("accommodation_range", []))
            G.nodes[name]["mid_acc_range"] = str(
                dest.get("mid_range_category", {}).get("accommodation_range", []))
            G.nodes[name]["luxury_acc_range"] = str(
                dest.get("luxury_category", {}).get("accommodation_range", []))
        except Exception:
            pass

        # ── Season nodes ───────────────────────────────────────────────────
        best_seasons = dest.get("best_seasons", [])
        avoid_seasons = dest.get("avoid_seasons", [])
        for season in best_seasons:
            season_node = f"Season_{season}"
            if season_node not in SEASON_NODES:
                G.add_node(season_node, type="Season", layer=3, season=season)
                SEASON_NODES.add(season_node)
            G.add_edge(name, season_node, relation="BEST_IN",
                       layer=3, source=src_weather)
        for season in avoid_seasons:
            season_node = f"Season_{season}"
            if season_node not in SEASON_NODES:
                G.add_node(season_node, type="Season", layer=3, season=season)
                SEASON_NODES.add(season_node)
            G.add_edge(name, season_node, relation="AVOID_IN",
                       layer=3, source=src_weather)

        # ── Accessibility node ─────────────────────────────────────────────
        acc = dest.get("accessibility", "Moderate")
        acc_node = f"Accessibility_{acc}"
        if acc_node not in ACCESS_NODES:
            G.add_node(acc_node, type="Accessibility", layer=3, level=acc)
            ACCESS_NODES.add(acc_node)
        G.add_edge(name, acc_node, relation="HAS_ACCESSIBILITY",
                   layer=3, source=src_budget)

        # ── Internet connectivity node ─────────────────────────────────────
        inet = dest.get("internet_connectivity", "Good")
        inet_node = f"Connectivity_{inet}"
        if inet_node not in CONNECTIVITY_NODES:
            G.add_node(inet_node, type="Connectivity", layer=3, level=inet)
            CONNECTIVITY_NODES.add(inet_node)
        G.add_edge(name, inet_node, relation="HAS_CONNECTIVITY",
                   layer=3, source=src_budget)

    print(f"  Climate nodes:       {len(CLIMATE_NODES)}")
    print(f"  Budget tier nodes:   {len(BUDGET_NODES)}")
    print(f"  Season nodes:        {len(SEASON_NODES)}")
    print(f"  Accessibility nodes: {len(ACCESS_NODES)}")
    print(f"  Connectivity nodes:  {len(CONNECTIVITY_NODES)}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n── Layer 4: Experience Nodes (TravelRAG-inspired) ──")

    TRIPTYPE_NODES    = set()
    ACTIVITY_NODES    = set()
    ATTRACTION_NODES  = set()
    FESTIVAL_NODES    = set()
    CUISINE_NODES     = set()
    LANGUAGE_NODES    = set()

    for dest in destinations:
        name     = dest["destination_name"]
        src_wiki = safe_source(dest, "trip_types")
        src_attr = safe_source(dest, "primary_attractions")
        src_food = safe_source(dest, "local_cuisine_must_try")
        src_fest = safe_source(dest, "festivals_events")
        src_lang = safe_source(dest, "language_spoken")
        src_act  = safe_source(dest, "activities_available")

        # ── Trip type nodes ────────────────────────────────────────────────
        for tt in dest.get("trip_types", []):
            tt_node = f"TripType_{tt}"
            if tt_node not in TRIPTYPE_NODES:
                G.add_node(tt_node, type="TripType", layer=4, trip_type=tt)
                TRIPTYPE_NODES.add(tt_node)
            G.add_edge(name, tt_node, relation="SUITS_TRIP_TYPE",
                       layer=4, source=src_wiki)
            # TravelRAG cross-level: TripType ← Budget (e.g., Luxury → Wellness)
            budget_node = f"Budget_{get_budget_tier(dest)}"
            if G.has_node(budget_node):
                G.add_edge(tt_node, budget_node,
                           relation="TYPICALLY_ASSOCIATED_BUDGET",
                           layer="cross", source="derived")

        # ── Activity nodes ─────────────────────────────────────────────────
        activities = dest.get("activities_available", [])
        if isinstance(activities, list):
            for act_raw in activities[:6]:
                act = act_raw[:60].strip()  # cap length
                act_node = f"Activity_{act}"
                if act_node not in ACTIVITY_NODES:
                    G.add_node(act_node, type="Activity", layer=4,
                               activity=act)
                    ACTIVITY_NODES.add(act_node)
                G.add_edge(name, act_node, relation="OFFERS_ACTIVITY",
                           layer=4, source=src_act)

        # ── Attraction nodes ───────────────────────────────────────────────
        attractions = dest.get("primary_attractions", [])
        if isinstance(attractions, list):
            for attr_raw in attractions[:5]:
                attr = attr_raw[:80].strip()
                attr_node = f"Attraction_{attr}"
                if attr_node not in ATTRACTION_NODES:
                    G.add_node(attr_node, type="Attraction", layer=4,
                               attraction=attr)
                    ATTRACTION_NODES.add(attr_node)
                G.add_edge(name, attr_node, relation="HAS_ATTRACTION",
                           layer=4, source=src_attr)

        # ── Cuisine nodes ──────────────────────────────────────────────────
        cuisines = dest.get("local_cuisine_must_try", [])
        if isinstance(cuisines, list):
            for c_raw in cuisines[:4]:
                c = c_raw[:60].strip()
                c_node = f"Cuisine_{c}"
                if c_node not in CUISINE_NODES:
                    G.add_node(c_node, type="Cuisine", layer=4, cuisine=c)
                    CUISINE_NODES.add(c_node)
                G.add_edge(name, c_node, relation="HAS_CUISINE",
                           layer=4, source=src_food)

        # ── Festival nodes ─────────────────────────────────────────────────
        festivals = dest.get("festivals_events", [])
        if isinstance(festivals, list):
            for f_raw in festivals[:4]:
                f = f_raw[:60].strip()
                f_node = f"Festival_{f}"
                if f_node not in FESTIVAL_NODES:
                    G.add_node(f_node, type="Festival", layer=4, festival=f)
                    FESTIVAL_NODES.add(f_node)
                G.add_edge(name, f_node, relation="HAS_FESTIVAL",
                           layer=4, source=src_fest)

        # ── Language nodes ─────────────────────────────────────────────────
        for lang in dest.get("language_spoken", []):
            lang_node = f"Language_{lang}"
            if lang_node not in LANGUAGE_NODES:
                G.add_node(lang_node, type="Language", layer=4, language=lang)
                LANGUAGE_NODES.add(lang_node)
            G.add_edge(name, lang_node, relation="SPOKEN_LANGUAGE",
                       layer=4, source=src_lang)

    print(f"  TripType nodes:   {len(TRIPTYPE_NODES)}")
    print(f"  Activity nodes:   {len(ACTIVITY_NODES)}")
    print(f"  Attraction nodes: {len(ATTRACTION_NODES)}")
    print(f"  Cuisine nodes:    {len(CUISINE_NODES)}")
    print(f"  Festival nodes:   {len(FESTIVAL_NODES)}")
    print(f"  Language nodes:   {len(LANGUAGE_NODES)}")

    # ──────────────────────────────────────────────────────────────────────────
    # TravelRAG: Cross-Level Semantic Associations
    # Connect destinations that share geographic proximity
    # (enables multi-hop "what else is nearby" traversal)
    print("\n── Cross-Level: Geographic Proximity Edges ──")

    proximity_edges = 0
    coords_list = [
        (n, dest_map[n].get("coordinates", {}))
        for n in dest_names
        if dest_map[n].get("coordinates", {}).get("latitude")
    ]

    for i in range(len(coords_list)):
        n1, c1 = coords_list[i]
        for j in range(i + 1, len(coords_list)):
            n2, c2 = coords_list[j]
            dist = haversine_km(
                c1["latitude"], c1["longitude"],
                c2["latitude"], c2["longitude"]
            )
            if dist <= 150:
                G.add_edge(n1, n2, relation="GEOGRAPHICALLY_NEAR",
                           distance_km=dist, layer="cross",
                           source="derived:haversine")
                G.add_edge(n2, n1, relation="GEOGRAPHICALLY_NEAR",
                           distance_km=dist, layer="cross",
                           source="derived:haversine")
                proximity_edges += 2

    print(f"  Proximity edges added (≤150km): {proximity_edges}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n── Layer 5: Destination Similarity Edges ──")

    vectors  = np.array([dest_vectors[n] for n in dest_names])
    norms    = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed   = vectors / norms
    sim_mat  = normed @ normed.T

    candidate_edges = []
    for i in range(len(dest_names)):
        for j in range(i + 1, len(dest_names)):
            s = float(sim_mat[i][j])
            if s >= SIMILARITY_THRESHOLD:
                candidate_edges.append((i, j, s))
    candidate_edges.sort(key=lambda x: x[2], reverse=True)

    edges_count = defaultdict(int)
    sim_edges_added = 0

    for i, j, sim in candidate_edges:
        n1, n2 = dest_names[i], dest_names[j]
        if (edges_count[n1] >= MAX_SIMILAR_PER_DEST or
                edges_count[n2] >= MAX_SIMILAR_PER_DEST):
            continue
        G.add_edge(n1, n2, relation="SIMILAR_TO", similarity=round(sim, 4),
                   weight=round(sim, 4), layer=5, source="derived:cosine")
        G.add_edge(n2, n1, relation="SIMILAR_TO", similarity=round(sim, 4),
                   weight=round(sim, 4), layer=5, source="derived:cosine")
        edges_count[n1] += 1
        edges_count[n2] += 1
        sim_edges_added += 2

    print(f"  Similarity edges added: {sim_edges_added}")

    # ──────────────────────────────────────────────────────────────────────────
    # TravelRAG: Community Detection
    # Louvain communities group semantically related destinations
    # stored as node attribute for retrieval routing
    print("\n── TravelRAG: Community Detection (Louvain) ──")

    try:
        from community import best_partition  # python-louvain
        G_undirected = G.to_undirected()
        # Run only on destination nodes for meaningful communities
        dest_subgraph = G_undirected.subgraph(dest_names)
        partition = best_partition(dest_subgraph)
        for node, community_id in partition.items():
            if node in G.nodes:
                G.nodes[node]["community_id"] = community_id
        n_communities = len(set(partition.values()))
        print(f"  Communities detected: {n_communities}")
    except ImportError:
        # Fallback: use connected components as crude communities
        print("  python-louvain not installed. Using fallback (greedy modularity).")
        G_undirected = G.to_undirected()
        try:
            communities = nx.community.greedy_modularity_communities(
                G_undirected.subgraph(dest_names))
            for cid, community in enumerate(communities):
                for node in community:
                    if node in G.nodes:
                        G.nodes[node]["community_id"] = cid
            print(f"  Communities detected: {len(communities)}")
        except Exception as e:
            print(f"  Community detection skipped: {e}")
            for node in dest_names:
                G.nodes[node]["community_id"] = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Centrality + Evidence-Aware Confidence Scoring
    print("\n── Computing Centrality & Confidence Scores ──")

    G_undi = G.to_undirected()
    degree_cent      = nx.degree_centrality(G_undi)
    betweenness_cent = nx.betweenness_centrality(
        G_undi, k=min(100, len(G_undi.nodes)))

    for name in dest_names:
        deg  = degree_cent.get(name, 0)
        bet  = betweenness_cent.get(name, 0)
        pop  = (dest_map[name].get("popularity_score") or 5) / 10.0
        safe = (dest_map[name].get("safety_rating") or 5) / 10.0
        cov  = (dest_map[name].get("_coverage_pct") or 50) / 100.0

        # Evidence-Aware Confidence Score
        # (from project's own architecture — quantifies recommendation reliability)
        # Higher coverage + higher safety + moderate centrality = more confident
        confidence_score = round(
            0.35 * cov          # data completeness
            + 0.25 * safe       # safety reliability
            + 0.20 * pop        # popularity evidence
            + 0.10 * bet        # graph connectivity (bridge value)
            + 0.10 * (1 - deg), # penalize over-connected hubs
            4
        )

        G.nodes[name]["degree_centrality"]      = round(deg, 4)
        G.nodes[name]["betweenness_centrality"] = round(bet, 6)
        G.nodes[name]["confidence_score"]       = confidence_score
        G.nodes[name]["pref_vector"]            = dest_vectors[name].tolist()

    print("  Confidence scores computed for all destinations.")

    return G, dest_vectors, dest_names, dest_map


# ══════════════════════════════════════════════════════════════════════════════
# SAVE + SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def save_graph(G: nx.DiGraph, dest_names: list, dest_map: dict):
    # Save as pickle (preserves all Python objects including numpy arrays)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save as GraphML (human-readable, interoperable)
    # GraphML can't store lists/dicts/None — convert to strings first
    G_export = G.copy()
    for node, data in G_export.nodes(data=True):
        for k, v in list(data.items()):
            if isinstance(v, (list, dict, np.ndarray)):
                G_export.nodes[node][k] = json.dumps(
                    v.tolist() if isinstance(v, np.ndarray) else v)
            elif v is None:
                G_export.nodes[node][k] = ""  # Convert None to empty string
    for u, v, data in G_export.edges(data=True):
        for k, val in list(data.items()):
            if isinstance(val, (list, dict)):
                G_export.edges[u, v][k] = json.dumps(val)
            elif val is None:
                G_export.edges[u, v][k] = ""  # Convert None to empty string
    nx.write_graphml(G_export, str(GRAPHML_PATH))

    # Save metadata JSON for retriever use
    node_type_counts = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_type_counts[data.get("type", "Unknown")] += 1

    relation_counts = defaultdict(int)
    for _, _, data in G.edges(data=True):
        relation_counts[data.get("relation", "Unknown")] += 1

    community_map = {}
    for name in dest_names:
        cid = G.nodes[name].get("community_id", -1)
        if cid not in community_map:
            community_map[cid] = []
        community_map[cid].append(name)

    metadata = {
        "total_nodes":        G.number_of_nodes(),
        "total_edges":        G.number_of_edges(),
        "destination_count":  len(dest_names),
        "destination_names":  sorted(dest_names),
        "node_type_counts":   dict(node_type_counts),
        "relation_counts":    dict(sorted(relation_counts.items(),
                                          key=lambda x: x[1], reverse=True)),
        "communities":        {str(k): v for k, v in community_map.items()},
        "graph_layers": {
            "1": "Geographic (Country/State/District/Transport)",
            "2": "Destination (Core nodes with all attributes)",
            "3": "Attribute (Climate/Budget/Season/Accessibility)",
            "4": "Experience (TripType/Activity/Attraction/Cuisine/Festival)",
            "5": "Similarity (Cross-destination cosine similarity)",
            "cross": "Cross-level (Proximity + Budget-TripType associations)"
        },
        "travelrag_features": [
            "Multi-layer hierarchical knowledge graph",
            "Cross-level semantic associations",
            "Geographic proximity edges (haversine ≤150km)",
            "Community detection (Louvain/greedy modularity)",
            "Parent-child hierarchy: India→State→District→Destination",
        ],
        "project_features": [
            "Source provenance on every edge (_sources from dataset)",
            "Evidence-Aware Confidence Scoring per destination",
            "Preference vectors for cosine similarity (Layer 5)",
            "Coverage-weighted confidence scores",
        ]
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\n  Saved pickle  → {GRAPH_PATH}")
    print(f"  Saved GraphML → {GRAPHML_PATH}")
    print(f"  Saved metadata → {METADATA_PATH}")


def print_summary(G: nx.DiGraph, dest_names: list):
    print("\n" + "═"*60)
    print("GRAPH BUILD COMPLETE")
    print("═"*60)
    print(f"  Total nodes : {G.number_of_nodes():,}")
    print(f"  Total edges : {G.number_of_edges():,}")
    print(f"  Destinations: {len(dest_names)}")
    print()
    print("  Node types:")
    type_counts = defaultdict(int)
    for _, d in G.nodes(data=True):
        type_counts[d.get("type","?")] += 1
    for ntype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ntype:<25} {count:>5}")
    print()
    print("  Top relations:")
    rel_counts = defaultdict(int)
    for _, _, d in G.edges(data=True):
        rel_counts[d.get("relation","?")] += 1
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1])[:12]:
        print(f"    {rel:<35} {count:>5}")
    print("═"*60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═"*60)
    print("Building Multi-Layer Knowledge Graph")
    print("(TravelRAG-inspired, India Travel Dataset)")
    print("═"*60)

    destinations = load_dataset()
    G, dest_vectors, dest_names, dest_map = build_graph(destinations)
    save_graph(G, dest_names, dest_map)
    print_summary(G, dest_names)


# Public API for retriever/app imports
def load_graph():
    """Load the pre-built graph. Call this from retriever.py."""
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    dest_names = metadata["destination_names"]
    return G, dest_names, metadata


if __name__ == "__main__":
    main()