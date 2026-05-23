import pandas as pd
import networkx as nx
import numpy as np
import json

PREFERENCE_COLUMNS = [
    "culture", "adventure", "nature", "beaches",
    "nightlife", "cuisine", "wellness", "urban", "seclusion"
]

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# ── Similarity Edge Controls ──
SIMILARITY_THRESHOLD = 0.85      # ↓ less restrictive
MAX_SIMILAR_PER_CITY = 15       # ↑ allow richer clustering


def categorize_climate(avg_temp):
    if avg_temp < 10:
        return "Cold"
    elif avg_temp <= 25:
        return "Moderate"
    else:
        return "Hot"


def preference_level(score):
    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"


def build_city_description(row):
    monthly_temps = json.loads(row["avg_temp_monthly"])
    yearly_avg = sum(m["avg"] for m in monthly_temps.values()) / 12

    desc = (
        f"{row['city']} is a city in {row['country']}, "
        f"located in {row['region']}. "
        f"It has a yearly average temperature of {yearly_avg:.1f}°C "
        f"({categorize_climate(yearly_avg)} climate). "
    )

    pref_parts = []
    for pref in PREFERENCE_COLUMNS:
        score = row[pref]
        level = preference_level(score)
        if level == "High":
            pref_parts.append(f"excellent {pref}")
        elif level == "Medium":
            pref_parts.append(f"moderate {pref}")

    if pref_parts:
        desc += "It is known for " + ", ".join(pref_parts) + "."

    return desc


def _normalize(arr):
    rng = arr.max() - arr.min()
    if rng == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / rng


def _compute_uniqueness_score(degree_cent, betweenness_cent, cities):
    if not cities:
        return {}

    deg_vals = np.array([degree_cent.get(c, 0) for c in cities])
    bet_vals = np.array([betweenness_cent.get(c, 0) for c in cities])

    deg_norm = _normalize(deg_vals)
    bet_norm = _normalize(bet_vals)

    uniqueness = {}

    for i, city in enumerate(cities):
        degree_component = 1.0 - deg_norm[i]      # penalize hubs
        bridge_component = bet_norm[i]           # reward bridges

        # Rebalanced weighting
        score = (
            0.5 * degree_component +
            0.5 * bridge_component
        )

        uniqueness[city] = round(float(score), 4)

    return uniqueness


def build_graph(df):

    G = nx.DiGraph()
    city_pref_vectors = {}
    city_descriptions = {}

    # ─────────────────────────────────────────────
    # PASS 1 — NODES + SEMANTIC STRUCTURE
    # ─────────────────────────────────────────────
    for _, row in df.iterrows():
        city = row["city"]
        country = row["country"]
        region = row["region"]

        monthly_temps = json.loads(row["avg_temp_monthly"])
        yearly_avg = sum(m["avg"] for m in monthly_temps.values()) / 12
        climate = categorize_climate(yearly_avg)

        G.add_node(
            city,
            type="City",
            yearly_avg_temp=round(yearly_avg, 1),
            climate_category=climate,
            country=country,
            region=region,
        )

        G.add_node(country, type="Country")
        G.add_node(region, type="Region")
        G.add_node(climate, type="Climate")

        G.add_edge(city, country, relation="LOCATED_IN")
        G.add_edge(city, region, relation="PART_OF")
        G.add_edge(region, country, relation="REGION_OF")  # NEW
        G.add_edge(city, climate, relation="HAS_CLIMATE")

        for month_num_str, temp_data in monthly_temps.items():
            month_idx = int(month_num_str) - 1
            month_node = f"Month_{MONTH_NAMES[month_idx]}"
            month_climate = categorize_climate(temp_data["avg"])

            G.add_node(month_node, type="Month")

            G.add_edge(
                city,
                month_node,
                relation="TEMP_IN_MONTH",
                avg_temp=temp_data["avg"],
                climate_in_month=month_climate,
            )

        # ─── Preference Layer (Expanded Semantic Anchors) ───
        pref_vector = []

        for pref in PREFERENCE_COLUMNS:
            score = row[pref]
            level = preference_level(score)

            pref_category_node = f"PrefCategory_{pref.capitalize()}"
            pref_level_node = f"{pref.capitalize()}_{level}"

            G.add_node(pref_category_node, type="PrefCategory")
            G.add_node(pref_level_node, type="TravelPreference")

            # Stronger semantic linking
            G.add_edge(city, pref_category_node, relation="HAS_CATEGORY")
            G.add_edge(city, pref_level_node, relation="HAS_PREFERENCE", weight=score)
            G.add_edge(pref_level_node, pref_category_node, relation="BELONGS_TO")

            pref_vector.append(score)

        city_pref_vectors[city] = np.array(pref_vector, dtype=np.float32)
        city_descriptions[city] = build_city_description(row)

    # ─────────────────────────────────────────────
    # PASS 2 — IMPROVED SIMILARITY STRUCTURE
    # ─────────────────────────────────────────────
    cities = list(city_pref_vectors.keys())
    vectors = np.array([city_pref_vectors[c] for c in cities])

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = vectors / norms
    sim_matrix = normalized @ normalized.T

    candidate_edges = []
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            sim = float(sim_matrix[i][j])
            if sim >= SIMILARITY_THRESHOLD:
                candidate_edges.append((i, j, sim))

    candidate_edges.sort(key=lambda x: x[2], reverse=True)

    edges_per_city = {c: 0 for c in cities}

    for i, j, sim in candidate_edges:
        city_i = cities[i]
        city_j = cities[j]

        if edges_per_city[city_i] >= MAX_SIMILAR_PER_CITY:
            continue
        if edges_per_city[city_j] >= MAX_SIMILAR_PER_CITY:
            continue

        G.add_edge(
            city_i, city_j,
            relation="SIMILAR_PROFILE",
            similarity=round(sim, 4),
            weight=round(sim * 2.0, 4)  # BOOSTED weight
        )
        G.add_edge(
            city_j, city_i,
            relation="SIMILAR_PROFILE",
            similarity=round(sim, 4),
            weight=round(sim * 2.0, 4)
        )

        edges_per_city[city_i] += 1
        edges_per_city[city_j] += 1

    # ─────────────────────────────────────────────
    # PASS 3 — CENTRALITY (FIXED)
    # ─────────────────────────────────────────────

    # CRITICAL FIX:
    # Use undirected projection for structural realism
    G_undirected = G.to_undirected()

    degree_cent = nx.degree_centrality(G_undirected)
    betweenness_cent = nx.betweenness_centrality(
        G_undirected,
        k=min(150, len(G_undirected.nodes))
    )

    uniqueness_scores = _compute_uniqueness_score(
        degree_cent,
        betweenness_cent,
        cities
    )

    for city in cities:
        deg = degree_cent.get(city, 0)
        bet = betweenness_cent.get(city, 0)
        uniq = uniqueness_scores.get(city, 0.5)

        # Graph Boost Score
        graph_boost = round(
            0.5 * uniq +
            0.3 * bet +
            0.2 * (1 - deg),
            4
        )

        G.nodes[city]["degree_centrality"] = round(deg, 4)
        G.nodes[city]["betweenness_centrality"] = round(bet, 6)
        G.nodes[city]["uniqueness_score"] = uniq
        G.nodes[city]["graph_boost_score"] = graph_boost

    print("Graph built with enhanced structural and semantic overlap.")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    return G, city_pref_vectors, cities, city_descriptions