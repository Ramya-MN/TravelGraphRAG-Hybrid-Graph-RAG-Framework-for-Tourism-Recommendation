from __future__ import annotations

from typing import Dict, List

import networkx as nx

from ..data.models import Destination


def build_knowledge_graph(destinations: List[Destination]) -> nx.DiGraph:
    g = nx.DiGraph()

    for d in destinations:
        dn = f"dest_{d.dest_id}"
        g.add_node(
            dn,
            node_type="destination",
            dest_id=d.dest_id,
            name=d.name,
            state=d.state,
            district=d.district,
            region=d.region,
            accessibility=d.accessibility,
            permits_required=d.permits_required,
            safety_rating=d.safety_rating,
            popularity_score=d.popularity_score,
            ideal_days=d.ideal_days,
            poi_type=d.poi_type,
            trip_types=d.trip_types,
            best_seasons=d.best_seasons,
            budget_tier=d.budget_tier,
        )

        state_n = f"state_{d.state}"
        region_n = f"region_{d.region}"
        budget_n = f"budget_{d.budget_tier}"
        access_n = f"access_{d.accessibility}"

        g.add_node(state_n, node_type="state", name=d.state)
        g.add_node(region_n, node_type="region", name=d.region)
        g.add_node(budget_n, node_type="budget_tier", name=d.budget_tier)
        g.add_node(access_n, node_type="accessibility", name=d.accessibility)

        g.add_edge(dn, state_n, edge_type="LOCATED_IN")
        g.add_edge(dn, region_n, edge_type="IN_REGION")
        g.add_edge(dn, budget_n, edge_type="HAS_BUDGET_TIER")
        g.add_edge(dn, access_n, edge_type="HAS_ACCESSIBILITY")

        for season in d.best_seasons:
            sn = f"season_{season}"
            g.add_node(sn, node_type="season", name=season)
            g.add_edge(dn, sn, edge_type="BEST_IN_SEASON")

        for season in d.avoid_seasons:
            sn = f"season_{season}"
            g.add_node(sn, node_type="season", name=season)
            g.add_edge(dn, sn, edge_type="AVOID_IN_SEASON")

        for trip in d.trip_types:
            tn = f"trip_{trip}"
            g.add_node(tn, node_type="trip_type", name=trip)
            g.add_edge(dn, tn, edge_type="SUITS_TRIP_TYPE")

    # Lightweight similarity edges for refinement.
    by_id: Dict[int, Destination] = {d.dest_id: d for d in destinations}
    for i, d1 in enumerate(destinations):
        for d2 in destinations[i + 1 :]:
            overlap = len(set(d1.trip_types) & set(d2.trip_types))
            if d1.state == d2.state or overlap >= 2:
                a = f"dest_{d1.dest_id}"
                b = f"dest_{d2.dest_id}"
                g.add_edge(a, b, edge_type="SIMILAR_TO")
                g.add_edge(b, a, edge_type="SIMILAR_TO")

    return g
