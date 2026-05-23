from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import osmium
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BAD_VALUES = {"yes", "no", "unknown", "private"}

# Strict tourist-only allowlists.
TOURISM_ALLOW = {
    "attraction",
    "museum",
    "gallery",
    "viewpoint",
    "zoo",
    "theme_park",
    "aquarium",
    "picnic_site",
    "artwork",
    "beach_resort",
    "resort",
    "hotel",
    "guest_house",
    "hostel",
    "camp_site",
    "caravan_site",
}

HISTORIC_ALLOW = {
    "monument",
    "fort",
    "castle",
    "archaeological_site",
    "ruins",
    "temple",
    "memorial",
}

NATURAL_ALLOW = {
    "beach",
    "waterfall",
    "peak",
    "cliff",
    "cave_entrance",
    "spring",
    "hot_spring",
    "bay",
    "cape",
    "island",
}

LEISURE_ALLOW = {
    "park",
    "nature_reserve",
    "garden",
    "marina",
    "water_park",
}

AMENITY_ALLOW = {
    # Travel-relevant stays/activities only.
    "restaurant",
    "cafe",
    "fast_food",
    "food_court",
    "theatre",
    "arts_centre",
}

AMENITY_DENY = {
    "hospital",
    "clinic",
    "police",
    "bank",
    "atm",
    "school",
    "college",
    "university",
    "courthouse",
    "fire_station",
    "post_office",
    "prison",
    "townhall",
    "government",
    "bus_station",
    "fuel",
    "pharmacy",
    "marketplace",
}

ICONIC_MUST_INCLUDE = {
    "taj mahal",
    "marina beach",
    "gateway of india",
    "qutub minar",
    "mysore palace",
    "hawa mahal",
    "charminar",
    "india gate",
    "red fort",
    "golden temple",
    "konark sun temple",
    "jagannath temple",
    "kedarnath",
    "badrinath",
    "manali",
    "leh",
    "goa",
    "kovalam beach",
    "radhanagar beach",
    "varkala beach",
    "munnar",
    "coorg",
    "darjeeling",
    "ooty",
    "hampi",
    "ajanta caves",
    "ellora caves",
    "khajuraho",
    "amer fort",
    "jaisalmer fort",
}

ICONIC_SEED_ROWS = [
    {"destination_name": "Taj Mahal", "state": "Uttar Pradesh", "district": "Agra", "lat": 27.1751, "lon": 78.0421, "poi_type": "monument", "trip_types": ["Cultural", "Photography"]},
    {"destination_name": "Agra Fort", "state": "Uttar Pradesh", "district": "Agra", "lat": 27.1795, "lon": 78.0211, "poi_type": "fort", "trip_types": ["Cultural", "Photography"]},
    {"destination_name": "Fatehpur Sikri", "state": "Uttar Pradesh", "district": "Agra", "lat": 27.0937, "lon": 77.6600, "poi_type": "archaeological_site", "trip_types": ["Cultural"]},
    {"destination_name": "Bangalore Palace", "state": "Karnataka", "district": "Bengaluru", "lat": 12.9987, "lon": 77.5920, "poi_type": "monument", "trip_types": ["Cultural", "Family"]},
    {"destination_name": "Lalbagh Botanical Garden", "state": "Karnataka", "district": "Bengaluru", "lat": 12.9507, "lon": 77.5848, "poi_type": "nature_reserve", "trip_types": ["Nature", "Family", "Photography"]},
    {"destination_name": "Cubbon Park", "state": "Karnataka", "district": "Bengaluru", "lat": 12.9763, "lon": 77.5929, "poi_type": "park", "trip_types": ["Nature", "Family"]},
    {"destination_name": "Amber Fort", "state": "Rajasthan", "district": "Jaipur", "lat": 26.9855, "lon": 75.8513, "poi_type": "fort", "trip_types": ["Cultural", "Photography"]},
    {"destination_name": "City Palace Jaipur", "state": "Rajasthan", "district": "Jaipur", "lat": 26.9258, "lon": 75.8237, "poi_type": "monument", "trip_types": ["Cultural"]},
    {"destination_name": "Meenakshi Temple", "state": "Tamil Nadu", "district": "Madurai", "lat": 9.9195, "lon": 78.1193, "poi_type": "temple", "trip_types": ["Spiritual", "Cultural"]},
    {"destination_name": "Munnar", "state": "Kerala", "district": "Idukki", "lat": 10.0889, "lon": 77.0595, "poi_type": "nature_reserve", "trip_types": ["Nature", "Photography"]},
    {"destination_name": "Marina Beach", "state": "Tamil Nadu", "district": "Chennai", "lat": 13.0487, "lon": 80.2824, "poi_type": "beach", "trip_types": ["Beach", "Nature", "Photography"]},
    {"destination_name": "Gateway of India", "state": "Maharashtra", "district": "Mumbai", "lat": 18.9220, "lon": 72.8347, "poi_type": "monument", "trip_types": ["Cultural", "Photography"]},
    {"destination_name": "Qutub Minar", "state": "Delhi", "district": "Delhi", "lat": 28.5244, "lon": 77.1855, "poi_type": "monument", "trip_types": ["Cultural", "Photography"]},
    {"destination_name": "India Gate", "state": "Delhi", "district": "Delhi", "lat": 28.6129, "lon": 77.2295, "poi_type": "monument", "trip_types": ["Cultural", "Photography"]},
    {"destination_name": "Red Fort", "state": "Delhi", "district": "Delhi", "lat": 28.6562, "lon": 77.2410, "poi_type": "fort", "trip_types": ["Cultural"]},
    {"destination_name": "Golden Temple", "state": "Punjab", "district": "Amritsar", "lat": 31.6200, "lon": 74.8765, "poi_type": "temple", "trip_types": ["Spiritual", "Cultural"]},
    {"destination_name": "Mysore Palace", "state": "Karnataka", "district": "Mysuru", "lat": 12.3052, "lon": 76.6552, "poi_type": "monument", "trip_types": ["Cultural"]},
    {"destination_name": "Charminar", "state": "Telangana", "district": "Hyderabad", "lat": 17.3616, "lon": 78.4747, "poi_type": "monument", "trip_types": ["Cultural"]},
    {"destination_name": "Hampi", "state": "Karnataka", "district": "Vijayanagara", "lat": 15.3350, "lon": 76.4600, "poi_type": "archaeological_site", "trip_types": ["Cultural", "Photography"]},
    {"destination_name": "Ajanta Caves", "state": "Maharashtra", "district": "Aurangabad", "lat": 20.5519, "lon": 75.7033, "poi_type": "archaeological_site", "trip_types": ["Cultural"]},
    {"destination_name": "Ellora Caves", "state": "Maharashtra", "district": "Aurangabad", "lat": 20.0268, "lon": 75.1770, "poi_type": "archaeological_site", "trip_types": ["Cultural"]},
    {"destination_name": "Konark Sun Temple", "state": "Odisha", "district": "Konark", "lat": 19.8876, "lon": 86.0945, "poi_type": "temple", "trip_types": ["Cultural", "Spiritual"]},
    {"destination_name": "Jagannath Temple", "state": "Odisha", "district": "Puri", "lat": 19.8049, "lon": 85.8186, "poi_type": "temple", "trip_types": ["Spiritual"]},
    {"destination_name": "Radhanagar Beach", "state": "Andaman and Nicobar Islands", "district": "Havelock", "lat": 11.9775, "lon": 92.9533, "poi_type": "beach", "trip_types": ["Beach", "Nature"]},
    {"destination_name": "Varkala Beach", "state": "Kerala", "district": "Varkala", "lat": 8.7379, "lon": 76.7163, "poi_type": "beach", "trip_types": ["Beach", "Nature"]},
    {"destination_name": "Kovalam Beach", "state": "Kerala", "district": "Thiruvananthapuram", "lat": 8.4004, "lon": 76.9784, "poi_type": "beach", "trip_types": ["Beach"]},
    {"destination_name": "Nainital", "state": "Uttarakhand", "district": "Nainital", "lat": 29.3919, "lon": 79.4542, "poi_type": "viewpoint", "trip_types": ["Nature", "Family"]},
    {"destination_name": "Manali", "state": "Himachal Pradesh", "district": "Manali", "lat": 32.2432, "lon": 77.1892, "poi_type": "peak", "trip_types": ["Adventure", "Nature"]},
    {"destination_name": "Darjeeling", "state": "West Bengal", "district": "Darjeeling", "lat": 27.0410, "lon": 88.2663, "poi_type": "viewpoint", "trip_types": ["Nature", "Photography"]},
    {"destination_name": "Ooty", "state": "Tamil Nadu", "district": "Nilgiris", "lat": 11.4102, "lon": 76.6950, "poi_type": "nature_reserve", "trip_types": ["Nature", "Family"]},
    {"destination_name": "Kaziranga National Park", "state": "Assam", "district": "Golaghat", "lat": 26.5775, "lon": 93.1711, "poi_type": "national_park", "trip_types": ["Wildlife", "Nature"]},
    {"destination_name": "Ranthambore National Park", "state": "Rajasthan", "district": "Sawai Madhopur", "lat": 26.0173, "lon": 76.5026, "poi_type": "national_park", "trip_types": ["Wildlife", "Nature"]},
]


class StopParsing(RuntimeError):
    pass

TRIPTYPE_RULES = {
    "Adventure": {"mountain", "trek", "hiking", "climbing", "adventure", "trail"},
    "Beach": {"beach", "coast"},
    "Cultural": {"museum", "monument", "heritage", "historic", "memorial", "fort"},
    "Spiritual": {"temple", "mosque", "church", "shrine", "gurudwara", "monastery"},
    "Family": {"zoo", "park", "theme_park", "aquarium"},
    "Wildlife": {"wildlife", "national_park", "sanctuary"},
    "Food": {"restaurant", "cafe", "food_court", "fast_food"},
    "Nature": {"garden", "forest", "lake", "waterfall", "nature_reserve"},
    "Wellness": {"spa", "resort", "hot_spring", "yoga"},
    "Photography": {"viewpoint", "scenic", "lookout"},
}


def _norm_text(v: str) -> str:
    return re.sub(r"\s+", " ", (v or "").strip())


def _infer_trip_types(tag_blob: str) -> List[str]:
    t = tag_blob.lower()
    out: List[str] = []
    for label, kws in TRIPTYPE_RULES.items():
        if any(k in t for k in kws):
            out.append(label)
    if not out:
        out.append("Cultural")
    return out


def _infer_accessibility(tags: Dict[str, str]) -> str:
    wheelchair = tags.get("wheelchair", "").lower()
    if wheelchair in {"yes", "designated"}:
        return "Easy"
    if wheelchair == "limited":
        return "Moderate"
    return "Moderate"


def _score_popularity(tags: Dict[str, str]) -> float:
    score = 4.0
    # Popular tourism classes
    for k in ("tourism", "historic", "amenity", "leisure"):
        v = tags.get(k, "").lower()
        if v in {"attraction", "museum", "monument", "theme_park", "zoo", "viewpoint"}:
            score += 1.2
    if tags.get("wikipedia"):
        score += 1.0
    if tags.get("wikidata"):
        score += 0.8
    if tags.get("website"):
        score += 0.3
    if tags.get("wikidata"):
        score += 0.8
    if tags.get("wikipedia"):
        score += 0.8
    if tags.get("website"):
        score += 0.3
    return min(10.0, round(score, 2))


def _norm_state_or_city(raw: str) -> str:
    v = _norm_text(raw)
    if not v:
        return "Unknown"
    return v


def _india_region(lat: float, lon: float) -> str:
    # Coarse regioning for balanced India-wide coverage.
    if lon > 88:
        return "East/Northeast"
    if lat > 26:
        return "North"
    if lat < 15 and lon > 72:
        return "South"
    if lon < 75:
        return "West"
    return "Central"


def _tourism_match(tags: Dict[str, str]) -> Tuple[bool, str]:
    tourism = str(tags.get("tourism", "")).lower()
    historic = str(tags.get("historic", "")).lower()
    natural = str(tags.get("natural", "")).lower()
    leisure = str(tags.get("leisure", "")).lower()
    amenity = str(tags.get("amenity", "")).lower()

    if amenity in AMENITY_DENY:
        return False, ""

    if tourism in TOURISM_ALLOW:
        return True, tourism
    if historic in HISTORIC_ALLOW:
        return True, historic
    if natural in NATURAL_ALLOW:
        return True, natural
    if leisure in LEISURE_ALLOW:
        return True, leisure
    if amenity in AMENITY_ALLOW:
        return True, amenity

    return False, ""


def _selection_score(row: Dict) -> float:
    score = float(row.get("popularity_score", 0.0))
    if row.get("wikidata"):
        score += 1.0
    if row.get("wikipedia"):
        score += 1.0

    poi_type = str(row.get("poi_type", "")).lower()
    if poi_type in {"attraction", "museum", "monument", "viewpoint", "theme_park", "zoo"}:
        score += 0.8

    if row.get("name_norm") in ICONIC_MUST_INCLUDE:
        score += 5.0

    return score


def _category_for_poi(poi_type: str) -> str:
    p = str(poi_type or "").lower()
    if p in {"beach", "bay", "cape", "island"}:
        return "beach"
    if p in {"peak", "waterfall", "cliff", "cave_entrance", "spring", "hot_spring", "nature_reserve", "garden", "park"}:
        return "nature"
    if p in {"national_park", "zoo"}:
        return "wildlife"
    if p in {"temple", "mosque", "church", "shrine", "gurudwara", "monastery"}:
        return "spiritual"
    if p in {"monument", "fort", "castle", "archaeological_site", "ruins", "memorial", "museum", "gallery"}:
        return "heritage"
    if p in {"hotel", "guest_house", "hostel", "resort", "camp_site", "caravan_site"}:
        return "stay"
    if p in {"restaurant", "cafe", "fast_food", "food_court"}:
        return "food"
    if p in {"theme_park", "aquarium", "theatre", "arts_centre"}:
        return "entertainment"
    return "other"


def _select_balanced_rows(rows: List[Dict], target_poi: int, per_state_cap: int | None = None) -> List[Dict]:
    if len(rows) <= target_poi:
        return rows

    if per_state_cap is None:
        per_state_cap = max(70, target_poi // 25)

    by_state: Dict[str, List[Dict]] = collections.defaultdict(list)
    by_region: Dict[str, List[Dict]] = collections.defaultdict(list)
    for r in rows:
        state = str(r.get("state") or "Unknown").strip()
        if not state:
            state = "Unknown"
        by_state[state].append(r)
        by_region[str(r.get("region") or "Central")].append(r)

    for state in by_state:
        by_state[state].sort(key=_selection_score, reverse=True)

    # Phase 0: guarantee iconic must-have destinations.
    selected: List[Dict] = []
    selected_ids = set()
    used_per_state = collections.Counter()
    used_per_type = collections.Counter()
    used_per_category = collections.Counter()

    type_caps = {
        "restaurant": max(90, target_poi // 18),
        "fast_food": max(45, target_poi // 35),
        "cafe": max(60, target_poi // 30),
        "hotel": max(170, target_poi // 12),
        "guest_house": max(90, target_poi // 20),
    }

    # Ensure representative mix of tourist spot categories.
    category_min = {
        "heritage": max(120, target_poi // 10),
        "nature": max(120, target_poi // 10),
        "beach": max(80, target_poi // 14),
        "wildlife": max(60, target_poi // 20),
        "spiritual": max(80, target_poi // 14),
        "stay": max(200, target_poi // 8),
        "food": max(200, target_poi // 8),
        "entertainment": max(60, target_poi // 20),
    }

    for r in rows:
        if r.get("name_norm") not in ICONIC_MUST_INCLUDE:
            continue
        rid = (r["destination_name"], r["coordinates"]["latitude"], r["coordinates"]["longitude"])
        if rid in selected_ids:
            continue
        selected.append(r)
        selected_ids.add(rid)
        used_per_state[str(r.get("state") or "Unknown")] += 1
        poi_type = str(r.get("poi_type") or "poi").lower()
        used_per_type[poi_type] += 1
        used_per_category[_category_for_poi(poi_type)] += 1
        if len(selected) >= target_poi:
            return selected[:target_poi]

    # Phase 1: guarantee at least one strong POI per state for geographic coverage.
    for state, state_rows in sorted(by_state.items(), key=lambda x: len(x[1]), reverse=True):
        if len(selected) >= target_poi:
            break
        if not state_rows:
            continue
        r = state_rows[0]
        rid = (r["destination_name"], r["coordinates"]["latitude"], r["coordinates"]["longitude"])
        if rid in selected_ids:
            continue
        poi_type = str(r.get("poi_type") or "poi").lower()
        cap = type_caps.get(poi_type)
        if cap is not None and used_per_type[poi_type] >= cap:
            continue

        selected.append(r)
        selected_ids.add(rid)
        used_per_state[state] += 1
        used_per_type[poi_type] += 1
        used_per_category[_category_for_poi(poi_type)] += 1

    # Phase 1b: enforce broad region-level presence.
    for region, region_rows in sorted(by_region.items(), key=lambda x: len(x[1]), reverse=True):
        if len(selected) >= target_poi:
            break
        added = 0
        for r in region_rows:
            rid = (r["destination_name"], r["coordinates"]["latitude"], r["coordinates"]["longitude"])
            if rid in selected_ids:
                continue
            poi_type = str(r.get("poi_type") or "poi").lower()
            cap = type_caps.get(poi_type)
            if cap is not None and used_per_type[poi_type] >= cap:
                continue

            selected.append(r)
            selected_ids.add(rid)
            used_per_state[str(r.get("state") or "Unknown")] += 1
            used_per_type[poi_type] += 1
            used_per_category[_category_for_poi(poi_type)] += 1
            added += 1
            if added >= 15:
                break

    # Phase 1c: ensure minimum coverage for each category.
    for category, target_min in category_min.items():
        if len(selected) >= target_poi:
            break
        if used_per_category[category] >= target_min:
            continue
        pool = [r for r in rows if _category_for_poi(r.get("poi_type")) == category]
        pool.sort(key=_selection_score, reverse=True)
        for r in pool:
            if used_per_category[category] >= target_min or len(selected) >= target_poi:
                break
            rid = (r["destination_name"], r["coordinates"]["latitude"], r["coordinates"]["longitude"])
            if rid in selected_ids:
                continue
            poi_type = str(r.get("poi_type") or "poi").lower()
            cap = type_caps.get(poi_type)
            if cap is not None and used_per_type[poi_type] >= cap:
                continue
            selected.append(r)
            selected_ids.add(rid)
            used_per_state[str(r.get("state") or "Unknown")] += 1
            used_per_type[poi_type] += 1
            used_per_category[category] += 1

    # Phase 2: round-robin fill from each state, honoring per-state cap.
    pointers = {state: 1 for state in by_state.keys()}
    state_order = [s for s, _ in sorted(by_state.items(), key=lambda x: len(x[1]), reverse=True)]

    progress = True
    while len(selected) < target_poi and progress:
        progress = False
        for state in state_order:
            if len(selected) >= target_poi:
                break
            if used_per_state[state] >= per_state_cap:
                continue

            idx = pointers[state]
            state_rows = by_state[state]
            if idx >= len(state_rows):
                continue

            r = state_rows[idx]
            pointers[state] += 1
            rid = (r["destination_name"], r["coordinates"]["latitude"], r["coordinates"]["longitude"])
            if rid in selected_ids:
                continue

            poi_type = str(r.get("poi_type") or "poi").lower()
            cap = type_caps.get(poi_type)
            if cap is not None and used_per_type[poi_type] >= cap:
                continue

            selected.append(r)
            used_per_type[poi_type] += 1
            selected_ids.add(rid)
            used_per_state[state] += 1
            used_per_category[_category_for_poi(poi_type)] += 1
            progress = True

    # Phase 3: global best fallback if still short.
    if len(selected) < target_poi:
        remaining = []
        for r in rows:
            rid = (r["destination_name"], r["coordinates"]["latitude"], r["coordinates"]["longitude"])
            if rid in selected_ids:
                continue
            remaining.append(r)
        remaining.sort(key=_selection_score, reverse=True)
        for r in remaining:
            if len(selected) >= target_poi:
                break
            selected.append(r)
            used_per_category[_category_for_poi(str(r.get("poi_type") or "poi").lower())] += 1

    return selected[:target_poi]


class POIHandler(osmium.SimpleHandler):
    def __init__(self, max_poi: int):
        super().__init__()
        self.max_poi = max_poi
        self.rows: List[Dict] = []

    def node(self, n: osmium.osm.Node) -> None:
        if len(self.rows) >= self.max_poi:
            raise StopParsing()
        if not n.location.valid():
            return

        tags = {t.k: t.v for t in n.tags}
        if not tags:
            return

        ok, poi_kind = _tourism_match(tags)
        if not ok:
            return

        name = _norm_text(tags.get("name") or tags.get("name:en") or "")
        if not name or name.lower() in BAD_VALUES:
            return

        city = _norm_state_or_city(tags.get("addr:city") or tags.get("is_in:city") or tags.get("district") or "")
        state = _norm_state_or_city(tags.get("addr:state") or tags.get("is_in:state") or tags.get("state") or city or "")

        # Skip rows where the POI name is just the state or district.
        name_norm = _norm_text(name).lower()
        if name_norm in {_norm_text(state).lower(), _norm_text(city).lower()}:
            return

        tag_blob = " ".join(f"{k}:{v}" for k, v in tags.items())
        trip_types = _infer_trip_types(tag_blob)

        region = _india_region(float(n.location.lat), float(n.location.lon))

        text_desc = " ".join(
            [
                name,
                tags.get("tourism", ""),
                tags.get("amenity", ""),
                tags.get("historic", ""),
                tags.get("description", ""),
                tags.get("name:en", ""),
            ]
        ).strip()

        self.rows.append(
            {
                "destination_name": name,
                "state": state,
                "district": city,
                "region": region,
                "name_norm": name_norm,
                "coordinates": {"latitude": float(n.location.lat), "longitude": float(n.location.lon)},
                "altitude_m": 0,
                "trip_types": trip_types,
                "best_seasons": ["Winter", "Summer", "Monsoon", "Post-Monsoon"],
                "avoid_seasons": [],
                "accessibility": _infer_accessibility(tags),
                "permits_required": False,
                "safety_rating": 7.0,
                "popularity_score": _score_popularity(tags),
                "ideal_days": 2,
                "minimum_days": 1,
                "maximum_days": 4,
                "unique_experiences": text_desc,
                "primary_attractions": [name],
                "activities_available": trip_types,
                "language_spoken": ["English", "Hindi"],
                "budget_category": {},
                "mid_range_category": {},
                "luxury_category": {},
                "poi_type": poi_kind,
                "osm_id": int(n.id),
                "wikidata": tags.get("wikidata", ""),
                "wikipedia": tags.get("wikipedia", ""),
                "_source": "OSM_Geofabrik_India_PBF",
            }
        )


def _build_queries(dataset: List[Dict], limit: int = 200) -> List[Dict]:
    regions = ["North", "South", "West", "East/Northeast", "Central"]
    trip_pool = sorted({t for d in dataset for t in d.get("trip_types", [])}) or ["Cultural", "Nature", "Food"]
    state_pool = sorted({d.get("state", "Unknown") for d in dataset if d.get("state") and d.get("state") != "Unknown"})

    queries: List[Dict] = []
    qid = 1

    for t in trip_pool[:12]:
        queries.append({"qid": qid, "category": "generic", "query": f"Best {t.lower()} places in India"})
        qid += 1

    for r in regions:
        queries.append({"qid": qid, "category": "generic", "query": f"Top attractions in {r}"})
        qid += 1

    # Explicit mainstream tourist intents.
    iconic_queries = [
        "Best beach destinations in India",
        "Top heritage and historical places in India",
        "Best hill stations in India",
        "Top spiritual destinations in India",
        "Best wildlife and nature destinations in India",
        "Top museums and monuments in India",
    ]
    for q in iconic_queries:
        queries.append({"qid": qid, "category": "generic", "query": q})
        qid += 1

    for s in state_pool[:60]:
        queries.append({"qid": qid, "category": "generic", "query": f"Best places to visit in {s}"})
        qid += 1

    personal_templates = [
        "I need family-friendly places in {state}",
        "Suggest a budget trip for me in {state}",
        "My parents need easy-access attractions in {state}",
        "I want nature and photography spots in {state}",
    ]
    for s in state_pool[:30]:
        for t in personal_templates:
            queries.append({"qid": qid, "category": "personal", "query": t.format(state=s)})
            qid += 1

    return queries[:limit]


def _inject_iconic_seeds(rows: List[Dict]) -> List[Dict]:
    existing_names = {str(r.get("destination_name", "")).strip().lower() for r in rows}
    for seed in ICONIC_SEED_ROWS:
        if seed["destination_name"].strip().lower() in existing_names:
            continue

        seed_row = {
            "destination_name": seed["destination_name"],
            "state": seed["state"],
            "district": seed["district"],
            "region": _india_region(float(seed["lat"]), float(seed["lon"])),
            "name_norm": seed["destination_name"].strip().lower(),
            "coordinates": {"latitude": float(seed["lat"]), "longitude": float(seed["lon"])},
            "altitude_m": 0,
            "trip_types": list(seed["trip_types"]),
            "best_seasons": ["Winter", "Summer", "Monsoon", "Post-Monsoon"],
            "avoid_seasons": [],
            "accessibility": "Moderate",
            "permits_required": False,
            "safety_rating": 7.5,
            "popularity_score": 9.2,
            "ideal_days": 2,
            "minimum_days": 1,
            "maximum_days": 4,
            "unique_experiences": f"{seed['destination_name']} iconic tourist destination in India",
            "primary_attractions": [seed["destination_name"]],
            "activities_available": list(seed["trip_types"]),
            "language_spoken": ["English", "Hindi"],
            "budget_category": {},
            "mid_range_category": {},
            "luxury_category": {},
            "poi_type": seed["poi_type"],
            "osm_id": 0,
            "wikidata": "",
            "wikipedia": "",
            "_source": "Curated_Iconic_India_Tourism",
        }
        rows.append(seed_row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-poi", type=int, default=30000)
    ap.add_argument("--target-poi", type=int, default=2000)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    pbf_path = (ROOT / cfg["data"]["osm_pbf"]).resolve()

    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF not found: {pbf_path}")

    handler = POIHandler(max_poi=args.max_poi)
    try:
        handler.apply_file(str(pbf_path), locations=False)
    except StopParsing:
        pass

    rows = []
    seen = set()
    for i, r in enumerate(handler.rows, start=1):
        key = (r["destination_name"].lower(), round(r["coordinates"]["latitude"], 5), round(r["coordinates"]["longitude"], 5))
        if key in seen:
            continue
        seen.add(key)
        r["id"] = len(rows) + 1
        rows.append(r)

    rows = _inject_iconic_seeds(rows)
    rows = _select_balanced_rows(rows, target_poi=args.target_poi)
    for i, r in enumerate(rows, start=1):
        r["id"] = i

    out_dir = ROOT / "data" / "osm_processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dataset = out_dir / "dataset_osm_india.json"
    out_dataset.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    queries = _build_queries(rows, limit=200)
    qdir = ROOT / "data" / "queries"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "thesis_queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    (qdir / "generic_queries.json").write_text(
        json.dumps([q["query"] for q in queries if q["category"] == "generic"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (qdir / "personal_queries.json").write_text(
        json.dumps([q["query"] for q in queries if q["category"] == "personal"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"pbf={pbf_path}")
    print(f"target_poi={args.target_poi}")
    print(f"poi_rows={len(rows)}")
    print(f"dataset={out_dataset}")
    print(f"queries={qdir / 'thesis_queries.json'}")


if __name__ == "__main__":
    main()
