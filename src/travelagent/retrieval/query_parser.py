from __future__ import annotations

import re
from typing import Any, Dict, List, Set

SEASONS = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
SEASON_ZH = {
    "冬": "Winter",
    "夏": "Summer",
    "春": "Spring",
    "秋": "Autumn",
    "雨季": "Monsoon",
}
TRIP_TYPES = [
    "Adventure",
    "Beach",
    "Cultural",
    "Spiritual",
    "Family",
    "Wildlife",
    "Food",
    "Nature",
    "Wellness",
    "Photography",
]
TRIP_TYPES_ZH = {
    "探险": "Adventure",
    "徒步": "Adventure",
    "登山": "Adventure",
    "海滩": "Beach",
    "文化": "Cultural",
    "历史": "Cultural",
    "宗教": "Spiritual",
    "亲子": "Family",
    "野生": "Wildlife",
    "美食": "Food",
    "自然": "Nature",
    "温泉": "Wellness",
    "摄影": "Photography",
}
TRIP_SYNONYMS = {
    "Adventure": ["adventure", "trek", "trekking", "hiking", "camping"],
    "Beach": ["beach", "coast", "coastal", "island"],
    "Cultural": ["culture", "cultural", "heritage", "historical", "history", "monument", "museum", "fort"],
    "Spiritual": ["spiritual", "pilgrim", "pilgrimage", "temple", "church", "mosque", "shrine"],
    "Family": ["family", "kids", "children", "parents"],
    "Wildlife": ["wildlife", "safari", "national park", "sanctuary"],
    "Food": ["food", "cuisine", "restaurant", "eat", "dining"],
    "Nature": ["nature", "scenic", "lake", "waterfall", "hill station", "mountain"],
    "Wellness": ["wellness", "spa", "yoga", "ayurveda"],
    "Photography": ["photography", "photo", "picturesque", "instagram"],
}
BUDGET_MAP = {
    "budget": "Budget",
    "affordable": "Budget",
    "cheap": "Budget",
    "mid": "MidRange",
    "moderate": "MidRange",
    "luxury": "Luxury",
    "premium": "Luxury",
}
ZH_BUDGET_MAP = {
    "预算": "Budget",
    "便宜": "Budget",
    "经济": "Budget",
    "中等": "MidRange",
    "适中": "MidRange",
    "高端": "Luxury",
    "豪华": "Luxury",
}
ACCESS_MAP = {"easy": "Easy", "moderate": "Moderate", "difficult": "Difficult"}
ZH_ACCESS_MAP = {"方便": "Easy", "轻松": "Easy", "中等": "Moderate", "困难": "Difficult"}


def parse_constraints(query: str) -> Dict[str, Any]:
    q = query.lower()
    constraints: Dict[str, Any] = {}

    for s in SEASONS:
        if s.lower() in q:
            constraints["season"] = s
            break
    if "season" not in constraints:
        for k, v in SEASON_ZH.items():
            if k in query:
                constraints["season"] = v
                break

    trip_hits: List[str] = []
    for t in TRIP_TYPES:
        if t.lower() in q:
            trip_hits.append(t)
    for k, v in TRIP_TYPES_ZH.items():
        if k in query and v not in trip_hits:
            trip_hits.append(v)
    for trip, keys in TRIP_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", q) for k in keys):
            if trip not in trip_hits:
                trip_hits.append(trip)
    if trip_hits:
        constraints["trip_type"] = trip_hits[0]
        constraints["trip_types"] = trip_hits

    for k, v in BUDGET_MAP.items():
        if re.search(rf"\b{k}\b", q):
            constraints["budget_tier"] = v
            break
    if "budget_tier" not in constraints:
        for k, v in ZH_BUDGET_MAP.items():
            if k in query:
                constraints["budget_tier"] = v
                break

    for k, v in ACCESS_MAP.items():
        if re.search(rf"\b{k}\b", q):
            constraints["accessibility"] = v
            break
    if "accessibility" not in constraints:
        for k, v in ZH_ACCESS_MAP.items():
            if k in query:
                constraints["accessibility"] = v
                break

    if "without permit" in q or "no permit" in q:
        constraints["permit"] = False

    # coarse region/state intent
    if "north india" in q:
        constraints["region"] = "North India"
    elif "south india" in q:
        constraints["region"] = "South India"
    elif "northeast" in q or "north east" in q:
        constraints["region"] = "Northeast India"
    elif "west india" in q:
        constraints["region"] = "West India"
    elif "east india" in q:
        constraints["region"] = "East India"
    elif "central india" in q:
        constraints["region"] = "Central India"

    # Extract lightweight geographic intent terms for lexical matching.
    location_terms: Set[str] = set()
    for pat in [r"\bin\s+([a-z][a-z\s\-,]{1,48})", r"\bnear\s+([a-z][a-z\s\-,]{1,48})"]:
        for m in re.findall(pat, q):
            term = re.sub(r"\s+", " ", m).strip(" ,.-")
            # Trim generic tails.
            term = re.sub(r"\b(for me|with family|for family|for kids|for parents|in india)\b", "", term).strip()
            if len(term) >= 3:
                location_terms.add(term)

    # Handle explicit district references.
    for m in re.findall(r"\bdistrict\s+([a-z][a-z\s\-]{1,40})", q):
        term = re.sub(r"\s+", " ", m).strip(" ,.-")
        if len(term) >= 3:
            location_terms.add(term)

    # Split comma forms and keep both local and parent geography.
    expanded_terms: Set[str] = set()
    for t in location_terms:
        expanded_terms.add(t)
        if "," in t:
            parts = [p.strip() for p in t.split(",") if p.strip()]
            expanded_terms.update(parts)
            if parts:
                expanded_terms.add(parts[-1])
    location_terms = expanded_terms

    generic_locs = {
        "india",
        "indian",
        "destination",
        "destinations",
        "attraction",
        "attractions",
        "place",
        "places",
        "visit",
        "trip",
    }
    location_terms = {t for t in location_terms if t not in generic_locs}

    # Add region/state abbreviations often used in travel queries.
    abbr_map = {
        "dl": "delhi",
        "mh": "maharashtra",
        "ka": "karnataka",
        "kl": "kerala",
        "tn": "tamil nadu",
        "up": "uttar pradesh",
        "mp": "madhya pradesh",
        "wb": "west bengal",
        "ap": "andhra pradesh",
    }
    for abbr, full in abbr_map.items():
        if re.search(rf"\b{abbr}\b", q):
            location_terms.add(full)

    city_aliases = {
        "cochin": "kochi",
        "fort cochin": "fort kochi",
        "bengaluru": "bangalore",
        "beṅgaḷūru": "bangalore",
        "bombay": "mumbai",
        "madras": "chennai",
        "benaras": "varanasi",
        "calcutta": "kolkata",
    }
    final_locs: Set[str] = set()
    for t in location_terms:
        final_locs.add(t)
        if t in city_aliases:
            final_locs.add(city_aliases[t])

    if final_locs:
        constraints["location_terms"] = sorted(final_locs)

    wants_food = any(k in q for k in ["food", "cuisine", "restaurant", "eat", "dining"])
    wants_stay = any(k in q for k in ["hotel", "stay", "resort", "accommodation", "budget trip"])
    wants_attraction = any(k in q for k in ["places to visit", "attraction", "heritage", "historical", "sightseeing", "destinations"])
    if wants_attraction and not wants_food and not wants_stay:
        constraints["intent_mode"] = "attraction_only"
    elif wants_food and not wants_attraction:
        constraints["intent_mode"] = "food_only"
    elif wants_stay and not wants_attraction:
        constraints["intent_mode"] = "stay_only"

    return constraints


def is_personal_query(query: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", query.lower()))
    markers: List[str] = ["i", "me", "my", "mine", "our", "we", "us"]
    if any(m in tokens for m in markers):
        return True
    zh_markers = ["我", "我们", "我的", "咱们", "俺"]
    return any(m in query for m in zh_markers)
