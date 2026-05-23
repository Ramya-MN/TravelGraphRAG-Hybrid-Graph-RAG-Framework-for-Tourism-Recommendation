from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEASON_MAP = {
    "春": "Spring",
    "夏": "Summer",
    "秋": "Autumn",
    "冬": "Winter",
    "spring": "Spring",
    "summer": "Summer",
    "autumn": "Autumn",
    "winter": "Winter",
}

TRIP_KEYWORDS = {
    "Adventure": ["adventure", "hiking", "trek", "mountain", "徒步", "登山", "探险"],
    "Cultural": ["museum", "history", "culture", "古迹", "博物馆", "文化", "历史"],
    "Nature": ["lake", "forest", "park", "nature", "自然", "公园", "湖", "山"],
    "Spiritual": ["temple", "church", "monastery", "寺", "庙", "教堂", "祈福"],
    "Food": ["food", "restaurant", "cuisine", "美食", "小吃", "餐厅"],
    "Family": ["family", "kids", "亲子", "家庭", "儿童"],
    "Wellness": ["wellness", "spa", "hot spring", "温泉", "疗养"],
    "Photography": ["photo", "scenic", "摄影", "风景", "打卡"],
}


def infer_trip_types(text: str) -> List[str]:
    t = text.lower()
    out: List[str] = []
    for label, kws in TRIP_KEYWORDS.items():
        if any(k in t for k in kws):
            out.append(label)
    if not out:
        out.append("Cultural")
    return out


def infer_season(query: str) -> List[str]:
    q = query.lower()
    out: List[str] = []
    for k, v in SEASON_MAP.items():
        if k in q and v not in out:
            out.append(v)
    return out


def parse_hours(raw: str) -> Tuple[int, int]:
    if not raw:
        return 1, 3
    nums = re.findall(r"\d+", str(raw))
    if not nums:
        return 1, 3
    if len(nums) == 1:
        n = int(nums[0])
        return max(1, n), max(2, min(7, n + 1))
    a, b = int(nums[0]), int(nums[1])
    return max(1, min(a, b)), max(2, min(10, max(a, b)))


def main() -> None:
    in_path = ROOT / "data" / "tprag_raw" / "tprag_dataset.json"
    out_path = ROOT / "data" / "tprag_processed" / "dataset_tprag.json"
    query_out = ROOT / "data" / "queries" / "thesis_queries.json"
    generic_out = ROOT / "data" / "queries" / "generic_queries.json"
    personal_out = ROOT / "data" / "queries" / "personal_queries.json"

    obj = json.loads(in_path.read_text(encoding="utf-8"))

    pois: Dict[Tuple[str, str], Dict] = {}
    query_rows: List[Dict] = []

    for i, (_, record) in enumerate(obj.items(), start=1):
        query = str(record.get("query", "")).strip()
        if query:
            category = "personal" if any(x in query for x in ["我", "我们", "我的", "俺", "咱们"]) else "generic"
            query_rows.append({"qid": i, "category": category, "query": query})

        seasons = infer_season(query)
        for poi in record.get("poi_list", []):
            name = str(poi.get("名称", "")).strip()
            city = str(poi.get("城市", "Unknown")).strip() or "Unknown"
            if not name:
                continue

            key = (name, city)
            desc = str(poi.get("描述", "")).strip()
            addr = str(poi.get("地址", "")).strip()
            play = str(poi.get("预计游玩时长", ""))
            min_days, ideal_days = parse_hours(play)
            lat = float(poi.get("纬度", 0.0) or 0.0)
            lon = float(poi.get("经度", 0.0) or 0.0)

            trip_types = infer_trip_types(" ".join([name, desc, addr]))

            if key not in pois:
                pois[key] = {
                    "name": name,
                    "city": city,
                    "desc": desc,
                    "addr": addr,
                    "lat": lat,
                    "lon": lon,
                    "trip_types": set(trip_types),
                    "seasons": set(seasons),
                    "count": 1,
                    "ideal_days": ideal_days,
                }
            else:
                p = pois[key]
                p["count"] += 1
                p["trip_types"].update(trip_types)
                p["seasons"].update(seasons)
                if not p["desc"] and desc:
                    p["desc"] = desc
                if not p["addr"] and addr:
                    p["addr"] = addr
                if p["lat"] == 0.0 and lat != 0.0:
                    p["lat"] = lat
                if p["lon"] == 0.0 and lon != 0.0:
                    p["lon"] = lon

    max_count = max(v["count"] for v in pois.values()) if pois else 1

    out: List[Dict] = []
    for idx, p in enumerate(sorted(pois.values(), key=lambda x: x["count"], reverse=True), start=1):
        pop = 1 + 9 * (math.log(1 + p["count"]) / math.log(1 + max_count))
        out.append(
            {
                "id": idx,
                "destination_name": p["name"],
                "state": p["city"],
                "district": p["city"],
                "coordinates": {"latitude": p["lat"], "longitude": p["lon"]},
                "altitude_m": 0,
                "trip_types": sorted(p["trip_types"]),
                "best_seasons": sorted(p["seasons"]) if p["seasons"] else ["Spring", "Summer", "Autumn", "Winter"],
                "avoid_seasons": [],
                "accessibility": "Moderate",
                "permits_required": False,
                "safety_rating": 7.0,
                "popularity_score": round(pop, 2),
                "ideal_days": int(p["ideal_days"]),
                "minimum_days": max(1, int(p["ideal_days"]) - 1),
                "maximum_days": min(10, int(p["ideal_days"]) + 2),
                "unique_experiences": p["desc"],
                "primary_attractions": [p["name"]],
                "activities_available": sorted(p["trip_types"]),
                "language_spoken": ["Chinese", "English"],
                "budget_category": {},
                "mid_range_category": {},
                "luxury_category": {},
                "address": p["addr"],
                "_source": "TP-RAG_GoogleDrive",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Use first 200 TP-RAG queries for evaluation/demos.
    query_rows = query_rows[:200]
    query_out.write_text(json.dumps(query_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    generic_queries = [r["query"] for r in query_rows if r["category"] == "generic"]
    personal_queries = [r["query"] for r in query_rows if r["category"] == "personal"]
    generic_out.write_text(json.dumps(generic_queries[:100], ensure_ascii=False, indent=2), encoding="utf-8")
    personal_out.write_text(json.dumps(personal_queries[:100], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pois={len(out)}")
    print(f"queries={len(query_rows)}")
    print(f"dataset={out_path}")
    print(f"queryset={query_out}")
    print(f"generic_queries={generic_out}")
    print(f"personal_queries={personal_out}")


if __name__ == "__main__":
    main()
