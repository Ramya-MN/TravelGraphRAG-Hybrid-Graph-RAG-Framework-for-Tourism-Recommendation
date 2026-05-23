from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List


def _normalize_query(text: str) -> str:
    q = re.sub(r"\s+", " ", str(text)).strip()
    q = re.sub(r"\s+,", ",", q)
    q = re.sub(r",\s*", ", ", q)
    return q


def _looks_invalid_location(query: str) -> bool:
    q = query.lower()
    if " in " not in q:
        return False
    tail = q.split(" in ", 1)[1].strip()
    if not tail:
        return True
    if tail.startswith(","):
        return True
    if re.fullmatch(r"[0-9\W]+", tail):
        return True
    if len(re.findall(r"[a-z]", tail)) < 2:
        return True
    return False


def load_queries(path: str) -> List[Dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Query file must be a list")

    rows: List[Dict[str, str]] = []
    seen = set()
    for i, item in enumerate(data, start=1):
        if isinstance(item, str):
            query = _normalize_query(item)
            if not query or _looks_invalid_location(query):
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"qid": i, "query": query, "category": "generic"})
        elif isinstance(item, dict):
            query = _normalize_query(item.get("query", ""))
            if not query or _looks_invalid_location(query):
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "qid": int(item.get("qid", i)),
                    "query": query,
                    "category": str(item.get("category", "generic")),
                }
            )
        else:
            raise ValueError("Invalid query row format")
    cleaned: List[Dict[str, str]] = []
    for i, row in enumerate(rows, start=1):
        row["qid"] = i
        cleaned.append(row)
    return cleaned
