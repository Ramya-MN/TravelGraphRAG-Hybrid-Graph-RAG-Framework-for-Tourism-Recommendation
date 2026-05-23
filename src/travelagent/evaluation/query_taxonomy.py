from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List


@dataclass
class TaxonomyRow:
    query: str
    query_type: str
    reason: str


def classify_query(query: str, personal_markers: Iterable[str]) -> TaxonomyRow:
    text = query.lower().strip()
    normalized = " " + re.sub(r"\s+", " ", text) + " "
    words = set(re.findall(r"[a-z']+", text))

    for marker in personal_markers:
        mk = marker.lower().strip()
        if not mk:
            continue

        # Single-word markers are matched as whole tokens only.
        if " " not in mk and mk in words:
            return TaxonomyRow(query=query, query_type="personal", reason=f"matched marker: {marker}")

        # Multi-word markers are matched as whole phrase spans.
        phrase = " " + mk + " "
        if " " in mk and phrase in normalized:
            return TaxonomyRow(query=query, query_type="personal", reason=f"matched marker: {marker}")

    return TaxonomyRow(query=query, query_type="generic", reason="no personal markers found")


def build_taxonomy(queries: List[str], personal_markers: Iterable[str]) -> List[TaxonomyRow]:
    return [classify_query(q, personal_markers) for q in queries]
