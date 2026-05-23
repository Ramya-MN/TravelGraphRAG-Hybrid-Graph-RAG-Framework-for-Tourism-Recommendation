from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.travelagent.pipeline import HybridGraphRAGPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="User query")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    pipeline = HybridGraphRAGPipeline(str(ROOT / "config" / "config.yaml")).initialise()
    out = pipeline.recommend(args.query, top_k=args.top_k)

    rows = []
    for rank, c in enumerate(out.accepted, start=1):
        rows.append(
            {
                "rank": rank,
                "dest_id": c.dest_id,
                "name": c.name,
                "confidence": round(c.confidence, 4),
                "s_sem": round(c.s_sem, 4),
                "s_graph": round(c.s_graph, 4),
                "matched_constraints": sum(1 for e in c.evidence if e.get("matched")),
            }
        )

    payload = {
        "query": out.query,
        "constraints": out.constraints,
        "discarded_count": out.discarded_count,
        "processing_time_sec": round(out.processing_time, 4),
        "results": rows,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
