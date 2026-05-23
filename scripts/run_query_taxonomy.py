from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.travelagent.evaluation.query_taxonomy import build_taxonomy


def load_json_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [str(x) for x in data]


def main() -> None:
    root = ROOT
    cfg_path = root / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    generic_path = (root / cfg["data"]["query_sets"]["generic"]).resolve()
    personal_path = (root / cfg["data"]["query_sets"]["personal"]).resolve()

    queries = load_json_list(generic_path) + load_json_list(personal_path)
    markers = cfg["query_taxonomy"]["personal_markers"]

    rows = build_taxonomy(queries, markers)

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "query_taxonomy_table.csv"

    lines = ["query,query_type,reason"]
    for r in rows:
        q = r.query.replace('"', "''")
        lines.append(f'"{q}",{r.query_type},"{r.reason}"')

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
