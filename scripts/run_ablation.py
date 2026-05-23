from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.travelagent.evaluation.evaluator import run_full_evaluation
from src.travelagent.evaluation.queryset import load_queries
from src.travelagent.pipeline import HybridGraphRAGPipeline


def main() -> None:
    root = ROOT
    cfg_path = root / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    settings = cfg["experiments"]["ablations"]

    query_file = root / cfg["data"]["query_sets"]["thesis"]
    queries = load_queries(str(query_file))

    pipeline = HybridGraphRAGPipeline(str(cfg_path)).initialise()
    out = run_full_evaluation(
        pipeline=pipeline,
        queries=queries,
        settings=settings,
        out_dir=str(root / "results"),
    )

    print("Ablation completed")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
