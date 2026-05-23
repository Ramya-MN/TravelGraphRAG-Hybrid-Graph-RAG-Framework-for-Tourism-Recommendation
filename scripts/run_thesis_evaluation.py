from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.travelagent.evaluation.evaluator import run_full_evaluation
from src.travelagent.evaluation.queryset import load_queries
from src.travelagent.pipeline import HybridGraphRAGPipeline


def main() -> None:
    cfg_path = ROOT / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    query_file = ROOT / cfg["data"]["query_sets"]["thesis"]
    queries = load_queries(str(query_file))

    pipeline = HybridGraphRAGPipeline(str(cfg_path)).initialise()

    settings = cfg["experiments"]["ablations"]
    out = run_full_evaluation(
        pipeline=pipeline,
        queries=queries,
        settings=settings,
        out_dir=str(ROOT / "results"),
    )

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_proof_plots.py")],
        cwd=str(ROOT),
        check=False,
    )

    print("Evaluation completed")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
