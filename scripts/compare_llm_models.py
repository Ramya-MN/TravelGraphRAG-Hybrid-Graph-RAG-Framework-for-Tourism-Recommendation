from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _create_default_template(path: Path) -> None:
    rows = [
        {
            "system": "TravelAgentV3",
            "model_family": "gpt-4o-mini",
            "setup": "HybridGraphRAG",
            "precision_at_5": 0.0,
            "recall_at_5": 0.0,
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
            "ndcg_at_5": 0.0,
            "mrr": 0.0,
            "map_at_5": 0.0,
            "hit_rate_at_5": 0.0,
            "groundedness": 0.0,
            "latency_sec": 0.0,
        },
        {
            "system": "TravelAgentV3",
            "model_family": "gpt-4o-mini",
            "setup": "LLMOnly_NoRAG",
            "precision_at_5": 0.0,
            "recall_at_5": 0.0,
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
            "ndcg_at_5": 0.0,
            "mrr": 0.0,
            "map_at_5": 0.0,
            "hit_rate_at_5": 0.0,
            "groundedness": 0.0,
            "latency_sec": 0.0,
        },
    ]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    in_path = ROOT / "results" / "llm_comparison_template.csv"
    out_path = ROOT / "results" / "llm_comparison_summary.json"

    if not in_path.exists():
        _create_default_template(in_path)
        print(f"created default template: {in_path}")

    df = pd.read_csv(in_path)
    required = {
        "system",
        "model_family",
        "setup",
        "precision_at_5",
        "recall_at_5",
        "recall_at_20",
        "recall_at_50",
        "ndcg_at_5",
        "mrr",
        "map_at_5",
        "hit_rate_at_5",
        "groundedness",
        "latency_sec",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        summary = {
            "rows": 0,
            "best_overall": [],
            "note": "Template exists but contains no data rows.",
        }
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {out_path}")
        return

    metric_cols = [
        "precision_at_5",
        "recall_at_5",
        "recall_at_20",
        "recall_at_50",
        "ndcg_at_5",
        "mrr",
        "map_at_5",
        "hit_rate_at_5",
        "groundedness",
        "latency_sec",
    ]
    if df[metric_cols].fillna(0.0).to_numpy().sum() <= 0.0:
        summary = {
            "rows": int(len(df)),
            "best_overall": [],
            "note": "All metrics are zero; upload real benchmark results to generate a comparison.",
        }
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {out_path}")
        return

    eval_path = ROOT / "results" / "evaluation_summary.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        summary_rows = eval_data.get("summary", [])
        full_row = next((r for r in summary_rows if r.get("setting") == "full"), None)
        if full_row:
            runtime = float(full_row.get("runtime_sec", 0.0))
            filled = {
                "system": "TravelAgentV3",
                "model_family": "n/a",
                "setup": "HybridGraphRAG",
                "precision_at_5": float(full_row.get("precision_at_5", 0.0)),
                "recall_at_5": float(full_row.get("recall_at_5", 0.0)),
                "recall_at_20": float(full_row.get("recall_at_20", 0.0)),
                "recall_at_50": float(full_row.get("recall_at_50", 0.0)),
                "ndcg_at_5": float(full_row.get("ndcg_at_5", 0.0)),
                "mrr": float(full_row.get("mrr", 0.0)),
                "map_at_5": float(full_row.get("map_at_5", 0.0)),
                "hit_rate_at_5": float(full_row.get("hit_rate_at_5", 0.0)),
                "groundedness": float(full_row.get("grounding_score", 0.0)),
                "latency_sec": runtime,
            }
            df = df[~((df["system"] == "TravelAgentV3") & (df["setup"] == "HybridGraphRAG"))]
            df = pd.concat([df, pd.DataFrame([filled])], ignore_index=True)

    ours = df[df["setup"].str.lower().str.contains("hybridgraphrag|rag", regex=True)]
    baseline = df[df["setup"].str.lower().str.contains("norag|llmonly", regex=True)]

    summary = {
        "rows": int(len(df)),
        "best_overall": df.sort_values("ndcg_at_5", ascending=False).head(1).to_dict(orient="records"),
    }

    if not ours.empty and not baseline.empty:
        ours_best = float(ours["ndcg_at_5"].max())
        base_best = float(baseline["ndcg_at_5"].max())
        summary["ndcg_advantage_vs_best_llm_only"] = round(ours_best - base_best, 4)

        ours_g = float(ours["groundedness"].max())
        base_g = float(baseline["groundedness"].max())
        summary["groundedness_advantage_vs_best_llm_only"] = round(ours_g - base_g, 4)
    else:
        summary["note"] = "Add LLM-only rows to compare against HybridGraphRAG."

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
