from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    summary_path = ROOT / "results" / "ablation_summary.csv"
    per_query_path = ROOT / "results" / "per_query_results.csv"
    out_dir = ROOT / "results" / "proof_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists() or not per_query_path.exists():
        raise FileNotFoundError("Run thesis evaluation first to generate result CSV files.")

    sdf = pd.read_csv(summary_path)
    qdf = pd.read_csv(per_query_path)

    fig1 = px.bar(
        sdf,
        x="setting",
        y=["ndcg_at_5", "map_at_5", "hit_rate_at_5"],
        barmode="group",
        title="Retrieval Quality Comparison",
    )
    fig1.write_html(out_dir / "quality_comparison.html")

    fig2 = px.bar(
        sdf,
        x="setting",
        y=["grounding_score", "violation_rate"],
        barmode="group",
        title="Grounding And Violation Proof",
    )
    fig2.write_html(out_dir / "grounding_violation.html")

    fig3 = px.scatter(
        sdf,
        x="runtime_sec",
        y="ndcg_at_5",
        color="setting",
        size="hit_rate_at_5",
        title="Latency vs NDCG@5",
    )
    fig3.write_html(out_dir / "latency_vs_ndcg.html")

    cat = qdf.groupby(["setting", "category"], as_index=False)["ndcg_at_5"].mean()
    fig4 = px.bar(cat, x="setting", y="ndcg_at_5", color="category", barmode="group", title="Category-wise NDCG@5")
    fig4.write_html(out_dir / "category_ndcg.html")

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
