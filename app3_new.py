"""
app3.py — GraphRAG Evaluation Dashboard (India Travel Edition)
Full evaluation metrics: Graph Boost, Climate Accuracy, Evidence Grounding,
Hallucination Reduction Ratio, Method Comparison, Refinement traces.
"""

import streamlit as st
import json
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
from PIL import Image
from typing import Optional
import pandas as pd

from src1.build_graph import load_graph
from src1.query_engine import SemanticQueryEngine
from src1.retriever import GraphRAGRetriever
from src1.evaluator import Evaluator
from src.query_expander import QueryExpander


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Travel GraphRAG Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #0068c9; }
    .metric-label { font-size: 14px; color: #555; }
    .expansion-tag {
        display: inline-block;
        background-color: #e8f4fd;
        border: 1px solid #0068c9;
        color: #0068c9;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 3px;
        font-size: 13px;
    }
    .expansion-tag-original {
        display: inline-block;
        background-color: #fff3cd;
        border: 1px solid #ffa500;
        color: #b35900;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 3px;
        font-size: 13px;
        font-weight: 600;
    }
    .expansion-box {
        background-color: #f8f9fa;
        border-left: 4px solid #0068c9;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-low  { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class ImageManager:
    def __init__(self, base_path="data/city_images"):
        self.base = Path(base_path)

    def get_images(self, name, limit=3):
        folder = self.base / name.replace(" ", "_")
        if not folder.exists():
            return []
        imgs = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            imgs.extend(sorted(folder.glob(ext)))
        return [str(p) for p in imgs[:limit]]

    def count(self, name):
        return len(self.get_images(name, limit=100))


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM INIT
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Knowledge Graph...")
def initialize_system():
    G, dest_names, metadata = load_graph()
    query_engine = SemanticQueryEngine(G, dest_names)
    retriever    = GraphRAGRetriever(G, dest_names, query_engine, metadata)
    evaluator    = Evaluator(G, retriever)
    return G, retriever, evaluator, dest_names, metadata

@st.cache_resource(show_spinner="Loading Image Manager...")
def init_images():
    return ImageManager()

@st.cache_resource(show_spinner="Loading Query Expander...")
def init_expander():
    return QueryExpander(max_synonyms_per_word=3, use_wordnet=True)

try:
    G_main, retriever, evaluator, dest_names, metadata = initialize_system()
    image_mgr = init_images()
    expander  = init_expander()
    st.sidebar.success(f"✅ Graph ready — {len(dest_names)} destinations")
except Exception as e:
    st.error(f"Startup error: {e}")
    import traceback; st.code(traceback.format_exc())
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def display_city_images(name, max_cols=3, limit=3):
    imgs = image_mgr.get_images(name, limit)
    if not imgs:
        st.caption("⚠️ No images available")
        return
    cols = st.columns(min(len(imgs), max_cols))
    for i, path in enumerate(imgs):
        try:
            with cols[i % len(cols)]:
                st.image(Image.open(path), use_column_width=True,
                         caption=f"Image {i+1}")
        except Exception:
            pass


def display_expansion_report(report: dict):
    original_set = set(report["original_tokens"])
    tags = ""
    for term in report["all_terms"]:
        css = ("expansion-tag-original" if term in original_set
               else "expansion-tag")
        tags += f'<span class="{css}">{term}</span>'
    st.markdown(
        f'<div class="expansion-box">'
        f'<strong>🔍 Expanded Query ({len(report["all_terms"])} terms):</strong><br><br>'
        f'{tags}<br><br>'
        f'<small>🟡 Original &nbsp;|&nbsp; 🔵 Synonyms</small>'
        f'</div>', unsafe_allow_html=True)


def plot_radar_chart(results: list):
    """Radar chart using Indian destination attributes."""
    DIMS = {
        "Popularity":    lambda r: (r.get("popularity_score") or 5) / 10,
        "Safety":        lambda r: (r.get("safety_rating") or 5) / 10,
        "Accessibility": lambda r: {"Easy": 1.0, "Moderate": 0.6,
                                     "Difficult": 0.3}.get(
                                    r.get("accessibility", "Moderate"), 0.6),
        "Confidence":    lambda r: r.get("confidence_score", 0.5),
        "Altitude":      lambda r: min((r.get("altitude_m") or 0) / 4000, 1.0),
        "Ideal Days":    lambda r: min((r.get("ideal_days") or 3) / 14, 1.0),
    }

    labels = list(DIMS.keys())
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ["#0068C9", "#FF4B4B", "#29B5E8", "#FFA500", "#28a745"]

    for idx, r in enumerate(results[:5]):
        node = G_main.nodes.get(r["city"], {})
        values = [DIMS[d](node) for d in labels]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                color=colors[idx], label=r["city"])
        ax.fill(angles, values, alpha=0.08, color=colors[idx])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"],
                       fontsize=7, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title("Destination Profile Comparison",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    st.pyplot(fig)
    plt.close(fig)


def get_pyvis_html(results: list, matched_nodes: list) -> Optional[str]:
    net = Network(height="500px", width="100%",
                  bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based(gravity=-80)

    TYPE_COLORS = {
        "TripType":      "#FF6B6B", "Climate":     "#4ECDC4",
        "Season":        "#45B7D1", "BudgetTier":  "#96CEB4",
        "Destination":   "#0068C9", "State":       "#FFEAA7",
        "Accessibility": "#DDA0DD",
    }
    seen = set()

    for m in matched_nodes:
        node = m["node"]
        if node not in seen:
            ntype = m.get("node_type", "")
            color = TYPE_COLORS.get(ntype, "#FF4B4B")
            net.add_node(node, label=node, color=color,
                         shape="star", size=20)
            seen.add(node)

    for r in results[:4]:
        city = r["city"]
        if city not in seen:
            net.add_node(city, label=city, color="#0068C9",
                         shape="dot", size=18)
            seen.add(city)
        for ep in r["evidence_paths"]:
            target = ep["target"]
            if target not in seen:
                net.add_node(target, label=target,
                             color="#29B5E8", size=12)
                seen.add(target)
            net.add_edge(city, target, color="#cccccc", width=1,
                         title=ep.get("relation", ""))

    try:
        net.save_graph("temp_graph.html")
        with open("temp_graph.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔬 Query Controls")

    BENCHMARK_QUERIES = [
        "adventure trekking in cold hill stations",
        "beach destination with budget accommodation",
        "spiritual pilgrimage in north India winter",
        "luxury wildlife safari south India",
        "family trip moderate climate easy accessibility",
        "offbeat nature destination in northeast India",
        "romantic honeymoon destination kerala monsoon",
        "cultural heritage tour Rajasthan winter",
        "wellness yoga retreat Uttarakhand",
        "photography landscape high altitude",
    ]

    selected = st.selectbox("Benchmark Queries:", BENCHMARK_QUERIES)
    custom   = st.text_input("Or type custom query:", "")
    query    = custom if custom else selected

    st.divider()
    st.markdown("### ⚙️ Retrieval Weights")
    v_weight = st.slider("Vector Weight", 0.0, 1.0, 0.35, 0.05)
    g_weight = round(1.0 - v_weight, 2)
    st.write(f"Graph Weight: **{g_weight}**")

    st.divider()
    st.markdown("### 🔍 Query Expansion")
    use_expansion = st.toggle("Enable Expansion", value=True)
    max_synonyms  = st.slider("Synonyms per term", 1, 5, 3)

    st.divider()
    st.markdown("### 🖼️ Display")
    images_per_city = st.slider("Images per city", 1, 5, 3)
    top_k = st.slider("Top K results", 3, 10, 5)

    st.divider()
    with st.expander("📊 Graph Stats"):
        st.write(f"**Destinations:** {len(dest_names)}")
        st.write(f"**Total nodes:** {metadata.get('total_nodes', 'N/A')}")
        st.write(f"**Total edges:** {metadata.get('total_edges', 'N/A')}")
        for layer, desc in metadata.get("graph_layers", {}).items():
            st.caption(f"Layer {layer}: {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.title("🇮🇳 India Travel — Graph-RAG Benchmarking Dashboard")
st.markdown(f"**Query:** `{query}`")

if st.button("🚀 Run Full Analysis", type="primary"):

    # ── Step 0: Query Expansion ───────────────────────────────────────────────
    expander.max_synonyms = max_synonyms
    if use_expansion:
        expansion_report = expander.expansion_report(query)
        retrieval_query  = expander.expand_to_string(query)
    else:
        expansion_report = None
        retrieval_query  = query

    # ── Step 1: Retrieval + All Evaluation Metrics ────────────────────────────
    with st.spinner("Running retrieval and evaluation pipeline..."):
        t0 = time.time()

        # Primary GraphRAG retrieval
        results, matched_nodes = retriever.retrieve_graph_rag(
            retrieval_query,
            top_k=top_k,
            vector_weight=v_weight,
            graph_weight=g_weight
        )
        rag_time = time.time() - t0

        # Vector-only baseline — needed for HRR and method comparison.
        # Called once here and reused; avoids running it twice.
        baseline_results = retriever.retrieve_vector_only(retrieval_query, top_k)

        # Evaluation metrics
        comparison  = evaluator.compare_methods(retrieval_query, top_k)
        clim_acc    = evaluator.climate_accuracy(retrieval_query, results)
        graph_boost = evaluator.graph_contribution(results)
        egs         = evaluator.evidence_grounding_strength(results)
        hrr         = evaluator.hallucination_reduction_ratio(
                          retrieval_query, results, baseline_results)

    # ── Step 2: Export to JSON ────────────────────────────────────────────────
    try:
        output_data = {
            "query":          query,
            "expanded_query": retrieval_query,
            "parameters": {
                "top_k":          top_k,
                "vector_weight":  v_weight,
                "graph_weight":   g_weight,
            },
            "metrics": {
                "retrieval_time_s":              rag_time,
                "climate_accuracy":              clim_acc,
                "graph_contribution":            graph_boost,
                "evidence_grounding_strength":   egs,
                "hallucination_reduction_ratio": hrr,
            },
            "matched_nodes": matched_nodes,
            "results":       results,
        }
        file_path = "graphrag_output.json"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                existing = [existing]
        except Exception:
            existing = []
        existing.append(output_data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, default=str)
    except Exception as e:
        st.warning(f"JSON export failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 0: Query Expansion Trace
    # ══════════════════════════════════════════════════════════════════════════
    if use_expansion and expansion_report:
        st.markdown("### 0. Query Expansion Trace")
        c1, c2 = st.columns([3, 1])
        with c1:
            display_expansion_report(expansion_report)
        with c2:
            st.metric("Terms Added",
                      expansion_report["expansion_count"],
                      f"+{expansion_report['expansion_count']}")
        st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Scorecard — all 5 metrics live
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 1. System Performance Scorecard")
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{rag_time:.3f}s</div>'
            f'<div class="metric-label">Latency</div>'
            f'</div>', unsafe_allow_html=True)

    with m2:
        boost_color = "green" if graph_boost["avg_boost"] > 0 else "red"
        st.markdown(
            f'<div class="metric-card" style="border-left:5px solid {boost_color}">'
            f'<div class="metric-value">{graph_boost["avg_boost"]:+.1%}</div>'
            f'<div class="metric-label">Graph Boost</div>'
            f'</div>', unsafe_allow_html=True)

    with m3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{clim_acc["accuracy"]:.0%}</div>'
            f'<div class="metric-label">Climate Accuracy</div>'
            f'</div>', unsafe_allow_html=True)

    with m4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{egs:.0%}</div>'
            f'<div class="metric-label">Evidence Grounding</div>'
            f'</div>', unsafe_allow_html=True)

    with m5:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{hrr:.0%}</div>'
            f'<div class="metric-label">Hallucination Reduction</div>'
            f'</div>', unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Method Comparison (Vector vs. Graph vs. RAG)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 2. Method Comparison (Vector vs. Graph vs. RAG)")
    c1, c2 = st.columns([2, 1])
    with c1:
        chart_data = pd.DataFrame({
            "Vector Only":      comparison["vector_only"]["scores"],
            "Graph Only":       comparison["graph_only"]["scores"],
            "Graph RAG (Ours)": comparison["graph_rag"]["scores"],
        })
        st.bar_chart(chart_data, color=["#A9A9A9", "#FFA500", "#0068C9"])
        st.caption("Y: Normalized Score | X: Rank Position")
    with c2:
        st.info("💡 **Insight**")
        if graph_boost["avg_boost"] > 0:
            st.markdown(
                f"Graph RAG outperforms Vector by "
                f"**{graph_boost['avg_boost']:.1%}**. "
                f"Knowledge graph adds factual grounding.")
        else:
            st.markdown(
                "Graph acting as **strict filter** — removing "
                "semantically similar but factually wrong results.")
        if results and results[0].get("filter_report"):
            st.markdown(
                f"**Constraint filtering:** "
                f"{results[0]['filter_report']['filtered_out']} "
                f"destinations eliminated")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: Evidence Subgraph
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 3. Reasoning Trace — Evidence Subgraph")
    st.caption("Shows the exact graph paths used to validate recommendations.")
    html = get_pyvis_html(results, matched_nodes)
    if html:
        components.html(html, height=520)
    else:
        st.warning("Could not render subgraph.")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Radar Chart
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 4. Destination Profile Radar")
    r1, r2 = st.columns([3, 1])
    with r1:
        plot_radar_chart(results)
    with r2:
        st.info("**Axes:**\n\n"
                "- **Popularity** — tourist demand\n"
                "- **Safety** — safety rating\n"
                "- **Accessibility** — ease of reach\n"
                "- **Confidence** — evidence strength\n"
                "- **Altitude** — elevation (hill stations)\n"
                "- **Ideal Days** — recommended stay")
        st.markdown("**Rankings:**")
        for i, r in enumerate(results[:5], 1):
            bar  = "█" * int(r["final_score"] * 10)
            conf = r["confidence_score"]
            css  = "confidence-high" if conf >= 0.6 else "confidence-low"
            st.markdown(
                f"`#{i}` **{r['city']}**  "
                f"`{bar}` {r['final_score']:.3f}  "
                f"<span class='{css}'>conf:{conf:.2f}</span>",
                unsafe_allow_html=True)
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: Detailed Results
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 5. Top Recommendations & Evidence")

    for i, r in enumerate(results, 1):
        city  = r["city"]
        node  = G_main.nodes.get(city, {})
        val   = r["validation"]
        conf  = r["confidence_score"]
        n_img = image_mgr.count(city)

        header = f"#### #{i} {city}"
        if n_img > 0:
            header += f" 🖼️ ({n_img} images)"
        conf_emoji = "🟢" if conf >= 0.6 else "🟡" if conf >= 0.4 else "🔴"
        header += f"  {conf_emoji} conf: {conf:.2f}"
        st.markdown(header)

        tab1, tab2, tab3 = st.tabs(
            ["📍 Destination Info", "🔗 Evidence Paths", "📸 Images"])

        with tab1:
            i1, i2, i3 = st.columns(3)
            with i1:
                st.metric("State",       node.get("state", "N/A"))
                st.metric("Climate",     node.get("climate_category", "N/A"))
                st.metric("Budget Tier", node.get("budget_tier", "N/A"))
            with i2:
                st.metric("Altitude",    f"{node.get('altitude_m') or 'N/A'} m")
                st.metric("Ideal Stay",  f"{node.get('ideal_days') or 'N/A'} days")
                st.metric("Peak Season", node.get("peak_tourist_season") or "N/A")
            with i3:
                st.metric("Popularity",
                          f"{node.get('popularity_score') or 'N/A'}/10")
                st.metric("Safety",
                          f"{node.get('safety_rating') or 'N/A'}/10")
                st.metric("Accessibility",
                          node.get("accessibility") or "N/A")

            if node.get("permits_required"):
                st.warning(
                    f"⚠️ Permit required: {node.get('permits_details', '')}")
            if node.get("unique_experiences"):
                st.caption(f"✨ {node['unique_experiences'][:200]}")

        with tab2:
            st.write(
                f"**Validation:** {val['passed']}/{val['total']} "
                f"criteria met ({val['pass_rate']:.0%})")
            st.write(
                f"**Final Score:** {r['final_score']:.4f} "
                f"(vec: {r['vector_score']:.3f} + "
                f"graph: {r['graph_score']:.3f})")
            st.markdown("**Evidence Paths:**")
            for ep in r["evidence_paths"]:
                icon = ("✅" if ep["evidence_type"] == "direct" else
                        "🔗" if "multi_hop" in ep["evidence_type"] else "❌")
                src  = f"[{ep['source']}]" if ep.get("source") else ""
                st.code(f"{icon} {ep['path_str']} {src}", language="text")
            if conf < 0.4:
                st.warning(
                    "⚠️ Low confidence — Self-Refining Loop "
                    "attempted to find better evidence.")

        with tab3:
            display_city_images(city, max_cols=images_per_city,
                                limit=images_per_city)

        st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: Ranked-Out Candidates
    # ══════════════════════════════════════════════════════════════════════════
    if results:
        retrieval_report  = results[0].get("retrieval_report", {})
        refinement_report = results[0].get("refinement_report", {})

        ranked_out = retrieval_report.get("ranked_out", [])
        if ranked_out:
            with st.expander(
                    f"📉 Ranked-Out Candidates — {len(ranked_out)}"):
                st.caption(
                    "These passed hard constraints but ranked below "
                    "the top results.")
                for item in ranked_out:
                    reasons = "; ".join(item.get("reasons", []))
                    stats = (
                        f"score={item['final_score']:.3f}, "
                        f"conf={item['confidence_score']:.2f}, "
                        f"direct={item['evidence_stats']['direct']}"
                    )
                    st.write(f"**{item['city']}** — {reasons} ({stats})")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 7: Self-Refining Loop Trace
        # ══════════════════════════════════════════════════════════════════════
        if refinement_report.get("iterations"):
            total_removed = sum(
                len(it.get("removed", []))
                for it in refinement_report["iterations"]
            )
            with st.expander(
                    f"🔁 Refinement Rejections — {total_removed}"):
                st.caption(
                    "Low-confidence recommendations replaced "
                    "during the self-refining loop.")
                for it in refinement_report["iterations"]:
                    st.markdown(f"**Iteration {it['iteration']}**")
                    for item in it.get("removed", []):
                        st.write(
                            f"❌ {item['city']} — {item['reason']} "
                            f"(conf={item['confidence_score']:.2f}, "
                            f"score={item['final_score']:.3f})")
                    for item in it.get("added", []):
                        st.write(
                            f"✅ {item['city']} — {item['reason']} "
                            f"(conf={item['confidence_score']:.2f}, "
                            f"score={item['final_score']:.3f})")

else:
    st.info("👈 Select a query from the sidebar and click "
            "**Run Full Analysis** to start.")

    with st.expander("📖 About this System"):
        st.markdown("""
        **Multi-Layer Knowledge Graph** for Indian travel recommendations.

        **Graph Layers:**
        - Layer 1: Geographic hierarchy (India → State → District → Destination)
        - Layer 2: Destination nodes with verified attributes
        - Layer 3: Attribute nodes (Climate, Budget, Season, Accessibility)
        - Layer 4: Experience nodes (TripType, Activity, Attraction, Cuisine)
        - Layer 5: Similarity edges (cosine similarity between destinations)

        **Architecture:**
        - Dual-Evidence Retrieval (vector + graph PPR)
        - Evidence-Aware Confidence Scoring
        - Self-Refining Retrieval Loop
        - Hard Constraint Filtering
        """)