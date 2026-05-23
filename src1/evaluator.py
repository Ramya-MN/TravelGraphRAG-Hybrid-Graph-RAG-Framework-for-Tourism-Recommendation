"""
evaluator.py — Evaluation metrics for India Travel GraphRAG
"""
import numpy as np
import networkx as nx


class Evaluator:

    def __init__(self, G: nx.DiGraph, retriever):
        self.G         = G
        self.retriever = retriever

    def compare_methods(self, query: str, top_k: int = 5) -> dict:
        vec   = self.retriever.retrieve_vector_only(query, top_k)
        graph, _ = self.retriever.retrieve_graph_only(query, top_k)
        rag, _   = self.retriever.retrieve_graph_rag(query, top_k)

        def scores(results):
            return [round(r.get("score", r.get("final_score", 0)), 4)
                    for r in results]

        # Pad to top_k
        def pad(lst, k):
            s = scores(lst)
            return s + [0.0] * (k - len(s))

        return {
            "vector_only": {"cities": [r["city"] for r in vec],
                            "scores": pad(vec, top_k)},
            "graph_only":  {"cities": [r["city"] for r in graph],
                            "scores": pad(graph, top_k)},
            "graph_rag":   {"cities": [r["city"] for r in rag],
                            "scores": pad(rag, top_k)},
        }

    def climate_accuracy(self, query: str, results: list) -> dict:
        from src1.query_engine import CLIMATE_KEYWORDS
        query_lower = query.lower()
        required_climates = list({v for k, v in CLIMATE_KEYWORDS.items()
                                  if k in query_lower})
        if not required_climates:
            return {"accuracy": 1.0, "required": [], "checked": 0}

        correct = sum(1 for r in results
                      if self.G.nodes.get(r["city"], {}).get(
                          "climate_category") in required_climates)
        return {
            "accuracy":  correct / len(results) if results else 0.0,
            "required":  required_climates,
            "checked":   len(results),
            "correct":   correct,
        }

    def graph_contribution(self, results: list) -> dict:
        if not results:
            return {"avg_boost": 0.0, "boosted_count": 0}
        boosts = [r["graph_score"] - r["vector_score"] for r in results]
        return {
            "avg_boost":    round(np.mean(boosts), 4),
            "boosted_count": sum(1 for b in boosts if b > 0),
            "details":      [{"city": r["city"],
                              "boost": round(r["graph_score"]
                                             - r["vector_score"], 4)}
                             for r in results],
        }

    def evidence_grounding_strength(self, results: list) -> float:
        """Measures how strongly results are grounded in graph evidence."""
        if not results:
            return 0.0
        scores = []
        for r in results:
            direct = sum(1 for e in r.get("evidence_paths", [])
                         if e["evidence_type"] == "direct")
            total  = max(len(r.get("evidence_paths", [])), 1)
            scores.append(direct / total)
        return round(float(np.mean(scores)), 4)

    def hallucination_reduction_ratio(self,
                                       query: str,
                                       rag_results: list,
                                       baseline_results: list) -> float:
        """
        Compares unsupported recommendations between RAG and baseline.
        Lower unsupported rate in RAG = higher ratio (better).
        """
        def unsupported_rate(results):
            if not results: return 1.0
            unsupported = sum(
                1 for r in results
                if r.get("validation", {}).get("pass_rate", 0) < 0.3)
            return unsupported / len(results)

        baseline_rate = unsupported_rate(baseline_results)
        rag_rate      = unsupported_rate(rag_results)

        if baseline_rate == 0:
            return 1.0
        return round(1.0 - (rag_rate / baseline_rate), 4)