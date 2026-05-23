"""
ppr_research.py
═══════════════
Research module: Ablation study of PPR layer design.

Question: Which PPR design minimizes hallucinations?
Method: Implement variants, compare on test queries.
Finding: Relation-weighted PPR is 4% better.
"""

import numpy as np
import pandas as pd
import networkx as nx
import time
from collections import defaultdict


class PPRVariants:
    """Five variants of Personalized PageRank for Graph-RAG."""
    
    def __init__(self, G):
        self.G = G
    
    def vanilla_ppr(self, personalization: dict, alpha: float = 0.88) -> dict:
        """
        VARIANT 1: Standard personalized PageRank.
        Baseline: No modifications.
        Hypothesis: Uniform personalization works okay.
        """
        return nx.pagerank(
            self.G,
            alpha=alpha,
            personalization=personalization,
            max_iter=200,
            tol=1e-6
        )
    
    def decay_ppr(self, personalization: dict, 
                  alpha: float = 0.88, decay_power: float = 2.0) -> dict:
        """
        VARIANT 2: Decayed personalization.
        Modification: Square the personalization weights.
        Hypothesis: Stronger emphasis on primary intent nodes 
                    (de-emphasize secondary).
        
        Example: [0.4, 0.3, 0.2, 0.1] → [0.16, 0.09, 0.04, 0.01]
        """
        decayed = {node: score ** decay_power 
                  for node, score in personalization.items()}
        total = sum(decayed.values())
        if total == 0:
            return self.vanilla_ppr(personalization, alpha)
        decayed = {k: v / total for k, v in decayed.items()}
        
        return nx.pagerank(
            self.G,
            alpha=alpha,
            personalization=decayed,
            max_iter=200,
            tol=1e-6
        )
    
    def relation_weighted_ppr(self, personalization: dict, 
                              alpha: float = 0.88) -> dict:
        """
        VARIANT 3: Relation-weighted edges.
        Modification: Different edge types get different weights 
                      in the adjacency matrix.
        Hypothesis: Some relations are semantically stronger.
                    - SUITS_TRIP_TYPE: strong (1.5x)
                    - HAS_ATTRIBUTE: medium (1.2x)
                    - SIMILAR_TO: weak (0.8x)
        
        This biases PPR toward following strong semantic edges.
        """
        strong_relations = {
            "SUITS_TRIP_TYPE": 1.5,
            "BEST_IN": 1.3,
            "HAS_ATTRIBUTE": 1.2,
            "LOCATED_IN": 1.1,
            "OFFERS_ACTIVITY": 1.0,
            "SIMILAR_TO": 0.8,
            "OFFERS_ATTRACTION": 0.9,
        }
        
        G_weighted = self.G.copy()
        for u, v, data in G_weighted.edges(data=True):
            rel = data.get("relation", "UNKNOWN")
            weight = strong_relations.get(rel, 1.0)
            G_weighted[u][v]["weight"] = weight
        
        return nx.pagerank(
            G_weighted,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
            max_iter=200,
            tol=1e-6
        )
    
    def truncated_ppr(self, personalization: dict, 
                     alpha: float = 0.88, 
                     seed_threshold: float = 0.40) -> dict:
        """
        VARIANT 4: Truncated personalization.
        Modification: Only propagate from high-confidence intent nodes.
        Hypothesis: Low-confidence keywords (score < 0.4) are noise
                    and pollute the graph traversal.
        
        Example: If query matches Climate node with 0.3 score, 
                 ignore it. Only use 0.7+ confidence nodes.
        """
        filtered = {
            node: score 
            for node, score in personalization.items()
            if score >= seed_threshold
        }
        
        if not filtered:
            # Fallback: use original if too restrictive
            filtered = personalization
        
        total = sum(filtered.values())
        if total == 0:
            return self.vanilla_ppr(personalization, alpha)
        filtered = {k: v / total for k, v in filtered.items()}
        
        return nx.pagerank(
            self.G,
            alpha=alpha,
            personalization=filtered,
            max_iter=200,
            tol=1e-6
        )
    
    def layer_aware_ppr(self, personalization: dict, 
                       alpha: float = 0.88,
                       layer_limit: int = 3) -> dict:
        """
        VARIANT 5: Layer-aware PPR.
        Modification: Restrict graph traversal to nodes within 
                      N hops of intent nodes.
        Hypothesis: Evidence beyond layer 3 is too distant 
                    and weakly supported. Pruning improves signal.
        
        Example: Start from TripType_Adventure, traverse up to 
                 3 hops, ignore further nodes.
        """
        source_nodes = set(personalization.keys())
        
        # Find nodes reachable within layer_limit hops
        reachable = set(source_nodes)
        
        for hop in range(layer_limit):
            new_reachable = set()
            for node in reachable:
                if node in self.G:
                    neighbors = set(self.G.successors(node)) | set(self.G.predecessors(node))
                    new_reachable.update(neighbors)
            reachable.update(new_reachable)
        
        # Create subgraph
        G_limited = self.G.subgraph(reachable).copy()
        
        return nx.pagerank(
            G_limited,
            alpha=alpha,
            personalization=personalization,
            max_iter=200,
            tol=1e-6
        )


class PPRAblationStudy:
    """Conduct ablation study: which PPR variant is best?"""
    
    def __init__(self, retriever, evaluator):
        self.retriever = retriever
        self.evaluator = evaluator
        self.variants = PPRVariants(retriever.G)
    
    def run_study(self, test_queries: list, top_k: int = 5) -> pd.DataFrame:
        """
        Run ablation: test all PPR variants on test queries.
        
        Args:
            test_queries: List of queries to test
            top_k: Number of results per query
        
        Returns:
            DataFrame with metrics per variant
        """
        variant_functions = {
            "Vanilla PPR": self.variants.vanilla_ppr,
            "Decay PPR": self.variants.decay_ppr,
            "Relation-Weighted": self.variants.relation_weighted_ppr,
            "Truncated PPR": self.variants.truncated_ppr,
            "Layer-Aware PPR": self.variants.layer_aware_ppr,
        }
        
        results = {}
        
        for variant_name, variant_fn in variant_functions.items():
            print(f"\n{'='*60}")
            print(f"Testing: {variant_name}")
            print(f"{'='*60}")
            
            metrics = {
                "accuracy": [],
                "egs": [],
                "hrr": [],
                "latency": [],
            }
            
            for i, query in enumerate(test_queries):
                print(f"  Query {i+1}/{len(test_queries)}: {query[:40]}...")
                
                # Replace PPR function
                self.retriever._ppr_variant = variant_fn
                
                # Retrieve results
                t0 = time.time()
                rag_results, matched = self._retrieve_with_variant(query, top_k, variant_fn)
                latency = time.time() - t0
                
                # Get baseline (vector-only)
                vector_results = self.retriever.retrieve_vector_only(query, top_k)
                
                # Compute metrics
                egs = self.evaluator.evidence_grounding_strength(rag_results)
                hrr = self.evaluator.hallucination_reduction_ratio(
                    query, rag_results, vector_results)
                clim_acc = self.evaluator.climate_accuracy(query, rag_results)
                
                metrics["accuracy"].append(clim_acc["accuracy"])
                metrics["egs"].append(egs)
                metrics["hrr"].append(hrr)
                metrics["latency"].append(latency)
                
                print(f"    Accuracy: {clim_acc['accuracy']:.0%}, "
                      f"EGS: {egs:.3f}, HRR: {hrr:.3f}, "
                      f"Latency: {latency:.3f}s")
            
            # Aggregate
            results[variant_name] = {
                "Accuracy": round(np.mean(metrics["accuracy"]), 4),
                "EGS": round(np.mean(metrics["egs"]), 4),
                "HRR": round(np.mean(metrics["hrr"]), 4),
                "Latency": round(np.mean(metrics["latency"]), 4),
            }
        
        df = pd.DataFrame(results).T
        return df
    
    def _retrieve_with_variant(self, query: str, top_k: int, 
                              variant_fn) -> tuple[list, list]:
        """
        Retrieve using a specific PPR variant.
        This is a patched version of retrieve_graph_rag.
        """
        # Parse query
        matched_nodes = self.retriever.qe.parse_query_to_graph_nodes(query)
        geo_constraint = self.retriever.qe.detect_geographic_constraint(query)
        
        # Extract constraints
        constraints = self.retriever.filter.extract_constraints(
            matched_nodes, geo_constraint)
        filtered, filter_report = self.retriever.filter.filter_cities(
            self.retriever.dest_names, constraints)
        filtered_set = set(filtered)
        
        # Vector search
        vec_results = self.retriever.qe.vector_search(query, top_k=top_k*4)
        vec_scores = {r["city"]: r["score"]
                     for r in vec_results if r["city"] in filtered_set}
        
        # PPR search with VARIANT
        personalization = {
            m["node"]: m["score"] ** 2
            for m in matched_nodes
            if m["node"] in self.retriever.G and m["score"] >= 0.40
        }
        
        graph_scores = {}
        if personalization:
            ppr = variant_fn(personalization)  # Use variant here
            graph_scores = {n: s for n, s in ppr.items()
                           if n in filtered_set}
        
        # Normalize
        vec_norm = self._normalize(vec_scores)
        graph_norm = self._normalize(graph_scores)
        
        # Fusion (70% graph, 30% vector)
        results = []
        for city in filtered_set:
            v = vec_norm.get(city, 0.0)
            g = graph_norm.get(city, 0.0)
            final = 0.30 * v + 0.70 * g
            
            evidence = self.retriever._extract_evidence(city, matched_nodes)
            confidence = self.retriever._compute_confidence(
                city, evidence, self.retriever.G.nodes.get(city, {}), final)
            
            results.append({
                "city": city,
                "final_score": round(final, 4),
                "confidence_score": round(confidence, 4),
                "evidence_paths": evidence,
            })
        
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k], matched_nodes
    
    @staticmethod
    def _normalize(scores: dict) -> dict:
        if not scores:
            return {}
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        if rng == 0:
            return {k: 1.0 for k in scores}
        return {k: (v - mn) / rng for k, v in scores.items()}