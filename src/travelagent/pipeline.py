from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .data.loader import load_destinations
from .graph.kg_builder import build_knowledge_graph
from .refinement.self_refiner import RefinementResult, SelfRefiner
from .retrieval.graph_retriever import GraphRetriever
from .retrieval.hybrid_retriever import Candidate, HybridRetriever
from .retrieval.query_parser import is_personal_query, parse_constraints
from .retrieval.semantic_retriever import SemanticRetriever
from .retrieval.sparse_retriever import SparseRetriever


@dataclass
class RecommendationOutput:
    query: str
    constraints: Dict[str, Any]
    raw_candidates: List[Candidate]
    accepted: List[Candidate]
    discarded_count: int
    refinement_log: List[Dict[str, object]]
    processing_time: float


class HybridGraphRAGPipeline:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self._ready = False

    def initialise(self) -> "HybridGraphRAGPipeline":
        if self._ready:
            return self

        data_path = (self.config_path.parent.parent / self.config["data"]["main_dataset"]).resolve()
        self.destinations = load_destinations(str(data_path))
        self.dest_map = {d.dest_id: d for d in self.destinations}
        self.graph = build_knowledge_graph(self.destinations)

        alpha = float(self.config.get("retrieval", {}).get("semantic_weight", 0.5))
        beta = float(self.config.get("retrieval", {}).get("graph_weight", 0.5))

        self.semantic = SemanticRetriever(self.destinations)
        self.sparse = SparseRetriever(self.destinations)
        self.graph_retriever = GraphRetriever(self.graph, self.destinations)
        self.hybrid = HybridRetriever(
            destinations=self.destinations,
            semantic=self.semantic,
            graph=self.graph_retriever,
            sparse=self.sparse,
            alpha=alpha,
            beta=beta,
        )

        self.refiner = SelfRefiner(
            retriever=self.hybrid,
            theta_accept=float(self.config.get("refinement", {}).get("theta_accept", 0.55)),
            theta_discard=float(self.config.get("refinement", {}).get("theta_discard", 0.20)),
            max_iter=int(self.config.get("refinement", {}).get("max_iterations", 2)),
            neighbor_bonus=float(self.config.get("refinement", {}).get("neighbor_bonus", 0.12)),
        )
        self._ready = True
        return self

    def _apply_personalization(self, candidates: List[Candidate], query: str, enabled: bool) -> List[Candidate]:
        if not enabled or not is_personal_query(query):
            return candidates
        # Lightweight personalization prior for user-centric asks.
        for c in candidates:
            c.confidence = min(1.0, c.confidence + 0.03)
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates

    def recommend(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_semantic: bool = True,
        use_graph: bool = True,
        use_refinement: bool = True,
        use_personalization: bool = True,
    ) -> RecommendationOutput:
        if not self._ready:
            self.initialise()

        t0 = time.time()
        constraints = parse_constraints(query)
        k = int(top_k or self.config.get("experiments", {}).get("top_k", 5))

        # Temporarily modify weights for ablation mode.
        old_alpha = self.hybrid.alpha
        old_beta = self.hybrid.beta

        if use_semantic and use_graph:
            pass
        elif use_semantic and not use_graph:
            self.hybrid.alpha = 1.0
            self.hybrid.beta = 0.0
        elif use_graph and not use_semantic:
            self.hybrid.alpha = 0.0
            self.hybrid.beta = 1.0
        else:
            self.hybrid.alpha = 0.0
            self.hybrid.beta = 1.0

        raw = self.hybrid.retrieve(query=query, constraints=constraints, top_k=max(15, k * 3))
        raw = self._apply_personalization(raw, query=query, enabled=use_personalization)

        if use_refinement:
            ref: RefinementResult = self.refiner.run(raw)
            accepted = ref.accepted[:k]
            discarded_count = len(ref.discarded)
            ref_log = ref.log
        else:
            accepted = raw[:k]
            discarded_count = 0
            ref_log = []

        # Restore weights.
        self.hybrid.alpha = old_alpha
        self.hybrid.beta = old_beta

        return RecommendationOutput(
            query=query,
            constraints=constraints,
            raw_candidates=raw,
            accepted=accepted,
            discarded_count=discarded_count,
            refinement_log=ref_log,
            processing_time=time.time() - t0,
        )
