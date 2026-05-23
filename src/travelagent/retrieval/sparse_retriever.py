from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

from ..data.models import Destination


class SparseRetriever:
    def __init__(self, destinations: List[Destination], k1: float = 1.5, b: float = 0.75):
        self.destinations = destinations
        self.k1 = k1
        self.b = b

        self._doc_terms: List[Dict[str, int]] = []
        self._doc_len: List[int] = []
        df: Dict[str, int] = {}

        for d in destinations:
            terms = self._tokenize(d.text_blob)
            tf: Dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            self._doc_terms.append(tf)
            doc_len = sum(tf.values())
            self._doc_len.append(doc_len)
            for t in tf.keys():
                df[t] = df.get(t, 0) + 1

        self._df = df
        self._avgdl = sum(self._doc_len) / max(1, len(self._doc_len))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        n = len(self.destinations)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _score_query(self, q_terms: List[str]) -> List[Tuple[int, float]]:
        if not q_terms:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, d in enumerate(self.destinations):
            tf = self._doc_terms[idx]
            dl = self._doc_len[idx]
            score = 0.0
            for t in q_terms:
                f = tf.get(t)
                if not f:
                    continue
                idf = self._idf(t)
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl))
                score += idf * (f * (self.k1 + 1.0)) / denom
            if score > 0:
                scores.append((d.dest_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _prf_expand(self, base_ranked: List[Tuple[int, float]], max_terms: int = 6) -> List[str]:
        if not base_ranked:
            return []

        feedback_docs = [did for did, _ in base_ranked[:5]]
        term_scores: Dict[str, float] = {}
        for did in feedback_docs:
            idx = did - 1
            if idx < 0 or idx >= len(self._doc_terms):
                continue
            tf = self._doc_terms[idx]
            for term, freq in tf.items():
                if len(term) < 3:
                    continue
                term_scores[term] = term_scores.get(term, 0.0) + (freq * self._idf(term))

        ranked_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked_terms[:max_terms]]

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        q_terms = self._tokenize(query)
        if not q_terms:
            return []

        base_ranked = self._score_query(q_terms)
        prf_terms = self._prf_expand(base_ranked)
        if prf_terms:
            q_terms = q_terms + prf_terms

        reranked = self._score_query(q_terms)
        return reranked[:top_k]
