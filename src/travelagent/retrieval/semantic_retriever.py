from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.models import Destination


class SemanticRetriever:
    def __init__(self, destinations: List[Destination]):
        self.destinations = destinations
        self.id_index = {d.dest_id: i for i, d in enumerate(destinations)}
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=60000,
            min_df=1,
            stop_words="english",
        )
        # Character n-grams are robust for short place names and spelling variants.
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=50000,
            min_df=2,
        )
        corpus = [d.text_blob for d in destinations]
        self.word_doc_matrix = self.word_vectorizer.fit_transform(corpus)
        self.char_doc_matrix = self.char_vectorizer.fit_transform(corpus)

    @staticmethod
    def _cosine_sparse(query_vec, mat) -> np.ndarray:
        num = mat @ query_vec.T
        num = num.toarray().reshape(-1)
        qn = np.linalg.norm(query_vec.toarray()) + 1e-12
        dn = np.sqrt(mat.multiply(mat).sum(axis=1)).A1 + 1e-12
        return num / (dn * qn)

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        qv_word = self.word_vectorizer.transform([query])
        qv_char = self.char_vectorizer.transform([query])

        word_sims = self._cosine_sparse(qv_word, self.word_doc_matrix)
        char_sims = self._cosine_sparse(qv_char, self.char_doc_matrix)
        sims = 0.65 * word_sims + 0.35 * char_sims

        idx = np.argsort(-sims)[:top_k]
        return [(self.destinations[i].dest_id, float(sims[i])) for i in idx]

    def score_single(self, query: str, dest_id: int) -> float:
        results = dict(self.retrieve(query, top_k=1000))
        return float(results.get(dest_id, 0.0))
