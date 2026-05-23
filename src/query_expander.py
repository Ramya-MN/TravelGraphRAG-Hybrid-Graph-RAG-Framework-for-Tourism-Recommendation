"""
query_expander.py
─────────────────
Expands a raw user query into a richer set of search terms before
retrieval. Uses two complementary strategies:

  1. Travel-domain keyword map  — hand-crafted synonyms for common
     travel intents (fast, zero dependencies, high precision)

  2. NLTK WordNet synonyms      — general-purpose synonym expansion
     for any word not in the domain map (broad coverage)

The result is a deduplicated list of terms that the retriever can
use to cast a wider net over the knowledge graph.
"""

from __future__ import annotations
import re
from typing import List, Dict

# ── Optional: NLTK WordNet ────────────────────────────────────────────────────
try:
    from nltk.corpus import wordnet
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False


# ── Travel-domain synonym map ─────────────────────────────────────────────────
# Keys are words that commonly appear in travel queries.
# Values are the extra terms to inject into the expanded query.
TRAVEL_DOMAIN_MAP: Dict[str, List[str]] = {
    # Climate / weather
    "moderate":     ["temperate", "mild", "pleasant", "warm"],
    "warm":         ["hot", "sunny", "tropical", "mediterranean"],
    "cold":         ["cool", "chilly", "arctic", "northern", "alpine"],
    "sunny":        ["warm", "bright", "clear", "mediterranean"],
    "temperate":    ["moderate", "mild", "comfortable"],

    # Terrain / geography
    "beach":        ["coastal", "seaside", "shore", "ocean", "sea", "waterfront"],
    "beaches":      ["coastal", "seaside", "shore", "ocean", "sea", "waterfront"],
    "mountain":     ["alpine", "highland", "hills", "peaks", "elevation"],
    "mountains":    ["alpine", "highland", "hills", "peaks", "elevation"],
    "ocean":        ["sea", "coast", "waterfront", "marine", "beach"],
    "sea":          ["ocean", "coast", "waterfront", "marine", "beach"],
    "nature":       ["wilderness", "outdoors", "landscape", "scenic", "green"],
    "forest":       ["woodland", "trees", "nature", "green", "wilderness"],
    "island":       ["coastal", "sea", "beach", "marine", "tropical"],

    # Vibe / atmosphere
    "nightlife":    ["clubs", "bars", "entertainment", "parties", "social"],
    "clubbing":     ["nightlife", "clubs", "bars", "dancing", "parties"],
    "romantic":     ["couples", "intimate", "scenic", "cozy", "peaceful"],
    "adventure":    ["outdoor", "extreme", "hiking", "sports", "active"],
    "peaceful":     ["quiet", "tranquil", "serene", "secluded", "calm"],
    "seclusion":    ["isolated", "quiet", "remote", "peaceful", "rural"],
    "luxury":       ["upscale", "premium", "exclusive", "high-end", "boutique"],
    "budget":       ["affordable", "cheap", "economical", "backpacker", "low-cost"],

    # Activities
    "hiking":       ["trekking", "trails", "outdoor", "nature", "walking"],
    "wellness":     ["spa", "relaxation", "health", "yoga", "retreat"],
    "culture":      ["history", "museums", "art", "heritage", "architecture"],
    "cuisine":      ["food", "gastronomy", "restaurants", "culinary", "dining"],
    "shopping":     ["markets", "boutiques", "stores", "retail", "malls"],
    "art":          ["museums", "galleries", "culture", "exhibitions", "creative"],
    "history":      ["heritage", "ancient", "historic", "culture", "monuments"],

    # Urban / rural
    "urban":        ["city", "metropolitan", "downtown", "cosmopolitan", "modern"],
    "city":         ["urban", "metropolitan", "downtown", "cosmopolitan"],
    "rural":        ["countryside", "village", "pastoral", "remote", "quiet"],
    "retreat":      ["peaceful", "secluded", "getaway", "quiet", "relaxing"],
    "getaway":      ["retreat", "escape", "vacation", "holiday", "trip"],
}


# ── Core Expander Class ───────────────────────────────────────────────────────
class QueryExpander:
    """
    Expands a user query into a richer set of search terms.

    Usage
    ─────
    expander = QueryExpander(max_synonyms_per_word=3)
    expanded = expander.expand("beaches with moderate climate")
    # → ["beaches", "coastal", "seaside", "shore", "ocean",
    #    "moderate", "temperate", "mild", "pleasant",
    #    "climate", "weather"]
    """

    def __init__(
        self,
        max_synonyms_per_word: int = 3,
        use_wordnet: bool = True,
        stopwords: set | None = None,
    ):
        self.max_synonyms = max_synonyms_per_word
        self.use_wordnet = use_wordnet and WORDNET_AVAILABLE

        # Words to skip during expansion
        self.stopwords = stopwords or {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
            "of", "with", "is", "are", "was", "were", "be", "been", "being",
            "i", "me", "my", "can", "go", "where", "what", "how", "which",
            "some", "any", "from", "that", "this", "it", "its", "have", "has",
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def expand(self, query: str) -> List[str]:
        """
        Return a deduplicated list of expanded terms for the query.

        Parameters
        ----------
        query : str
            Raw user query, e.g. "beaches with moderate climate"

        Returns
        -------
        List[str]
            Original tokens + synonyms, deduplicated, lower-cased.
        """
        tokens = self._tokenize(query)
        expanded: List[str] = list(tokens)  # always keep originals

        for token in tokens:
            if token in self.stopwords:
                continue
            # 1. Domain map first (higher precision)
            domain_syns = TRAVEL_DOMAIN_MAP.get(token, [])
            expanded.extend(domain_syns[: self.max_synonyms])

            # 2. WordNet fallback for words not in domain map
            if not domain_syns and self.use_wordnet:
                wordnet_syns = self._wordnet_synonyms(token)
                expanded.extend(wordnet_syns[: self.max_synonyms])

        return self._deduplicate(expanded)

    def expand_to_string(self, query: str) -> str:
        """Return expanded terms joined as a single string (useful for vector search)."""
        return " ".join(self.expand(query))

    def expansion_report(self, query: str) -> Dict:
        """
        Returns a structured report showing original vs added terms.
        Useful for displaying in the Streamlit dashboard.
        """
        tokens = self._tokenize(query)
        original = set(tokens)
        all_expanded = self.expand(query)
        added = [t for t in all_expanded if t not in original]

        return {
            "original_query": query,
            "original_tokens": list(original - self.stopwords),
            "added_terms": added,
            "all_terms": all_expanded,
            "expansion_count": len(added),
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase and split on non-alpha characters."""
        return [w for w in re.split(r"[^a-z]+", text.lower()) if w]

    def _wordnet_synonyms(self, word: str) -> List[str]:
        """Get synonyms from WordNet for a single word."""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name != word and name not in self.stopwords:
                    synonyms.append(name)
        return list(dict.fromkeys(synonyms))  # deduplicate, preserve order

    @staticmethod
    def _deduplicate(terms: List[str]) -> List[str]:
        """Remove duplicates while preserving insertion order."""
        seen = set()
        result = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result