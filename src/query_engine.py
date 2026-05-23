import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ─── Semantic descriptions for every graph node type ───
NODE_SEMANTIC_MAP = {
    # ─── Preferences - High ───
    "Culture_High": (
        "culture cultural rich history museums art galleries heritage "
        "landmarks ancient ruins historical sites civilization traditions "
        "architecture monuments cultural experiences"
    ),
    "Adventure_High": (
        "adventure adventurous extreme sports hiking trekking "
        "rock climbing zip-lining kayaking outdoor thrills "
        "exploration expedition adrenaline bungee jumping rafting"
    ),
    "Nature_High": (
        "nature natural environment wildlife forests mountains "
        "waterfalls scenic scenery landscape greenery outdoors "
        "national parks nature reserves flora fauna pristine wilderness "
        "nature beauty natural wonders"
    ),
    "Beaches_High": (
        "beaches beach coastal ocean seaside surfing sand shore "
        "tropical island paradise beachfront waterfront seashore "
        "beach vacation swimming snorkeling beach resort"
    ),
    "Nightlife_High": (
        "nightlife clubs clubbing bars parties dancing entertainment "
        "pub crawl nightclub DJ music scene vibrant nightlife "
        "party scene late night bars discos going out"
    ),
    "Cuisine_High": (
        "cuisine food culinary gastronomy restaurants fine dining "
        "street food world-class food diverse restaurants "
        "food culture cooking local dishes food scene gourmet"
    ),
    "Wellness_High": (
        "wellness spa spas yoga retreats meditation health resorts "
        "relaxation thermal baths detox wellness retreat "
        "rejuvenation healing holistic massage hot springs"
    ),
    "Urban_High": (
        "urban city metropolitan modern city skyscrapers shopping "
        "cosmopolitan bustling downtown cityscape metropolis "
        "urban life city center city experience urban exploration"
    ),
    "Seclusion_High": (
        "seclusion secluded isolated remote peaceful quiet serene "
        "off-the-beaten-path private solitude privacy retreat "
        "getaway escape tranquil undiscovered hidden gem "
        "away from crowds peaceful retreat"
    ),

    # ─── Preferences - Medium ───
    "Culture_Medium": (
        "culture moderate some cultural attractions historical sites "
        "a few museums decent cultural offerings some heritage"
    ),
    "Adventure_Medium": (
        "adventure moderate outdoor activities some adventure options "
        "light hiking easy trails some outdoor recreation"
    ),
    "Nature_Medium": (
        "nature moderate some natural attractions parks gardens "
        "some greenery decent nature access some scenic spots"
    ),
    "Beaches_Medium": (
        "beaches moderate some beach access coastal areas waterfront "
        "some shoreline decent beach nearby water access"
    ),
    "Nightlife_Medium": (
        "nightlife moderate some bars restaurants evening entertainment "
        "decent night scene some pubs casual nightlife options"
    ),
    "Cuisine_Medium": (
        "cuisine moderate good food options local cuisine "
        "variety of restaurants decent dining some good food"
    ),
    "Wellness_Medium": (
        "wellness moderate some wellness facilities spa options "
        "a few spas decent relaxation options some health services"
    ),
    "Urban_Medium": (
        "urban moderate mid-sized city some urban amenities "
        "town center decent infrastructure some city features"
    ),
    "Seclusion_Medium": (
        "seclusion moderate somewhat quiet moderate crowds "
        "semi-private some peaceful areas not too busy"
    ),

    # ─── Preferences - Low ───
    "Culture_Low": (
        "culture low limited cultural attractions few museums "
        "minimal heritage sparse historical sites"
    ),
    "Adventure_Low": (
        "adventure low limited adventure activities calm "
        "flat terrain minimal outdoor recreation no thrills"
    ),
    "Nature_Low": (
        "nature low limited natural scenery urban landscape "
        "concrete minimal greenery sparse nature"
    ),
    "Beaches_Low": (
        "beaches low landlocked no beach access far from ocean "
        "no coastline no waterfront inland location"
    ),
    "Nightlife_Low": (
        "nightlife low quiet evenings limited nightlife "
        "early closing few bars minimal entertainment"
    ),
    "Cuisine_Low": (
        "cuisine low limited dining options few restaurants "
        "basic food minimal food variety sparse dining"
    ),
    "Wellness_Low": (
        "wellness low limited wellness options no spas "
        "minimal relaxation facilities no retreat options"
    ),
    "Urban_Low": (
        "urban low rural small town village countryside "
        "minimal urban features sparse infrastructure"
    ),
    "Seclusion_Low": (
        "seclusion low crowded touristy busy popular overcrowded "
        "no privacy heavy tourism packed with tourists"
    ),

    # ─── Climate ───
    "Cold": (
        "cold cool chilly freezing winter snow ice arctic sub-zero "
        "cold weather cold climate frigid frost snowy icy "
        "cold temperature cold region"
    ),
    "Moderate": (
        "moderate mild temperate pleasant comfortable warm cool "
        "spring-like moderate climate mild weather "
        "moderate temperature nice weather year-round comfort"
    ),
    "Hot": (
        "hot tropical warm scorching summer heat humid sunny "
        "arid desert hot weather hot climate sweltering "
        "hot temperature tropical heat blazing"
    ),

    # ─── Months ───
    "Month_Jan": (
        "january jan winter new year cold season "
        "start of year beginning of year first month"
    ),
    "Month_Feb": (
        "february feb winter valentines cold month "
        "second month late winter"
    ),
    "Month_Mar": (
        "march mar spring beginning equinox "
        "early spring third month"
    ),
    "Month_Apr": (
        "april apr spring warm flowers bloom "
        "mid spring fourth month"
    ),
    "Month_May": (
        "may late spring pleasant warm "
        "end of spring fifth month"
    ),
    "Month_Jun": (
        "june jun summer solstice hot sunny "
        "early summer sixth month start of summer"
    ),
    "Month_Jul": (
        "july jul summer peak hot vacation "
        "mid summer seventh month peak summer"
    ),
    "Month_Aug": (
        "august aug summer hot holiday "
        "late summer eighth month end of summer"
    ),
    "Month_Sep": (
        "september sep autumn fall cooling "
        "early autumn ninth month start of fall"
    ),
    "Month_Oct": (
        "october oct autumn fall halloween cool "
        "mid autumn tenth month"
    ),
    "Month_Nov": (
        "november nov late autumn cold "
        "end of autumn eleventh month pre-winter"
    ),
    "Month_Dec": (
        "december dec winter christmas holiday cold snow "
        "last month end of year twelfth month"
    ),
}

# ─── Conflict pairs ───
CONFLICT_PAIRS = [
    ("Cold", "Hot"),
    ("Cold", "Moderate"),
    ("Moderate", "Hot"),
    ("Culture_High", "Culture_Low"),
    ("Adventure_High", "Adventure_Low"),
    ("Nature_High", "Nature_Low"),
    ("Beaches_High", "Beaches_Low"),
    ("Nightlife_High", "Nightlife_Low"),
    ("Cuisine_High", "Cuisine_Low"),
    ("Wellness_High", "Wellness_Low"),
    ("Urban_High", "Urban_Low"),
    ("Seclusion_High", "Seclusion_Low"),
    ("Urban_High", "Seclusion_High"),
]

# ─── Matching thresholds ───
SEMANTIC_MATCH_THRESHOLD = 0.42
MAX_MATCHED_NODES = 6

# ─── Geographic keyword mapping ───
# Maps query keywords to region values in the dataset
# Keys are lowercase. Values are lists of region substrings to match against.
GEOGRAPHIC_KEYWORDS = {
    # Europe
    "europe": ["Europe"],
    "european": ["Europe"],
    "western europe": ["Western Europe"],
    "eastern europe": ["Eastern Europe"],
    "scandinavia": ["Northern Europe"],
    "scandinavian": ["Northern Europe"],
    "mediterranean": ["Southern Europe"],
    "nordic": ["Northern Europe"],
    # Asia
    "asia": ["Asia"],
    "asian": ["Asia"],
    "southeast asia": ["South-Eastern Asia", "Southeast Asia"],
    "east asia": ["Eastern Asia", "East Asia"],
    "south asia": ["Southern Asia", "South Asia"],
    "middle east": ["Western Asia", "Middle East"],
    # Americas
    "north america": ["North America", "Northern America"],
    "south america": ["South America"],
    "latin america": ["South America", "Central America", "Caribbean"],
    "central america": ["Central America"],
    "caribbean": ["Caribbean"],
    # Africa
    "africa": ["Africa"],
    "african": ["Africa"],
    "north africa": ["Northern Africa"],
    "sub-saharan": ["Sub-Saharan Africa"],
    "west africa": ["Western Africa"],
    "east africa": ["Eastern Africa"],
    "southern africa": ["Southern Africa"],
    # Oceania
    "oceania": ["Oceania", "Australia and New Zealand"],
    "australia": ["Australia and New Zealand"],
    "pacific": ["Oceania", "Melanesia", "Polynesia"],
}


class SemanticQueryEngine:
    """
    Converts natural language queries into graph entry points
    using semantic similarity.

    Features:
    - Higher matching threshold (0.42) to eliminate noise
    - Conflict resolution between contradictory matched nodes
    - Keyword-dense semantic descriptions for accurate matching
    - Geographic keyword detection for region filtering
    """

    def __init__(self, G, city_names, city_descriptions,
                 model_name="all-MiniLM-L6-v2"):
        self.G = G
        self.city_names = city_names
        self.city_descriptions = city_descriptions
        self.model = SentenceTransformer(model_name)

        # ── Build FAISS index for city descriptions ──
        city_texts = [city_descriptions[c] for c in city_names]
        self.city_embeddings = self.model.encode(
            city_texts, convert_to_numpy=True
        )
        faiss.normalize_L2(self.city_embeddings)

        d = self.city_embeddings.shape[1]
        self.city_index = faiss.IndexFlatIP(d)
        self.city_index.add(self.city_embeddings)

        # ── Build FAISS index for graph nodes ──
        self.node_names = list(NODE_SEMANTIC_MAP.keys())
        node_texts = [NODE_SEMANTIC_MAP[n] for n in self.node_names]
        self.node_embeddings = self.model.encode(
            node_texts, convert_to_numpy=True
        )
        faiss.normalize_L2(self.node_embeddings)

        self.node_index = faiss.IndexFlatIP(d)
        self.node_index.add(self.node_embeddings)

        # ── Pre-build conflict lookup ──
        self.conflict_lookup = {}
        for a, b in CONFLICT_PAIRS:
            self.conflict_lookup.setdefault(a, set()).add(b)
            self.conflict_lookup.setdefault(b, set()).add(a)

        print(
            f"  Query engine ready: {len(city_names)} cities, "
            f"{len(self.node_names)} semantic nodes indexed"
        )

    def parse_query_to_graph_nodes(self, query, top_k=10, threshold=None):
        """
        Given a natural language query, find the most relevant
        graph nodes using semantic similarity.
        """
        if threshold is None:
            threshold = SEMANTIC_MATCH_THRESHOLD

        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        fetch_k = min(top_k * 2, len(self.node_names))
        scores, indices = self.node_index.search(q_emb, fetch_k)

        raw_matches = []
        for idx_pos, node_idx in enumerate(indices[0]):
            score = float(scores[0][idx_pos])
            if score >= threshold:
                node_name = self.node_names[node_idx]
                raw_matches.append({
                    "node": node_name,
                    "score": score,
                    "description": NODE_SEMANTIC_MAP[node_name][:60],
                })

        resolved = self._resolve_conflicts(raw_matches)
        resolved = resolved[:MAX_MATCHED_NODES]

        return resolved

    def detect_geographic_constraint(self, query):
        """
        Detect geographic/regional constraints from query text.

        Uses keyword matching (not semantic) because geographic names
        are proper nouns that should be matched exactly, not fuzzily.

        Returns:
            dict with:
            - "detected": bool
            - "keyword": str, the keyword found in query
            - "regions": list of region substrings to filter against
            - "detail": str, human-readable summary
        """
        query_lower = query.lower()

        # Check longest keywords first to match "southeast asia" before "asia"
        sorted_keywords = sorted(
            GEOGRAPHIC_KEYWORDS.keys(),
            key=len,
            reverse=True,
        )

        for keyword in sorted_keywords:
            if keyword in query_lower:
                regions = GEOGRAPHIC_KEYWORDS[keyword]
                return {
                    "detected": True,
                    "keyword": keyword,
                    "regions": regions,
                    "detail": (
                        f"Geographic filter: '{keyword}' → "
                        f"regions containing {regions}"
                    ),
                }

        return {
            "detected": False,
            "keyword": None,
            "regions": [],
            "detail": "No geographic constraint detected",
        }

    def _resolve_conflicts(self, matches):
        """Remove contradictory nodes, keeping the higher-scoring one."""
        if not matches:
            return matches

        sorted_matches = sorted(
            matches, key=lambda x: x["score"], reverse=True
        )

        resolved = []
        excluded = set()

        for match in sorted_matches:
            node = match["node"]

            if node in excluded:
                continue

            resolved.append(match)

            conflicting_nodes = self.conflict_lookup.get(node, set())
            for conflict in conflicting_nodes:
                if conflict not in excluded:
                    excluded.add(conflict)

        return resolved

    def vector_search(self, query, top_k=10):
        """Standard FAISS vector search over city descriptions."""
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = self.city_index.search(q_emb, top_k)

        results = []
        for idx_pos, city_idx in enumerate(indices[0]):
            results.append({
                "city": self.city_names[city_idx],
                "score": float(scores[0][idx_pos]),
            })
        return results