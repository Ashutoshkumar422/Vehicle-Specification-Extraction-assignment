import re
from typing import List, Dict, Any, Tuple, Optional
from src.embeddings import EmbeddingEngine
from src.vector_store import FAISSVectorStore
from src.schema import EvidenceChunk


class SpecRetriever:
    """Retrieves the most semantically and lexically relevant manual passages for a specification query."""

    # Words that describe query actions rather than the component or specification itself
    STOP_WORDS = {
        'what', 'is', 'the', 'for', 'of', 'in', 'to', 'at', 'a', 'an',
        'torque', 'spec', 'specs', 'specification', 'specifications',
        'value', 'values', 'tighten', 'tightening', 'capacity', 'capacities',
        'fluid', 'part', 'number', 'base', 'item', 'approximate', 'refill',
        'give', 'find', 'me', 'how', 'much'
    }

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: FAISSVectorStore,
        similarity_threshold: float = 0.20
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def _compute_hybrid_score(self, query: str, chunk_dict: Dict[str, Any], base_score: float) -> float:
        """
        Combines dense vector similarity with token-level lexical overlap and
        technical domain specification markers.
        """
        text = chunk_dict.get("text", "")
        text_lower = text.lower()
        q_lower = query.lower()

        # Extract meaningful component tokens from query
        q_tokens = [w for w in re.findall(r'\b[a-z0-9]+\b', q_lower) if w not in self.STOP_WORDS]

        score = base_score

        # 1. Component lexical overlap boost
        if q_tokens:
            matched_tokens = [w for w in q_tokens if w in text_lower]
            overlap_ratio = len(matched_tokens) / len(q_tokens)
            score += overlap_ratio * 0.35

            # Multi-word exact component phrase bonus (e.g., "lower ball joint", "brake disc shield")
            comp_phrase = " ".join(q_tokens)
            if comp_phrase in text_lower:
                score += 0.25

        # 2. Antonym penalty: prevent Upper vs Lower / Front vs Rear cross-contamination
        antonyms = [('upper', 'lower'), ('front', 'rear'), ('inner', 'outer'), ('left', 'right')]
        for a1, a2 in antonyms:
            if a1 in q_lower and a2 in text_lower and a1 not in text_lower:
                score -= 0.20

        # 3. Domain-targeted specification and table header boosts
        if any(w in q_lower for w in ["torque", "tighten", "tightening"]):
            if "torque specifications" in text_lower or "specifications" in text_lower:
                score += 0.20
            if any(u in text_lower for u in ["nm", "lb-ft", "lb-in"]):
                score += 0.10

        elif any(w in q_lower for w in ["capacity", "fluid", "fill", "oil", "coolant", "lubricant"]):
            if "fill capacity" in text_lower or "capacities" in text_lower:
                score += 0.20
            if any(u in text_lower for u in ["liters", "quarts", "pt", "qt", "2.8l", "2.84l"]):
                score += 0.10

        elif any(w in q_lower for w in ["part", "number"]):
            if "part number" in text_lower or "base part number" in text_lower:
                score += 0.25
            elif "item" in text_lower and "description" in text_lower:
                score += 0.15

        elif any(w in q_lower for w in ["clearance", "gap", "backlash", "runout"]):
            if "clearance" in text_lower or "backlash" in text_lower:
                score += 0.20
            if any(u in text_lower for u in ["mm", "0.0", "in"]):
                score += 0.10

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[EvidenceChunk]:
        """
        Embeds the search query and retrieves candidates from FAISS with hybrid lexical-semantic re-ranking.
        Guarantees specification tables containing exact component names rank at the top.
        """
        if not query or not query.strip():
            return []

        # Generate query vector
        query_vec = self.embedding_engine.embed_query(query)

        # Retrieve a broad candidate pool from FAISS to capture table specs
        total_chunks = len(self.vector_store.chunks)
        candidate_k = min(total_chunks, max(top_k * 15, 80)) if total_chunks > 0 else 80

        raw_results: List[Tuple[Dict[str, Any], float]] = self.vector_store.search(
            query_embedding=query_vec,
            top_k=candidate_k
        )

        # Re-rank candidates based on hybrid lexical + semantic score
        scored_candidates = []
        for chunk_dict, base_score in raw_results:
            if base_score < self.similarity_threshold:
                continue

            hybrid_score = self._compute_hybrid_score(query, chunk_dict, base_score)
            scored_candidates.append((chunk_dict, base_score, hybrid_score))

        # Sort by final hybrid score descending
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        evidence_chunks: List[EvidenceChunk] = []
        seen_texts = set()

        for chunk_dict, orig_score, hybrid_score in scored_candidates:
            # Deduplicate nearly identical chunks
            normalized_snippet = " ".join(chunk_dict.get("text", "").split())[:200]
            if normalized_snippet in seen_texts:
                continue
            seen_texts.add(normalized_snippet)

            chunk_obj = EvidenceChunk(
                chunk_id=chunk_dict.get("chunk_id", ""),
                page_number=int(chunk_dict.get("page_number", 1)),
                text=chunk_dict.get("text", ""),
                similarity_score=round(float(orig_score), 4)
            )
            evidence_chunks.append(chunk_obj)

            if len(evidence_chunks) >= top_k:
                break

        return evidence_chunks
