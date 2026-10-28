import pytest
import numpy as np
from pathlib import Path
from src.vector_store import FAISSVectorStore
from src.retriever import SpecRetriever
from src.schema import EvidenceChunk


class DummyEmbeddingEngine:
    """Deterministic dummy embedding engine for fast unit tests without model download."""
    dimension = 4

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def embed_texts(self, texts, **kwargs) -> np.ndarray:
        # Give first text high similarity with query, second text orthogonal
        vectors = []
        for i, t in enumerate(texts):
            if "caliper" in t.lower():
                v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            else:
                v = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
            vectors.append(v / np.linalg.norm(v))
        return np.array(vectors, dtype=np.float32)


def test_vector_store_build_and_search(tmp_path: Path):
    store = FAISSVectorStore()
    chunks = [
        {"chunk_id": "c1", "page_number": 1, "text": "Brake caliper bolt torque is 35 Nm."},
        {"chunk_id": "c2", "page_number": 2, "text": "Windshield wiper washer fluid capacity."}
    ]
    engine = DummyEmbeddingEngine()
    embeddings = engine.embed_texts([c["text"] for c in chunks])

    store.build_index(chunks, embeddings, save_dir=tmp_path)
    assert store.is_index_ready(tmp_path)

    # Search
    query_vec = engine.embed_query("brake caliper")
    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    # The first result should be the caliper chunk with high cosine similarity
    best_chunk, score = results[0]
    assert best_chunk["chunk_id"] == "c1"
    assert score > 0.9

    # Test load
    store2 = FAISSVectorStore()
    loaded = store2.load(tmp_path)
    assert loaded is True
    assert len(store2.chunks) == 2


def test_spec_retriever_filtering(tmp_path: Path):
    store = FAISSVectorStore()
    chunks = [
        {"chunk_id": "c1", "page_number": 10, "text": "Brake caliper torque: 35 Nm"},
        {"chunk_id": "c2", "page_number": 11, "text": "Unrelated tire pressure instructions"}
    ]
    engine = DummyEmbeddingEngine()
    embeddings = engine.embed_texts([c["text"] for c in chunks])
    store.build_index(chunks, embeddings, save_dir=tmp_path)

    retriever = SpecRetriever(
        embedding_engine=engine,
        vector_store=store,
        similarity_threshold=0.5
    )

    results = retriever.retrieve("brake caliper", top_k=2)
    assert len(results) >= 1
    assert results[0].page_number == 10
    assert "Brake caliper" in results[0].text
