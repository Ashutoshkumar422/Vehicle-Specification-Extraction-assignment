import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Persistent vector store using FAISS IndexFlatIP with JSON metadata storage."""

    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = Path(index_dir) if index_dir else None
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.dimension: int = 0

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        save_dir: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Builds a FAISS inner product index from normalized embeddings and stores chunks.
        """
        import faiss

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        if len(embeddings) == 0:
            raise ValueError("Cannot build FAISS index from empty embeddings")

        self.dimension = embeddings.shape[1]
        self.chunks = list(chunks)

        # IndexFlatIP calculates inner product (cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        # Ensure float32 and contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vectors)

        target_dir = Path(save_dir) if save_dir else self.index_dir
        if target_dir:
            self.save(target_dir, metadata=metadata)

    def save(self, target_dir: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persists the FAISS index and chunk metadata to disk."""
        import faiss

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise ValueError("No FAISS index to save")

        index_file = target_dir / "index.faiss"
        chunks_file = target_dir / "chunks.json"
        meta_file = target_dir / "meta.json"

        faiss.write_index(self.index, str(index_file))

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        meta_info = {
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
            "custom_metadata": metadata or {}
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved FAISS index with {len(self.chunks)} chunks to {target_dir}")

    def load(self, index_dir: Path) -> bool:
        """
        Loads the FAISS index and chunks from disk.
        Returns True if successful, False if files are missing or corrupted.
        """
        import faiss

        index_dir = Path(index_dir)
        index_file = index_dir / "index.faiss"
        chunks_file = index_dir / "chunks.json"

        if not index_file.exists() or not chunks_file.exists():
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            self.dimension = self.index.d
            self.index_dir = index_dir

            if self.index.ntotal != len(self.chunks):
                logger.warning(
                    f"Index count ({self.index.ntotal}) != chunks count ({len(self.chunks)})"
                )
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {index_dir}: {e}")
            return False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs inner product search.
        Returns list of (chunk_dict, similarity_score) sorted by descending similarity.
        """
        if self.index is None:
            raise RuntimeError("Index has not been initialized or loaded")

        if len(self.chunks) == 0 or self.index.ntotal == 0:
            return []

        # Prepare query vector: 2D array of shape (1, dimension)
        if query_embedding.ndim == 1:
            q = np.expand_dims(query_embedding, axis=0)
        else:
            q = query_embedding

        q = np.ascontiguousarray(q, dtype=np.float32)
        k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    @staticmethod
    def is_index_ready(index_dir: Path) -> bool:
        """Checks if a valid index directory exists with index.faiss and chunks.json."""
        p = Path(index_dir)
        return (p / "index.faiss").exists() and (p / "chunks.json").exists()

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the current index."""
        return {
            "is_loaded": self.index is not None,
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
            "faiss_total": self.index.ntotal if self.index else 0
        }
