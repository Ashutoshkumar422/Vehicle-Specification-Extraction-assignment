from typing import List, Optional, Callable, Union
import numpy as np


class EmbeddingEngine:
    """Manages sentence-transformer embeddings with lazy model loading and batch processing."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """Lazy-loads the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load sentence-transformers model '{self.model_name}': {e}"
                )
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimension."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy loading
        return self._dimension or 384

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Generates normalized dense embeddings for a list of texts.
        Returns np.ndarray of shape (len(texts), dimension) in float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generates a normalized embedding vector for a single search query.
        Returns 1D np.ndarray of shape (dimension,) in float32.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        embedding = self.model.encode(
            [query.strip()],
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embedding[0].astype(np.float32)
