"""Embedding model for semantic search."""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from config import config


class EmbeddingModel:
    """Wrapper for sentence transformer embedding model."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize embedding model.

        Args:
            model_name: Name of the sentence transformer model to use.
                       If None, uses config default.
        """
        self.model_name = model_name or config.model.embedder_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {self.model_name}: {e}")

    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings for input texts.

        Args:
            texts: Single text string or list of texts to embed.
            convert_to_numpy: Convert output to numpy array.
            normalize_embeddings: Normalize embeddings to unit length.
            show_progress_bar: Show encoding progress bar.
            **kwargs: Additional arguments for encode method.

        Returns:
            Array of embeddings, shape (n_texts, embedding_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            **kwargs
        ).astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            query: Query text to encode.

        Returns:
            Single embedding vector.
        """
        embeddings = self.encode([query])
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embeddings."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Create a dummy embedding to get dimension
        dummy = self.encode(["test"])
        return dummy.shape[1]

    def __repr__(self) -> str:
        return f"EmbeddingModel(name='{self.model_name}')"
