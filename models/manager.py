"""Model manager for coordinating embedding and generation models."""

from typing import Optional
from contextlib import contextmanager
from models.embeddings import EmbeddingModel
from models.generation import TextGenerationModel
from config import config


class ModelManager:
    """Manages the lifecycle of ML models used in the application."""

    def __init__(self):
        """Initialize model manager with configuration."""
        self.config = config
        self._embedding_model: Optional[EmbeddingModel] = None
        self._generation_model: Optional[TextGenerationModel] = None

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get or create embedding model."""
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel()
        return self._embedding_model

    @property
    def generation_model(self) -> TextGenerationModel:
        """Get or create generation model."""
        if self._generation_model is None:
            self._generation_model = TextGenerationModel()
        return self._generation_model

    def reload_models(self):
        """Reload both models (useful for memory cleanup)."""
        self._embedding_model = None
        self._generation_model = None
        # Access properties to trigger loading
        _ = self.embedding_model
        _ = self.generation_model

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the embedding model."""
        return self.embedding_model.get_embedding_dimension()

    @contextmanager
    def inference_context(self):
        """Context manager for inference operations."""
        try:
            yield self
        finally:
            # Cleanup can be added here if needed
            pass

    def __repr__(self) -> str:
        return "ModelManager()"
