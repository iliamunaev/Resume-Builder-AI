"""RAG service for semantic search and generation."""

import re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from models.manager import ModelManager
from config import config
from exceptions import (
    DataError, ModelError, ValidationError, SearchError,
    handle_error, create_error_response
)


class RAGService:
    """Service for RAG operations with semantic search and generation."""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        """Initialize RAG service.

        Args:
            model_manager: Model manager instance. If None, creates new one.
        """
        self.model_manager = model_manager or ModelManager()
        self.embeddings = None
        self.metadata = None
        self.index = None
        self._load_data()

    def _load_data(self):
        """Load required data files."""
        try:
            if not config.validate_data_files():
                missing_files = [
                    str(path) for path, exists in config.validate_data_files().items() if not exists
                ]
                raise DataError(
                    "Required data files not found. Run data processing first.",
                    details={"missing_files": missing_files}
                )

            import faiss
            import json

            # Load embeddings and metadata
            self.embeddings = np.load(config.embeddings_path).astype("float32")
            self.metadata = json.loads(config.metadata_path.read_text())
            self.index = faiss.read_index(str(config.faiss_index_path))

        except (FileNotFoundError, json.JSONDecodeError, np.linalg.LinAlgError) as e:
            raise DataError(f"Failed to load data files: {str(e)}", details={"original_error": str(e)})
        except Exception as e:
            raise handle_error(e)

    def search_similar(self, query: str, k: int = None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Search for similar content in the vector database.

        Args:
            query: Search query string.
            k: Number of results to return. If None, uses config default.

        Returns:
            Tuple of (results_list, scores_array).

        Raises:
            ValidationError: If query is invalid.
            SearchError: If search operation fails.
        """
        try:
            if not query or not isinstance(query, str):
                raise ValidationError("Query must be a non-empty string")

            if k is not None and (k <= 0 or k > 1000):  # Reasonable upper bound
                raise ValidationError(f"Invalid k value: {k}. Must be between 1 and 1000.")

            k = k or config.model.top_k

            # Encode query
            query_emb = self.model_manager.embedding_model.encode_query(query)

            # Search
            scores, indices = self.index.search(query_emb.reshape(1, -1), k)

            # Format results
            results = []
            for i, score in zip(indices[0], scores[0]):
                if 0 <= i < len(self.metadata):
                    source, text = self.metadata[i]
                    results.append({
                        "source": source,
                        "text": text,
                        "score": float(score),
                        "index": int(i)
                    })

            return results, scores

        except ValidationError:
            raise  # Re-raise validation errors as-is
        except Exception as e:
            raise SearchError(f"Search operation failed: {str(e)}", details={"query": query})

    def generate_skills(self, query: str, k: int = None, show_scores: bool = False) -> str:
        """Generate CV skills section based on job requirements.

        Args:
            query: Job requirements query.
            k: Number of context items to use. If None, uses config default.
            show_scores: Whether to print similarity scores.

        Returns:
            Generated skills section as formatted string.

        Raises:
            ValidationError: If query is invalid.
            ModelError: If generation fails.
        """
        try:
            if not query or not isinstance(query, str):
                raise ValidationError("Query must be a non-empty string")

            if k is not None and (k <= 0 or k > 1000):
                raise ValidationError(f"Invalid k value: {k}. Must be between 1 and 1000.")

            k = k or config.model.top_k

            # Get relevant context
            results, scores = self.search_similar(query, k)

            if not results:
                if show_scores:
                    print("[RAG] No relevant context found.")
                return "Skills:\n- (no relevant data found)"

            # Show scores if requested
            if show_scores:
                print(f"\n[RAG] Top-{k} matches for: {query}\n")
                for i, result in enumerate(results, 1):
                    print(f"#{i:<2} score={result['score']:.4f} [{result['source']}] {result['text']}")
                print()

            # Build context for generation
            context = self._build_context(results)

            # Generate skills using language model
            return self._generate_from_context(query, context)

        except ValidationError:
            raise  # Re-raise validation errors
        except SearchError:
            raise  # Re-raise search errors
        except Exception as e:
            raise ModelError(f"Skills generation failed: {str(e)}", details={"query": query})

    def _build_context(self, results: List[Dict[str, Any]], max_items: int = 2) -> str:
        """Build context string from search results.

        Args:
            results: List of search result dictionaries.
            max_items: Maximum number of items to include in context.

        Returns:
            Formatted context string.
        """
        lines = []
        for result in results[:max_items]:
            text = result["text"]
            # Filter out code-like content
            if not any(kw in text.lower() for kw in ["import", "```", ".py", "print"]):
                source = result.get("source", "unknown")
                lines.append(f"{source}: {text}")

        return "\n".join(lines) if lines else "(empty)"

    def _generate_from_context(self, query: str, context: str) -> str:
        """Generate skills from query and context.

        Args:
            query: Original job requirements query.
            context: Retrieved context for generation.

        Returns:
            Generated skills section.
        """
        # Create generation prompt
        prompt_template = (
            "Generate a CV Skills section tailored to the job requirements: \"{query}\"\n"
            "Use ONLY this data, DO NOT add unlisted details (e.g., years of experience, unlisted frameworks like TensorFlow), avoid markdown, and use exact terms from the data (e.g., RAG for Retrieval-Augmented Generation, FastAPI, FastAI, agentic AI):\n"
            "{context}\n"
            "Return 2-3 unique, relevant bullets matching the job requirements:\n"
            "Skills:\n- "
        )

        prompt = self.model_manager.generation_model.truncate_prompt(
            prompt_template.format(context=context, query=query)
        )

        # Generate text
        output = self.model_manager.generation_model.generate(prompt)

        # Extract and format skills
        return self._extract_skills_from_output(output)

    def _extract_skills_from_output(self, output: str) -> str:
        """Extract skills section from generated output.

        Args:
            output: Raw generated text.

        Returns:
            Formatted skills section.
        """
        # Find skills section
        skills_start = output.find("Skills:")
        if skills_start == -1:
            return "Skills:\n- (no skills generated)"

        output = output[skills_start:]

        # Extract bullet points
        bullets = [
            line.strip()[2:] for line in output.split("\n")
            if line.strip().startswith("- ") and len(line.strip()) > 5
        ]

        # Clean and deduplicate
        unique_bullets, seen = [], set()
        for bullet in bullets:
            skill = re.sub(r'[`*]', '', bullet.strip())  # Remove markdown
            if skill and skill.lower() not in seen and len(unique_bullets) < 3:
                unique_bullets.append(f"- {skill}")
                seen.add(skill.lower())

        if not unique_bullets:
            return "Skills:\n- (no relevant skills generated)"

        return "Skills:\n" + "\n".join(unique_bullets)

    def health_check(self) -> Dict[str, Any]:
        """Check if the RAG service is healthy and ready.

        Returns:
            Health status dictionary.
        """
        return {
            "status": "healthy" if self._is_ready() else "unhealthy",
            "embedding_model": self.model_manager.embedding_model.model_name if self.model_manager.embedding_model.model else None,
            "generation_model": self.model_manager.generation_model.model_name if self.model_manager.generation_model.model else None,
            "data_files": config.validate_data_files(),
            "embedding_dimension": self.model_manager.get_embedding_dimension(),
            "data_size": len(self.metadata) if self.metadata else 0
        }

    def _is_ready(self) -> bool:
        """Check if service is ready for operation."""
        try:
            return (
                self.embeddings is not None and
                self.metadata is not None and
                self.index is not None and
                self.model_manager.embedding_model.model is not None and
                self.model_manager.generation_model.model is not None
            )
        except Exception:
            return False
