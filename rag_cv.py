import json
import re
import numpy as np
import faiss
from typing import List, Tuple, Optional
from pathlib import Path

from config import config
from models.manager import ModelManager
from exceptions import DataError, ValidationError, SearchError


class RAGSystem:
    """RAG (Retrieval-Augmented Generation) system for CV skills generation."""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        """Initialize RAG system.

        Args:
            model_manager: Model manager instance. If None, creates new one.
        """
        self.model_manager = model_manager or ModelManager()
        self.config = config

        # Setup environment
        self.config.setup_environment()

        # Load data files
        self._load_data_files()

        # Initialize models
        self._initialize_models()

    def _load_data_files(self):
        """Load required data files for RAG system."""
        try:
            if not self.config.validate_data_files():
                missing_files = [
                    str(path) for path, exists in self.config.validate_data_files().items() if not exists
                ]
                raise DataError(
                    "Required data files missing. Run data processing pipeline first.",
                    details={"missing_files": missing_files}
                )

            # Load embeddings and metadata
            self.embeddings = np.load(self.config.embeddings_path).astype("float32")
            self.metadata = json.loads(self.config.metadata_path.read_text())
            self.index = faiss.read_index(str(self.config.faiss_index_path))

        except (FileNotFoundError, json.JSONDecodeError, np.linalg.LinAlgError) as e:
            raise DataError(f"Failed to load data files: {str(e)}", details={"original_error": str(e)})
        except Exception as e:
            raise DataError(f"Unexpected error loading data: {str(e)}")

    def _initialize_models(self):
        """Initialize embedding and generation models."""
        # Models are loaded lazily through the model manager
        pass

    def _truncate_prompt(self, text: str) -> str:
        """Truncate prompt text to maximum input length."""
        return self.model_manager.generation_model.truncate_prompt(text)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using the embedding model.

        Args:
            query: Query string to encode.

        Returns:
            Query embedding vector.
        """
        return self.model_manager.embedding_model.encode_query(query)

    def search_similar(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings.

        Args:
            query: Query string.
            k: Number of similar items to retrieve.

        Returns:
            Tuple of (scores, indices) arrays.

        Raises:
            ValidationError: If query is invalid.
            SearchError: If search operation fails.
        """
        try:
            if not query or not isinstance(query, str):
                raise ValidationError("Query must be a non-empty string")

            if k <= 0 or k > 1000:  # Reasonable upper bound
                raise ValidationError(f"Invalid k value: {k}. Must be between 1 and 1000.")

            query_emb = self.encode_query(query)
            return self.index.search(query_emb.reshape(1, -1), k)

        except ValidationError:
            raise  # Re-raise validation errors as-is
        except Exception as e:
            raise SearchError(f"Search operation failed: {str(e)}", details={"query": query})

    def build_context(self, indices: np.ndarray, max_items: int = 2) -> str:
        """Build context from retrieved indices.

        Args:
            indices: Array of indices from search results.
            max_items: Maximum number of context items to include.

        Returns:
            Formatted context string.
        """
        lines = []
        for i in indices[0]:
            if 0 <= i < len(self.metadata):
                src, sent = self.metadata[i]
                # Filter out code-like content
                if not any(kw in sent.lower() for kw in ["import", "```", ".py", "print"]):
                    lines.append(f"{src}: {sent}")

        return "\n".join(lines[:max_items]) if lines else "(empty)"

    def extract_skills(self, output: str) -> List[str]:
        """Extract skills from generated output.

        Args:
            output: Raw generated text.

        Returns:
            List of cleaned skill bullets.
        """
        # Find skills section
        skills_start = output.find("Skills:")
        if skills_start == -1:
            return []

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

        return unique_bullets

    def generate_skills(self, query: str, k: int = None, show_scores: bool = True) -> str:
        """Generate CV skills section for a job query.

        Args:
            query: Job requirements or query string.
            k: Number of similar items to retrieve. If None, uses config default.
            show_scores: Whether to print retrieval scores.

        Returns:
            Formatted skills section string.

        Raises:
            ValidationError: If query is invalid.
        """
        try:
            if not query or not isinstance(query, str):
                raise ValidationError("Query must be a non-empty string")

            if k is not None and (k <= 0 or k > 1000):
                raise ValidationError(f"Invalid k value: {k}. Must be between 1 and 1000.")

            k = k or self.config.model.top_k

            # Search for relevant context
            scores, indices = self.search_similar(query, k)

            # Check if we got any results
            if indices is None or indices.size == 0 or len(indices[0]) == 0:
                if show_scores:
                    print("[RAG] No hits found.")
                return "Skills:\n- (no relevant data found)"

            # Show retrieval results if requested
            if show_scores:
                print(f"\n[RAG] Top-{k} retrieval for: {query}\n")
                for rank, (i, s) in enumerate(zip(indices[0], scores[0]), start=1):
                    if 0 <= i < len(self.metadata):
                        src, sent = self.metadata[i]
                        print(f"#{rank:<2} score={s:.4f} [{src}] {sent}")
                print()

            # Build context from relevant results
            context = self.build_context(indices)

            # Create prompt for generation
            prompt_template = (
                "Generate a CV Skills section tailored to the job requirements: \"{query}\"\n"
                "Use ONLY this data, DO NOT add unlisted details (e.g., years of experience, unlisted frameworks like TensorFlow), avoid markdown, and use exact terms from the data (e.g., RAG for Retrieval-Augmented Generation, FastAPI, FastAI, agentic AI):\n"
                "{context}\n"
                "Return 2-3 unique, relevant bullets matching the job requirements:\n"
                "Skills:\n- "
            )

            prompt = self._truncate_prompt(prompt_template.format(context=context, query=query))

            # Generate text using the model
            output = self.model_manager.generation_model.generate(prompt)

            # Extract and format skills
            skills = self.extract_skills(output)

            if not skills:
                return "Skills:\n- (no relevant skills generated)"

            return "Skills:\n" + "\n".join(skills)

        except (ValidationError, SearchError):
            raise  # Re-raise our custom exceptions
        except Exception as e:
            raise ValidationError(f"Skills generation failed: {str(e)}")


# Global RAG system instance for backward compatibility
_rag_system = None


def get_rag_system() -> RAGSystem:
    """Get or create global RAG system instance."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system


def rag_generate(query: str, k: int = None, show_scores: bool = True) -> str:
    """Generate CV skills section for a job query (backward compatibility function).

    Args:
        query: Job requirements or query string.
        k: Number of similar items to retrieve. If None, uses config default.
        show_scores: Whether to print retrieval scores.

    Returns:
        Formatted skills section string.
    """
    return get_rag_system().generate_skills(query, k, show_scores)


def main():
    """Main function for testing RAG system."""
    query = "Strong Python skills and familiarity with AI/LLM concepts"
    cv_skills = rag_generate(query, show_scores=True)
    print("[RESULT] Generated CV Skills Section:\n")
    print(cv_skills)


if __name__ == "__main__":
    main()
