"""Data service for data processing and management."""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import config
from utils.utils import clean_text


class DataService:
    """Service for data processing and file operations."""

    def __init__(self):
        """Initialize data service."""
        self.config = config

    def load_data(self) -> Dict[str, Any]:
        """Load main data file.

        Returns:
            Dictionary containing user and job data.
        """
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")

        with open(self.config.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_data(self, data: Dict[str, Any]):
        """Save data to main data file.

        Args:
            data: Data dictionary to save.
        """
        self.config.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_texts(self) -> List[str]:
        """Load text corpus for embedding.

        Returns:
            List of text strings.
        """
        if not self.config.texts_path.exists():
            raise FileNotFoundError(f"Texts file not found: {self.config.texts_path}")

        with open(self.config.texts_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_metadata(self) -> List[List[str]]:
        """Load metadata for texts.

        Returns:
            List of [source, text] pairs.
        """
        if not self.config.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.config.metadata_path}")

        with open(self.config.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_github_data(self, github_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process GitHub profile and repository data.

        Args:
            github_data: Raw GitHub data from API.

        Returns:
            Processed data with tokens and entities.
        """
        return self._extract_tokens_and_entities(github_data)

    def _extract_tokens_and_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tokens and entities from text data using spaCy.

        Args:
            data: Raw data dictionary.

        Returns:
            Processed data with NLP features.
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError) as e:
            raise RuntimeError(f"spaCy not available: {e}")

        def process_text(text: str) -> Dict[str, Any]:
            """Process single text with spaCy."""
            if not text:
                return {"tokens": [], "entities": []}

            cleaned = clean_text(text)
            doc = nlp(cleaned if cleaned.strip() else "no content")

            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_punct and not token.is_stop
            ]
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            return {"tokens": tokens, "entities": entities}

        processed = {"username": data.get("username", ""), "repos": []}

        # Process profile
        profile_text = data.get("profile", "")
        processed["profile"] = process_text(profile_text)

        # Process repositories
        for repo in data.get("repos", []):
            repo_processed = process_text(repo.get("readme", ""))
            repo_processed["name"] = repo.get("name", "")
            processed["repos"].append(repo_processed)

        return processed

    def validate_data_files(self) -> Dict[str, bool]:
        """Validate existence of all required data files.

        Returns:
            Dictionary mapping file names to existence status.
        """
        files = {
            "data_file": self.config.data_path,
            "embeddings": self.config.embeddings_path,
            "texts": self.config.texts_path,
            "metadata": self.config.metadata_path,
            "faiss_index": self.config.faiss_index_path
        }

        return {name: path.exists() for name, path in files.items()}

    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the data files.

        Returns:
            Dictionary with data statistics.
        """
        stats = {"files_exist": self.validate_data_files()}

        try:
            if stats["files_exist"]["texts"]:
                texts = self.load_texts()
                stats["text_count"] = len(texts)

            if stats["files_exist"]["metadata"]:
                metadata = self.load_metadata()
                stats["metadata_count"] = len(metadata)

            if stats["files_exist"]["embeddings"]:
                embeddings = np.load(self.config.embeddings_path)
                stats["embedding_shape"] = embeddings.shape

        except Exception as e:
            stats["error"] = str(e)

        return stats
