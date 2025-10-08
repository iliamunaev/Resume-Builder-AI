"""Data processing service for text preprocessing and data generation."""

import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import spacy

from config import config
from utils.utils import clean_text
from exceptions import DataError, ValidationError


class DataProcessingService:
    """Service for data processing, preprocessing, and generation."""

    def __init__(self):
        """Initialize data processing service."""
        self.config = config
        self._nlp = None

    @property
    def nlp(self):
        """Get spaCy NLP model, loading if necessary."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError) as e:
                raise DataError(f"Failed to load spaCy model: {e}")
        return self._nlp

    def fetch_github_data(self, username: Optional[str] = None, token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch GitHub user profile and repository data.

        Args:
            username: GitHub username. If None, uses config default.
            token: GitHub API token. If None, uses config default.

        Returns:
            Dictionary containing GitHub profile and repository data.

        Raises:
            DataError: If GitHub API request fails.
        """
        username = username or self.config.github.username
        token = token or self.config.github.token

        if not username or not token:
            raise ValidationError("GitHub username and token are required")

        headers = {"Authorization": f"token {token}"}
        github_data: Dict[str, Any] = {"username": username, "repos": []}

        try:
            # Fetch user profile
            profile_url = f"https://api.github.com/users/{username}"
            profile_resp = requests.get(profile_url, headers=headers)
            profile_resp.raise_for_status()
            github_data["profile"] = profile_resp.json().get("bio", "")

            # Fetch public repos
            repos_url = f"https://api.github.com/users/{username}/repos"
            repos_resp = requests.get(repos_url, headers=headers)
            repos_resp.raise_for_status()
            repos = repos_resp.json()

            # Fetch README for each repo (limit to 3 for performance)
            for repo in repos[:3]:
                repo_name = repo["name"]
                readme_url = f"https://api.github.com/repos/{username}/{repo_name}/readme"
                readme_resp = requests.get(readme_url, headers=headers)

                if readme_resp.status_code == 200:
                    readme_content = clean_text(
                        base64.b64decode(readme_resp.json()["content"]).decode("utf-8")
                    )
                    github_data["repos"].append({"name": repo_name, "readme": readme_content})
                else:
                    github_data["repos"].append({"name": repo_name, "readme": ""})

            return github_data

        except requests.RequestException as e:
            raise DataError(f"GitHub API request failed: {e}")
        except Exception as e:
            raise DataError(f"Failed to fetch GitHub data: {e}")

    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocess text using spaCy for tokenization and entity extraction.

        Args:
            text: Input text to process.

        Returns:
            Dictionary with tokens, entities, and other NLP features.

        Raises:
            DataError: If text processing fails.
        """
        try:
            if not text:
                return {"tokens": [], "entities": [], "raw_text": ""}

            cleaned = clean_text(text)
            if not cleaned.strip():
                return {"tokens": [], "entities": [], "raw_text": cleaned}

            doc = self.nlp(cleaned)

            # Extract tokens (lemmas, excluding punctuation and stopwords)
            tokens = [
                token.lemma_.lower()
                for token in doc
                if not token.is_punct and not token.is_stop and token.lemma_.strip()
            ]

            # Extract entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            return {
                "tokens": tokens,
                "entities": entities,
                "raw_text": cleaned
            }

        except Exception as e:
            raise DataError(f"Text preprocessing failed: {e}")

    def preprocess_file(self, file_path: str) -> Dict[str, Any]:
        """Preprocess a text file.

        Args:
            file_path: Path to the text file.

        Returns:
            Dictionary with file metadata and processed content.

        Raises:
            FileNotFoundError: If file doesn't exist.
            DataError: If file processing fails.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            raw_text = path.read_text(encoding="utf-8")
            processed = self.preprocess_text(raw_text)

            return {
                "file": file_path,
                "raw_text": raw_text,
                **processed
            }

        except (UnicodeDecodeError, OSError) as e:
            raise DataError(f"Failed to read file {file_path}: {e}")

    def generate_dataset(self, vacancy_path: str, user_bio_path: str) -> Dict[str, Any]:
        """Generate complete dataset from vacancy, bio, and GitHub data.

        Args:
            vacancy_path: Path to vacancy description file.
            user_bio_path: Path to user biography file.

        Returns:
            Complete dataset dictionary.

        Raises:
            DataError: If dataset generation fails.
        """
        try:
            # Load and process vacancy and user bio
            vacancy_data = self.preprocess_file(vacancy_path)
            user_data = self.preprocess_file(user_bio_path)

            # Fetch and process GitHub data
            github_data = self.fetch_github_data()
            github_processed = self._process_github_data(github_data)

            # Structure the complete dataset
            dataset = {
                "vacancy": {
                    "raw_text": vacancy_data["raw_text"],
                    "tokens": vacancy_data["tokens"],
                    "entities": vacancy_data["entities"]
                },
                "user_bio": {
                    "raw_text": user_data["raw_text"],
                    "tokens": user_data["tokens"],
                    "entities": user_data["entities"]
                },
                "github_profile": github_processed["profile"],
                "github_repos": github_processed["repos"]
            }

            return dataset

        except Exception as e:
            raise DataError(f"Dataset generation failed: {e}")

    def _process_github_data(self, github_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process GitHub profile and repository data.

        Args:
            github_data: Raw GitHub data.

        Returns:
            Processed GitHub data with NLP features.
        """
        processed = {
            "username": github_data.get("username", ""),
            "profile": self.preprocess_text(github_data.get("profile", "")),
            "repos": []
        }

        for repo in github_data.get("repos", []):
            repo_processed = self.preprocess_text(repo.get("readme", ""))
            repo_processed["name"] = repo.get("name", "")
            processed["repos"].append(repo_processed)

        return processed

    def save_dataset(self, dataset: Dict[str, Any], output_path: Optional[str] = None):
        """Save dataset to JSON file.

        Args:
            dataset: Dataset dictionary to save.
            output_path: Output file path. If None, uses config default.

        Raises:
            DataError: If saving fails.
        """
        output_path = output_path or str(self.config.data_path)

        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        except (OSError, TypeError) as e:
            raise DataError(f"Failed to save dataset: {e}")

    def load_dataset(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Load dataset from JSON file.

        Args:
            data_path: Path to dataset file. If None, uses config default.

        Returns:
            Loaded dataset dictionary.

        Raises:
            FileNotFoundError: If file doesn't exist.
            DataError: If loading fails.
        """
        data_path = data_path or str(self.config.data_path)

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            raise DataError(f"Failed to load dataset: {e}")

    def get_dataset_stats(self, dataset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get statistics about the dataset.

        Args:
            dataset: Dataset to analyze. If None, loads from file.

        Returns:
            Dictionary with dataset statistics.
        """
        if dataset is None:
            try:
                dataset = self.load_dataset()
            except (FileNotFoundError, DataError):
                return {"error": "No dataset available"}

        stats = {
            "vacancy_tokens": len(dataset.get("vacancy", {}).get("tokens", [])),
            "user_tokens": len(dataset.get("user_bio", {}).get("tokens", [])),
            "github_repos": len(dataset.get("github_repos", [])),
            "profile_entities": len(dataset.get("github_profile", {}).get("entities", [])),
        }

        # Count total tokens across repos
        repo_tokens = sum(
            len(repo.get("tokens", []))
            for repo in dataset.get("github_repos", [])
        )
        stats["repo_tokens"] = repo_tokens

        return stats
