import re

def clean_text(text: str) -> str:
    """Clean raw text, return only words, letters."""
    # Remove punctuation and non-letters, keep only letters + spaces
    cleaned = re.sub(r"[^A-Za-z\s]", "", text)

    # Normalize multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
