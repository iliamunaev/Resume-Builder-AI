import re
from pathlib import Path

def clean_text(text: str) -> str:
    # Remove punctuation and non-letters, keep only letters + spaces
    cleaned = re.sub(r"[^A-Za-z\s]", "", text)

    # Normalize multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned

def read_file(file_path: str) -> str:
    """Read and preprocess a text file, extracting tokens and entities."""
    # Read file
    text = Path(file_path).read_text(encoding="utf-8").strip()

    cleaned = clean_text(text)

    return cleaned

def main():
    vacancy_path = "vacancy.txt"

    vacancy_data = read_file(vacancy_path)

    # Print results
    print(vacancy_data)


if __name__ == "__main__":
    main()
