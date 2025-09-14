import spacy
from pathlib import Path

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(file_path: str) -> dict:
    """Read and preprocess a text file, extracting tokens and entities."""
    # Read file
    text = Path(file_path).read_text(encoding="utf-8").strip()

    # Process with spaCy
    doc = nlp(text)

    # Extract tokens (words, excluding punctuation/stopwords)
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]

    # Extract entities (e.g., skills, roles, organizations)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return {
        "file": file_path,
        "tokens": tokens,
        "entities": entities,
        "raw_text": text
    }

def main():
    # Paths to sample files
    vacancy_path = "data/vacancy.txt"
    user_bio_path = "data/user_bio.txt"

    # Preprocess both files
    vacancy_data = preprocess_text(vacancy_path)
    user_data = preprocess_text(user_bio_path)

    # Print results
    print("Vacancy Preprocessing:")
    print(f"Tokens: {vacancy_data['tokens']}")
    print(f"Entities: {vacancy_data['entities']}")

    print("User Bio Preprocessing:")
    print(f"Tokens: {user_data['tokens']}")
    print(f"Entities: {user_data['entities']}")

    print("\nPreprocessing successful!")

if __name__ == "__main__":
    main()
