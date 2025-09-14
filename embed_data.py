import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_input_data(file_path: str) -> dict:
    """Load input texts from JSON."""
    return json.loads(Path(file_path).read_text(encoding="utf-8"))

def generate_embeddings(texts: list) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create a FAISS index for embeddings."""
    dim = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def main():
    # Load input data
    input_file = "data/inputs.json"
    data = load_input_data(input_file)

    # Collect texts to embed (split into sentences for better granularity)
    texts = []
    metadata = []
    for key, value in [
        ("vacancy", data["vacancy"]),
        ("user_bio", data["user_bio"]),
        ("github_profile", data["github_profile"])
    ]:
        sentences = value.split("\n")
        texts.extend([s for s in sentences if s.strip()])  # Skip empty lines
        metadata.extend([(key, s) for s in sentences if s.strip()])

    for repo in data["github_repos"]:
        if repo["readme"]:
            sentences = repo["readme"].split("\n")
            texts.extend([s for s in sentences if s.strip()])
            metadata.extend([(f"repo_{repo['name']}", s) for s in sentences if s.strip()])

    # Generate embeddings
    embeddings = generate_embeddings(texts)

    # Create FAISS index
    index = create_faiss_index(embeddings)

    # Save embeddings and metadata
    np.save("data/embeddings.npy", embeddings)
    with open("data/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Test search: Find top-3 matches for a vacancy requirement
    query = "Strong Python skills and familiarity with AI/LLM concepts"
    query_embedding = generate_embeddings([query])
    distances, indices = index.search(query_embedding, k=3)

    print("Top-3 matches for query:", query)
    for idx, dist in zip(indices[0], distances[0]):
        source, text = metadata[idx]
        print(f"Source: {source}, Text: {text}, Distance: {dist:.4f}")

    print("\nEmbedding and indexing successful!")

if __name__ == "__main__":
    main()
