import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = Path("data")
INPUTS_JSON = DATA_DIR / "inputs.json"
EMB_NPY = DATA_DIR / "embeddings.npy"
TEXTS_JSON = DATA_DIR / "texts.json"
META_JSON = DATA_DIR / "metadata.json"
FAISS_INDEX = DATA_DIR / "faiss.index"

# NLP setup (full model)
_nlp = spacy.load("en_core_web_sm")

# SentenceTransformer
_model = SentenceTransformer(MODEL_NAME)


def load_input_data(file_path: Path) -> Dict:
    """Load input JSON with basic validation."""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    data.setdefault("vacancy", "")
    data.setdefault("user_bio", "")
    data.setdefault("github_profile", "")
    data.setdefault("github_repos", [])
    return data


def split_into_sentences(text: str) -> List[str]:
    """Use spaCy's full pipeline sentence segmentation."""
    if not text:
        return []
    doc = _nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def collect_texts_and_metadata(data: Dict) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Flatten inputs into sentences with (source, sentence) metadata."""
    pairs: List[Tuple[str, str]] = []
    for key in ("vacancy", "user_bio", "github_profile"):
        for s in split_into_sentences(data.get(key, "")):
            pairs.append((key, s))

    for repo in data.get("github_repos", []):
        name = repo.get("name", "unknown")
        readme = repo.get("readme", "")
        if readme:
            for s in split_into_sentences(readme):
                pairs.append((f"repo_{name}", s))

    # Deduplicate exact duplicates while preserving order
    seen = set()
    dedup_pairs = []
    for src, sent in pairs:
        if (src, sent) not in seen:
            seen.add((src, sent))
            dedup_pairs.append((src, sent))

    texts = [sent for _, sent in dedup_pairs]
    metadata = dedup_pairs
    return texts, metadata


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate normalized embeddings for cosine similarity."""
    if not texts:
        return np.empty((0, _model.get_sentence_embedding_dimension()), dtype="float32")
    embs = _model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embs.astype("float32")


def create_faiss_index_ip(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS index for cosine similarity using inner product."""
    if embeddings.size == 0:
        dim = _model.get_sentence_embedding_dimension()
        return faiss.IndexFlatIP(dim)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_artifacts(embeddings: np.ndarray, texts: List[str], metadata: List[Tuple[str, str]], index: faiss.IndexFlatIP):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_NPY, embeddings)
    TEXTS_JSON.write_text(json.dumps(texts, indent=2, ensure_ascii=False), encoding="utf-8")
    META_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(index, str(FAISS_INDEX))


def load_index_and_context():
    index = faiss.read_index(str(FAISS_INDEX)) if FAISS_INDEX.exists() else None
    texts = json.loads(TEXTS_JSON.read_text(encoding="utf-8")) if TEXTS_JSON.exists() else []
    metadata = json.loads(META_JSON.read_text(encoding="utf-8")) if META_JSON.exists() else []
    return index, texts, metadata


def search(query: str, k: int = 3):
    """Search helper that re-encodes the query and runs FAISS IP (cosine)."""
    index, texts, metadata = load_index_and_context()
    if index is None or not texts:
        print("Index or texts not found. Build the index first.")
        return

    q_emb = _model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k=k)

    print(f"Top-{k} matches for query: {query}\n")
    for i, score in zip(idxs[0], scores[0]):
        src, sent = metadata[i]
        print(f"Score: {score:.4f} | Source: {src} | Text: {sent}\n")


def build_index(input_file: Path):
    data = load_input_data(input_file)
    texts, metadata = collect_texts_and_metadata(data)
    if not texts:
        print("No texts found to embed.")
        return
    embs = generate_embeddings(texts)
    index = create_faiss_index_ip(embs)
    save_artifacts(embs, texts, metadata, index)
    print("\nIndexing complete.\n")

    # Small test
    demo_query = "Strong Python skills and familiarity with AI/LLM concepts"
    search(demo_query, k=3)

def main():
    build_index(INPUTS_JSON)
    return

if __name__ == "__main__":
    main()
