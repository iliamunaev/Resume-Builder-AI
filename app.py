from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path

app = FastAPI(title="Resume Builder Semantic Matcher")

# Config
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = Path("data")
EMB_NPY = DATA_DIR / "embeddings.npy"
TEXTS_JSON = DATA_DIR / "texts.json"
META_JSON = DATA_DIR / "metadata.json"
FAISS_INDEX = DATA_DIR / "faiss.index"

# Load Sentence Transformer model
model = SentenceTransformer(MODEL_NAME)

# Load embeddings and metadata
if not all(p.exists() for p in [EMB_NPY, TEXTS_JSON, META_JSON, FAISS_INDEX]):
    raise FileNotFoundError("Required files missing. Run embed_data.py first.")

embeddings = np.load(EMB_NPY)
with open(TEXTS_JSON, "r", encoding="utf-8") as f:
    texts = json.load(f)
with open(META_JSON, "r", encoding="utf-8") as f:
    metadata = json.load(f)
index = faiss.read_index(str(FAISS_INDEX))

class QueryInput(BaseModel):
    vacancy_text: str

@app.post("/match")
async def match_skills(query_input: QueryInput):
    try:
        # Split vacancy into sentences using simple split (or use spaCy if preferred)
        sentences = [s.strip() for s in query_input.vacancy_text.split("\n") if s.strip()]
        if not sentences:
            raise HTTPException(status_code=400, detail="Vacancy text is empty")

        # Generate embeddings for vacancy sentences
        query_embeddings = model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        ).astype("float32")

        # Search for top-5 matches per query sentence
        results = []
        for query, query_emb in zip(sentences, query_embeddings):
            scores, indices = index.search(query_emb.reshape(1, -1), k=5)
            matches = [
                {"source": metadata[idx][0], "text": metadata[idx][1], "score": float(score)}
                for idx, score in zip(indices[0], scores[0])
            ]
            results.append({"query": query, "matches": matches})

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
