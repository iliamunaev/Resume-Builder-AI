import os, re, json, sys
from pathlib import Path
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------------------
# Config
# ----------------------------
DATA_DIR       = Path("./data")  # must contain embeddings.npy, texts.json, metadata.json, faiss.index
EMBEDDER_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME     = "Qwen/Qwen2.5-0.5B-Instruct"

TOP_K          = 3
MAX_NEW_TOKENS = 120
INPUT_MAX_LEN  = 512
SEED           = 42

# ----------------------------
# Reproducibility
# ----------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Device select
# ----------------------------
if torch.cuda.is_available():
    device_map, dtype = "auto", torch.float16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device_map, dtype = {"": "mps"}, torch.float32
else:
    device_map, dtype = None, torch.float32

# ----------------------------
# Require files
# ----------------------------
required = ["embeddings.npy", "texts.json", "metadata.json", "faiss.index"]
for f in required:
    p = DATA_DIR / f
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Place your {f} inside {DATA_DIR}")

embeddings = np.load(DATA_DIR / "embeddings.npy").astype("float32")
texts = json.loads((DATA_DIR / "texts.json").read_text(encoding="utf-8"))
metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
index = faiss.read_index(str(DATA_DIR / "faiss.index"))

# ----------------------------
# Embedding model
# ----------------------------
embed_model = SentenceTransformer(EMBEDDER_NAME)

# Sanity check
embed_dim = embed_model.get_sentence_embedding_dimension()
if index.d != embed_dim:
    raise ValueError(f"FAISS index dim {index.d} != embedder dim {embed_dim}")

# ----------------------------
# Load TinyLlama
# ----------------------------
print(f"[INFO] Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    low_cpu_mem_usage=True,
    device_map=device_map,
)

text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.pad_token_id,
)

# ----------------------------
# Helpers
# ----------------------------
def _truncate_prompt(text: str) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > INPUT_MAX_LEN:
        ids = ids[-INPUT_MAX_LEN:]
    return tokenizer.decode(ids, skip_special_tokens=True)

def encode_query(q: str) -> np.ndarray:
    return embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# ----------------------------
# RAG generate
# ----------------------------
def rag_generate(query: str, k: int = TOP_K, show_scores: bool = True) -> str:
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    q_emb = encode_query(query)
    scores, idxs = index.search(q_emb, k=k)

    if idxs.size == 0 or all(i < 0 for i in idxs[0]):
        return "Skills:\n- (no relevant data found)"

    # Retrieval preview
    if show_scores:
        print(f"\n[RAG] Top-{k} retrieval for: {query}\n")
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), start=1):
            if 0 <= i < len(metadata):
                src, sent = metadata[i]
                print(f"#{rank:<2} score={s:0.4f} [{src}] {sent}")
        print()

    # Context build
    lines = []
    for i in idxs[0]:
        if 0 <= i < len(metadata):
            src, sent = metadata[i]
            if not any(kw in sent.lower() for kw in ("import", "```", ".py", "print")):
                lines.append(f"{src}: {sent}")
    context = "\n".join(lines[:2]) if lines else "(empty)"

    # Prompt
    prompt_template = (
        "Generate a CV Skills section tailored to the job requirements: \"{query}\"\n"
        "Use ONLY this data, do not add unlisted details, avoid markdown.\n"
        "{context}\n"
        "Return 2-3 unique bullets:\n"
        "Skills:\n- "
    )
    prompt = _truncate_prompt(prompt_template.format(context=context, query=query))

    with torch.inference_mode():
        result = text_gen(prompt)
        output = result[0]["generated_text"].strip()

    # Extract bullets
    skills_start = output.find("Skills:")
    if skills_start == -1:
        return "Skills:\n- (no skills generated)"
    output = output[skills_start:]
    bullets = [
        line.strip()[2:] for line in output.split("\n")
        if line.strip().startswith("- ") and len(line.strip()) > 3
    ]
    unique, seen = [], set()
    for b in bullets:
        clean = re.sub(r'[`*]', '', b.strip())
        if clean and clean.lower() not in seen:
            unique.append(f"- {clean}")
            seen.add(clean.lower())
        if len(unique) >= 3:
            break
    return "Skills:\n" + "\n".join(unique) if unique else "Skills:\n- (no relevant skills generated)"

# ----------------------------
# Demo
# ----------------------------
def main():
    query = "Strong Python skills and familiarity with AI/LLM concepts"
    cv_skills = rag_generate(query, k=TOP_K, show_scores=True)
    print("\n[RESULT] Generated CV Skills Section:\n")
    print(cv_skills)

if __name__ == "__main__":
    main()
