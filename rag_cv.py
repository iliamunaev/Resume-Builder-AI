#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ============================
# Config
# ============================
# A tiny, instruction-tuned model that runs on CPU reasonably well.
# You can switch to "distilgpt2" if you prefer, but instruct models follow prompts better.
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DATA_DIR = Path("data")
EMB_NPY = DATA_DIR / "embeddings.npy"
TEXTS_JSON = DATA_DIR / "texts.json"
META_JSON = DATA_DIR / "metadata.json"
FAISS_INDEX = DATA_DIR / "faiss.index"

EMBEDDER_NAME = "all-MiniLM-L6-v2"  # must match what you used to build the FAISS index

TOP_K = 5
MAX_NEW_TOKENS = 160
INPUT_MAX_LEN = 2048  # truncate long prompts safely

# ============================
# Load artifacts
# ============================
def _require_file(p: Path, hint: str = ""):
    if not p.exists():
        msg = f"Missing required file: {p}"
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)

_require_file(EMB_NPY, "Create it with the same embedder used below.")
_require_file(TEXTS_JSON, "A list of text chunks.")
_require_file(META_JSON, "A list of [source, sentence] or similar tuples.")
_require_file(FAISS_INDEX, "A FAISS index built for cosine/IP using normalized embeddings.")

embeddings = np.load(EMB_NPY)
texts = json.loads(TEXTS_JSON.read_text(encoding="utf-8"))
metadata = json.loads(META_JSON.read_text(encoding="utf-8"))
index = faiss.read_index(str(FAISS_INDEX))

# Basic sanity checks
if embeddings.dtype != np.float32:
    embeddings = embeddings.astype("float32")

# ============================
# Embedder for queries
# ============================
embed_model = SentenceTransformer(EMBEDDER_NAME)

def encode_query(q: str) -> np.ndarray:
    q_emb = embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return q_emb

# ============================
# LLM setup (pure Transformers)
# ============================
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'cuda' if device == 0 else 'cpu'}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Some models lack a pad token; use EOS to prevent warnings
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Use bfloat16 if available on your hardware, else float32 on CPU
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
)

text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,            # -1 = CPU, 0 = CUDA:0
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,          # deterministic; avoids temperature usage
    top_p=1.0,                # ignored when do_sample=False, kept explicit
    repetition_penalty=1.1,   # light penalty to reduce loops
    pad_token_id=tokenizer.pad_token_id,
    return_full_text=False,
    truncation=True
)

# ============================
# Prompt template
# ============================
def build_prompt(context: str, query: str) -> str:
    # Keep the instruction crisp; instruct models respond better.
    return (
        "You are drafting a concise CV 'Skills' section for the user.\n"
        "RULES:\n"
        "- Use ONLY facts found in the provided User Data.\n"
        "- Do NOT invent or infer missing facts.\n"
        "- Be brief, specific, and grouped by themes.\n"
        f"- Target requirement to address: \"{query}\"\n\n"
        f"User Data:\n{context}\n\n"
        "Return output exactly in this format:\n"
        "Skills:\n"
        "- <skill or tool>\n"
        "- <skill or tool>\n"
        "- <skill or tool>\n"
    )

def safe_truncate(s: str, max_len: int = INPUT_MAX_LEN) -> str:
    # Token-based truncation via tokenizer would be better; keep simple for CPU:
    # We’ll rely on pipeline's truncation too, but this guards massive contexts.
    return s if len(s) <= max_len else s[:max_len]

# ============================
# RAG generate
# ============================
def rag_generate(query: str, k: int = TOP_K, show_scores: bool = True) -> str:
    q_emb = encode_query(query)
    scores, idxs = index.search(q_emb, k=k)

    if idxs.size == 0:
        if show_scores:
            print("[RAG] No hits found.")
        return "Skills:\n- (no relevant data found)"

    if show_scores:
        print(f"\n[RAG] Top-{k} retrieval for: {query}\n")
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), start=1):
            if 0 <= i < len(metadata):
                try:
                    src, sent = metadata[i]
                except Exception:
                    src, sent = "unknown_source", str(metadata[i])
                print(f"#{rank:<2} score={s:0.4f}  [{src}]  {sent}")
        print()

    # Build concise context from top-k hits
    lines = []
    for i in idxs[0]:
        if 0 <= i < len(metadata):
            try:
                src, sent = metadata[i]
            except Exception:
                src, sent = "unknown_source", str(metadata[i])
            lines.append(f"{src}: {sent}")
    context = "\n".join(lines) if lines else "(empty)"
    prompt = build_prompt(context=context, query=query)
    prompt = safe_truncate(prompt, INPUT_MAX_LEN)

    # Generate
    result = text_gen(prompt)
    completion = result[0]["generated_text"].strip()
    return completion

# ============================
# Main
# ============================
def main():
    query = "Strong Python skills and familiarity with AI/LLM concepts"
    print("Generated CV Skills Section:\n")
    cv_skills = rag_generate(query, k=TOP_K, show_scores=True)
    print(cv_skills)
    print("\nRAG generation successful!")

if __name__ == "__main__":
    main()
