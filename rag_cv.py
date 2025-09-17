import os, json, re
from pathlib import Path
import numpy as np
import faiss
import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Cache setup ---
cache_dir = Path("/kaggle/working/cache")
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))
os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Load HF token ---
env_path = Path("/kaggle/input/env-vars/.env")
raw = env_path.read_text(encoding="utf-8")
m = re.search(r'hf_[A-Za-z0-9_-]+', raw)
if not m:
    raise RuntimeError("No hf_ token found in /kaggle/input/env-vars/.env")
HF_TOKEN = m.group(0)
login(token=HF_TOKEN)

# --- Config ---
MODEL_NAME     = "meta-llama/Llama-3.2-3B-Instruct"
EMBEDDER_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K          = 3
MAX_NEW_TOKENS = 120
INPUT_MAX_LEN  = 512
DATA_DIR       = Path("/kaggle/input/data-resume")

# --- Require files ---
for f in ["embeddings.npy", "texts.json", "metadata.json", "faiss.index"]:
    p = DATA_DIR / f
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Upload data to Kaggle dataset.")

embeddings = np.load(DATA_DIR / "embeddings.npy").astype("float32")
metadata = json.loads((DATA_DIR / "metadata.json").read_text())
index = faiss.read_index(str(DATA_DIR / "faiss.index"))

# Sentence embeddings
embed_model = SentenceTransformer(EMBEDDER_NAME)

# --- Tokenizer & model ---
cuda = torch.cuda.is_available()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16 if cuda else torch.float32,
    device_map="auto" if cuda else None,
)

# --- Text generation pipeline ---
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

# --- Helpers ---
def _truncate_prompt(text: str) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > INPUT_MAX_LEN:
        ids = ids[-INPUT_MAX_LEN:]
    return tokenizer.decode(ids, skip_special_tokens=True)

def encode_query(q: str) -> np.ndarray:
    return embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def rag_generate(query: str, k: int = TOP_K, show_scores: bool = True) -> str:
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    q_emb = encode_query(query)
    scores, idxs = index.search(q_emb, k=k)

    if idxs.size == 0:
        if show_scores:
            print("[RAG] No hits found.")
        return "Skills:\n- (no relevant data found)"

    # Retrieval preview
    if show_scores:
        print(f"\n[RAG] Top-{k} retrieval for: {query}\n")
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), start=1):
            if 0 <= i < len(metadata):
                src, sent = metadata[i]
                print(f"#{rank:<2} score={s:0.4f} [{src}] {sent}")
        print()

    # Build concise context, avoid code-y lines
    lines = []
    for i in idxs[0]:
        if 0 <= i < len(metadata):
            src, sent = metadata[i]
            if not any(kw in sent.lower() for kw in ["import", "```", ".py", "print"]):
                lines.append(f"{src}: {sent}")
    context = "\n".join(lines[:2]) if lines else "(empty)"

    # Dynamic prompt
    prompt_template = (
        "Generate a CV Skills section tailored to the job requirements: \"{query}\"\n"
        "Use ONLY this data, DO NOT add unlisted details (e.g., years of experience, unlisted frameworks like TensorFlow), avoid markdown, and use exact terms from the data (e.g., RAG for Retrieval-Augmented Generation, FastAPI, FastAI, agentic AI):\n"
        "{context}\n"
        "Return 2-3 unique, relevant bullets matching the job requirements:\n"
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
        if line.strip().startswith("- ") and len(line.strip()) > 5
    ]
    unique_bullets, seen = [], set()
    for bullet in bullets:
        skill = re.sub(r'[`*]', '', bullet.strip())  # Remove markdown
        if skill and skill.lower() not in seen and len(unique_bullets) < 3:
            unique_bullets.append(f"- {skill}")
            seen.add(skill.lower())
    if not unique_bullets:
        return "Skills:\n- (no relevant skills generated)"
    return "Skills:\n" + "\n".join(unique_bullets[:3])

def main():
    query = "Strong Python skills and familiarity with AI/LLM concepts"
    cv_skills = rag_generate(query, k=TOP_K, show_scores=True)
    print("[RESULT] Generated CV Skills Section:\n")
    print(cv_skills)

if __name__ == "__main__":
    main()
