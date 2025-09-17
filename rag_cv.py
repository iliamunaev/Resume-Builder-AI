import os
from pathlib import Path
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

# --- Safety: cache setup, cap threads ---
cache_dir = Path.home() / ".cache" / "ai_models"
os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))
os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# Model Configuration
MODEL_CONFIGS = {
    "tiny": {
        "name": "microsoft/DialoGPT-small",  # ~117MB
        "max_tokens": 50,
        "size": "117MB",
        "description": "Smallest option, perfect for WSL with limited RAM"
    },
    "small": {
        "name": "gpt2-medium",  # ~1.5GB
        "max_tokens": 60,
        "size": "1.5GB",
        "description": "Balanced size and performance"
    },
    "medium": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ~2.2GB
        "max_tokens": 80,
        "size": "2.2GB",
        "description": "Original model (requires more RAM)"
    },
    "alternative_tiny": {
        "name": "distilgpt2",  # ~353MB
        "max_tokens": 50,
        "size": "353MB",
        "description": "Alternative tiny model, good text generation"
    }
}

# Select model size
SELECTED_MODEL = "small"  # Switched to distilgpt2
MODEL_NAME     = MODEL_CONFIGS[SELECTED_MODEL]["name"]
EMBEDDER_NAME  = "all-MiniLM-L6-v2"
TOP_K          = 3  # Reduced to avoid token limit
MAX_NEW_TOKENS = MODEL_CONFIGS[SELECTED_MODEL]["max_tokens"]
INPUT_MAX_LEN  = 512
DATA_DIR       = Path("data")

# --- Require files ---
for f in ["embeddings.npy", "texts.json", "metadata.json", "faiss.index"]:
    p = DATA_DIR / f
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")

try:
    embeddings = np.load(DATA_DIR / "embeddings.npy").astype("float32")
    metadata   = json.loads((DATA_DIR / "metadata.json").read_text())
    index      = faiss.read_index(str(DATA_DIR / "faiss.index"))
except FileNotFoundError as e:
    raise FileNotFoundError(f"Failed to load required data files: {e}")
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in metadata.json: {e}")
except Exception as e:
    raise RuntimeError(f"Error loading data files: {e}")

try:
    embed_model = SentenceTransformer(EMBEDDER_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model '{EMBEDDER_NAME}': {e}")

cpu_only = not torch.cuda.is_available()
device   = 0 if not cpu_only else -1

# Memory optimization settings
USE_QUANTIZATION = SELECTED_MODEL in ["tiny", "alternative_tiny"] and not cpu_only
QUANTIZATION_BITS = 8

print("=" * 50)
print(f"[INFO] Using model: {MODEL_CONFIGS[SELECTED_MODEL]['description']}")
print(f"[INFO] Model: {MODEL_NAME} ({MODEL_CONFIGS[SELECTED_MODEL]['size']})")
print(f"[INFO] Device: {'CPU' if cpu_only else 'GPU'}")
if USE_QUANTIZATION:
    print(f"[INFO] Using {QUANTIZATION_BITS}-bit quantization")
else:
    print("[INFO] Running without quantization")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"low_cpu_mem_usage": True}
    if USE_QUANTIZATION:
        try:
            from transformers import BitsAndBytesConfig
            print(f"[INFO] Using {QUANTIZATION_BITS}-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=(QUANTIZATION_BITS == 8),
                load_in_4bit=(QUANTIZATION_BITS == 4),
            )
            model_kwargs["quantization_config"] = quantization_config
        except ImportError:
            print("[WARNING] bitsandbytes not available, loading without quantization")
            model_kwargs["dtype"] = torch.float16 if not cpu_only else torch.float32
    else:
        model_kwargs["dtype"] = torch.float16 if not cpu_only else torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
except Exception as e:
    raise RuntimeError(f"Failed to load model '{MODEL_NAME}': {e}")

try:
    pipeline_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id
    }
    if not USE_QUANTIZATION:
        pipeline_kwargs["device"] = device

    text_gen = pipeline("text-generation", **pipeline_kwargs)
except Exception as e:
    raise RuntimeError(f"Failed to create text generation pipeline: {e}")

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0

def check_system_memory():
    """Check if system has enough memory for selected model"""
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
        memory_requirements = {
            "tiny": 0.5,
            "alternative_tiny": 0.8,
            "small": 2.5,
            "medium": 4.0
        }
        required = memory_requirements.get(SELECTED_MODEL, 2.0)
        if available_memory < required:
            print(f"[WARNING] Low memory detected: {available_memory:.1f}GB available, "
                  f"{required:.1f}GB recommended for {SELECTED_MODEL} model")
            print("[SUGGESTION] Consider using 'tiny' model or adding more RAM")
        else:
            print(f"[INFO] Memory check passed: {available_memory:.1f}GB available")
    except ImportError:
        print("[INFO] psutil not available for memory monitoring")

def encode_query(q): return embed_model.encode([q], convert_to_numpy=True,
                                               normalize_embeddings=True).astype("float32")

def rag_generate(query: str, k: int = TOP_K, show_scores: bool = True) -> str:
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    try:
        q_emb = encode_query(query)
        scores, idxs = index.search(q_emb, k=k)

        if idxs.size == 0:
            if show_scores:
                print("[RAG] No hits found.")
            return "Skills:\n- (no relevant data found)"

        # Pretty-print retrieval results with scores
        if show_scores:
            print(f"\n[RAG] Top-{k} retrieval for: {query}\n")
            for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), start=1):
                if 0 <= i < len(metadata):
                    src, sent = metadata[i]
                    print(f"#{rank:<2} score={s:0.4f}  [{src}]  {sent}")
            print()

        # Build context from top-k hits
        lines = [f"{metadata[i][0]}: {metadata[i][1]}" for i in idxs[0] if 0 <= i < len(metadata)]
        context = "\n".join(lines) if lines else "(empty)"

        # Structured prompt for bullet list
        prompt_template = (
            "You are drafting a concise CV Skills section.\n"
            "Use ONLY the provided user data. Do not invent facts.\n"
            "Target requirement: \"{query}\"\n\n"
            "User Data:\n{context}\n\n"
            "Return a short bullet list under the heading 'Skills':\n"
            "Skills:\n- "
        )
        prompt = prompt_template.format(context=context, query=query)

        with torch.inference_mode():
            result = text_gen(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            output = result[0]["generated_text"].strip()
            print(f"[DEBUG] Raw LLM Output: {repr(output)}")  # Debug print
            skills_start = output.find("Skills:")
            if skills_start != -1:
                output = output[skills_start:]
            return output

    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")

def main():
    print(f"[INFO] HF cache: {os.environ['HF_HOME']}")
    print(f"[INFO] Torch cache: {os.environ['TORCH_HOME']}")
    check_system_memory()
    initial_memory = get_memory_usage()
    print(f"[INFO] Initial memory usage: {initial_memory:.1f}MB")

    print("\nTest run:\n")
    try:
        query = "Strong Python skills and familiarity with AI/LLM concepts"
        cv_skills = rag_generate(query, k=TOP_K, show_scores=True)
        print("Generated CV Skills Section:\n")
        print(cv_skills)
        final_memory = get_memory_usage()
        print(f"[INFO] Memory usage delta: {final_memory - initial_memory:+.1f}MB")
        print("\nRAG generation successful!")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        print("[SUGGESTION] Try switching to 'alternative_tiny' model")

if __name__ == "__main__":
    main()
