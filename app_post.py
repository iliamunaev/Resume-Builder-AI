# app_post.py
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_cv_test import rag_generate  # your RAG generator

app = FastAPI(title="Resume Builder RAG API")

class QueryRequest(BaseModel):
    query: str

# Accepts "-", "*", "1.", "2)", etc.
_BULLET_RE = re.compile(r'^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$')

def parse_skills(text: str, limit: int = 3):
    skills, seen = [], set()
    for raw in text.splitlines():
        m = _BULLET_RE.match(raw)
        if not m:
            continue
        item = m.group(1)
        # strip simple markdown / trailing punct
        item = re.sub(r'[`*_]', '', item)
        item = re.sub(r'[;.,:]+$', '', item).strip()
        key = item.lower()
        if item and key not in seen:
            skills.append(item)
            seen.add(key)
        if len(skills) >= limit:
            break
    return skills

@app.post("/generate-skills")
async def generate_skills(request: QueryRequest):
    try:
        cv_skills = rag_generate(request.query, k=3, show_scores=False)
        skills = parse_skills(cv_skills, limit=3)

        # Fallback: if model returned plain lines after "Skills:" with "- "
        if not skills:
            lines = [ln.strip() for ln in cv_skills.splitlines()]
            try:
                start = next(i for i, ln in enumerate(lines) if ln.lower().startswith("skills"))
                tail = lines[start+1:]
            except StopIteration:
                tail = lines
            skills = parse_skills("\n".join(tail), limit=3)

        return {"query": request.query, "skills": skills}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating skills: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Keep workers=1 so the model isn't loaded multiple times
    uvicorn.run("app_post:app", host="127.0.0.1", port=8000, workers=1)
