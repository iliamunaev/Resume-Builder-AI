"""RAG-powered skills generation API."""

import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from config import config
from services.skills_service import SkillsService

# Initialize services
skills_service = SkillsService()

app = FastAPI(
    title="Resume Builder RAG API",
    description="API for generating tailored CV skills using RAG",
    version="2.0.0"
)

class QueryRequest(BaseModel):
    """Request model for skills generation."""
    query: str
    max_skills: int = 3

class SkillsResponse(BaseModel):
    """Response model for skills generation."""
    query: str
    skills: List[str]
    success: bool

@app.post("/generate-skills", response_model=SkillsResponse)
async def generate_skills(request: QueryRequest):
    """Generate skills section tailored to job requirements.

    Args:
        request: Query request with job requirements and max skills.

    Returns:
        Generated skills matching the job requirements.
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Generate skills using the service
        skills_section = skills_service.generate_skills_section(
            request.query,
            max_skills=request.max_skills
        )

        # Parse skills from the response
        skills = _parse_skills_from_response(skills_section, request.max_skills)

        return SkillsResponse(
            query=request.query,
            skills=skills,
            success=True
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating skills: {str(e)}")

@app.get("/health")
async def health_check():
    """Check API health and service status."""
    try:
        health = skills_service.rag_service.health_check()
        return {
            "status": "healthy",
            "services": health,
            "config": {
                "model": config.model.generator_name,
                "embedder": config.model.embedder_name,
                "data_dir": str(config.data.data_dir)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def _parse_skills_from_response(skills_text: str, max_skills: int = 3) -> List[str]:
    """Parse skills from skills section text.

    Args:
        skills_text: Raw skills section text.
        max_skills: Maximum number of skills to return.

    Returns:
        List of individual skill strings.
    """
    # Regex pattern for bullet points
    bullet_pattern = re.compile(r'^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$', re.MULTILINE)

    skills, seen = [], set()
    for match in bullet_pattern.finditer(skills_text):
        skill = match.group(1)
        # Clean the skill
        skill = re.sub(r'[`*_]', '', skill)  # Remove markdown
        skill = re.sub(r'[;.,:]+$', '', skill).strip()  # Remove trailing punctuation

        key = skill.lower()
        if skill and key not in seen:
            skills.append(skill)
            seen.add(key)

        if len(skills) >= max_skills:
            break

    return skills

if __name__ == "__main__":
    import uvicorn

    # Use configuration for server settings
    uvicorn.run(
        "app_post:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers
    )
