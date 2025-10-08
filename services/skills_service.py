"""Skills service for skills processing and formatting."""

import re
from typing import List, Dict, Any, Optional
from services.rag_service import RAGService


class SkillsService:
    """Service for skills-related operations."""

    def __init__(self, rag_service: Optional[RAGService] = None):
        """Initialize skills service.

        Args:
            rag_service: RAG service instance. If None, creates new one.
        """
        self.rag_service = rag_service or RAGService()

    def generate_skills_section(self, job_requirements: str, max_skills: int = 3) -> str:
        """Generate a formatted skills section for a job.

        Args:
            job_requirements: Job requirements text.
            max_skills: Maximum number of skills to generate.

        Returns:
            Formatted skills section string.
        """
        if not job_requirements or not isinstance(job_requirements, str):
            raise ValueError("Job requirements must be a non-empty string")

        # Generate skills using RAG
        raw_skills = self.rag_service.generate_skills(job_requirements)

        # Parse and format skills
        skills = self._parse_skills(raw_skills, max_skills)

        return self._format_skills_section(skills)

    def extract_skills_from_text(self, text: str, limit: int = 3) -> List[str]:
        """Extract skills from text using bullet point parsing.

        Args:
            text: Text containing skills (possibly with bullets).
            limit: Maximum number of skills to extract.

        Returns:
            List of extracted skill strings.
        """
        # Regex pattern for bullet points
        bullet_pattern = re.compile(r'^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$', re.MULTILINE)

        skills, seen = [], set()
        for match in bullet_pattern.finditer(text):
            skill = match.group(1)
            # Clean the skill
            skill = re.sub(r'[`*_]', '', skill)  # Remove markdown
            skill = re.sub(r'[;.,:]+$', '', skill).strip()  # Remove trailing punctuation

            key = skill.lower()
            if skill and key not in seen:
                skills.append(skill)
                seen.add(key)

            if len(skills) >= limit:
                break

        return skills

    def _parse_skills(self, raw_text: str, max_skills: int = 3) -> List[str]:
        """Parse skills from RAG output.

        Args:
            raw_text: Raw skills text from RAG.
            max_skills: Maximum number of skills to return.

        Returns:
            List of parsed skill strings.
        """
        # First try to extract from Skills: section
        skills_start = raw_text.find("Skills:")
        if skills_start != -1:
            skills_text = raw_text[skills_start:]
            skills = self.extract_skills_from_text(skills_text, max_skills)
            if skills:
                return skills

        # Fallback: look for bullet points after "Skills:" in any line
        lines = [line.strip() for line in raw_text.splitlines()]
        try:
            start_idx = next(i for i, line in enumerate(lines) if line.lower().startswith("skills"))
            remaining_lines = lines[start_idx + 1:]
        except StopIteration:
            remaining_lines = lines

        # Extract skills from remaining lines
        return self.extract_skills_from_text("\n".join(remaining_lines), max_skills)

    def _format_skills_section(self, skills: List[str]) -> str:
        """Format skills list into a proper section.

        Args:
            skills: List of skill strings.

        Returns:
            Formatted skills section.
        """
        if not skills:
            return "Skills:\n- (No relevant skills found)"

        formatted_skills = [f"- {skill}" for skill in skills]
        return "Skills:\n" + "\n".join(formatted_skills)

    def validate_skills(self, skills: List[str], job_requirements: str) -> Dict[str, Any]:
        """Validate if skills match job requirements.

        Args:
            skills: List of skills to validate.
            job_requirements: Job requirements text.

        Returns:
            Validation results dictionary.
        """
        if not skills or not job_requirements:
            return {"valid": False, "reason": "Missing skills or requirements"}

        # Simple validation - check for keyword overlap
        job_words = set(re.findall(r'\b\w+\b', job_requirements.lower()))
        skill_words = set()

        for skill in skills:
            skill_words.update(re.findall(r'\b\w+\b', skill.lower()))

        overlap = job_words.intersection(skill_words)
        overlap_ratio = len(overlap) / len(job_words) if job_words else 0

        return {
            "valid": overlap_ratio > 0.1,  # At least 10% overlap
            "overlap_ratio": overlap_ratio,
            "overlapping_terms": list(overlap),
            "job_terms": len(job_words),
            "skill_terms": len(skill_words)
        }
