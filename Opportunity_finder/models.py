"""User profile models for scholarship matching."""

from pydantic import BaseModel
from typing import List, Optional


class UserProfile(BaseModel):
    """User profile for scholarship matching."""
    name: Optional[str] = None
    cgpa: float
    cgpa_scale: float
    degree: str
    field: str
    research_papers: int
    experience_years: float
    internships: int
    extracurriculars: List[str]
    english_test: Optional[str] = None
    english_score: Optional[float] = None
    target_degree: str


# Alias for backward compatibility
userProfile = UserProfile
