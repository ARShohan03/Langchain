"""Opportunity Finder - Scholarship matching system using RAG."""

from .models import UserProfile, userProfile
from .scrapers import fetch_programs, fetch_erasmus_programs
from .ingestion import ingest
from .vectorstore import build_vectorstore
from .reasoning import build_reasoning_chain
from .engine import analyze
from .database import ScholarshipDB, seed_database

__all__ = [
    "UserProfile",
    "userProfile",
    "fetch_programs",
    "fetch_erasmus_programs",
    "ingest",
    "build_vectorstore",
    "build_reasoning_chain",
    "analyze",
    "ScholarshipDB",
    "seed_database",
]

__version__ = "4.0.0"

