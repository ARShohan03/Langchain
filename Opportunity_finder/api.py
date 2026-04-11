"""FastAPI backend for the Global STEM Opportunity Finder (Zero-Dependency AI)."""

import os
import json
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import models
from models import UserProfile
from database import ScholarshipDB, seed_database

# Paths
_here = Path(__file__).parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")

# Global state
vectorstore = None
reasoning_chain = None
scholarship_db = None
MOCK_MODE = False

app = FastAPI(title="Global STEM Opportunity Finder API", version="4.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_resources():
    global vectorstore, reasoning_chain, scholarship_db, MOCK_MODE
    try:
        # 1. Initialize SQLite database
        scholarship_db = ScholarshipDB()
        
        # Seed if empty or stale
        if scholarship_db.count() == 0 or scholarship_db.needs_refresh(max_age_days=30):
            print("[API] Seeding/refreshing scholarship database...")
            seed_database(scholarship_db)
        
        print(f"[API] SQLite DB loaded: {scholarship_db.count()} scholarships")
        
        # 2. Build vectorstore from DB
        doc_dicts = scholarship_db.to_documents()
        
        if doc_dicts:
            # Simple document abstraction
            class Doc:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata
            
            docs = [Doc(d['page_content'], d['metadata']) for d in doc_dicts]
            
            from vectorstore import build_vectorstore
            vectorstore = build_vectorstore(docs)
            print(f"[API] Vectorstore built with {len(docs)} documents from DB.")

        # Also export to JSON for backward compatibility
        json_path = _here / "data" / "processed" / "scholarships.json"
        os.makedirs(json_path.parent, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc_dicts, f, indent=2)
        print(f"[API] Exported {len(doc_dicts)} scholarships to JSON backup.")
        
        # 3. Initialize LLM (Pure requests-based client, no torch)
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not hf_token:
            print("[API] WARNING: No HuggingFace token found. Entering MOCK_MODE.")
            MOCK_MODE = True
            return

        from llm_client import MistralClient
        from reasoning import build_reasoning_chain
        
        llm = MistralClient(token=hf_token)
        reasoning_chain = build_reasoning_chain(llm)
        print("[API] Global STEM reasoning engine ready.")
        MOCK_MODE = False
    except Exception as e:
        print(f"[API] Initialization failed: {e}. Falling back to MOCK_MODE.")
        import traceback
        traceback.print_exc()
        MOCK_MODE = True

@app.on_event("startup")
async def startup_event():
    init_resources()

@app.get("/status")
def get_status():
    return {
        "engine": "Hybrid (Algorithmic 60% + LLM 40%)",
        "mock_mode": MOCK_MODE,
        "scholarships_count": scholarship_db.count() if scholarship_db else 0,
        "vectorstore_docs": len(vectorstore.docs) if vectorstore else 0,
        "regions": ["Europe", "South Korea", "Japan", "China", "Russia", "North America", "Oceania"],
        "scoring": "Multi-dimensional: GPA(25%), Field(25%), Degree(15%), Research(15%), Language(10%), Experience(10%)"
    }

@app.post("/analyze")
async def analyze_profile(profile: UserProfile):
    if MOCK_MODE:
        return {"results": get_mock_results(profile), "mock": True}
    
    try:
        from engine import analyze
        results = analyze(profile, vectorstore, reasoning_chain)
        if not results:
            return {"results": get_mock_results(profile), "mock": True, "message": "No specific matches found, showing examples."}
        return {"results": results, "mock": False, "total": len(results)}
    except Exception as e:
        print(f"[API] Real-time analysis failed: {e}")
        import traceback
        traceback.print_exc()
        # Final fallback with explicit error reporting
        return {"results": get_mock_results(profile), "mock": True, "error": str(e)}


# ── New Endpoints ─────────────────────────────────────────────────────

@app.get("/scholarships")
def list_scholarships(region: str = None):
    """View all cached scholarships, optionally filtered by region."""
    if not scholarship_db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    if region:
        scholarships = scholarship_db.get_scholarships_by_region(region)
    else:
        scholarships = scholarship_db.get_all_scholarships()
    
    return {
        "count": len(scholarships),
        "scholarships": scholarships
    }

@app.get("/db-status")
def db_status():
    """Check database health and last refresh time."""
    if not scholarship_db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    return {
        "total_scholarships": scholarship_db.count(),
        "last_updated": scholarship_db.get_last_updated(),
        "needs_refresh": scholarship_db.needs_refresh(),
        "db_path": str(scholarship_db.db_path),
    }

@app.post("/refresh")
async def refresh_database():
    """Trigger a database refresh — re-seeds with latest curated data."""
    global vectorstore
    
    if not scholarship_db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        seed_database(scholarship_db)
        
        # Rebuild vectorstore
        doc_dicts = scholarship_db.to_documents()
        class Doc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        
        docs = [Doc(d['page_content'], d['metadata']) for d in doc_dicts]
        from vectorstore import build_vectorstore
        vectorstore = build_vectorstore(docs)
        
        return {
            "status": "success",
            "total_scholarships": scholarship_db.count(),
            "last_updated": scholarship_db.get_last_updated(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Mock fallback ─────────────────────────────────────────────────────

def get_mock_results(profile: UserProfile):
    """Fallback high-quality mock results for STEM focus."""
    return [
        {
            "title": "Global Korea Scholarship (GKS) - Graduate STEM Track",
            "region": "South Korea",
            "url": "https://www.studyinkorea.go.kr",
            "eligibility": "Eligible",
            "score": 88,
            "analysis": f"Strong fit for your {profile.field} background. Korea focuses heavily on STEM development.",
            "gaps": "Maintain your high CGPA and consider learning basic Korean."
        },
        {
            "title": "MEXT Scholarship (Monbukagakusho) - Engineering",
            "region": "Japan",
            "url": "https://www.studyinjapan.go.jp",
            "eligibility": "Eligible",
            "score": 82,
            "analysis": f"Matches your target degree ({profile.target_degree}). Japan offers world-class robotics and CS research.",
            "gaps": "Prepare for the competitive research proposal requirement."
        }
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
