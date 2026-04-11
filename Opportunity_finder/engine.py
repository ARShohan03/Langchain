"""Analysis engine — evaluates scholarships against user profiles using real-time AI.

Uses a hybrid scoring approach:
  60% Algorithmic (GPA, field, degree, research, language, experience)  
  40% LLM-analyzed (AI evaluates nuances the algorithm can't catch)
"""

import re
import json
from typing import List, Dict, Any
from models import UserProfile


# ────────────────────────────────────────────────────────────────────────
#  FIELD SYNONYM MAP  — maps related fields so "CS" matches "AI" etc.
# ────────────────────────────────────────────────────────────────────────
_FIELD_SYNONYMS = {
    "computer science": {"cs", "computing", "software", "programming", "informatics", "computer", "it"},
    "ai": {"artificial intelligence", "machine learning", "deep learning", "ml", "neural networks", "nlp"},
    "data science": {"data analytics", "big data", "data engineering", "data mining", "statistics"},
    "electrical engineering": {"electronics", "power systems", "signal processing", "vlsi", "semiconductor", "ee"},
    "mechanical engineering": {"mechanics", "thermodynamics", "manufacturing", "automotive", "me"},
    "civil engineering": {"structural", "construction", "geotechnical", "transportation engineering"},
    "physics": {"applied physics", "quantum", "photonics", "optics", "astrophysics"},
    "mathematics": {"applied mathematics", "math", "numerical analysis", "algebra", "calculus", "statistics"},
    "chemistry": {"chemical engineering", "biochemistry", "organic chemistry", "materials chemistry"},
    "biology": {"biotechnology", "biomedical", "bioinformatics", "life sciences", "microbiology"},
    "robotics": {"automation", "mechatronics", "robot", "autonomous systems"},
    "engineering": {"stem", "technology", "technical", "applied science"},
    "environmental science": {"sustainability", "climate", "ecology", "renewable energy", "green tech"},
    "materials science": {"nanotechnology", "nanomaterials", "composites", "metallurgy"},
}


def _expand_field(field: str) -> set:
    """Expand a user's field into a set of related keywords."""
    field_lower = field.lower().strip()
    keywords = set(field_lower.split())
    keywords.add(field_lower)
    
    for canonical, synonyms in _FIELD_SYNONYMS.items():
        # If user's field matches canonical OR any synonym
        if field_lower in synonyms or canonical in field_lower or any(s in field_lower for s in synonyms):
            keywords.add(canonical)
            keywords.update(synonyms)
    
    return keywords


# ────────────────────────────────────────────────────────────────────────
#  MULTI-DIMENSIONAL SCORING (60% of final score)
# ────────────────────────────────────────────────────────────────────────
def _compute_algorithmic_score(profile: UserProfile, doc: Any) -> Dict[str, Any]:
    """
    Compute real fit score from 6 dimensions.
    
    Returns dict with individual dimension scores and weighted total.
    """
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
    content = (doc.page_content if hasattr(doc, 'page_content') else "").lower()
    
    # Parse metadata
    field_keywords = metadata.get("field_keywords", [])
    if isinstance(field_keywords, str):
        try:
            field_keywords = json.loads(field_keywords)
        except Exception:
            field_keywords = [field_keywords]
    
    min_gpa = float(metadata.get("min_gpa", 0))
    research_required = metadata.get("research_required", False)
    if isinstance(research_required, str):
        research_required = research_required.lower() in ("true", "1", "yes")
    
    lang_req = metadata.get("language_requirement", "").lower()
    scholarship_degree = metadata.get("degree", "").lower()
    
    scores = {}
    
    # ── 1. GPA FIT (25%) ──────────────────────────────────────────────
    gpa_ratio = profile.cgpa / profile.cgpa_scale if profile.cgpa_scale > 0 else 0
    gpa_on_4 = gpa_ratio * 4.0  # Normalize to 4.0 scale
    
    if min_gpa > 0:
        if gpa_on_4 >= min_gpa + 0.5:
            scores["gpa"] = 100  # Well above minimum
        elif gpa_on_4 >= min_gpa + 0.2:
            scores["gpa"] = 85   # Comfortably above minimum
        elif gpa_on_4 >= min_gpa:
            scores["gpa"] = 70   # Meets minimum exactly
        elif gpa_on_4 >= min_gpa - 0.3:
            scores["gpa"] = 45   # Slightly below — risky
        else:
            scores["gpa"] = 15   # Significantly below minimum
    else:
        # No specific min GPA — use general tiers
        if gpa_ratio >= 0.90:
            scores["gpa"] = 95
        elif gpa_ratio >= 0.80:
            scores["gpa"] = 75
        elif gpa_ratio >= 0.70:
            scores["gpa"] = 55
        elif gpa_ratio >= 0.60:
            scores["gpa"] = 35
        else:
            scores["gpa"] = 15
    
    # ── 2. FIELD ALIGNMENT (25%) ──────────────────────────────────────
    user_fields = _expand_field(profile.field)
    
    # Compare against scholarship field_keywords and content
    scholarship_fields = set()
    for kw in field_keywords:
        scholarship_fields.update(_expand_field(kw))
    
    # Also extract from content
    for canonical, synonyms in _FIELD_SYNONYMS.items():
        if canonical in content or any(s in content for s in synonyms):
            scholarship_fields.add(canonical)
            scholarship_fields.update(synonyms)
    
    if scholarship_fields:
        overlap = user_fields & scholarship_fields
        # Check for "all" fields indicator
        if any(kw.lower() == "all stem fields" for kw in field_keywords):
            scores["field"] = 80  # Accepts all STEM — good but not perfect field-specific match
        elif len(overlap) >= 5:
            scores["field"] = 100
        elif len(overlap) >= 3:
            scores["field"] = 85
        elif len(overlap) >= 1:
            scores["field"] = 60
        else:
            scores["field"] = 10
    else:
        # No field info — use keyword matching on content
        field_words = profile.field.lower().split()
        matches = sum(1 for w in field_words if w in content)
        scores["field"] = min(80, matches * 30)  # Up to 80 via content matching
    
    # ── 3. DEGREE MATCH (15%) ─────────────────────────────────────────
    target = profile.target_degree.lower().strip()
    
    # Normalize degree abbreviations to canonical forms
    def _normalize_degree(d):
        d = d.lower().strip()
        d = d.replace("'", "").replace("\u2019", "")  # Remove apostrophes
        # Map abbreviations
        for abbr, canon in [("msc", "master"), ("m.sc", "master"), ("ma", "master"),
                            ("meng", "master"), ("m.eng", "master"), ("mba", "master"),
                            ("bsc", "bachelor"), ("b.sc", "bachelor"), ("ba", "bachelor"),
                            ("beng", "bachelor"), ("b.eng", "bachelor"),
                            ("phd", "doctoral"), ("ph.d", "doctoral"),
                            ("doctorate", "doctoral"), ("postdoc", "postdoctoral")]:
            if abbr == d or abbr in d.split():
                d = d + " " + canon
        return d
    
    target_normalized = _normalize_degree(target)
    degree_normalized = _normalize_degree(scholarship_degree)
    
    # Check if user wants master and scholarship offers master
    target_is_master = "master" in target_normalized
    target_is_phd = "doctoral" in target_normalized or "phd" in target_normalized
    target_is_bachelor = "bachelor" in target_normalized
    
    schol_has_master = "master" in degree_normalized
    schol_has_phd = "doctoral" in degree_normalized or "phd" in degree_normalized
    schol_has_bachelor = "bachelor" in degree_normalized
    
    if not scholarship_degree:
        scores["degree"] = 60  # Unknown scholarship degree
    elif (target_is_master and schol_has_master) or (target_is_phd and schol_has_phd) or (target_is_bachelor and schol_has_bachelor):
        scores["degree"] = 100  # Direct match
    elif target_is_master and schol_has_phd and not schol_has_master:
        scores["degree"] = 40   # Want Masters but it's PhD-only
    elif target_is_phd and schol_has_master and not schol_has_phd:
        scores["degree"] = 40   # Want PhD but it's Masters-only
    elif target_is_bachelor and (schol_has_master or schol_has_phd):
        scores["degree"] = 15   # Want Bachelor but it's grad-level
    elif (target_is_master or target_is_phd) and schol_has_bachelor:
        scores["degree"] = 10   # Want grad but it's undergrad
    else:
        scores["degree"] = 50
    
    # ── 4. RESEARCH STRENGTH (15%) ────────────────────────────────────
    papers = profile.research_papers
    
    if research_required:
        if papers >= 3:
            scores["research"] = 100
        elif papers >= 1:
            scores["research"] = 70
        elif papers == 0:
            scores["research"] = 20  # Research required but no papers — big gap
    else:
        # Research not required but still a bonus
        if papers >= 3:
            scores["research"] = 100
        elif papers >= 1:
            scores["research"] = 80
        elif papers == 0:
            scores["research"] = 60  # Not penalized much since not required
    
    # ── 5. LANGUAGE READINESS (10%) ───────────────────────────────────
    user_test = (profile.english_test or "").lower().strip()
    user_score_val = profile.english_score or 0
    
    if not lang_req or lang_req == "none":
        scores["language"] = 85  # No requirement — good
    elif not user_test or user_score_val == 0:
        scores["language"] = 30  # No test provided — uncertain
    else:
        # Extract required score from string like "IELTS 6.5"
        req_match = re.search(r'(\d+\.?\d*)', lang_req)
        req_score = float(req_match.group(1)) if req_match else 6.0
        
        if "ielts" in user_test:
            if "ielts" in lang_req or "toefl" not in lang_req:
                if user_score_val >= req_score + 1.0:
                    scores["language"] = 100
                elif user_score_val >= req_score:
                    scores["language"] = 85
                elif user_score_val >= req_score - 0.5:
                    scores["language"] = 60
                else:
                    scores["language"] = 25
            else:
                scores["language"] = 50  # Different test type — uncertain
        elif "toefl" in user_test:
            if "toefl" in lang_req:
                if user_score_val >= req_score + 10:
                    scores["language"] = 100
                elif user_score_val >= req_score:
                    scores["language"] = 85
                elif user_score_val >= req_score - 5:
                    scores["language"] = 60
                else:
                    scores["language"] = 25
            else:
                scores["language"] = 50
        else:
            scores["language"] = 50  # Other test — partial credit
    
    # ── 6. EXPERIENCE & EXTRAS (10%) ──────────────────────────────────
    exp_score = 0
    
    # Internships
    if profile.internships >= 3:
        exp_score += 50
    elif profile.internships >= 1:
        exp_score += 30
    else:
        exp_score += 10
    
    # Extracurriculars — check relevance
    if profile.extracurriculars:
        relevant = 0
        for ec in profile.extracurriculars:
            ec_lower = ec.lower()
            if any(kw in ec_lower for kw in ["research", "ai", "coding", "robotics", "lab", 
                                               "hackathon", "stem", "science", "engineering",
                                               "programming", "competition", "olympiad"]):
                relevant += 1
        if relevant >= 2:
            exp_score += 50
        elif relevant >= 1:
            exp_score += 35
        else:
            exp_score += 15  # Has extracurriculars but not STEM-relevant
    else:
        exp_score += 5  # No extracurriculars listed
    
    scores["experience"] = min(100, exp_score)
    
    # ── WEIGHTED TOTAL ────────────────────────────────────────────────
    weights = {
        "gpa": 0.25,
        "field": 0.25,
        "degree": 0.15,
        "research": 0.15,
        "language": 0.10,
        "experience": 0.10,
    }
    
    weighted_total = sum(scores[k] * weights[k] for k in weights)
    
    return {
        "total": round(weighted_total, 1),
        "breakdown": scores,
    }


# ────────────────────────────────────────────────────────────────────────
#  LLM SCORE PARSING
# ────────────────────────────────────────────────────────────────────────
def _parse_llm_score(content: str) -> int:
    """Extract numeric score from LLM response. Returns -1 if not found."""
    if not content:
        return -1
    
    # Look for [SCORE] section first
    score_section_match = re.search(
        r'\[SCORE\]\s*(.*?)(?=\[|$)', content, re.DOTALL | re.IGNORECASE
    )
    if score_section_match:
        digits = re.search(r'(\d{1,3})', score_section_match.group(1))
        if digits:
            val = int(digits.group(1))
            if 0 <= val <= 100:
                return val
    
    # Fallback: look for "score: XX" or "XX/100" or "XX%" anywhere
    patterns = [
        r'score[:\s]+(\d{1,3})',
        r'(\d{1,3})\s*/\s*100',
        r'(\d{1,3})\s*%',
    ]
    for pat in patterns:
        match = re.search(pat, content, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return val
    
    return -1


def _parse_ai_output(content: str, title: str) -> Dict[str, Any]:
    """Parse structured fields from AI response."""
    data = {
        "eligibility": "Partially Eligible",
        "score": -1,
        "analysis": f"Analysis for {title} based on your STEM background.",
        "gaps": "Strengthen your research profile."
    }
    
    if not content:
        return data

    # Sections to parse
    sections = {
        "eligibility": ["ELIGIBILITY", "Eligibility"],
        "score": ["SCORE", "Score"],
        "analysis": ["ANALYSIS", "Analysis"],
        "gaps": ["GAPS", "Strategic Advice", "Advice", "Gaps"]
    }

    def get_section_text(keys):
        for key in keys:
            pattern = rf"\[?{key}\]?:?\s*(.*?)(?=\[|Score|Analysis|Gaps|\*\*|\n\n|\Z)"
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match and len(match.group(1).strip()) > 5:
                return match.group(1).strip(" *#:\n\t")
        return None

    # Eligibility
    val = get_section_text(sections["eligibility"])
    if val:
        data["eligibility"] = val.split('\n')[0].strip()[:100]

    # Analysis
    val = get_section_text(sections["analysis"])
    if val:
        data["analysis"] = val[:600]

    # Gaps
    val = get_section_text(sections["gaps"])
    if val:
        data["gaps"] = val[:450]

    # Score
    data["score"] = _parse_llm_score(content)

    return data


# ────────────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS — HYBRID SCORING
# ────────────────────────────────────────────────────────────────────────
def analyze(profile: UserProfile, vectorstore: Any, reasoning_chain: Any) -> List[Dict[str, Any]]:
    """
    Search and analyze matching scholarships with hybrid scoring:
      - Phase 1: Compute algorithmic score for ALL scholarships (instant)
      - Phase 2: Call LLM for top 10 algorithmic matches only (saves time)
      - Return ALL results sorted by final score
    """
    # 1. Get ALL scholarships from vectorstore
    total_docs = len(vectorstore.docs) if hasattr(vectorstore, 'docs') else 48
    query = f"{profile.field} {profile.target_degree} STEM scholarship"
    docs = vectorstore.invoke(query, k=total_docs)
    
    # 2. Build profile string for AI (only need to do this once)
    profile_str = (
        f"Candidate: {profile.name or 'Student'}\n"
        f"Academic Field: {profile.field}\n"
        f"CGPA: {profile.cgpa}/{profile.cgpa_scale} (GPA on 4.0 scale: {profile.cgpa / profile.cgpa_scale * 4.0 if profile.cgpa_scale > 0 else 0:.2f})\n"
        f"Research Papers: {profile.research_papers}\n"
        f"Internships: {profile.internships}\n"
        f"Extracurriculars: {', '.join(profile.extracurriculars) if profile.extracurriculars else 'None'}\n"
        f"English: {profile.english_test or 'Not provided'} {profile.english_score or ''}\n"
        f"Target Degree: {profile.target_degree}\n"
        f"Experience: {profile.experience_years} years"
    )
    
    # ── PHASE 1: Algorithmic scoring for ALL programs ─────────────────
    print(f"[Engine] Phase 1: Computing algorithmic scores for {len(docs)} scholarships...")
    
    scored_docs = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown Program")
        algo_result = _compute_algorithmic_score(profile, doc)
        scored_docs.append({
            "doc": doc,
            "title": title,
            "algo_score": algo_result["total"],
            "breakdown": algo_result["breakdown"],
        })
    
    # Sort by algo score to find top matches for LLM analysis
    scored_docs.sort(key=lambda x: x["algo_score"], reverse=True)
    
    # ── PHASE 2: LLM analysis for top 10 only ────────────────────────
    LLM_TOP_N = 10
    top_for_llm = scored_docs[:LLM_TOP_N]
    
    print(f"[Engine] Phase 2: Running AI analysis on top {LLM_TOP_N} matches...")
    
    llm_results = {}  # title -> ai_data
    for item in top_for_llm:
        title = item["title"]
        doc = item["doc"]
        print(f"  [LLM] Analyzing: {title}...")
        
        try:
            ai_input = {
                "profile": profile_str,
                "scholarship": f"Program: {title}\nSummary: {doc.page_content}"
            }
            ai_response = reasoning_chain.invoke(ai_input)
            ai_data = _parse_ai_output(ai_response.content, title)
            llm_results[title] = ai_data
        except Exception as e:
            print(f"  [LLM] Failed for {title}: {e}")
    
    # ── PHASE 3: Combine scores and build results ─────────────────────
    results = []
    
    for item in scored_docs:
        title = item["title"]
        doc = item["doc"]
        algo_score = item["algo_score"]
        breakdown = item["breakdown"]
        
        # Get LLM data if available
        ai_data = llm_results.get(title)
        
        if ai_data:
            llm_score = ai_data["score"]
            eligibility = ai_data["eligibility"]
            analysis = ai_data["analysis"]
            gaps = ai_data["gaps"]
            
            # Hybrid: 60% algo + 40% LLM
            if llm_score >= 0:
                final_score = round(0.6 * algo_score + 0.4 * llm_score)
            else:
                final_score = round(algo_score)
        else:
            # No LLM data — pure algorithmic
            llm_score = -1
            final_score = round(algo_score)
            
            # Generate simple eligibility from algo score
            if algo_score >= 70:
                eligibility = "Likely Eligible"
            elif algo_score >= 40:
                eligibility = "Partially Eligible"
            else:
                eligibility = "Low Match"
            
            analysis = _generate_algo_analysis(profile, doc, algo_score, breakdown)
            gaps = _generate_algo_gaps(profile, breakdown)
        
        final_score = max(0, min(100, final_score))
        
        # Build dimension breakdown string
        dim_str = " | ".join(
            f"{k.upper()}: {v}" for k, v in breakdown.items()
        )
        
        results.append({
            "title": title,
            "region": doc.metadata.get("region", "Global"),
            "url": doc.metadata.get("url", "#"),
            "eligibility": eligibility,
            "score": final_score,
            "algo_score": round(algo_score),
            "llm_score": llm_score if llm_score >= 0 else None,
            "score_breakdown": dim_str,
            "analysis": analysis,
            "gaps": gaps,
            "degree": doc.metadata.get("degree", "Master's"),
            "countries": doc.metadata.get("countries", ""),
            "universities": doc.metadata.get("universities", ""),
            "funding_type": doc.metadata.get("funding_type", ""),
        })
    
    # Sort by final score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _generate_algo_analysis(profile: UserProfile, doc: Any, score: float, breakdown: dict) -> str:
    """Generate human-readable analysis from algorithmic scores alone."""
    title = doc.metadata.get("title", "this program")
    strengths = []
    weaknesses = []
    
    if breakdown.get("gpa", 0) >= 85:
        strengths.append("Your GPA exceeds the minimum requirement")
    elif breakdown.get("gpa", 0) < 50:
        weaknesses.append("your GPA is below the typical threshold")
    
    if breakdown.get("field", 0) >= 85:
        strengths.append(f"your {profile.field} background aligns well with the program focus")
    elif breakdown.get("field", 0) < 40:
        weaknesses.append("your field of study doesn't closely match the program's target disciplines")
    
    if breakdown.get("research", 0) < 40:
        weaknesses.append("the program expects research experience which you currently lack")
    
    if breakdown.get("degree", 0) >= 80:
        strengths.append(f"the {profile.target_degree} level matches what's offered")
    
    parts = []
    if strengths:
        parts.append(f"For {title}: {', '.join(strengths)}.")
    if weaknesses:
        parts.append(f"However, {', '.join(weaknesses)}.")
    
    return " ".join(parts) if parts else f"Algorithmic fit analysis for {title} based on your profile."


def _generate_algo_gaps(profile: UserProfile, breakdown: dict) -> str:
    """Generate the most impactful advice based on weakest dimension."""
    weakest = min(breakdown, key=breakdown.get)
    
    advice = {
        "gpa": "Focus on improving your academic performance to meet the GPA requirements.",
        "field": "Consider highlighting transferable skills or interdisciplinary experience to strengthen your field alignment.",
        "degree": "Check if your target degree level matches what this scholarship offers.",
        "research": "Start building your research portfolio with publications or conference papers.",
        "language": "Focus on achieving the required English proficiency score before applying.",
        "experience": "Gain relevant experience through internships, projects, or STEM-related extracurriculars.",
    }
    
    return advice.get(weakest, "Strengthen your overall profile to improve your chances.")

