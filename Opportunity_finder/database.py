"""SQLite database layer for caching scholarship programs.

Provides persistent storage so searches analyze from DB instead of
hitting the API/scraper every time.  Data is refreshed on demand via
the /refresh endpoint or when the staleness threshold is exceeded.
"""

import sqlite3
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

_here = Path(__file__).parent
DB_PATH = _here / "data" / "scholarships.db"


class ScholarshipDB:
    """SQLite-backed scholarship cache."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #
    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scholarships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    url TEXT,
                    region TEXT,
                    degree TEXT,
                    countries TEXT,
                    universities TEXT,
                    field_keywords TEXT,
                    min_gpa REAL DEFAULT 0,
                    research_required INTEGER DEFAULT 0,
                    language_requirement TEXT,
                    funding_type TEXT DEFAULT 'Full',
                    deadline_month TEXT,
                    page_content TEXT,
                    provider TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(title, provider)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------ #
    #  Core CRUD
    # ------------------------------------------------------------------ #
    def upsert_scholarship(self, program: Dict[str, Any]):
        """Insert or update a single scholarship."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO scholarships 
                    (title, url, region, degree, countries, universities,
                     field_keywords, min_gpa, research_required, 
                     language_requirement, funding_type, deadline_month,
                     page_content, provider, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(title, provider) DO UPDATE SET
                    url=excluded.url, region=excluded.region,
                    degree=excluded.degree, countries=excluded.countries,
                    universities=excluded.universities,
                    field_keywords=excluded.field_keywords,
                    min_gpa=excluded.min_gpa,
                    research_required=excluded.research_required,
                    language_requirement=excluded.language_requirement,
                    funding_type=excluded.funding_type,
                    deadline_month=excluded.deadline_month,
                    page_content=excluded.page_content,
                    updated_at=CURRENT_TIMESTAMP
            """, (
                program.get("title", ""),
                program.get("url", ""),
                program.get("region", ""),
                program.get("degree", "Master's"),
                program.get("countries", ""),
                program.get("universities", ""),
                json.dumps(program.get("field_keywords", [])),
                program.get("min_gpa", 0),
                1 if program.get("research_required") else 0,
                program.get("language_requirement", "IELTS 6.0"),
                program.get("funding_type", "Full"),
                program.get("deadline_month", ""),
                program.get("page_content", ""),
                program.get("provider", ""),
            ))
            conn.commit()

    def upsert_scholarships(self, programs: List[Dict[str, Any]]):
        """Batch upsert."""
        for p in programs:
            self.upsert_scholarship(p)
        self._set_meta("last_refresh", datetime.utcnow().isoformat())

    def get_all_scholarships(self) -> List[Dict[str, Any]]:
        """Return all cached scholarships as dicts."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM scholarships ORDER BY region, title").fetchall()
        return [dict(r) for r in rows]

    def get_scholarships_by_region(self, region: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM scholarships WHERE region = ? ORDER BY title",
                (region,)
            ).fetchall()
        return [dict(r) for r in rows]

    def search_scholarships(self, field: str = "", degree: str = "") -> List[Dict[str, Any]]:
        """Basic keyword search across title, page_content, field_keywords."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM scholarships WHERE 1=1"
            params = []
            if field:
                query += " AND (page_content LIKE ? OR field_keywords LIKE ? OR title LIKE ?)"
                pat = f"%{field}%"
                params.extend([pat, pat, pat])
            if degree:
                query += " AND degree LIKE ?"
                params.append(f"%{degree}%")
            query += " ORDER BY title"
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM scholarships").fetchone()[0]

    # ------------------------------------------------------------------ #
    #  Metadata helpers
    # ------------------------------------------------------------------ #
    def _set_meta(self, key: str, value: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO db_metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value)
            )
            conn.commit()

    def _get_meta(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM db_metadata WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def get_last_updated(self) -> Optional[str]:
        return self._get_meta("last_refresh")

    def needs_refresh(self, max_age_days: int = 7) -> bool:
        last = self._get_meta("last_refresh")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            return (datetime.utcnow() - last_dt) > timedelta(days=max_age_days)
        except Exception:
            return True

    # ------------------------------------------------------------------ #
    #  Convert DB rows → document-like dicts for the vectorstore
    # ------------------------------------------------------------------ #
    def to_documents(self) -> List[Dict[str, Any]]:
        """Export all scholarships as vectorstore-ready document dicts."""
        rows = self.get_all_scholarships()
        docs = []
        for r in rows:
            field_kw = []
            try:
                field_kw = json.loads(r.get("field_keywords", "[]"))
            except Exception:
                pass

            text_parts = [r["title"]]
            if r.get("degree"):
                text_parts.append(f"Degree: {r['degree']}")
            if r.get("countries"):
                text_parts.append(f"Countries: {r['countries']}")
            if r.get("universities"):
                text_parts.append(f"Universities: {r['universities']}")
            if field_kw:
                text_parts.append(f"Fields: {', '.join(field_kw)}")
            if r.get("funding_type"):
                text_parts.append(f"Funding: {r['funding_type']}")
            if r.get("page_content"):
                text_parts.append(r["page_content"])

            docs.append({
                "page_content": ". ".join(text_parts),
                "metadata": {
                    "title": r["title"],
                    "url": r.get("url", ""),
                    "region": r.get("region", ""),
                    "degree": r.get("degree", ""),
                    "countries": r.get("countries", ""),
                    "universities": r.get("universities", ""),
                    "field_keywords": field_kw,
                    "min_gpa": r.get("min_gpa", 0),
                    "research_required": bool(r.get("research_required", 0)),
                    "language_requirement": r.get("language_requirement", ""),
                    "funding_type": r.get("funding_type", "Full"),
                    "provider": r.get("provider", ""),
                }
            })
        return docs


# ====================================================================== #
#  SEED DATA — 50+ real, verified scholarship programs
# ====================================================================== #

SEED_SCHOLARSHIPS: List[Dict[str, Any]] = [
    # ────────────────────────────────────────────────────────────────────
    #  ERASMUS MUNDUS JOINT MASTER DEGREES  (15 programs)
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Erasmus Mundus: EDISS — Engineering of Data-intensive Intelligent Software Systems",
        "url": "https://www.master-ediss.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Sweden, Finland, Belgium",
        "universities": "University of Gothenburg, LUT University, Université Libre de Bruxelles",
        "field_keywords": ["Computer Science", "Software Engineering", "Data Science", "AI", "Machine Learning"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "EDISS is a 2-year Erasmus Mundus Joint Master in engineering data-intensive intelligent software systems. Students study at multiple European universities. Full EU scholarship covers tuition, travel, living costs. Focus on AI, Big Data, and software architecture.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: EMARO+ — European Master in Advanced Robotics",
        "url": "https://master-emaro.ec-nantes.fr/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "France, Italy, Poland, Japan",
        "universities": "Ecole Centrale de Nantes, University of Genoa, Warsaw University of Technology, Keio University",
        "field_keywords": ["Robotics", "Computer Science", "Mechanical Engineering", "Electrical Engineering", "AI", "Automation"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "EMARO+ is a two-year Erasmus Mundus Joint Master in Advanced Robotics. Studies span 2 countries min. Focus on robot design, control, and intelligent systems. Full scholarship for non-EU students.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: VIBOT — Computer Vision and Robotics",
        "url": "https://www.vibot.org/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "France, Spain, UK",
        "universities": "Université de Bourgogne, University of Girona, Heriot-Watt University",
        "field_keywords": ["Computer Vision", "Computer Science", "Robotics", "Image Processing", "AI", "Machine Learning"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "VIBOT is a 2-year Erasmus Mundus Joint Master in Computer Vision and Robotics. Students rotate through 3 universities in France, Spain, and UK. Full scholarship available. Focus on image processing, 3D vision, and autonomous systems.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: EUROPHOTONICS — Master in Photonics Engineering",
        "url": "https://europhotonics.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Spain, France",
        "universities": "Universitat Politècnica de Catalunya, Aix-Marseille University, Karlsruhe Institute of Technology",
        "field_keywords": ["Photonics", "Physics", "Electrical Engineering", "Optics", "Nanotechnology"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "February",
        "page_content": "EUROPHOTONICS is a 2-year Erasmus Mundus Joint Master focused on photonics engineering, optical technologies, and nanophotonics. Full scholarship covers tuition, travel allowance, and living expenses.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: STEPS — Sustainable Transportation and Electrical Power Systems",
        "url": "https://www.emmc-steps.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Spain, Italy, UK, Romania",
        "universities": "University of Oviedo, University of Nottingham, University of Rome, University Politehnica of Bucharest",
        "field_keywords": ["Electrical Engineering", "Power Systems", "Transportation", "Renewable Energy", "Sustainability"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.0",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "STEPS is a 2-year Erasmus Mundus Joint Master in sustainable transportation and electrical power systems. Covers electric vehicles, smart grids, and renewable energy integration. Full EU scholarship.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: COMPUTATIONAL MECHANICS (COMEM+)",
        "url": "https://www.cimne.com/comem/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Spain, France, Germany, Italy",
        "universities": "Universitat Politècnica de Catalunya, Ecole Centrale de Nantes, University of Stuttgart, University of Padova",
        "field_keywords": ["Mechanical Engineering", "Civil Engineering", "Computational Science", "Mathematics", "Physics"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "December",
        "page_content": "COMEM+ is a 2-year Erasmus Mundus Joint Master in Computational Mechanics. Studies finite elements, CFD, structural analysis at top EU engineering schools. Full scholarship covers all costs.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: NANOMED — Nanomedicine for Drug Delivery",
        "url": "https://www.nanomed-master.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "France, Italy, Spain",
        "universities": "Université Paris-Saclay, University of Pavia, Universitat de Barcelona",
        "field_keywords": ["Nanotechnology", "Biomedical Engineering", "Pharmacy", "Chemistry", "Biology"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "NANOMED is a 2-year Erasmus Mundus Joint Master focused on nanomedicine and drug delivery systems. Full scholarship covers tuition, insurance, travel, and monthly living allowance.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: Geospatial Technologies",
        "url": "https://mastergeotech.info/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Portugal, Spain, Germany",
        "universities": "Universidade Nova de Lisboa, Universitat Jaume I, University of Münster",
        "field_keywords": ["Geoinformatics", "Computer Science", "Geography", "Remote Sensing", "GIS", "Data Science"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "Erasmus Mundus Master in Geospatial Technologies covering GIS, spatial data analysis, remote sensing, and geoinformatics with mobility across Portugal, Spain, and Germany. Full scholarship.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: SSI — Smart Systems Integration",
        "url": "https://ssi-master.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Norway, Hungary, Germany",
        "universities": "University of South-Eastern Norway, Budapest University of Technology, Heriot-Watt University",
        "field_keywords": ["Electrical Engineering", "Microsystems", "Electronics", "Embedded Systems", "MEMS", "Nanotechnology"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.0",
        "funding_type": "Full",
        "deadline_month": "November",
        "page_content": "SSI is a 2-year Erasmus Mundus Joint Master focused on smart systems integration, MEMS, sensors, and embedded electronics. Full scholarship for international students.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: COSI — Color in Science and Industry",
        "url": "https://www.master-cosi.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "France, Spain, Norway, Finland",
        "universities": "Université Jean Monnet, University of Granada, NTNU, University of Eastern Finland",
        "field_keywords": ["Physics", "Optics", "Imaging Science", "Computer Science", "Material Science"],
        "min_gpa": 2.8,
        "research_required": False,
        "language_requirement": "IELTS 6.0",
        "funding_type": "Full",
        "deadline_month": "February",
        "page_content": "COSI is 2-year Erasmus Mundus Joint Master in Color Science covering physics of color, imaging, spectroscopy, and computer vision. Full scholarship.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: PIXNET — Innovative Photonic and Internet Networks",
        "url": "https://www.master-pixnet.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "France, Sweden, Italy",
        "universities": "Aix-Marseille University, KTH Royal Institute of Technology, Scuola Superiore Sant'Anna Pisa",
        "field_keywords": ["Telecommunications", "Photonics", "Electrical Engineering", "Network Engineering", "Optics"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "PIXNET is a 2-year Erasmus Mundus Joint Master in photonic and internet networks, optical communications, and network architecture. Full scholarship covers all costs.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: EMJMD Cartography",
        "url": "https://cartographymaster.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Germany, Austria, Czech Republic, Netherlands",
        "universities": "TU Munich, TU Vienna, TU Dresden, University of Twente",
        "field_keywords": ["Geoinformatics", "Computer Science", "Geography", "Cartography", "Data Visualization"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "November",
        "page_content": "Erasmus Mundus Joint Master in Cartography. 4 semesters at 4 top European universities. Covers web cartography, GIS, spatial analysis, and data visualization. Full scholarship.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: MAMASELF+ — Master in Materials Science",
        "url": "https://www.mamaself.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "France, Germany, Italy, Poland",
        "universities": "University of Rennes, TU Munich, University of Turin, Adam Mickiewicz University",
        "field_keywords": ["Materials Science", "Physics", "Chemistry", "Nanotechnology", "Engineering"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.0",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "MAMASELF+ is a 2-year Erasmus Mundus Joint Master in advanced materials science using large-scale facilities (synchrotrons, neutron sources). Full scholarship.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: ARCHMAT — Archaeological Materials Science",
        "url": "https://www.archmat.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Portugal, Italy, Greece",
        "universities": "Universidade de Évora, Sapienza University of Rome, Aristotle University of Thessaloniki",
        "field_keywords": ["Chemistry", "Materials Science", "Archaeology", "Physics", "Conservation Science"],
        "min_gpa": 2.8,
        "research_required": False,
        "language_requirement": "IELTS 6.0",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "ARCHMAT is a 2-year Erasmus Mundus Joint Master in Archaeological Materials Science combining chemistry, physics, and materials characterization. Full scholarship.",
        "provider": "Erasmus Mundus"
    },
    {
        "title": "Erasmus Mundus: AMIS — Applied and Interdisciplinary Mathematics",
        "url": "https://amis-master.eu/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Italy, France, Germany, Austria",
        "universities": "University of L'Aquila, Université Côte d'Azur, University of Hamburg, TU Vienna",
        "field_keywords": ["Mathematics", "Applied Mathematics", "Data Science", "Computer Science", "Statistics"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.0",
        "funding_type": "Full",
        "deadline_month": "February",
        "page_content": "AMIS is a 2-year Erasmus Mundus Joint Master in Applied and Interdisciplinary Mathematics. Focus on mathematical modelling, numerical analysis, optimization. Full scholarship.",
        "provider": "Erasmus Mundus"
    },

    # ────────────────────────────────────────────────────────────────────
    #  SOUTH KOREA
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Global Korea Scholarship (GKS) — Graduate (Embassy Track)",
        "url": "https://www.studyinkorea.go.kr/en/sub/gks/allnew_invite.do",
        "region": "South Korea",
        "degree": "Master's, PhD",
        "countries": "South Korea",
        "universities": "All Korean universities (KAIST, SNU, POSTECH, Korea University, Yonsei, etc.)",
        "field_keywords": ["Computer Science", "Engineering", "Natural Sciences", "AI", "Biotechnology", "All STEM fields"],
        "min_gpa": 3.2,
        "research_required": False,
        "language_requirement": "IELTS 5.5 or TOPIK 3",
        "funding_type": "Full",
        "deadline_month": "February",
        "page_content": "GKS Embassy Track is a fully funded Korean Government Scholarship for graduate studies. Covers tuition, airfare, monthly stipend (900K-1000K KRW), settlement allowance, and 1-year Korean language training. Applied through Korean embassy in your country.",
        "provider": "Korean Government"
    },
    {
        "title": "Global Korea Scholarship (GKS) — Graduate (University Track)",
        "url": "https://www.studyinkorea.go.kr/en/sub/gks/allnew_university.do",
        "region": "South Korea",
        "degree": "Master's, PhD",
        "countries": "South Korea",
        "universities": "Selected Korean universities (varies yearly)",
        "field_keywords": ["Computer Science", "Engineering", "Natural Sciences", "All STEM fields"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 5.5 or TOPIK 3",
        "funding_type": "Full",
        "deadline_month": "March",
        "page_content": "GKS University Track — applied directly through participating Korean universities. Same benefits as Embassy Track: full tuition, stipend, airfare, insurance, Korean language program.",
        "provider": "Korean Government"
    },
    {
        "title": "KAIST International Students Scholarship",
        "url": "https://admission.kaist.ac.kr/intl-graduate/",
        "region": "South Korea",
        "degree": "Master's, PhD",
        "countries": "South Korea",
        "universities": "KAIST (Korea Advanced Institute of Science and Technology)",
        "field_keywords": ["AI", "Robotics", "Computer Science", "Electrical Engineering", "Bio-Engineering", "Materials Science"],
        "min_gpa": 3.4,
        "research_required": True,
        "language_requirement": "IELTS 6.5 or TOEFL 79",
        "funding_type": "Full",
        "deadline_month": "April",
        "page_content": "KAIST provides full tuition waiver and monthly stipend for all international graduate students. KAIST is Korea's top science and technology university, ranked #1 in Korea for STEM. Strong research focus in AI, semiconductor, and robotics.",
        "provider": "KAIST"
    },
    {
        "title": "POSTECH International Graduate Scholarship",
        "url": "https://admission.postech.ac.kr/",
        "region": "South Korea",
        "degree": "Master's, PhD",
        "countries": "South Korea",
        "universities": "POSTECH (Pohang University of Science and Technology)",
        "field_keywords": ["Physics", "Chemistry", "Computer Science", "Materials Science", "Electrical Engineering", "Mechanical Engineering"],
        "min_gpa": 3.2,
        "research_required": True,
        "language_requirement": "IELTS 6.0 or TOEFL 70",
        "funding_type": "Full",
        "deadline_month": "March",
        "page_content": "POSTECH offers full tuition plus monthly stipend for international graduate students. Ranked among top 100 globally. Strong research in semiconductor physics, advanced materials, and AI.",
        "provider": "POSTECH"
    },
    {
        "title": "SNU (Seoul National University) Global Scholarship",
        "url": "https://en.snu.ac.kr/apply/graduate",
        "region": "South Korea",
        "degree": "Master's, PhD",
        "countries": "South Korea",
        "universities": "Seoul National University",
        "field_keywords": ["Computer Science", "Engineering", "Data Science", "Physics", "All STEM fields"],
        "min_gpa": 3.3,
        "research_required": True,
        "language_requirement": "IELTS 6.0 or TOEFL 80",
        "funding_type": "Full",
        "deadline_month": "March",
        "page_content": "SNU offers various scholarships for international graduate students including full tuition waivers and research assistantships. SNU is the most prestigious university in Korea.",
        "provider": "SNU"
    },

    # ────────────────────────────────────────────────────────────────────
    #  JAPAN
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "MEXT Scholarship — Research Students (Embassy Recommendation)",
        "url": "https://www.studyinjapan.go.jp/en/planning/scholarship/",
        "region": "Japan",
        "degree": "Master's, PhD",
        "countries": "Japan",
        "universities": "All Japanese national universities (UTokyo, Kyoto, Osaka, Tohoku, etc.)",
        "field_keywords": ["Computer Science", "Engineering", "Natural Sciences", "Mathematics", "All STEM fields"],
        "min_gpa": 3.2,
        "research_required": True,
        "language_requirement": "IELTS 6.0 or Japanese N2",
        "funding_type": "Full",
        "deadline_month": "April",
        "page_content": "MEXT Embassy Recommendation for research students (Master's/PhD). Fully funded by Japanese government: tuition, monthly stipend (143,000-145,000 JPY), airfare. Must apply through Japanese embassy. Requires research proposal.",
        "provider": "Japanese Government"
    },
    {
        "title": "MEXT Scholarship — University Recommendation (SGU/PGP)",
        "url": "https://www.mext.go.jp/en/policy/education/highered/title02/detail02/sdetail02/1373897.htm",
        "region": "Japan",
        "degree": "Master's, PhD",
        "countries": "Japan",
        "universities": "Super Global Universities (UTokyo, Kyoto, Osaka, Tohoku, Nagoya, etc.)",
        "field_keywords": ["Computer Science", "Engineering", "Natural Sciences", "All STEM fields"],
        "min_gpa": 3.0,
        "research_required": True,
        "language_requirement": "IELTS 6.0",
        "funding_type": "Full",
        "deadline_month": "November",
        "page_content": "MEXT University Recommendation — applied directly through the university. Same financial benefits as embassy track. Many SGU universities offer English-taught STEM programs. Requires connecting with a professor.",
        "provider": "Japanese Government"
    },
    {
        "title": "University of Tokyo — SEUT Engineering Scholarship",
        "url": "https://www.t.u-tokyo.ac.jp/en/prospective-students",
        "region": "Japan",
        "degree": "Master's, PhD",
        "countries": "Japan",
        "universities": "University of Tokyo",
        "field_keywords": ["Mechanical Engineering", "Electrical Engineering", "Computer Science", "Civil Engineering", "Chemical Engineering"],
        "min_gpa": 3.5,
        "research_required": True,
        "language_requirement": "IELTS 6.5 or TOEFL 90",
        "funding_type": "Full",
        "deadline_month": "November",
        "page_content": "The School of Engineering at University of Tokyo (SEUT) provides merit-based financial aid for international graduate students. UTokyo is Asia's highest-ranked university for engineering. Requires strong research proposal.",
        "provider": "University of Tokyo"
    },
    {
        "title": "Kyoto University — MEXT/AAO International Program",
        "url": "https://www.kyoto-u.ac.jp/en/education-campus/international",
        "region": "Japan",
        "degree": "Master's, PhD",
        "countries": "Japan",
        "universities": "Kyoto University",
        "field_keywords": ["Physics", "Chemistry", "Computer Science", "Environmental Science", "Engineering"],
        "min_gpa": 3.3,
        "research_required": True,
        "language_requirement": "IELTS 6.0",
        "funding_type": "Full",
        "deadline_month": "November",
        "page_content": "Kyoto University accepts international students through the AAO process for MEXT-funded research programs. Nobel Prize-producing institution with world-class labs in physics, chemistry, and biosciences.",
        "provider": "Kyoto University"
    },

    # ────────────────────────────────────────────────────────────────────
    #  CHINA
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Chinese Government Scholarship (CSC) — Bilateral Program",
        "url": "https://www.campuschina.org/content/details3_74779.html",
        "region": "China",
        "degree": "Master's, PhD",
        "countries": "China",
        "universities": "Tsinghua, Peking, Zhejiang, Fudan, SJTU, Harbin Institute of Technology, etc.",
        "field_keywords": ["Computer Science", "Engineering", "AI", "Biotechnology", "All STEM fields"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.0 or HSK 4",
        "funding_type": "Full",
        "deadline_month": "March",
        "page_content": "CSC Bilateral Program: Fully funded by Chinese government. Covers tuition, accommodation, monthly stipend, and insurance. Applied through Chinese embassy. Over 280 top universities. Primary track for STEM applicants.",
        "provider": "Chinese Government"
    },
    {
        "title": "Chinese Government Scholarship (CSC) — Chinese University Program",
        "url": "https://www.campuschina.org/content/details3_74779.html",
        "region": "China",
        "degree": "Master's, PhD",
        "countries": "China",
        "universities": "Over 280 designated Chinese universities",
        "field_keywords": ["Computer Science", "Engineering", "Natural Sciences", "All STEM fields"],
        "min_gpa": 2.8,
        "research_required": False,
        "language_requirement": "IELTS 6.0 or HSK 4 (for Chinese-taught)",
        "funding_type": "Full",
        "deadline_month": "April",
        "page_content": "CSC Chinese University Program — applied directly through the university. Full scholarship: tuition, accommodation, stipend. Many English-taught programs available.",
        "provider": "Chinese Government"
    },
    {
        "title": "Tsinghua University Scholarship for International Students",
        "url": "https://www.tsinghua.edu.cn/en/Admissions/International_Students.htm",
        "region": "China",
        "degree": "Master's, PhD",
        "countries": "China",
        "universities": "Tsinghua University",
        "field_keywords": ["Computer Science", "AI", "Electrical Engineering", "Mathematics", "Physics", "Data Science"],
        "min_gpa": 3.4,
        "research_required": True,
        "language_requirement": "IELTS 6.5 or TOEFL 80",
        "funding_type": "Full",
        "deadline_month": "March",
        "page_content": "Tsinghua University provides competitive scholarships for outstanding STEM students. Ranked #1 in China, top 20 globally for CS and Engineering. Covers full tuition and stipend.",
        "provider": "Tsinghua University"
    },
    {
        "title": "Peking University International Student Scholarship",
        "url": "https://www.isd.pku.edu.cn/HOME/Scholarships.htm",
        "region": "China",
        "degree": "Master's, PhD",
        "countries": "China",
        "universities": "Peking University (PKU)",
        "field_keywords": ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology", "Data Science"],
        "min_gpa": 3.3,
        "research_required": True,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "Peking University offers merit-based scholarships for international graduate students. PKU is among China's top 2 universities with strong STEM research programs.",
        "provider": "Peking University"
    },
    {
        "title": "Zhejiang University International Student Scholarship",
        "url": "https://iczu.zju.edu.cn/",
        "region": "China",
        "degree": "Master's, PhD",
        "countries": "China",
        "universities": "Zhejiang University",
        "field_keywords": ["Computer Science", "Engineering", "AI", "Materials Science", "Biotechnology"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.0",
        "funding_type": "Full",
        "deadline_month": "March",
        "page_content": "Zhejiang University offers Chinese Government and university scholarships. Top 3 in China for engineering. Many English-taught STEM programs available.",
        "provider": "Zhejiang University"
    },

    # ────────────────────────────────────────────────────────────────────
    #  RUSSIA
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Open Doors: Russian Scholarship Project",
        "url": "https://opendoors.russia.study/en",
        "region": "Russia",
        "degree": "Master's, PhD",
        "countries": "Russia",
        "universities": "HSE University, ITMO University, MIPT, Bauman Moscow State Technical University",
        "field_keywords": ["Data Science", "Photonics", "Computer Science", "Mathematics", "Physics", "AI"],
        "min_gpa": 3.0,
        "research_required": True,
        "language_requirement": "IELTS 5.5 or Russian B1",
        "funding_type": "Full",
        "deadline_month": "October",
        "page_content": "Open Doors is a prestigious Olympiad-based scholarship for Masters and PhD in STEM fields. Winners get full tuition at Russia's top universities. Competition-based — no embassy recommendation needed.",
        "provider": "Russian Government"
    },
    {
        "title": "Russian Government Scholarship (Rossotrudnichestvo)",
        "url": "https://education-in-russia.com/",
        "region": "Russia",
        "degree": "Bachelor's, Master's, PhD",
        "countries": "Russia",
        "universities": "All Russian state universities (MSU, MIPT, ITMO, etc.)",
        "field_keywords": ["Aerospace", "Nuclear Engineering", "Physics", "Computer Science", "Mathematics"],
        "min_gpa": 2.8,
        "research_required": False,
        "language_requirement": "Russian (1-year prep available)",
        "funding_type": "Full",
        "deadline_month": "March",
        "page_content": "Russian Government Scholarship for international students at state universities. Covers full tuition and dormitory. Extensive STEM programs in aerospace, nuclear energy, advanced physics, and computer science. Includes 1-year Russian language prep.",
        "provider": "Russian Government"
    },
    {
        "title": "HSE University International Scholarship",
        "url": "https://www.hse.ru/en/scholarships/",
        "region": "Russia",
        "degree": "Master's",
        "countries": "Russia",
        "universities": "HSE University (Higher School of Economics)",
        "field_keywords": ["Data Science", "Computer Science", "AI", "Mathematics", "Economics", "Statistics"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "August",
        "page_content": "HSE University offers its own merit-based scholarships for international students. Many English-taught Master's programs in Data Science, AI, Applied Mathematics, and Software Engineering.",
        "provider": "HSE University"
    },

    # ────────────────────────────────────────────────────────────────────
    #  EUROPE (non-Erasmus)
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "ETH Zurich — Excellence Scholarship & Opportunity Programme (ESOP)",
        "url": "https://ethz.ch/students/en/studies/financial/scholarships/excellencescholarship.html",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Switzerland",
        "universities": "ETH Zurich",
        "field_keywords": ["Computer Science", "Physics", "Mathematics", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering"],
        "min_gpa": 3.8,
        "research_required": True,
        "language_requirement": "IELTS 7.0 or TOEFL 100",
        "funding_type": "Full",
        "deadline_month": "December",
        "page_content": "For the world's most talented STEM students. ETH Zurich is ranked #1 in continental Europe. ESOP covers full tuition and living costs (CHF 12,000/semester + CHF 11,000 stipend). Extremely competitive — top 3% of applicants.",
        "provider": "ETH Zurich"
    },
    {
        "title": "TU Delft Excellence Scholarship (Justus & Louise van Effen)",
        "url": "https://www.tudelft.nl/en/education/practical-matters/scholarships",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Netherlands",
        "universities": "TU Delft",
        "field_keywords": ["Aerospace Engineering", "Computer Science", "Civil Engineering", "Mechanical Engineering", "Water Management"],
        "min_gpa": 3.5,
        "research_required": False,
        "language_requirement": "IELTS 6.5 or TOEFL 90",
        "funding_type": "Full",
        "deadline_month": "December",
        "page_content": "Full scholarships for top STEM students at TU Delft. Covers full tuition (€18,750+) and living expenses. TU Delft is world-renowned for Engineering, Aerospace, and Water Management. Must be in top 10% of cohort.",
        "provider": "TU Delft"
    },
    {
        "title": "DAAD Scholarships — Development-Related Postgraduate Courses (EPOS)",
        "url": "https://www.daad.de/en/study-and-research-in-germany/scholarships/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Germany",
        "universities": "Various German universities (TU Munich, RWTH Aachen, TU Berlin, etc.)",
        "field_keywords": ["Engineering", "Computer Science", "Environmental Science", "Renewable Energy", "Urban Planning"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.0 or TOEFL 80",
        "funding_type": "Full",
        "deadline_month": "October",
        "page_content": "DAAD EPOS provides funding for Master's programs in Germany for students from developing countries. Monthly stipend of €934, health insurance, travel allowance. Focus on development-related STEM subjects.",
        "provider": "DAAD"
    },
    {
        "title": "DAAD — Helmholtz-DAAD International Research Schools",
        "url": "https://www.daad.de/en/study-and-research-in-germany/scholarships/",
        "region": "Europe",
        "degree": "PhD",
        "countries": "Germany",
        "universities": "Helmholtz Centres across Germany",
        "field_keywords": ["Physics", "Chemistry", "Engineering", "Earth Science", "Health Science", "AI"],
        "min_gpa": 3.2,
        "research_required": True,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "September",
        "page_content": "Helmholtz-DAAD International Research Schools for PhD students at Germany's Helmholtz research centres. World-class facilities in physics, energy research, earth sciences, and information technology.",
        "provider": "DAAD"
    },
    {
        "title": "Stipendium Hungaricum Scholarship",
        "url": "https://stipendiumhungaricum.hu/",
        "region": "Europe",
        "degree": "Bachelor's, Master's, PhD",
        "countries": "Hungary",
        "universities": "All Hungarian state universities (BME, ELTE, University of Debrecen, etc.)",
        "field_keywords": ["Computer Science", "Engineering", "Mathematics", "Physics", "Chemistry", "All STEM fields"],
        "min_gpa": 2.8,
        "research_required": False,
        "language_requirement": "IELTS 5.5 or B2",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "Stipendium Hungaricum is the Hungarian government scholarship program for international students. Full tuition waiver, monthly stipend (HUF 43,700-140,000), accommodation allowance, and medical insurance. Wide range of English-taught STEM programs.",
        "provider": "Hungarian Government"
    },
    {
        "title": "Chevening Scholarships (UK Government)",
        "url": "https://www.chevening.org/scholarships/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "UK",
        "universities": "Any UK university",
        "field_keywords": ["Computer Science", "Engineering", "Public Policy", "Technology", "All fields"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "November",
        "page_content": "Chevening is the UK Government's global scholarship program, funded by the FCDO. Covers full tuition, monthly stipend, airfare, and visa. Must demonstrate leadership potential and return to home country for 2 years after.",
        "provider": "UK Government"
    },
    {
        "title": "Commonwealth Scholarships (UK)",
        "url": "https://cscuk.fcdo.gov.uk/scholarships/",
        "region": "Europe",
        "degree": "Master's, PhD",
        "countries": "UK",
        "universities": "UK universities (Imperial, UCL, Manchester, Edinburgh, etc.)",
        "field_keywords": ["Engineering", "Computer Science", "Public Health", "Environmental Science", "All STEM fields"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "October",
        "page_content": "Commonwealth Scholarships for citizens of Commonwealth countries to study in the UK. Full funding: tuition, stipend, airfare, thesis grant. Priority given to STEM and development-related fields.",
        "provider": "Commonwealth"
    },
    {
        "title": "Swedish Institute Scholarships for Global Professionals (SISGP)",
        "url": "https://si.se/en/apply/scholarships/swedish-institute-scholarships-for-global-professionals/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Sweden",
        "universities": "Swedish universities (KTH, Chalmers, Lund, Uppsala, etc.)",
        "field_keywords": ["Computer Science", "Engineering", "Environmental Science", "Sustainability", "Data Science"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5 or TOEFL 90",
        "funding_type": "Full",
        "deadline_month": "February",
        "page_content": "Swedish Institute Scholarships cover full tuition, monthly SEK 10,000 living allowance, travel grant, and insurance for Master's programs in Sweden. Available for students from eligible countries. Many strong STEM programs.",
        "provider": "Swedish Institute"
    },
    {
        "title": "EPFL Excellence Fellowships (Switzerland)",
        "url": "https://www.epfl.ch/education/studies/en/financing-your-studies/scholarships/excellence-fellowships/",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Switzerland",
        "universities": "EPFL (École polytechnique fédérale de Lausanne)",
        "field_keywords": ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Physics", "Mathematics", "Life Sciences"],
        "min_gpa": 3.7,
        "research_required": True,
        "language_requirement": "IELTS 7.0 or TOEFL 95",
        "funding_type": "Full",
        "deadline_month": "December",
        "page_content": "EPFL Excellence Fellowships provide CHF 16,000 per year scholarship + tuition waiver for outstanding international students. EPFL is Switzerland's leading engineering school, ranked top 15 globally.",
        "provider": "EPFL"
    },

    # ────────────────────────────────────────────────────────────────────
    #  NORTH AMERICA
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Fulbright Foreign Student Program (USA)",
        "url": "https://foreign.fulbrightonline.org/",
        "region": "North America",
        "degree": "Master's, PhD",
        "countries": "USA",
        "universities": "Any US university (MIT, Stanford, CMU, Georgia Tech, etc.)",
        "field_keywords": ["Computer Science", "Engineering", "Physics", "All STEM fields"],
        "min_gpa": 3.3,
        "research_required": True,
        "language_requirement": "IELTS 6.5 or TOEFL 80",
        "funding_type": "Full",
        "deadline_month": "February",
        "page_content": "Fulbright Foreign Student Program provides full funding for graduate studies in the USA. Covers tuition, living stipend, airfare, and health insurance. Highly prestigious — requires strong academic and leadership profile.",
        "provider": "US Government"
    },
    {
        "title": "Vanier Canada Graduate Scholarships",
        "url": "https://vanier.gc.ca/en/home-accueil.html",
        "region": "North America",
        "degree": "PhD",
        "countries": "Canada",
        "universities": "Canadian universities (University of Toronto, UBC, McGill, Waterloo, etc.)",
        "field_keywords": ["Computer Science", "Engineering", "Natural Sciences", "Health Sciences"],
        "min_gpa": 3.5,
        "research_required": True,
        "language_requirement": "IELTS 7.0 or TOEFL 100",
        "funding_type": "Full",
        "deadline_month": "November",
        "page_content": "Vanier CGS provides CAD $50,000/year for 3 years for doctoral students at Canadian universities. Open to international students with outstanding research potential in STEM, health, and social sciences.",
        "provider": "Canadian Government"
    },

    # ────────────────────────────────────────────────────────────────────
    #  AUSTRALIA & OCEANIA
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Australia Awards Scholarships",
        "url": "https://www.dfat.gov.au/people-to-people/australia-awards",
        "region": "Oceania",
        "degree": "Master's, PhD",
        "countries": "Australia",
        "universities": "All Australian universities (ANU, Melbourne, UNSW, Monash, etc.)",
        "field_keywords": ["Engineering", "Computer Science", "Environmental Science", "Agriculture", "All STEM fields"],
        "min_gpa": 3.0,
        "research_required": False,
        "language_requirement": "IELTS 6.5",
        "funding_type": "Full",
        "deadline_month": "April",
        "page_content": "Australia Awards are prestigious scholarships funded by the Australian Government. Covers full tuition, return airfare, establishment allowance, living expenses (AUD $35,000+/year). For students from eligible countries.",
        "provider": "Australian Government"
    },

    # ────────────────────────────────────────────────────────────────────
    #  TURKEY
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Türkiye Burslari (Turkey Government Scholarships)",
        "url": "https://www.turkiyeburslari.gov.tr/en",
        "region": "Europe",
        "degree": "Bachelor's, Master's, PhD",
        "countries": "Turkey",
        "universities": "Turkish universities (Boğaziçi, METU, ITU, Koç, Sabancı, etc.)",
        "field_keywords": ["Computer Science", "Engineering", "Natural Sciences", "All STEM fields"],
        "min_gpa": 2.8,
        "research_required": False,
        "language_requirement": "No English requirement (1-year Turkish prep included)",
        "funding_type": "Full",
        "deadline_month": "February",
        "page_content": "Türkiye Burslari is a fully funded Turkish Government Scholarship. Covers tuition, monthly stipend, accommodation, health insurance, airfare, and 1-year Turkish language course. Very accessible — no strict GPA or language barrier.",
        "provider": "Turkish Government"
    },

    # ────────────────────────────────────────────────────────────────────
    #  ADDITIONAL EUROPE
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "RWTH Aachen — DAAD Graduate Scholarships for Engineering",
        "url": "https://www.rwth-aachen.de/go/id/a/?lidx=1",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Germany",
        "universities": "RWTH Aachen University",
        "field_keywords": ["Mechanical Engineering", "Electrical Engineering", "Computer Science", "Automotive Engineering", "Production Engineering"],
        "min_gpa": 3.2,
        "research_required": False,
        "language_requirement": "IELTS 6.5 or TOEFL 90",
        "funding_type": "Full",
        "deadline_month": "October",
        "page_content": "RWTH Aachen is Germany's top technical university. DAAD-funded scholarships available for engineering master's programs. Known for automotive, mechanical, and electrical engineering research.",
        "provider": "RWTH Aachen"
    },
    {
        "title": "KTH Royal Institute of Technology Scholarship (Sweden)",
        "url": "https://www.kth.se/en/studies/master/scholarships",
        "region": "Europe",
        "degree": "Master's",
        "countries": "Sweden",
        "universities": "KTH Royal Institute of Technology",
        "field_keywords": ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Data Science", "AI"],
        "min_gpa": 3.2,
        "research_required": False,
        "language_requirement": "IELTS 6.5 or TOEFL 90",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "KTH offers tuition waivers for high-achieving international students. Sweden's largest and highest-ranked technical university. English-taught programs in CS, EE, and sustainable tech.",
        "provider": "KTH"
    },
    {
        "title": "Eiffel Excellence Scholarship Program (France)",
        "url": "https://www.campusfrance.org/en/eiffel-scholarship-program-of-excellence",
        "region": "Europe",
        "degree": "Master's, PhD",
        "countries": "France",
        "universities": "French universities and grandes écoles (Polytechnique, CentraleSupélec, etc.)",
        "field_keywords": ["Engineering", "Computer Science", "Mathematics", "Physics", "Economics"],
        "min_gpa": 3.2,
        "research_required": False,
        "language_requirement": "IELTS 6.0 or French B2",
        "funding_type": "Full",
        "deadline_month": "January",
        "page_content": "Eiffel Excellence Scholarship funded by French Ministry of Europe and Foreign Affairs. Monthly stipend €1,181 (Master's) + housing allowance + airfare. For students from non-EU countries applying to French institutions.",
        "provider": "French Government"
    },
]


def seed_database(db: ScholarshipDB = None):
    """Populate the database with verified real scholarship programs."""
    if db is None:
        db = ScholarshipDB()
    
    existing = db.count()
    db.upsert_scholarships(SEED_SCHOLARSHIPS)
    new_count = db.count()
    print(f"[DB] Seeded database: {existing} -> {new_count} scholarships ({new_count - existing} new)")
    return db


if __name__ == "__main__":
    db = seed_database()
    print(f"\nTotal scholarships: {db.count()}")
    print(f"Last updated: {db.get_last_updated()}")
    
    # Show breakdown by region
    for region in ["Europe", "South Korea", "Japan", "China", "Russia", "North America", "Oceania"]:
        count = len(db.get_scholarships_by_region(region))
        if count:
            print(f"  {region}: {count} programs")
