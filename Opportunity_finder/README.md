# Opportunity Finder - Scholarship Matching System

A RAG (Retrieval Augmented Generation) based scholarship matching system that analyzes student profiles and recommends suitable Erasmus Mundus programs.

## Project Structure

```
opportunity_finder/
├── __init__.py                  # Package initialization
├── models.py                    # User profile data models
├── scrapers.py                  # Web scraping for programs
├── ingestion.py                 # Document processing
├── vectorstore.py               # Vector database setup
├── reasoning.py                 # LLM reasoning chain
├── engine.py                    # Analysis engine
├── main.py                      # Command-line entry point
├── main.ipynb                   # Jupyter notebook (interactive)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Module Descriptions

### `models.py`
Defines the `UserProfile` Pydantic model that represents a student's academic and personal information:
- CGPA, degree, field of study
- Research papers, internships, extracurriculars
- Language test scores
- Target degree level

### `scrapers.py`
Handles web scraping of Erasmus Mundus programs:
- Uses Selenium for JavaScript-heavy websites
- Fetches program details (title, URL, countries, universities, degree type)
- Functions: `fetch_programs()`, `fetch_erasmus_programs()`

### `ingestion.py`
Converts programs into LangChain Documents:
- Structures data for vector storage
- Exports to JSON for persistence
- Function: `ingest(programs)`

### `vectorstore.py`
Creates semantic search capabilities:
- Uses HuggingFace embeddings (sentence-transformers)
- Implements FAISS for fast similarity search
- Function: `build_vectorstore(docs)`

### `reasoning.py`
Sets up the LLM reasoning chain:
- Defines prompt template for scholarship analysis
- Evaluates eligibility and competitiveness
- Function: `build_reasoning_chain(llm)`

### `engine.py`
Main analysis logic:
- Retrieves relevant scholarships
- Runs LLM analysis on each match
- Generates personalized recommendations
- Function: `analyze(profile, vectorstore, reasoning_chain)`

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install ChromeDriver** (for web scraping):
   ```bash
   # Automatically handled by webdriver-manager, but ensure Chrome is installed
   ```

## Usage

### As a Python Module

```python
from models import UserProfile
from scrapers import fetch_programs
from ingestion import ingest
from vectorstore import build_vectorstore
from reasoning import build_reasoning_chain
from engine import analyze
from langchain_huggingface import HuggingFaceEndpoint

# Define profile
profile = UserProfile(
    cgpa=3.75,
    cgpa_scale=4.0,
    degree="BSc",
    field="Computer Science",
    research_papers=1,
    experience_years=0.5,
    internships=2,
    extracurriculars=["AI research", "Contest programming"],
    english_test="IELTS",
    english_score=7.5,
    target_degree="MSc"
)

# Run pipeline
programs = fetch_programs()
docs = ingest(programs)
vectorstore = build_vectorstore(docs)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.2,
    max_new_tokens=512
)

reasoning_chain = build_reasoning_chain(llm)
results = analyze(profile, vectorstore, reasoning_chain)

for r in results:
    print(f"[{r['region']}] {r['title']}\n{r['analysis']}")
```

### From Command Line

```bash
python main.py
```

### In Jupyter Notebook

Open `main.ipynb` and run cells sequentially. The notebook provides interactive exploration and visualization capabilities.

## Dependencies

- **LangChain**: LLM orchestration framework
- **Selenium**: Web automation for scraping
- **FAISS**: Vector similarity search
- **HuggingFace**: Embeddings and LLM endpoints
- **Pydantic**: Data validation
- **WebDriver Manager**: Chrome driver management

See `requirements.txt` for complete list.

## Features

✅ Semantic search on scholarship data  
✅ LLM-powered eligibility analysis  
✅ Personalized recommendations  
✅ Multi-field scholarship matching  
✅ Competitive scoring  
✅ Production-ready module structure  

## Future Improvements

- [ ] Add more scholarship sources
- [ ] Implement caching for embeddings
- [ ] Add unit tests
- [ ] Create web API (FastAPI)
- [ ] Add database for storing results
- [ ] Implement user authentication
- [ ] Add visualization dashboard

## License

MIT
