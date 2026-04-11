# """
# Consolidates functions from various notebooks for Pylance compatibility.
# This module imports and re-exports all functions used in main.ipynb.
# """

# from pydantic import BaseModel
# from typing import List, Optional
# from langchain_core.documents import Document
# import json
# import os
# import time
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# from langchain_core.vectorstores import VectorStore
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings


# # ==================== Profile ====================
# class UserProfile(BaseModel):
#     """User profile for scholarship matching"""
#     name: Optional[str] = None
#     cgpa: float
#     cgpa_scale: float
#     degree: str
#     field: str
#     research_papers: int
#     experience_years: float
#     internships: int
#     extracurriculars: List[str]
#     english_test: Optional[str] = None
#     english_score: Optional[float] = None
#     target_degree: str


# class userProfile(BaseModel):
#     """Alias for UserProfile (lowercase version used in notebooks)"""
#     name: Optional[str] = None
#     cgpa: float
#     cgpa_scale: float
#     degree: str
#     field: str
#     research_papers: int
#     experience_years: float
#     internships: int
#     extracurriculars: List[str]
#     english_test: Optional[str] = None
#     english_score: Optional[float] = None
#     target_degree: str


# # ==================== Scraping ====================
# def fetch_programs():
#     """Fetch Erasmus Mundus programs from official website"""
#     chrome_options = Options()
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--headless")

#     driver = webdriver.Chrome(
#         service=Service(ChromeDriverManager().install()),
#         options=chrome_options
#     )

#     try:
#         driver.get("https://www.eacea.ec.europa.eu/scholarships/erasmus-mundus-catalogue_en")
#         wait = WebDriverWait(driver, 30)

#         # Scroll to trigger lazy loading
#         time.sleep(3)
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(3)

#         # Step 1: Collect all unique URLs first
#         link_elements = wait.until(
#             EC.presence_of_all_elements_located(
#                 (By.XPATH, "//a[contains(@href,'erasmus-plus.ec.europa.eu/projects')]")
#             )
#         )

#         urls = []
#         seen_urls = set()
#         for link in link_elements:
#             href = link.get_attribute("href")
#             if href and href not in seen_urls:
#                 urls.append(href)
#                 seen_urls.add(href)

#         programs = []

#         # Step 2: Visit each URL separately
#         for url in urls:
#             driver.get(url)
#             time.sleep(2)

#             try:
#                 title = driver.find_element(By.TAG_NAME, "h1").text.strip()
#             except:
#                 title = "Erasmus Mundus Programme"

#             # Extract details
#             try:
#                 countries = ", ".join([c.text.strip() for c in driver.find_elements(By.CSS_SELECTOR, ".programme-country")])
#             except:
#                 countries = ""
#             try:
#                 universities = ", ".join([u.text.strip() for u in driver.find_elements(By.CSS_SELECTOR, ".programme-university")])
#             except:
#                 universities = ""
#             try:
#                 degree = driver.find_element(By.CSS_SELECTOR, ".programme-degree").text.strip()
#             except:
#                 degree = ""
#             try:
#                 duration = driver.find_element(By.CSS_SELECTOR, ".programme-duration").text.strip()
#             except:
#                 duration = ""

#             programs.append({
#                 "title": title,
#                 "url": url,
#                 "region": "Europe",
#                 "countries": countries,
#                 "universities": universities,
#                 "degree": degree,
#                 "duration": duration,
#                 "text": title  # For ingestion
#             })

#         return programs

#     finally:
#         driver.quit()


# def fetch_erasmus_programs():
#     """Alias for fetch_programs"""
#     return fetch_programs()


# # ==================== Ingestion ====================
# def ingest(programs, out_path="data/processed/scholarships.json"):
#     """Ingest programs and convert to LangChain Documents"""
#     docs = []

#     for p in programs:
#         docs.append(
#             Document(
#                 page_content=p.get("text", p.get("title", "")),
#                 metadata={
#                     "region": p.get("region", ""),
#                     "title": p.get("title", "")
#                 }
#             )
#         )

#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump([d.model_dump() if hasattr(d, 'model_dump') else d.dict() for d in docs], f, indent=2)

#     return docs


# # ==================== Vectorstore ====================
# def build_vectorstore(docs) -> VectorStore:
#     """Build vectorstore from documents"""
#     from langchain_community.vectorstores import FAISS
    
#     # Split documents
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     split_docs = text_splitter.split_documents(docs)
    
#     # Create embeddings
#     embeddings = HuggingFaceEmbeddings()
    
#     # Build vectorstore
#     vectorstore = FAISS.from_documents(split_docs, embeddings)
    
#     return vectorstore


# # ==================== Reasoning ====================
# def build_reasoning_chain(llm):
#     """Build reasoning chain for analyzing scholarship fit"""
#     from langchain.prompts import PromptTemplate
#     from langchain.chains import LLMChain
    
#     prompt_template = """
#     Based on the student profile and scholarship information, assess the fit:
    
#     Student Profile:
#     {profile}
    
#     Scholarship:
#     {scholarship}
    
#     Provide a brief analysis of whether this scholarship is a good fit and why.
#     """
    
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["profile", "scholarship"]
#     )
    
#     chain = LLMChain(llm=llm, prompt=prompt)
#     return chain


# # ==================== Analysis ====================
# def analyze(profile, vectorstore, reasoning_chain):
#     """Analyze scholarships and provide recommendations"""
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
#     query = f""""
#     Field: {profile.field}
#     Target degree: {profile.target_degree}
#     CGPA: {profile.cgpa}/{profile.cgpa_scale}
#     Research papers: {profile.research_papers}
#     """
#     docs = retriever.invoke(query)

#     results = []

#     for d in docs:
#         response = reasoning_chain.invoke(
#             {
#                 "profile": profile.model_dump_json() if hasattr(profile, 'model_dump_json') else profile.json(),
#                 "scholarship": d.page_content
#             }
#         )

#         results.append(
#             {
#                 "title": d.metadata.get("title"),
#                 "region": d.metadata.get("region"),
#                 "analysis": response.get("text", str(response))
#             }
#         )

#     return results


# __all__ = [
#     "UserProfile",
#     "userProfile",
#     "fetch_programs",
#     "fetch_erasmus_programs",
#     "ingest",
#     "build_vectorstore",
#     "build_reasoning_chain",
#     "analyze",
# ]
