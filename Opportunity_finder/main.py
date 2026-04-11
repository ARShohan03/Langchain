"""Main entry point for the Opportunity Finder scholarship matching system."""

# from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub

from models import UserProfile
from scrapers import fetch_programs
from ingestion import ingest
from vectorstore import build_vectorstore
from reasoning import build_reasoning_chain
from engine import analyze


def main():
    """Run the scholarship matching pipeline."""
    
    # Define user profile
    profile = UserProfile(
        name="Md Abdur Rahaman",
        cgpa=3.60,
        cgpa_scale=4.0,
        degree="BSc",
        field="Computer Science",
        research_papers=0,
        experience_years=0,
        internships=0,
        extracurriculars=["AI research"],
        english_test="IELTS",
        english_score=6.5,
        target_degree="MSc"
    )

    # Fetch programs
    print("Fetching programs...")
    programs = fetch_programs()
    print(f"Found {len(programs)} programs")

    # Ingest programs
    print("Processing programs...")
    docs = ingest(programs)
    print(f"Created {len(docs)} documents")

    # Build vectorstore
    print("Building vectorstore...")
    vectorstore = build_vectorstore(docs)

    # Initialize LLM

    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )


    # Build reasoning chain
    reasoning_chain = build_reasoning_chain(llm)

    # Analyze scholarships
    print("Analyzing scholarships...")
    results = analyze(profile, vectorstore, reasoning_chain)

    # Display results
    print("\n" + "="*60)
    print("SCHOLARSHIP RECOMMENDATIONS")
    print("="*60)
    for r in results:
        print(f"\n[{r['region']}] {r['title']}")
        print("-" * 60)
        print(r['analysis'])

    return results


if __name__ == "__main__":
    main()
