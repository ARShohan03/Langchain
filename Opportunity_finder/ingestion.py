"""Data ingestion module for processing programs into documents."""

import json
import os
import re
from typing import List, Dict, Any
from langchain_core.documents import Document


def ingest(programs: List[Dict[str, Any]], out_path: str = "data/processed/scholarships.json") -> List[Document]:
    """
    Ingest programs and convert to LangChain Documents.

    Args:
        programs: List of program dictionaries
        out_path: Output path for processed scholarships JSON

    Returns:
        List of LangChain Document objects
    """
    docs = []

    for p in programs:
        title = p.get("title", "Unknown Program")
        url = p.get("url", "")
        region = p.get("region", "Europe")
        degree = p.get("degree", "Master's")
        countries = p.get("countries", "")
        universities = p.get("universities", "")

        # Build rich text content for semantic search
        text_parts = [title]
        if degree:
            text_parts.append(f"Degree: {degree}")
        if countries:
            text_parts.append(f"Countries: {countries}")
        if universities:
            text_parts.append(f"Universities: {universities}")
        if url:
            text_parts.append(f"URL: {url}")

        page_content = ". ".join(text_parts)

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "region": region,
                    "title": title,
                    "url": url,
                    "degree": degree,
                    "countries": countries,
                    "universities": universities,
                }
            )
        )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [d.model_dump() if hasattr(d, 'model_dump') else d.dict() for d in docs],
            f,
            indent=2
        )

    return docs


def load_docs_from_json(json_path: str = "data/processed/scholarships.json") -> List[Document]:
    """
    Load previously ingested documents from JSON.

    Args:
        json_path: Path to the processed scholarships JSON

    Returns:
        List of LangChain Document objects
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        metadata = item.get("metadata", {})
        page_content = item.get("page_content", "")

        # Extract URL from page_content if not in metadata (legacy format)
        if not metadata.get("url"):
            url_match = re.search(r'https?://\S+', page_content)
            if url_match:
                metadata["url"] = url_match.group(0)

        docs.append(Document(
            page_content=page_content,
            metadata=metadata
        ))

    return docs
