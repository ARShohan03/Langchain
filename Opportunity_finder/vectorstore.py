"""Vector store module using scikit-learn for memory-efficient text matching."""

from typing import List
from langchain_core.documents import Document
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    """Simple and stable retriever using TF-IDF and Cosine Similarity."""
    
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Prepare content for indexing
        texts = [doc.page_content for doc in docs]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def invoke(self, query: str, k: int = 10) -> List[Document]:
        """Search for top k documents matching the query."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Filter out zero similarity results if possible, but keep at least k/2 for reasoning
        results = [self.docs[i] for i in top_indices]
        return results

    def as_retriever(self, search_kwargs: dict = None):
        """Mock LangChain as_retriever interface."""
        k = search_kwargs.get("k", 10) if search_kwargs else 10
        
        class MockRetriever:
            def __init__(self, parent, k):
                self.parent = parent
                self.k = k
            def invoke(self, query):
                return self.parent.invoke(query, k=self.k)
        
        return MockRetriever(self, k)

def build_vectorstore(docs: List[Document]) -> TfidfRetriever:
    """Build a stable TF-IDF retriever."""
    return TfidfRetriever(docs)
