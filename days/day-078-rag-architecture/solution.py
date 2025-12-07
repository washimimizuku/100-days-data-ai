"""Day 78: RAG Architecture - Solutions
NOTE: Requires sentence-transformers and ollama
pip install sentence-transformers ollama numpy"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Sample documents for testing
SAMPLE_DOCS = [
    "Python is a high-level, interpreted programming language known for its simplicity and readability.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "RAG (Retrieval-Augmented Generation) combines information retrieval with language model generation.",
    "Vector databases store embeddings and enable fast similarity search for semantic retrieval.",
    "LangChain is a framework for building applications with large language models.",
    "Transformers are neural network architectures that use self-attention mechanisms.",
    "Fine-tuning adapts pre-trained models to specific tasks by training on domain data.",
    "Prompt engineering involves crafting effective inputs to guide LLM behavior.",
]


# Exercise 1: Basic RAG System
class BasicRAG:
    """Simple RAG system with indexing, retrieval, and generation."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def index_documents(self, documents: List[str]):
        self.documents = documents
        self.embeddings = self.embedding_model.encode(documents)
        print(f"Indexed {len(documents)} documents")
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if self.embeddings is None:
            return []
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_indices]
    
    def generate(self, query: str, k: int = 3, model: str = 'mistral') -> Dict:
        context_docs = self.retrieve(query, k)
        if not context_docs:
            return {"answer": "No relevant documents found.", "sources": []}
        
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)])
        prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {query}

Answer (be concise and cite sources using [1], [2], etc.):"""
        
        try:
            response = ollama.generate(model=model, prompt=prompt, options={'temperature': 0.3, 'num_predict': 150})
            return {'answer': response['response'].strip(), 'sources': context_docs}
        except Exception as e:
            return {'answer': f"Error generating response: {e}", 'sources': context_docs}


def exercise_1_basic_rag():
    print("Exercise 1: Basic RAG System\n" + "-" * 40)
    rag = BasicRAG()
    rag.index_documents(SAMPLE_DOCS)
    
    queries = ["What is RAG?", "Explain machine learning", "What is Python?"]
    for query in queries:
        print(f"\nQuery: {query}")
        result = rag.generate(query, k=2)
        print(f"Answer: {result['answer'][:150]}...\nSources: {len(result['sources'])} documents")


# Exercise 2: Hybrid Retrieval
class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def index_documents(self, documents: List[str]):
        self.documents = documents
        self.embeddings = self.embedding_model.encode(documents)
    
    def semantic_search(self, query: str) -> np.ndarray:
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = np.dot(self.embeddings, query_embedding)
        return (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-10)
    
    def keyword_search(self, query: str) -> np.ndarray:
        query_terms = set(query.lower().split())
        scores = []
        for doc in self.documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / len(query_terms) if query_terms else 0
            scores.append(score)
        scores = np.array(scores)
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores
    
    def hybrid_search(self, query: str, k: int = 3, alpha: float = 0.7) -> List[Tuple[str, float]]:
        semantic_scores = self.semantic_search(query)
        keyword_scores = self.keyword_search(query)
        combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        return [(self.documents[i], combined_scores[i]) for i in top_indices]


def exercise_2_hybrid_retrieval():
    print("\nExercise 2: Hybrid Retrieval\n" + "-" * 40)
    retriever = HybridRetriever()
    retriever.index_documents(SAMPLE_DOCS)
    query = "language model generation"
    print(f"\nQuery: {query}\n\nHybrid Results (alpha=0.7):")
    results = retriever.hybrid_search(query, k=3, alpha=0.7)
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}\n   {doc[:80]}...")


# Exercise 3: Query Rewriting
class QueryRewriter:
    """Rewrite queries for better retrieval."""
    
    def __init__(self, model: str = 'mistral'):
        self.model = model
    
    def rewrite(self, query: str) -> str:
        prompt = f"""Rewrite this query to be more specific and include related technical terms.
Keep it concise (one sentence).

Original query: {query}

Rewritten query:"""
        try:
            response = ollama.generate(model=self.model, prompt=prompt, options={'temperature': 0.5, 'num_predict': 50})
            return response['response'].strip()
        except Exception as e:
            return query
    
    def expand(self, query: str) -> List[str]:
        variations = [query]
        rewritten = self.rewrite(query)
        if rewritten != query:
            variations.append(rewritten)
        if not query.endswith('?'):
            variations.append(f"What is {query}?")
        return variations


def exercise_3_query_rewriting():
    print("\nExercise 3: Query Rewriting\n" + "-" * 40)
    rewriter = QueryRewriter()
    queries = ["RAG", "ML models", "vector search"]
    for query in queries:
        print(f"\nOriginal: {query}")
        print(f"Rewritten: {rewriter.rewrite(query)}")


# Exercise 4: Reranking
class Reranker:
    """Rerank retrieved documents by relevance."""
    
    def __init__(self, model: str = 'mistral'):
        self.model = model
    
    def score_relevance(self, query: str, document: str) -> float:
        prompt = f"""Rate how relevant this document is to the query on a scale of 0-10.
Respond with only a number.

Query: {query}
Document: {document}

Relevance score (0-10):"""
        try:
            response = ollama.generate(model=self.model, prompt=prompt, options={'temperature': 0.1, 'num_predict': 5})
            score_text = response['response'].strip().split()[0]
            score = float(score_text)
            return max(0, min(10, score))
        except:
            return 5.0
    
    def rerank(self, query: str, documents: List[str], threshold: float = 5.0) -> List[Tuple[str, float]]:
        scored_docs = []
        for doc in documents:
            score = self.score_relevance(query, doc)
            if score >= threshold:
                scored_docs.append((doc, score))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs


def exercise_4_reranking():
    print("\nExercise 4: Reranking Pipeline\n" + "-" * 40)
    rag = BasicRAG()
    rag.index_documents(SAMPLE_DOCS)
    query = "How do language models work?"
    initial_docs = rag.retrieve(query, k=4)
    print(f"\nQuery: {query}\n\nInitial retrieval: {len(initial_docs)} documents")
    reranker = Reranker()
    reranked = reranker.rerank(query, initial_docs, threshold=6.0)
    print(f"\nAfter reranking (threshold=6.0): {len(reranked)} documents")
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"{i}. Score: {score:.1f}\n   {doc[:80]}...")


# Exercise 5: Citation System
class CitationRAG(BasicRAG):
    """RAG system with automatic citations."""
    
    def generate_with_citations(self, query: str, k: int = 3) -> Dict:
        context_docs = self.retrieve(query, k)
        if not context_docs:
            return {"answer": "No relevant documents found.", "sources": [], "citations": []}
        
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)])
        prompt = f"""Answer the question based on the context below. 
Add citations [1], [2], etc. after each claim to indicate which source supports it.

Context:
{context}

Question: {query}

Answer with citations:"""
        
        try:
            response = ollama.generate(model='mistral', prompt=prompt, options={'temperature': 0.3, 'num_predict': 200})
            answer = response['response'].strip()
            sources = [f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)]
            return {'answer': answer, 'sources': sources, 'num_citations': answer.count('[')}
        except Exception as e:
            return {'answer': f"Error: {e}", 'sources': [], 'num_citations': 0}


def exercise_5_citation_system():
    print("\nExercise 5: Citation System\n" + "-" * 40)
    rag = CitationRAG()
    rag.index_documents(SAMPLE_DOCS)
    query = "What is the relationship between RAG and language models?"
    print(f"\nQuery: {query}")
    result = rag.generate_with_citations(query, k=3)
    print(f"\nAnswer:\n{result['answer']}\n\nSources:")
    for source in result['sources']:
        print(f"  {source}")
    print(f"\nCitations found: {result['num_citations']}")


if __name__ == "__main__":
    print("Day 78: RAG Architecture - Solutions\n" + "=" * 60)
    try:
        exercise_1_basic_rag()
        exercise_2_hybrid_retrieval()
        exercise_3_query_rewriting()
        exercise_4_reranking()
        exercise_5_citation_system()
        print("\n" + "=" * 60 + "\nAll exercises completed!")
    except Exception as e:
        print(f"\nError: {e}\n\nMake sure you have installed:\npip install sentence-transformers ollama numpy\n\nAnd Ollama is running with mistral model:\nollama pull mistral")
