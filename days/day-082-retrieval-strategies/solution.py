"""
Day 82: Retrieval Strategies - Solutions
"""

from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import ollama


# Sample documents
SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Python is a popular programming language for data science",
    "Natural language processing enables computers to understand text",
    "Computer vision allows machines to interpret images",
    "Reinforcement learning trains agents through rewards",
    "Data preprocessing is crucial for model performance",
    "Neural networks are inspired by biological neurons"
]


# Exercise 1: Hybrid Retriever
class HybridRetriever:
    """Hybrid retrieval combining dense and sparse."""
    
    def __init__(self, documents: List[str], alpha: float = 0.7):
        self.documents = documents
        self.alpha = alpha
        
        # Dense retrieval
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embeddings = self.model.encode(documents)
        
        # Normalize embeddings
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        self.doc_embeddings = self.doc_embeddings / (norms + 1e-10)
        
        # Sparse retrieval
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Hybrid retrieval."""
        # Dense scores
        query_emb = self.model.encode(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        dense_scores = np.dot(self.doc_embeddings, query_emb)
        
        # Normalize to 0-1
        dense_scores = (dense_scores - dense_scores.min()) / \
                      (dense_scores.max() - dense_scores.min() + 1e-10)
        
        # Sparse scores
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize to 0-1
        if sparse_scores.max() > 0:
            sparse_scores = (sparse_scores - sparse_scores.min()) / \
                           (sparse_scores.max() - sparse_scores.min() + 1e-10)
        
        # Combine
        combined = self.alpha * dense_scores + (1 - self.alpha) * sparse_scores
        
        # Top-k
        top_indices = np.argsort(combined)[-k:][::-1]
        
        return [
            {
                'document': self.documents[idx],
                'score': float(combined[idx]),
                'dense_score': float(dense_scores[idx]),
                'sparse_score': float(sparse_scores[idx])
            }
            for idx in top_indices
        ]


def exercise_1_hybrid_retriever():
    """Exercise 1: Hybrid Retriever"""
    print("Exercise 1: Hybrid Retriever")
    print("-" * 40)
    
    retriever = HybridRetriever(SAMPLE_DOCS, alpha=0.7)
    
    query = "neural network learning"
    print(f"\nQuery: {query}")
    print("\nTop 3 Results:")
    
    results = retriever.retrieve(query, k=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Dense: {result['dense_score']:.3f}, Sparse: {result['sparse_score']:.3f}")
        print(f"   {result['document']}")


# Exercise 2: Query Expander
class QueryExpander:
    """Expand queries for better retrieval."""
    
    def __init__(self):
        pass
    
    def expand_with_llm(self, query: str) -> str:
        """Expand query using LLM."""
        prompt = f"""Expand this search query by adding related terms. Keep it concise.

Original: {query}

Expanded:"""
        
        try:
            response = ollama.generate(
                model='mistral',
                prompt=prompt,
                options={'temperature': 0.5, 'num_predict': 50}
            )
            return response['response'].strip()
        except:
            return query
    
    def expand_with_prf(self, query: str, retriever, top_k: int = 3) -> str:
        """Expand using pseudo-relevance feedback."""
        # Get top results
        results = retriever.retrieve(query, k=top_k)
        
        # Extract key terms (simplified)
        top_docs = [r['document'] for r in results]
        all_text = ' '.join(top_docs)
        
        # Add to query
        words = all_text.lower().split()
        # Get unique words not in original query
        query_words = set(query.lower().split())
        new_words = [w for w in words if w not in query_words][:5]
        
        expanded = query + ' ' + ' '.join(new_words)
        return expanded


def exercise_2_query_expander():
    """Exercise 2: Query Expander"""
    print("\nExercise 2: Query Expander")
    print("-" * 40)
    
    expander = QueryExpander()
    retriever = HybridRetriever(SAMPLE_DOCS)
    
    query = "ML"
    
    print(f"\nOriginal query: {query}")
    
    # LLM expansion
    llm_expanded = expander.expand_with_llm(query)
    print(f"\nLLM expanded: {llm_expanded}")
    
    # PRF expansion
    prf_expanded = expander.expand_with_prf(query, retriever)
    print(f"\nPRF expanded: {prf_expanded}")


# Exercise 3: Reranker
class Reranker:
    """Rerank retrieved documents."""
    
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """Rerank documents by relevance."""
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'document': doc, 'score': float(score)}
            for doc, score in ranked[:top_k]
        ]


def exercise_3_reranker():
    """Exercise 3: Reranker"""
    print("\nExercise 3: Reranker")
    print("-" * 40)
    
    retriever = HybridRetriever(SAMPLE_DOCS)
    reranker = Reranker()
    
    query = "how do neural networks work"
    
    # Initial retrieval
    print(f"\nQuery: {query}")
    print("\nInitial retrieval (top 5):")
    initial = retriever.retrieve(query, k=5)
    for i, result in enumerate(initial, 1):
        print(f"{i}. {result['document'][:60]}...")
    
    # Rerank
    docs = [r['document'] for r in initial]
    reranked = reranker.rerank(query, docs, top_k=5)
    
    print("\nAfter reranking:")
    for i, result in enumerate(reranked, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   {result['document'][:60]}...")


# Exercise 4: Multi-Stage Pipeline
class AdvancedRetriever:
    """Multi-stage retrieval pipeline."""
    
    def __init__(self, documents: List[str]):
        self.hybrid = HybridRetriever(documents)
        self.expander = QueryExpander()
        self.reranker = Reranker()
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Full retrieval pipeline."""
        print(f"\nStage 1: Query expansion")
        expanded = self.expander.expand_with_llm(query)
        print(f"  Expanded: {expanded[:80]}...")
        
        print(f"\nStage 2: Hybrid retrieval")
        candidates = self.hybrid.retrieve(expanded, k=10)
        print(f"  Retrieved {len(candidates)} candidates")
        
        print(f"\nStage 3: Reranking")
        docs = [c['document'] for c in candidates]
        reranked = self.reranker.rerank(query, docs, top_k=k)
        print(f"  Reranked to top {k}")
        
        return reranked


def exercise_4_multi_stage_pipeline():
    """Exercise 4: Multi-Stage Pipeline"""
    print("\nExercise 4: Multi-Stage Pipeline")
    print("-" * 40)
    
    retriever = AdvancedRetriever(SAMPLE_DOCS)
    
    query = "deep learning"
    print(f"\nQuery: {query}")
    
    results = retriever.retrieve(query, k=3)
    
    print("\nFinal Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   {result['document']}")


# Exercise 5: Retrieval Evaluator
class RetrievalEvaluator:
    """Evaluate retrieval quality."""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate precision@k."""
        top_k = retrieved[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant))
        return relevant_in_top_k / k if k > 0 else 0
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate recall@k."""
        top_k = retrieved[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant))
        return relevant_in_top_k / len(relevant) if relevant else 0
    
    @staticmethod
    def mrr(retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    reciprocal_ranks.append(1 / i)
                    break
            else:
                reciprocal_ranks.append(0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    def evaluate(self, retrieved: List[str], relevant: List[str], k_values: List[int]) -> Dict:
        """Comprehensive evaluation."""
        results = {}
        
        for k in k_values:
            results[f'precision@{k}'] = self.precision_at_k(retrieved, relevant, k)
            results[f'recall@{k}'] = self.recall_at_k(retrieved, relevant, k)
        
        return results


def exercise_5_retrieval_evaluator():
    """Exercise 5: Retrieval Evaluator"""
    print("\nExercise 5: Retrieval Evaluator")
    print("-" * 40)
    
    evaluator = RetrievalEvaluator()
    
    # Simulate retrieval results
    retrieved = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Deep learning uses neural networks with multiple layers",
        "Computer vision allows machines to interpret images",
        "Natural language processing enables computers to understand text"
    ]
    
    # Ground truth relevant documents
    relevant = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Neural networks are inspired by biological neurons"
    ]
    
    # Evaluate
    results = evaluator.evaluate(retrieved, relevant, k_values=[1, 3, 5])
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.3f}")
    
    # MRR example
    retrieved_lists = [retrieved]
    relevant_lists = [relevant]
    mrr_score = evaluator.mrr(retrieved_lists, relevant_lists)
    print(f"  MRR: {mrr_score:.3f}")


if __name__ == "__main__":
    print("Day 82: Retrieval Strategies - Solutions\n")
    print("=" * 60)
    
    try:
        exercise_1_hybrid_retriever()
        exercise_2_query_expander()
        exercise_3_reranker()
        exercise_4_multi_stage_pipeline()
        exercise_5_retrieval_evaluator()
        
        print("\n" + "=" * 60)
        print("All exercises completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed:")
        print("pip install sentence-transformers rank-bm25 ollama")
