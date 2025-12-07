# Day 82: Retrieval Strategies

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Master advanced retrieval techniques for RAG systems
- Implement hybrid search combining dense and sparse retrieval
- Learn query expansion and rewriting strategies
- Build multi-stage retrieval pipelines with reranking
- Optimize retrieval quality and performance

---

## Retrieval Fundamentals

**Goal**: Find most relevant documents for a query from large corpus.

**Challenges**: Vocabulary mismatch, ambiguous queries, mixed topics, precision/recall balance, speed/accuracy trade-offs.

**Pipeline**: Query → Retrieval (Stage 1) → Reranking (Stage 2) → Context → LLM → Response

---

## Dense vs Sparse Retrieval

```python
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

# Dense Retrieval (Semantic) - captures meaning, works with paraphrases
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = ["ML is AI", "Python programming", "Data science"]
doc_embeddings = model.encode(documents)
query_embedding = model.encode("machine learning")
similarities = np.dot(doc_embeddings, query_embedding)
top_k = np.argsort(similarities)[-3:][::-1]

# Sparse Retrieval (BM25) - fast, exact keywords, no embedding needed
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
scores = bm25.get_scores("machine learning".lower().split())
top_k = np.argsort(scores)[-3:][::-1]
```

---

## Hybrid Search

### Combining Dense and Sparse

**Strategy**: Combine semantic and keyword search for best of both worlds.

```python
class HybridRetriever:
    """Hybrid retrieval combining dense and sparse."""
    
    def __init__(self, documents, alpha=0.7):
        self.documents = documents
        self.alpha = alpha  # Weight for dense retrieval
        
        # Dense retrieval
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embeddings = self.model.encode(documents)
        
        # Sparse retrieval
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def retrieve(self, query, k=5):
        """Hybrid retrieval."""
        # Dense scores
        query_emb = self.model.encode(query)
        dense_scores = np.dot(self.doc_embeddings, query_emb)
        dense_scores = (dense_scores - dense_scores.min()) / \
                      (dense_scores.max() - dense_scores.min() + 1e-10)
        
        # Sparse scores
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_scores = (sparse_scores - sparse_scores.min()) / \
                       (sparse_scores.max() - sparse_scores.min() + 1e-10)
        
        # Combine
        combined = self.alpha * dense_scores + (1 - self.alpha) * sparse_scores
        
        # Top-k
        top_indices = np.argsort(combined)[-k:][::-1]
        
        return [
            {
                'document': self.documents[idx],
                'score': combined[idx],
                'dense_score': dense_scores[idx],
                'sparse_score': sparse_scores[idx]
            }
            for idx in top_indices
        ]
```

**Alpha Parameter**:
- α = 1.0: Pure dense (semantic)
- α = 0.0: Pure sparse (keyword)
- α = 0.7: Balanced (typical)

---

## Query Expansion

**Why**: Short queries lack context ("ML" → "machine learning algorithms and techniques")

```python
# 1. Synonym Expansion
from nltk.corpus import wordnet

def expand_with_synonyms(query):
    words = query.split()
    expanded = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return ' '.join(expanded)

# 2. LLM-Based Expansion
import ollama

def expand_with_llm(query):
    prompt = f"Expand this search query with related terms: {query}\nExpanded:"
    response = ollama.generate(model='mistral', prompt=prompt, 
                               options={'temperature': 0.5, 'num_predict': 50})
    return response['response'].strip()

# 3. Pseudo-Relevance Feedback
def expand_with_prf(query, retriever, top_k=3):
    results = retriever.retrieve(query, k=top_k)
    top_docs = [r['document'] for r in results]
    return query + ' ' + ' '.join(top_docs)[:100]
```

---

## Query Rewriting & Multi-Query

```python
# Query Rewriting - clarify ambiguous queries
def rewrite_query(query):
    prompt = f"Rewrite this search query to be more specific: {query}\nRewritten:"
    response = ollama.generate(model='mistral', prompt=prompt, 
                               options={'temperature': 0.3, 'num_predict': 30})
    return response['response'].strip()

# Multi-Query Retrieval - use multiple variations
def multi_query_retrieval(query, retriever, n_variations=3):
    variations = [query]
    for i in range(n_variations - 1):
        prompt = f"Generate alternative way to ask: {query}\nAlternative:"
        response = ollama.generate(model='mistral', prompt=prompt, 
                                   options={'temperature': 0.7, 'num_predict': 30})
        variations.append(response['response'].strip())
    
    all_results = {}
    for var in variations:
        for result in retriever.retrieve(var, k=5):
            doc = result['document']
            all_results[doc] = all_results.get(doc, 0) + result['score']
    
    return sorted([{'document': d, 'score': s} for d, s in all_results.items()],
                  key=lambda x: x[1], reverse=True)
```

---

## Reranking & Multi-Stage Pipeline

```python
from sentence_transformers import CrossEncoder

# Reranking - Stage 1: fast retrieval (top 100), Stage 2: accurate reranking (top 10)
class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query, documents, top_k=5):
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [{'document': doc, 'score': float(score)} for doc, score in ranked[:top_k]]

# Multi-Stage Pipeline
class AdvancedRetriever:
    def __init__(self, documents):
        self.hybrid = HybridRetriever(documents)
        self.reranker = Reranker()
    
    def retrieve(self, query, k=5):
        expanded = expand_with_llm(query)  # Stage 1: Query expansion
        candidates = self.hybrid.retrieve(expanded, k=20)  # Stage 2: Hybrid retrieval
        docs = [c['document'] for c in candidates]
        return self.reranker.rerank(query, docs, top_k=k)  # Stage 3: Reranking
```

---

## Retrieval Evaluation

```python
# Precision@k - fraction of top-k that are relevant
def precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    return len(set(top_k) & set(relevant)) / k

# Recall@k - fraction of relevant docs in top-k
def recall_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    return len(set(top_k) & set(relevant)) / len(relevant)

# MRR (Mean Reciprocal Rank)
def mrr(retrieved_lists, relevant_lists):
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                reciprocal_ranks.append(1 / i)
                break
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)
```

---

## 💻 Exercises (40 min)

### Exercise 1: Hybrid Retriever
Implement hybrid search combining dense and sparse retrieval.

### Exercise 2: Query Expander
Build query expansion using LLM and synonyms.

### Exercise 3: Reranker
Create a reranking system using cross-encoders.

### Exercise 4: Multi-Stage Pipeline
Build complete retrieval pipeline with expansion and reranking.

### Exercise 5: Retrieval Evaluator
Implement evaluation metrics (precision, recall, MRR).

---

## ✅ Quiz

Test your understanding in `quiz.md`

---

## 🎯 Key Takeaways

- Dense retrieval captures semantics, sparse captures keywords
- Hybrid search combines both for best results
- Query expansion adds context to short queries
- Query rewriting clarifies ambiguous queries
- Multi-query retrieval uses multiple variations
- Reranking improves accuracy in second stage
- Two-stage: fast retrieval → accurate reranking
- Cross-encoders better than bi-encoders for reranking
- Alpha parameter balances dense/sparse (0.7 typical)
- Evaluation metrics: precision@k, recall@k, MRR
- Multi-stage pipelines improve quality
- Trade-offs: quality vs speed, complexity vs simplicity

---

## 📚 Resources

- [Dense Passage Retrieval Paper](https://arxiv.org/abs/2004.04906)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Cross-Encoders for Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Hybrid Search Guide](https://www.pinecone.io/learn/hybrid-search-intro/)
- [Query Expansion Techniques](https://nlp.stanford.edu/IR-book/html/htmledition/query-expansion-1.html)

---

## Tomorrow: Day 83 - LangChain Basics

Learn about LangChain framework for building LLM applications, including chains, agents, and memory management.
