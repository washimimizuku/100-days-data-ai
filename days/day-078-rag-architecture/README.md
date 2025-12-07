# Day 78: RAG Architecture

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Understand Retrieval-Augmented Generation (RAG) architecture
- Learn the components of a RAG system
- Compare RAG with fine-tuning and prompt engineering
- Master retrieval strategies for context selection

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with LLM generation to provide accurate, up-to-date responses grounded in external knowledge.

### The Problem RAG Solves

**LLM Limitations**: Knowledge cutoff, hallucinations, no private data access, cannot update without retraining

**RAG Solution**: Retrieve relevant information → Augment prompt → Generate grounded response → Update by updating corpus

---

## RAG Architecture

### Core Components

```
Query → Retrieval → Reranking → Context → LLM → Response
```

**1. Knowledge Base**: Documents + metadata + embeddings  
**2. Retriever**: Converts query to embeddings, searches, returns top-k  
**3. Reranker**: Scores and filters results  
**4. Generator**: LLM produces response with citations

---

## RAG vs Alternatives

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Knowledge Update | Easy (update corpus) | Hard (retrain) |
| Cost | Low | High |
| Transparency | High (cite sources) | Low |
| Latency | Higher | Lower |

### RAG vs Prompt Engineering

| Aspect | RAG | Prompts |
|--------|-----|---------|
| Context Size | Unlimited | Limited |
| Knowledge | External DB | Model params |
| Maintenance | Update DB | Update prompts |

### When to Use RAG

**Good**: Q&A over documents, customer support, research, legal/medical analysis  
**Not Ideal**: Creative writing, simple classification, real-time constraints

---

## Retrieval Strategies

### 1. Dense Retrieval (Semantic)

```python
# Embed and find similar
query_emb = model.encode(query)
doc_embs = model.encode(documents)
similarities = np.dot(doc_embs, query_emb)
top_k = np.argsort(similarities)[-k:][::-1]
```

**Pros**: Semantic meaning, paraphrases  
**Cons**: Expensive, may miss keywords

### 2. Sparse Retrieval (Keyword)

```python
# BM25 scoring
scores = bm25(query, documents)
top_k = np.argsort(scores)[-k:]
```

**Pros**: Fast, exact matching  
**Cons**: Misses semantics, vocabulary mismatch

### 3. Hybrid Retrieval

```python
# Combine both
final = alpha * dense + (1 - alpha) * sparse
```

**Pros**: Best of both, robust  
**Cons**: More complex, tuning needed

---

## Building a Simple RAG

### Complete Example

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

# 1. Index documents
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "Python is a programming language.",
    "Machine learning is AI.",
    "RAG combines retrieval with generation."
]
embeddings = model.encode(documents)

# 2. Retrieve
def retrieve(query, k=2):
    query_emb = model.encode([query])[0]
    sims = np.dot(embeddings, query_emb)
    top = np.argsort(sims)[-k:][::-1]
    return [documents[i] for i in top]

# 3. Generate
def rag_query(query):
    docs = retrieve(query)
    context = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(docs)])
    
    prompt = f"""Answer based on context:

Context: {context}

Question: {query}

Answer:"""
    
    response = ollama.generate(model='mistral', prompt=prompt)
    return {'answer': response['response'], 'sources': docs}

# Use
result = rag_query("What is RAG?")
```

---

## Advanced Techniques

### Query Rewriting

```python
def rewrite_query(query):
    prompt = f"Rewrite for better search: {query}\n\nRewritten:"
    return ollama.generate(model='mistral', prompt=prompt)['response']
```

### Multi-Query

```python
def multi_query(query):
    variations = [query, rewrite_query(query), f"Explain {query}"]
    all_docs = set()
    for var in variations:
        all_docs.update(retrieve(var))
    return list(all_docs)
```

### Reranking

```python
def rerank(query, documents):
    scores = []
    for doc in documents:
        prompt = f"Rate relevance 0-10:\nQuery: {query}\nDoc: {doc}\nScore:"
        score = float(ollama.generate(model='mistral', prompt=prompt)['response'].split()[0])
        scores.append(score)
    return [d for d, s in sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)]
```

---

## 💻 Exercises (40 min)

### Exercise 1: Basic RAG System
Build RAG with indexing, retrieval, and generation.

### Exercise 2: Hybrid Retrieval
Combine semantic and keyword search.

### Exercise 3: Query Rewriting
Improve retrieval with query reformulation.

### Exercise 4: Reranking Pipeline
Score and filter retrieved documents.

### Exercise 5: Citation System
Add source citations to responses.

---

## ✅ Quiz

Test your understanding in `quiz.md`

---

## 🎯 Key Takeaways

- RAG combines retrieval + generation for factual responses
- Core: knowledge base, retriever, reranker, generator
- RAG vs fine-tuning: easier updates, transparent, higher latency
- Retrieval: dense (semantic), sparse (keyword), hybrid (both)
- Advanced: query rewriting, multi-query, reranking, citations
- Best for: Q&A, support, research, document analysis
- Trade-off: latency vs accuracy and maintainability

---

## 📚 Resources

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Sentence Transformers](https://www.sbert.net/)
- [Vector Databases](https://www.pinecone.io/learn/vector-database/)

---

## Tomorrow: Day 79 - Document Chunking

Learn strategies for splitting documents into optimal chunks for RAG systems.
