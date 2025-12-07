# Day 80: Vector Embeddings

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Understand vector embeddings and their role in semantic search
- Learn about embedding models and architectures
- Master similarity metrics for comparing embeddings
- Implement embedding generation for text and documents
- Optimize embedding dimensionality and quality

---

## What are Vector Embeddings?

**Vector Embeddings** are numerical representations of data in high-dimensional space where semantic similarity is captured by geometric proximity.

```
Text → Embedding Model → Vector [0.23, -0.45, 0.67, ...]
Similar texts → Similar vectors (close in space)
```

**Why for RAG**: Traditional keyword search fails on synonyms ("ML algorithms" vs "machine learning techniques"). Embeddings capture semantic meaning, enabling similarity-based retrieval that handles synonyms, paraphrases, and works across languages

---

## Embedding Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General, local |
| all-mpnet-base-v2 | 768 | Medium | Better | Quality-focused |
| text-embedding-ada-002 | 1536 | API | Best | Production |
| multilingual-e5-large | 1024 | Slow | Best | Multilingual |

---

## Generating Embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Single text
embedding = model.encode("Machine learning is a subset of AI")
print(f"Shape: {embedding.shape}")  # (384,)

# Batch processing
texts = ["ML is powerful", "Deep learning uses neural networks"]
embeddings = model.encode(texts)  # Shape: (2, 384)

# Normalization (enables cosine similarity via dot product)
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding
```

---

## Similarity Metrics

```python
# 1. Cosine Similarity (most common) - measures angle
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# Range: [-1, 1], invariant to magnitude

# 2. Euclidean Distance - geometric distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
# Range: [0, ∞), sensitive to magnitude

# 3. Dot Product (fastest for normalized vectors)
def dot_product_similarity(a, b):
    return np.dot(a, b)
# For normalized: equivalent to cosine

# Comparison
texts = ["Machine learning algorithms", "ML techniques", "Pizza recipe"]
embeddings = model.encode(texts)
norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

for i in range(1, len(texts)):
    cos_sim = np.dot(norm_embeddings[0], norm_embeddings[i])
    euc_dist = np.linalg.norm(embeddings[0] - embeddings[i])
    print(f"'{texts[0]}' vs '{texts[i]}': cos={cos_sim:.3f}, euc={euc_dist:.3f}")
```

---

## Dimensionality

**Trade-offs**: Higher dims (768, 1536) = better quality but slower/more storage. Lower dims (128, 384) = faster but may lose nuance. Sweet spot: 384-768.

```python
from sklearn.decomposition import PCA

def reduce_dimensions(embeddings, target_dim=128):
    """Reduce embedding dimensions."""
    pca = PCA(n_components=target_dim)
    reduced = pca.fit_transform(embeddings)
    print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")
    return reduced
```

---

## Embedding Quality

**Factors**: Model selection (larger/domain-specific better), input quality (clean, appropriate length), normalization (consistent preprocessing).

```python
def evaluate_embeddings(model, test_pairs):
    """Evaluate embedding quality on test pairs."""
    scores = []
    for text1, text2, expected_sim in test_pairs:
        emb1, emb2 = model.encode([text1, text2])
        emb1, emb2 = emb1 / np.linalg.norm(emb1), emb2 / np.linalg.norm(emb2)
        actual_sim = np.dot(emb1, emb2)
        scores.append({'text1': text1, 'text2': text2, 
                      'expected': expected_sim, 'actual': actual_sim})
    return scores

# Test: (text1, text2, expected_similarity)
test_pairs = [("dog", "puppy", 0.9), ("dog", "cat", 0.7), ("dog", "car", 0.1)]
scores = evaluate_embeddings(model, test_pairs)
```

---

## Batch Processing & Caching

```python
import pickle

def batch_encode(texts, model, batch_size=32):
    """Encode texts in batches for efficiency."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.append(model.encode(batch, show_progress_bar=False))
    return np.vstack(embeddings)

def cache_embeddings(texts, model, cache_file='embeddings.pkl'):
    """Cache embeddings to avoid recomputation."""
    try:
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
            if cached['texts'] == texts:
                return cached['embeddings']
    except FileNotFoundError:
        pass
    
    embeddings = model.encode(texts)
    with open(cache_file, 'wb') as f:
        pickle.dump({'texts': texts, 'embeddings': embeddings}, f)
    return embeddings
```

---

## Advanced Techniques

```python
# Query Expansion
def expand_query_embedding(query, model, expansion_terms):
    """Expand query with related terms."""
    query_emb = model.encode(query)
    expansion_embs = model.encode(expansion_terms)
    expanded = 0.7 * query_emb + 0.3 * np.mean(expansion_embs, axis=0)
    return expanded / np.linalg.norm(expanded)

# Multi-Vector for long text
def multi_vector_embedding(text, model, chunk_size=100):
    """Create multiple embeddings for long text."""
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return model.encode(chunks)
```

---

## 💻 Exercises (40 min)

### Exercise 1: Embedding Generator
Build a system to generate and cache embeddings efficiently.

### Exercise 2: Similarity Calculator
Implement multiple similarity metrics and compare results.

### Exercise 3: Embedding Evaluator
Create a tool to evaluate embedding quality on test datasets.

### Exercise 4: Dimensionality Reducer
Implement PCA-based dimensionality reduction with quality analysis.

### Exercise 5: Semantic Search Engine
Build a simple semantic search using embeddings and similarity.

---

## ✅ Quiz

Test your understanding in `quiz.md`

---

## 🎯 Key Takeaways

- Vector embeddings represent text as numerical vectors in semantic space
- Similar meanings → similar vectors (close in space)
- Sentence Transformers are popular for local embedding generation
- Cosine similarity is most common metric (angle between vectors)
- Euclidean distance measures geometric distance
- Dot product is fastest for normalized vectors
- Higher dimensions = better quality but slower/more storage
- Dimensionality: 384-768 is typical sweet spot
- Normalize embeddings for consistent similarity comparisons
- Batch processing improves efficiency for large datasets
- Cache embeddings to avoid recomputation
- Model selection depends on quality, speed, and domain needs

---

## 📚 Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Embedding Models Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Vector Similarity Metrics](https://www.pinecone.io/learn/vector-similarity/)
- [Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html)

---

## Tomorrow: Day 81 - Vector Databases

Learn about specialized databases for storing and querying vector embeddings, including Pinecone, Weaviate, Chroma, and FAISS.
