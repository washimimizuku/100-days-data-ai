# Day 81: Vector Databases

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Understand vector databases and their role in RAG systems
- Learn about popular vector database solutions
- Master indexing strategies for efficient similarity search
- Implement CRUD operations with vector databases
- Compare local vs cloud vector database options

---

## What are Vector Databases?

**Vector Databases** are specialized databases optimized for storing, indexing, and querying high-dimensional vector embeddings with fast similarity search.

**Why Not Regular DBs**: Traditional databases are optimized for exact matches, slow for similarity search, lack vector-specific indexing. Vector DBs use approximate nearest neighbor (ANN) search with specialized indexing (HNSW, IVF) for fast similarity queries.

**Use Cases**: Semantic search, recommendations, RAG systems, image search, anomaly detection

---

## Popular Vector Databases

### 1. FAISS (Local Library)

```python
import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatL2(dimension)
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)
```

**Pros**: Free, fast, many index types | **Cons**: No persistence, no metadata | **Best For**: Research, prototyping

### 2. Chroma (Local Embedded)

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

collection.add(
    documents=["ML is AI", "Python is a language"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)

results = collection.query(query_texts=["What is ML?"], n_results=2)
```

**Pros**: Easy, built-in embeddings, metadata | **Cons**: Limited scalability | **Best For**: Development

### 3. Pinecone (Cloud Service)

```python
import pinecone

pinecone.init(api_key="key", environment="us-west1-gcp")
pinecone.create_index("docs", dimension=384, metric="cosine")
index = pinecone.Index("docs")

index.upsert(vectors=[("id1", [0.1, 0.2, ...], {"text": "doc1"})])
results = index.query(vector=[0.15, 0.25, ...], top_k=5, include_metadata=True)
```

**Pros**: Managed, auto-scaling, high performance | **Cons**: Paid, API latency | **Best For**: Production

---

## Indexing Strategies

```python
# 1. Flat Index (Brute Force) - 100% accurate, slow O(n)
index = faiss.IndexFlatL2(dimension)  # Use: < 10K vectors

# 2. IVF (Inverted File) - Cluster vectors, search relevant clusters
nlist = 100
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(vectors)
index.add(vectors)
index.nprobe = 10  # clusters to search
# Use: 10K-1M vectors, good accuracy/speed trade-off

# 3. HNSW (Hierarchical Navigable Small World) - Graph-based
index = faiss.IndexHNSWFlat(dimension, M=32)
index.add(vectors)
# Use: 100K+ vectors, fast queries, no training needed

# 4. Product Quantization (PQ) - Compress vectors
index = faiss.IndexPQ(dimension, m=8, nbits=8)
index.train(vectors)
# Use: Millions of vectors, memory constrained
```

---

## CRUD Operations

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

# Create/Insert
collection.add(
    documents=["Text 1", "Text 2"],
    metadatas=[{"source": f"doc{i}"} for i in range(2)],
    ids=[f"id{i}" for i in range(2)]
)

# Read/Query
results = collection.query(
    query_texts=["What is ML?"],
    n_results=5,
    where={"source": "doc1"}  # Metadata filter
)
doc = collection.get(ids=["id1"])

# Update
collection.update(ids=["id1"], documents=["Updated"], metadatas=[{"updated": True}])

# Delete
collection.delete(ids=["id1"])
collection.delete(where={"source": "doc1"})
```

---

## Metadata Filtering

```python
# Without metadata - returns all similar
results = collection.query(query_texts=["ML"], n_results=5)

# With metadata - narrows search scope
results = collection.query(
    query_texts=["ML"],
    n_results=5,
    where={"source": "research_papers", "year": {"$gte": 2020}}
)

# Filter operators
where={"category": "tech"}  # Equality
where={"year": {"$gte": 2020, "$lte": 2024}}  # Comparison
where={"source": {"$in": ["doc1", "doc2"]}}  # In list
where={"$and": [{"category": "tech"}, {"year": {"$gte": 2020}}]}  # And/Or
```

---

## Performance Optimization

```python
# 1. Batch Operations (not one-at-a-time)
collection.add(
    documents=[doc.text for doc in documents],
    ids=[doc.id for doc in documents]
)

# 2. Index Selection
index = faiss.IndexFlatL2(dimension)  # < 10K
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)  # 10K-100K
index = faiss.IndexHNSWFlat(dimension, M=32)  # 100K+

# 3. Query Optimization
results = collection.query(query_texts=["ML"], n_results=5, where={"category": "tech"})
index.nprobe = 10  # Lower = faster, less accurate
```

---

## Local vs Cloud

**Local (FAISS, Chroma)**:
- Pros: No costs, full control, low latency, privacy
- Cons: Manual scaling, infrastructure management
- Use: Development, < 1M vectors, privacy needs

**Cloud (Pinecone, Weaviate)**:
- Pros: Auto-scaling, managed, high availability
- Cons: Ongoing costs, API latency, vendor lock-in
- Use: Production, > 1M vectors, managed service

---

## 💻 Exercises (40 min)

### Exercise 1: FAISS Index
Build a FAISS-based vector search system with different index types.

### Exercise 2: Chroma Database
Implement CRUD operations with Chroma including metadata filtering.

### Exercise 3: Index Comparison
Compare performance of different index types (Flat, IVF, HNSW).

### Exercise 4: Metadata Filtering
Build a filtered search system using metadata queries.

### Exercise 5: Batch Operations
Optimize insertion and querying with batch processing.

---

## ✅ Quiz

Test your understanding in `quiz.md`

---

## 🎯 Key Takeaways

- Vector databases are optimized for similarity search on embeddings
- FAISS is fast and free but requires manual management
- Chroma is easy for development with built-in features
- Pinecone is managed cloud service for production
- Weaviate offers rich features and hybrid search
- Indexing strategies trade accuracy for speed
- Flat index: accurate but slow (< 10K vectors)
- IVF: good balance (10K-1M vectors)
- HNSW: fast queries (100K+ vectors)
- Metadata filtering narrows search scope
- Batch operations improve performance
- Local DBs: control and privacy, manual scaling
- Cloud DBs: managed service, auto-scaling, costs

---

## 📚 Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Vector Database Comparison](https://www.pinecone.io/learn/vector-database/)

---

## Tomorrow: Day 82 - Retrieval Strategies

Learn advanced retrieval techniques including hybrid search, reranking, query expansion, and multi-stage retrieval pipelines.
