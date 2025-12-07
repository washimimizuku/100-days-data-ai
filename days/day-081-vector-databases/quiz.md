# Day 81: Vector Databases - Quiz

Test your understanding of vector databases for similarity search.

---

## Question 1
What is the primary purpose of vector databases?

A) To store relational data efficiently  
B) To perform fast similarity search on high-dimensional vector embeddings  
C) To compress large files  
D) To manage user authentication  

**Correct Answer: B**

---

## Question 2
Which indexing strategy provides 100% accurate results but is slowest for large datasets?

A) HNSW (Hierarchical Navigable Small World)  
B) IVF (Inverted File Index)  
C) Flat Index (brute force)  
D) Product Quantization  

**Correct Answer: C**

---

## Question 3
What is the main advantage of FAISS?

A) It's a managed cloud service  
B) It's free, open source, and very fast for local deployments  
C) It has the best user interface  
D) It only works with small datasets  

**Correct Answer: B**

---

## Question 4
Which vector database is best suited for production applications requiring auto-scaling?

A) FAISS  
B) Chroma (embedded)  
C) Pinecone (cloud service)  
D) SQLite  

**Correct Answer: C**

---

## Question 5
What does IVF (Inverted File Index) do to improve search speed?

A) Compresses vectors to reduce size  
B) Clusters vectors and searches only relevant clusters  
C) Uses graph-based navigation  
D) Removes duplicate vectors  

**Correct Answer: B**

---

## Question 6
Which index type is best for datasets with 100K+ vectors when fast queries are critical?

A) Flat Index  
B) IVF Index  
C) HNSW Index  
D) No index needed  

**Correct Answer: C**

---

## Question 7
What is the purpose of metadata filtering in vector databases?

A) To compress metadata  
B) To narrow search scope by filtering on document properties before similarity search  
C) To encrypt sensitive data  
D) To sort results alphabetically  

**Correct Answer: B**

---

## Question 8
Why is batch processing important for vector database operations?

A) It produces more accurate results  
B) It improves efficiency by processing multiple operations together  
C) It reduces vector dimensions  
D) It enables real-time updates  

**Correct Answer: B**

---

## Question 9
What is a key disadvantage of cloud vector databases compared to local ones?

A) Lower accuracy  
B) Slower query times  
C) Ongoing API costs and potential vendor lock-in  
D) Cannot handle large datasets  

**Correct Answer: C**

---

## Question 10
Which statement about HNSW index is TRUE?

A) It requires training on data before use  
B) It provides 100% accurate results  
C) It uses graph-based navigation for fast queries without training  
D) It's only suitable for small datasets  

**Correct Answer: C**

---

## Scoring Guide
- 9-10 correct: Expert - You've mastered vector databases!
- 7-8 correct: Proficient - Strong understanding with minor gaps
- 5-6 correct: Intermediate - Good foundation, review key concepts
- Below 5: Beginner - Review the material and practice more

## Key Takeaways
- Vector databases optimize similarity search on embeddings
- FAISS: fast, free, local (requires manual management)
- Chroma: easy development with built-in features
- Pinecone: managed cloud service for production
- Weaviate: rich features and hybrid search
- Flat index: accurate but slow (< 10K vectors)
- IVF: clusters vectors for speed (10K-1M vectors)
- HNSW: graph-based, fast queries (100K+ vectors)
- Product Quantization: compresses vectors (memory constrained)
- Metadata filtering narrows search scope
- Batch operations improve performance significantly
- Local DBs: control, privacy, manual scaling
- Cloud DBs: managed, auto-scaling, ongoing costs
- Index selection depends on dataset size and requirements
