# Day 80: Vector Embeddings - Quiz

Test your understanding of vector embeddings for semantic search.

---

## Question 1
What are vector embeddings?

A) Compressed versions of text files  
B) Numerical representations of data in high-dimensional space where semantic similarity is captured by proximity  
C) Binary encodings of words  
D) Database indexes for faster search  

**Correct Answer: B**

---

## Question 2
Which similarity metric is MOST commonly used for comparing embeddings?

A) Manhattan distance  
B) Hamming distance  
C) Cosine similarity  
D) Levenshtein distance  

**Correct Answer: C**

---

## Question 3
What is the range of cosine similarity values?

A) 0 to 1  
B) -1 to 1  
C) 0 to infinity  
D) -infinity to infinity  

**Correct Answer: B**

---

## Question 4
Why should embeddings be normalized before using dot product for similarity?

A) To reduce storage space  
B) To make dot product equivalent to cosine similarity  
C) To speed up computation  
D) To remove negative values  

**Correct Answer: B**

---

## Question 5
What is a typical embedding dimension for the all-MiniLM-L6-v2 model?

A) 128  
B) 256  
C) 384  
D) 1536  

**Correct Answer: C**

---

## Question 6
What is the main trade-off of using higher-dimensional embeddings?

A) Lower accuracy but faster search  
B) Better quality but more storage and slower search  
C) Simpler implementation but less flexibility  
D) More languages supported but higher cost  

**Correct Answer: B**

---

## Question 7
What technique can be used to reduce embedding dimensions while preserving information?

A) Hashing  
B) Compression  
C) PCA (Principal Component Analysis)  
D) Encryption  

**Correct Answer: C**

---

## Question 8
Why is batch processing important when generating embeddings?

A) It produces more accurate embeddings  
B) It improves efficiency by processing multiple texts together  
C) It reduces the embedding dimensions  
D) It enables multilingual support  

**Correct Answer: B**

---

## Question 9
What is the advantage of caching embeddings?

A) Reduces model size  
B) Improves embedding quality  
C) Avoids recomputation for the same texts, saving time  
D) Enables real-time updates  

**Correct Answer: C**

---

## Question 10
Which statement about Euclidean distance is TRUE?

A) It measures the angle between vectors  
B) It measures the straight-line distance between vectors in space  
C) It only works for normalized vectors  
D) It always returns values between -1 and 1  

**Correct Answer: B**

---

## Scoring Guide
- 9-10 correct: Expert - You've mastered vector embeddings!
- 7-8 correct: Proficient - Strong understanding with minor gaps
- 5-6 correct: Intermediate - Good foundation, review key concepts
- Below 5: Beginner - Review the material and practice more

## Key Takeaways
- Vector embeddings represent text as numerical vectors in semantic space
- Similar meanings result in similar vectors (close in space)
- Cosine similarity is the most common metric (measures angle)
- Euclidean distance measures geometric distance between vectors
- Dot product is fastest for normalized vectors
- Normalization enables consistent similarity comparisons
- Higher dimensions = better quality but more storage/slower search
- Typical dimensions: 384-768 for good balance
- PCA can reduce dimensions while preserving information
- Batch processing improves efficiency for large datasets
- Caching avoids recomputation and saves time
- Sentence Transformers are popular for local embedding generation
