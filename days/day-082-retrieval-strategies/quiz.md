# Day 82: Retrieval Strategies - Quiz

Test your understanding of advanced retrieval techniques for RAG systems.

---

## Question 1
What is the main advantage of hybrid search over pure dense retrieval?

A) It's faster to compute  
B) It combines semantic understanding with exact keyword matching  
C) It requires less storage  
D) It works without embeddings  

**Correct Answer: B**

---

## Question 2
In hybrid search, what does the alpha parameter control?

A) The number of results returned  
B) The weight balance between dense and sparse retrieval scores  
C) The embedding dimension  
D) The query length  

**Correct Answer: B**

---

## Question 3
What is the purpose of query expansion?

A) To make queries longer for better storage  
B) To add context and related terms to short queries for better retrieval  
C) To translate queries to different languages  
D) To compress queries for faster processing  

**Correct Answer: B**

---

## Question 4
What is pseudo-relevance feedback (PRF)?

A) User feedback on search results  
B) Expanding queries using terms from top retrieved documents  
C) A type of neural network  
D) A database indexing method  

**Correct Answer: B**

---

## Question 5
Why use a two-stage retrieval pipeline?

A) To reduce storage requirements  
B) To enable fast initial retrieval followed by accurate reranking  
C) To support multiple languages  
D) To compress embeddings  

**Correct Answer: B**

---

## Question 6
What is the advantage of cross-encoders for reranking?

A) They're faster than bi-encoders  
B) They see query and document together for better relevance scoring  
C) They require less memory  
D) They work without training  

**Correct Answer: B**

---

## Question 7
What does precision@k measure?

A) The speed of retrieval  
B) The fraction of top-k results that are relevant  
C) The total number of relevant documents  
D) The embedding quality  

**Correct Answer: B**

---

## Question 8
What is MRR (Mean Reciprocal Rank)?

A) The average position of relevant documents  
B) The average of reciprocal ranks of first relevant result  
C) The maximum retrieval rate  
D) The model reranking ratio  

**Correct Answer: B**

---

## Question 9
When should you use multi-query retrieval?

A) When you have only one query  
B) When you want to retrieve using multiple query variations for comprehensive results  
C) When storage is limited  
D) When speed is the only priority  

**Correct Answer: B**

---

## Question 10
What is the typical alpha value for balanced hybrid search?

A) 0.0 (pure sparse)  
B) 0.3  
C) 0.7 (balanced toward dense)  
D) 1.0 (pure dense)  

**Correct Answer: C**

---

## Scoring Guide
- 9-10 correct: Expert - You've mastered retrieval strategies!
- 7-8 correct: Proficient - Strong understanding with minor gaps
- 5-6 correct: Intermediate - Good foundation, review key concepts
- Below 5: Beginner - Review the material and practice more

## Key Takeaways
- Hybrid search combines dense (semantic) and sparse (keyword) retrieval
- Alpha parameter balances dense/sparse (0.7 typical for dense-heavy)
- Query expansion adds context to improve retrieval
- Query rewriting clarifies ambiguous queries
- Multi-query retrieval uses multiple variations
- Two-stage: fast retrieval → accurate reranking
- Cross-encoders better for reranking (see query+doc together)
- Precision@k: fraction of top-k that are relevant
- Recall@k: fraction of relevant docs in top-k
- MRR: average reciprocal rank of first relevant result
- Multi-stage pipelines improve quality
- Trade-offs: quality vs speed, complexity vs simplicity
