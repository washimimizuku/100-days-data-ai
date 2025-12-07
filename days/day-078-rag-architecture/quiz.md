# Day 78: RAG Architecture - Quiz

Test your understanding of Retrieval-Augmented Generation systems.

---

## Question 1
What is the primary purpose of RAG (Retrieval-Augmented Generation)?

A) To make LLMs generate text faster  
B) To combine information retrieval with LLM generation for factual responses  
C) To reduce the size of language models  
D) To eliminate the need for training data  

**Correct Answer: B**

---

## Question 2
Which component is NOT part of a typical RAG system?

A) Knowledge base with documents  
B) Retriever for finding relevant information  
C) Model training pipeline  
D) Generator (LLM) for producing responses  

**Correct Answer: C**

---

## Question 3
What is the main advantage of RAG over fine-tuning for updating knowledge?

A) RAG is faster at inference time  
B) RAG requires less memory  
C) RAG can update knowledge by simply updating the document corpus  
D) RAG produces more creative responses  

**Correct Answer: C**

---

## Question 4
In dense retrieval (semantic search), how are documents and queries compared?

A) By counting exact keyword matches  
B) By comparing their embedding vectors using similarity metrics  
C) By analyzing their grammatical structure  
D) By measuring their length  

**Correct Answer: B**

---

## Question 5
What is hybrid retrieval in RAG systems?

A) Using multiple LLMs simultaneously  
B) Combining dense (semantic) and sparse (keyword) retrieval methods  
C) Retrieving from multiple databases  
D) Using both training and inference data  

**Correct Answer: B**

---

## Question 6
What is the purpose of reranking in a RAG pipeline?

A) To sort documents alphabetically  
B) To score and reorder retrieved documents by relevance to improve quality  
C) To remove duplicate documents  
D) To compress documents for faster processing  

**Correct Answer: B**

---

## Question 7
Why is query rewriting useful in RAG systems?

A) To make queries shorter  
B) To translate queries to different languages  
C) To expand and clarify queries for better retrieval results  
D) To encrypt sensitive queries  

**Correct Answer: C**

---

## Question 8
What is a key benefit of adding citations to RAG responses?

A) Makes responses longer  
B) Provides transparency and allows verification of sources  
C) Reduces inference time  
D) Eliminates the need for retrieval  

**Correct Answer: B**

---

## Question 9
When is RAG NOT the best choice?

A) Question answering over large document collections  
B) Customer support with knowledge bases  
C) Creative writing without factual constraints  
D) Legal document analysis  

**Correct Answer: C**

---

## Question 10
What is the trade-off of using RAG compared to direct LLM prompting?

A) RAG is less accurate but faster  
B) RAG is more accurate but has higher latency due to retrieval step  
C) RAG uses less memory but more compute  
D) RAG requires smaller models but more training  

**Correct Answer: B**

---

## Scoring Guide
- 9-10 correct: Expert - You've mastered RAG architecture!
- 7-8 correct: Proficient - Strong understanding with minor gaps
- 5-6 correct: Intermediate - Good foundation, review key concepts
- Below 5: Beginner - Review the material and practice more

## Key Takeaways
- RAG combines retrieval with generation for factual, grounded responses
- Core components: knowledge base, retriever, reranker, generator
- RAG enables easy knowledge updates without model retraining
- Dense retrieval uses embeddings, sparse uses keywords, hybrid combines both
- Reranking improves relevance of retrieved documents
- Query rewriting enhances retrieval quality
- Citations provide transparency and source attribution
- RAG excels at knowledge-intensive tasks but adds latency
- Trade-offs: accuracy and maintainability vs speed
- Best for Q&A, support, research, and document analysis
