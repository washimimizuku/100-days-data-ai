# Day 79: Document Chunking - Quiz

Test your understanding of document chunking strategies for RAG systems.

---

## Question 1
Why is document chunking necessary for RAG systems?

A) To make documents load faster  
B) To fit documents within LLM context limits and improve retrieval granularity  
C) To compress documents for storage  
D) To translate documents to different languages  

**Correct Answer: B**

---

## Question 2
What is the main disadvantage of fixed-size chunking?

A) It's too slow to implement  
B) It may split sentences or paragraphs mid-thought, breaking context  
C) It requires machine learning models  
D) It only works with English text  

**Correct Answer: B**

---

## Question 3
What does semantic chunking use to determine chunk boundaries?

A) Character count  
B) Paragraph markers  
C) Semantic similarity between sentences using embeddings  
D) Random splitting  

**Correct Answer: C**

---

## Question 4
What is the purpose of chunk overlap?

A) To reduce storage requirements  
B) To preserve context at chunk boundaries and avoid information loss  
C) To make chunks the same size  
D) To speed up retrieval  

**Correct Answer: B**

---

## Question 5
What is a typical recommended overlap percentage for chunks?

A) 0-5% (minimal overlap)  
B) 10-20% (moderate overlap)  
C) 50-60% (high overlap)  
D) 80-90% (maximum overlap)  

**Correct Answer: B**

---

## Question 6
How does recursive chunking work?

A) It splits text randomly until chunks are small enough  
B) It uses a hierarchy of separators (paragraphs, sentences, words) to preserve structure  
C) It repeats the same splitting method multiple times  
D) It only splits at the end of documents  

**Correct Answer: B**

---

## Question 7
What is the typical optimal chunk size range for RAG systems?

A) 50-100 tokens  
B) 100-200 tokens  
C) 300-800 tokens  
D) 1000-2000 tokens  

**Correct Answer: C**

---

## Question 8
What is an advantage of paragraph-based chunking?

A) All chunks are exactly the same size  
B) It preserves logical structure and complete thoughts  
C) It's the fastest chunking method  
D) It works without any text processing  

**Correct Answer: B**

---

## Question 9
What metadata is useful to track for each chunk?

A) Only the text content  
B) Chunk ID, source document, position, and statistics  
C) Just the chunk size  
D) Only the creation timestamp  

**Correct Answer: B**

---

## Question 10
When is semantic chunking most beneficial?

A) For simple, uniform documents  
B) When speed is the only priority  
C) For multi-topic documents where quality matters more than speed  
D) For documents under 100 words  

**Correct Answer: C**

---

## Scoring Guide
- 9-10 correct: Expert - You've mastered document chunking!
- 7-8 correct: Proficient - Strong understanding with minor gaps
- 5-6 correct: Intermediate - Good foundation, review key concepts
- Below 5: Beginner - Review the material and practice more

## Key Takeaways
- Chunking is critical for RAG quality and LLM context limits
- Fixed-size is simple but may break context
- Sentence/paragraph chunking preserves structure
- Semantic chunking groups by topic similarity (best quality)
- Recursive chunking balances size and coherence
- Overlap (10-20%) preserves context at boundaries
- Optimal chunk size: 300-800 tokens (domain-dependent)
- Metadata tracking improves retrieval and debugging
- Test different strategies for your specific use case
- Trade-offs: coherence vs size, quality vs speed
