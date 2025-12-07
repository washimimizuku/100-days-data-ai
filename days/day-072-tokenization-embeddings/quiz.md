# Day 72: Tokenization & Embeddings - Quiz

Test your understanding of tokenization algorithms and word embeddings.

## Questions

### 1. What is the main advantage of subword tokenization over word-level tokenization?
a) It's faster to compute
b) It handles rare and out-of-vocabulary words better
c) It produces shorter sequences
d) It requires less memory

**Correct Answer: b**

### 2. How does Byte Pair Encoding (BPE) work?
a) It splits words into individual bytes
b) It iteratively merges the most frequent character pairs
c) It uses a pre-defined dictionary
d) It randomly splits words

**Correct Answer: b**

### 3. What does the ## prefix indicate in BERT's WordPiece tokenization?
a) A special token
b) A continuation of the previous token
c) A rare word
d) A punctuation mark

**Correct Answer: b**

### 4. Which tokenization method is language-agnostic and works without pre-tokenization?
a) BPE
b) WordPiece
c) SentencePiece
d) Character-level

**Correct Answer: c**

### 5. What is the typical vocabulary size for modern LLMs?
a) 1,000 - 5,000 tokens
b) 10,000 - 20,000 tokens
c) 30,000 - 50,000 tokens
d) 100,000+ tokens

**Correct Answer: c**

### 6. How do contextual embeddings (like BERT) differ from static embeddings (like Word2Vec)?
a) Contextual embeddings are larger
b) Contextual embeddings vary based on context
c) Contextual embeddings are faster to compute
d) Contextual embeddings use fewer dimensions

**Correct Answer: b**

### 7. What metric is commonly used to measure semantic similarity between embeddings?
a) Euclidean distance
b) Manhattan distance
c) Cosine similarity
d) Hamming distance

**Correct Answer: c**

### 8. What is the purpose of special tokens like [CLS] and [SEP] in BERT?
a) To mark sentence boundaries and classification positions
b) To pad sequences
c) To represent unknown words
d) To indicate punctuation

**Correct Answer: a**

### 9. What happens when text exceeds a model's maximum context length?
a) The model automatically compresses it
b) It must be truncated or processed in chunks
c) The model increases its context window
d) An error is always raised

**Correct Answer: b**

### 10. Why is mean pooling used when creating sentence embeddings from token embeddings?
a) To reduce memory usage
b) To aggregate token-level information into a single vector
c) To normalize the embeddings
d) To remove special tokens

**Correct Answer: b**

## Scoring Guide
- 9-10 correct: Excellent! You understand tokenization and embeddings well.
- 7-8 correct: Good job! Review the topics you missed.
- 5-6 correct: Fair. Revisit the README and practice more.
- Below 5: Review the material and work through the exercises again.

## Answer Key
1. b - Subword tokenization handles rare/OOV words by breaking them into known subwords
2. b - BPE iteratively merges the most frequent character pairs
3. b - ## indicates the token continues the previous word
4. c - SentencePiece treats text as raw bytes, no pre-tokenization needed
5. c - Most modern LLMs use 30k-50k token vocabularies
6. b - Contextual embeddings change based on surrounding context
7. c - Cosine similarity measures angle between vectors
8. a - [CLS] marks classification position, [SEP] separates sentences
9. b - Long text must be truncated or processed in chunks
10. b - Mean pooling aggregates token embeddings into sentence embedding
