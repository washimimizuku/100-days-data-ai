# Day 79: Document Chunking

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Understand why document chunking is critical for RAG systems
- Learn different chunking strategies and their trade-offs
- Master fixed-size, semantic, and recursive chunking methods
- Implement chunk overlap for context preservation
- Optimize chunk size for retrieval quality

---

## Why Chunking Matters

**LLM Context Limits**: Models have maximum context windows (4K-32K tokens). Long documents must be split into manageable pieces.

**Retrieval Granularity**: Chunks must be sized correctly - too large dilutes context with irrelevant info, too small fragments information and loses context.

**Impact on RAG**: Good chunking preserves complete, coherent information with context at boundaries. Poor chunking splits mid-thought, loses context, and degrades retrieval quality

---

## Chunking Strategies

### 1. Fixed-Size Chunking

**Concept**: Split text into chunks of fixed character/token count.

```python
def fixed_size_chunking(text: str, chunk_size: int = 500, 
                       overlap: int = 50) -> List[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap for context
    
    return chunks
```

**Pros**:
- Simple and fast
- Predictable chunk sizes
- Easy to implement

**Cons**:
- May split sentences/paragraphs
- No semantic awareness
- Can break context

**Best For**:
- Uniform documents
- Quick prototyping
- When speed matters

---

### 2. Sentence/Paragraph Chunking

**Concept**: Split at natural boundaries (sentences or paragraphs).

```python
import re

def sentence_chunking(text: str, max_sentences: int = 5) -> List[str]:
    """Split text into chunks of sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []
    
    for sentence in sentences:
        current.append(sentence)
        if len(current) >= max_sentences:
            chunks.append(' '.join(current))
            current = []
    
    if current:
        chunks.append(' '.join(current))
    return chunks
```

**Pros**: Preserves boundaries, coherent chunks, better readability
**Cons**: Variable sizes, may not respect structure
**Best For**: Natural language text, general documents

### 3. Semantic Chunking

**Concept**: Split based on semantic similarity between sentences.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunking(text: str, similarity_threshold: float = 0.7) -> List[str]:
    """Split text based on semantic similarity."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = re.split(r'(?<=[.!?])\s+', text)
    embeddings = model.encode(sentences)
    
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i-1], embeddings[i])
        if similarity >= similarity_threshold:
            current.append(sentences[i])
        else:
            chunks.append(' '.join(current))
            current = [sentences[i]]
    
    if current:
        chunks.append(' '.join(current))
    return chunks
```

**Pros**: Semantically coherent, adapts to content, better topic boundaries
**Cons**: Computationally expensive, requires embedding model
**Best For**: Multi-topic documents, when quality > speed

### 4. Recursive Chunking

**Concept**: Split hierarchically using multiple separators (paragraphs → sentences → words).

```python
def recursive_chunking(text: str, chunk_size: int = 500, 
                      separators: List[str] = None) -> List[str]:
    """Recursively split text using hierarchy of separators."""
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ']
    
    chunks = []
    
    def split_text(text: str, sep_index: int = 0):
        if len(text) <= chunk_size:
            chunks.append(text)
            return
        
        if sep_index >= len(separators):
            chunks.append(text[:chunk_size])
            split_text(text[chunk_size:], 0)
            return
        
        separator = separators[sep_index]
        parts = text.split(separator)
        current = ""
        
        for part in parts:
            if len(current) + len(part) + len(separator) <= chunk_size:
                current += part + separator
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    split_text(part, sep_index + 1)
                else:
                    current = part + separator
        
        if current:
            chunks.append(current.strip())
    
    split_text(text)
    return chunks
```

**Pros**: Respects structure, flexible, balances size and coherence
**Cons**: More complex, requires tuning
**Best For**: Structured documents, code, mixed content

---

## Chunk Overlap

**Why**: Preserves context at boundaries, prevents information loss when sentences span chunks.

```python
def chunk_with_overlap(text: str, chunk_size: int = 500, 
                      overlap: int = 100) -> List[str]:
    """Create overlapping chunks."""
    chunks, start = [], 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if end == len(text):
            break
    
    return chunks
```

**Guidelines**: Use 10-20% overlap (typical). More overlap = more context but more storage

---

## Chunk Size Optimization

**Factors**:
- **Model Context**: Must fit within LLM limits (typical: 500-1000 tokens)
- **Retrieval Quality**: Sweet spot is 300-800 tokens
- **Domain**: Technical docs (smaller), narratives (larger), Q&A (medium)

```python
def evaluate_chunk_size(documents: List[str], queries: List[str], 
                       chunk_sizes: List[int]) -> Dict:
    """Test different chunk sizes."""
    results = {}
    for size in chunk_sizes:
        chunks = []
        for doc in documents:
            chunks.extend(fixed_size_chunking(doc, chunk_size=size))
        avg_relevance = test_retrieval(chunks, queries)
        results[size] = {'num_chunks': len(chunks), 'avg_relevance': avg_relevance}
    return results
```

---

## Advanced Techniques

```python
# Metadata Enrichment
def chunk_with_metadata(text: str, source: str, chunk_size: int = 500) -> List[Dict]:
    """Create chunks with metadata."""
    chunks = fixed_size_chunking(text, chunk_size)
    return [{
        'text': chunk,
        'metadata': {'source': source, 'chunk_id': i, 'total_chunks': len(chunks)}
    } for i, chunk in enumerate(chunks)]

# Sliding Window
def sliding_window_chunking(text: str, window_size: int = 500, 
                           step_size: int = 250) -> List[str]:
    """Create chunks with sliding window."""
    chunks = []
    for start in range(0, len(text), step_size):
        end = min(start + window_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
    return chunks
```

---

## 💻 Exercises (40 min)

### Exercise 1: Fixed-Size Chunker
Implement fixed-size chunking with configurable overlap.

### Exercise 2: Semantic Chunker
Build a semantic chunker using sentence embeddings.

### Exercise 3: Recursive Chunker
Create a recursive chunker with custom separators.

### Exercise 4: Chunk Optimizer
Develop a tool to find optimal chunk size for a dataset.

### Exercise 5: Metadata Enrichment
Add metadata tracking to chunks for better retrieval.

---

## ✅ Quiz

Test your understanding in `quiz.md`

---

## 🎯 Key Takeaways

- Chunking is critical for RAG quality and performance
- Fixed-size is simple but may break context
- Sentence/paragraph chunking preserves structure
- Semantic chunking groups by topic similarity
- Recursive chunking balances size and coherence
- Overlap preserves context at boundaries (10-20% typical)
- Optimal chunk size: 300-800 tokens (depends on use case)
- Metadata enrichment improves retrieval and tracking
- Test different strategies for your specific domain
- Trade-offs: coherence vs size, quality vs speed

---

## 📚 Resources

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Chunking Strategies Paper](https://arxiv.org/abs/2307.03172)
- [Sentence Transformers](https://www.sbert.net/)
- [Optimal Chunk Size Study](https://www.pinecone.io/learn/chunking-strategies/)

---

## Tomorrow: Day 80 - Vector Embeddings

Learn about vector embeddings for semantic search, including embedding models, dimensionality, and similarity metrics.
