# Day 84: Mini Project - RAG System - Detailed Specification

## Project Goal

Build a complete RAG (Retrieval-Augmented Generation) system that can:
1. Process and chunk documents intelligently
2. Generate and store vector embeddings
3. Retrieve relevant context using multiple strategies
4. Generate accurate answers with citations
5. Provide an interactive query interface

---

## Technical Requirements

### 1. Document Processing Module

**Class**: `DocumentChunker`

**Methods**:
- `chunk_text(text, chunk_size=500, overlap=50)` - Split text into overlapping chunks
- `chunk_by_sentences(text, max_tokens=500)` - Chunk by sentence boundaries
- `extract_metadata(text, source)` - Extract metadata from document
- `load_document(filepath)` - Load from file (txt, md)

**Requirements**:
- Preserve sentence boundaries
- Maintain context with overlap
- Extract source information
- Handle edge cases (empty docs, very short docs)

---

### 2. Embedding & Storage Module

**Class**: `VectorStore`

**Methods**:
- `__init__(collection_name, model_name)` - Initialize with ChromaDB
- `add_documents(chunks, metadatas)` - Add documents with embeddings
- `search(query, top_k, filter)` - Semantic search
- `hybrid_search(query, top_k, alpha)` - Combine semantic + keyword
- `get_stats()` - Return collection statistics

**Requirements**:
- Use sentence-transformers for embeddings
- Store in ChromaDB with metadata
- Support batch operations
- Handle duplicate detection

---

### 3. Retrieval Module

**Class**: `Retriever`

**Methods**:
- `semantic_search(query, top_k)` - Pure vector search
- `keyword_search(query, top_k)` - BM25-style search
- `hybrid_search(query, top_k, alpha)` - Weighted combination
- `rerank(query, results)` - Rerank by relevance
- `filter_by_metadata(results, filters)` - Apply metadata filters

**Requirements**:
- Multiple retrieval strategies
- Configurable weights for hybrid
- Metadata filtering support
- Return scores with results

---

### 4. Generation Module

**Class**: `Generator`

**Methods**:
- `generate_answer(query, context, model)` - Generate answer from context
- `extract_citations(answer, sources)` - Find citation markers
- `score_confidence(answer, context)` - Estimate confidence
- `format_response(answer, sources)` - Format with citations

**Requirements**:
- Use Ollama for generation
- Include citation markers [1], [2], etc.
- Track source documents
- Handle generation errors

---

### 5. Main RAG System

**Class**: `RAGSystem`

**Methods**:
- `__init__(collection_name, model_name)` - Initialize all components
- `index_documents(documents, metadatas)` - Process and index
- `query(question, top_k, method)` - End-to-end query
- `get_stats()` - System statistics
- `clear()` - Clear all data

**Workflow**:
1. Chunk documents
2. Generate embeddings
3. Store in vector DB
4. Retrieve relevant chunks
5. Generate answer with citations

---

## Implementation Details

### Document Chunking Strategy

```python
# Sentence-based chunking with overlap
def chunk_by_sentences(text, max_tokens=500, overlap_sentences=2):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        tokens = count_tokens(sentence)
        if current_tokens + tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Keep last N sentences for overlap
            current_chunk = current_chunk[-overlap_sentences:]
            current_tokens = sum(count_tokens(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_tokens += tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### Hybrid Search Algorithm

```python
def hybrid_search(query, top_k=5, alpha=0.7):
    # Semantic scores (0-1)
    semantic_scores = vector_search(query, top_k * 2)
    
    # Keyword scores (0-1)
    keyword_scores = bm25_search(query, top_k * 2)
    
    # Combine with weighted average
    combined = {}
    for doc_id in set(semantic_scores.keys()) | set(keyword_scores.keys()):
        sem_score = semantic_scores.get(doc_id, 0)
        key_score = keyword_scores.get(doc_id, 0)
        combined[doc_id] = alpha * sem_score + (1 - alpha) * key_score
    
    # Return top-k
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

### Answer Generation Prompt

```python
prompt_template = """Answer the question based on the context below. 
Include citations [1], [2], etc. to indicate which source supports each claim.

Context:
{context}

Question: {question}

Answer with citations:"""

# Format context with source numbers
context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])
```

---

## File Structure

### rag_system.py (< 400 lines)

Contains:
- `DocumentChunker` class
- `VectorStore` class
- `Retriever` class
- `Generator` class
- `RAGSystem` class (main orchestrator)

### query.py (< 200 lines)

Interactive CLI:
- Command parser
- Result formatter
- Query history
- Statistics display

### test_rag_system.sh (< 100 lines)

Test script:
- Dependency checks
- Unit tests for each component
- Integration test
- Performance benchmarks

### requirements.txt

```
chromadb>=0.4.0
sentence-transformers>=2.2.0
ollama>=0.1.0
numpy>=1.24.0
```

---

## Testing Strategy

### Unit Tests

1. **Document Chunking**
   - Test chunk size limits
   - Verify overlap
   - Check metadata extraction

2. **Embedding Generation**
   - Test batch processing
   - Verify embedding dimensions
   - Check normalization

3. **Vector Search**
   - Test semantic search
   - Test keyword search
   - Test hybrid search
   - Verify ranking

4. **Answer Generation**
   - Test citation extraction
   - Verify source tracking
   - Check error handling

### Integration Test

```python
# End-to-end test
rag = RAGSystem("test_collection")

# Index sample documents
docs = [
    "Python is a programming language.",
    "Machine learning uses algorithms to learn from data.",
    "RAG combines retrieval with generation."
]
rag.index_documents(docs)

# Query
result = rag.query("What is Python?", top_k=2)

# Verify
assert "Python" in result['answer']
assert len(result['sources']) > 0
assert result['confidence'] > 0
```

---

## Performance Targets

- **Indexing**: < 1 second per 1000 tokens
- **Retrieval**: < 100ms for top-10 results
- **Generation**: < 3 seconds per answer
- **Memory**: < 500MB for 10k documents

---

## Error Handling

1. **Ollama not available**: Graceful fallback message
2. **Empty query**: Return helpful error
3. **No results found**: Suggest query refinement
4. **Generation timeout**: Retry with shorter context
5. **Invalid document format**: Skip with warning

---

## Success Metrics

- ✅ All components implemented
- ✅ All tests passing
- ✅ Files under 400 lines
- ✅ Interactive CLI working
- ✅ Citations tracked correctly
- ✅ Performance targets met
- ✅ Error handling robust

---

## Deliverables

1. `rag_system.py` - Complete implementation
2. `query.py` - Interactive interface
3. `test_rag_system.sh` - Test suite
4. `requirements.txt` - Dependencies
5. Working demo with sample documents

---

## Timeline

- **Hour 1**: Document chunking + embedding storage
- **Hour 2**: Retrieval engine + generation
- **Hour 3**: Query interface + testing + refinement

---

## Next Steps After Completion

1. Test with real documents
2. Experiment with different chunk sizes
3. Try different retrieval strategies
4. Optimize for your use case
5. Add custom features
6. Deploy for production use
