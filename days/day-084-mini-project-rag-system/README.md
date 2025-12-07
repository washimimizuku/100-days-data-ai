# Day 84: Mini Project - RAG System

## 🎯 Project Overview

Build a production-ready Retrieval-Augmented Generation (RAG) system that combines document chunking, vector embeddings, similarity search, and LLM generation. This project integrates concepts from Days 78-83.

**Time**: 2-3 hours

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   RAG System                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Document    │  │  Embedding   │  │   Vector     │ │
│  │  Chunker     │  │  Generator   │  │   Store      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Retrieval Engine                        │  │
│  │  - Semantic Search  - Hybrid Search  - Reranking │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Generation Engine (Ollama)              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 Requirements

### Core Features

1. **Document Processing**
   - Load documents from multiple formats (txt, md, pdf)
   - Intelligent chunking with overlap
   - Metadata extraction and preservation
   - Chunk size optimization

2. **Embedding & Indexing**
   - Generate embeddings using sentence-transformers
   - Store vectors in ChromaDB
   - Batch processing for efficiency
   - Incremental updates

3. **Retrieval System**
   - Semantic search using cosine similarity
   - Hybrid search (semantic + keyword)
   - Metadata filtering
   - Result reranking

4. **Generation Engine**
   - Context-aware answer generation
   - Citation tracking
   - Source attribution
   - Confidence scoring

5. **Query Interface**
   - Interactive CLI
   - Query history
   - Result formatting
   - Performance metrics

---

## 🚀 Getting Started

### Prerequisites

```bash
# Install Ollama
# Visit: https://ollama.com/download

# Pull required model
ollama pull mistral

# Install Python dependencies
pip install -r requirements.txt
```

### Project Structure

```
day-084-mini-project-rag-system/
├── README.md                    # This file
├── project.md                   # Detailed specification
├── rag_system.py               # Main RAG implementation
├── query.py                    # Query interface
├── test_rag_system.sh          # Test script
└── requirements.txt            # Dependencies
```

---

## 💻 Implementation Guide

### Step 1: Document Chunker (30 min)

Implement intelligent document chunking:
- Split by sentences with overlap
- Preserve context boundaries
- Extract metadata (source, page, section)
- Handle multiple formats

### Step 2: Embedding & Storage (30 min)

Build vector storage system:
- Generate embeddings with sentence-transformers
- Store in ChromaDB with metadata
- Batch processing for large documents
- Support incremental updates

### Step 3: Retrieval Engine (40 min)

Create flexible retrieval:
- Semantic search with cosine similarity
- Hybrid search combining semantic + keyword
- Metadata filtering (date, source, type)
- Rerank results by relevance

### Step 4: Generation Engine (30 min)

Implement answer generation:
- Build context from retrieved chunks
- Generate answers with Ollama
- Track citations and sources
- Score confidence

### Step 5: Query Interface (20 min)

Build user interface:
- Interactive CLI with commands
- Display results with sources
- Show performance metrics
- Save query history

---

## 🎓 Learning Objectives

By completing this project, you will:
- Build end-to-end RAG pipeline
- Implement document chunking strategies
- Work with vector databases
- Apply retrieval techniques
- Integrate LLMs for generation
- Handle citations and sources
- Optimize for performance

---

## 🧪 Testing

Run the test script to validate your implementation:

```bash
chmod +x test_rag_system.sh
./test_rag_system.sh
```

The test script will:
1. Check dependencies (Ollama, ChromaDB)
2. Test document chunking
3. Test embedding generation
4. Test vector storage and retrieval
5. Test answer generation
6. Validate citations

---

## 📊 Example Usage

### Index Documents

```python
from rag_system import RAGSystem

# Initialize system
rag = RAGSystem(collection_name="my_docs")

# Index documents
documents = [
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers for complex patterns.",
    "RAG combines retrieval with generation for knowledge-intensive tasks."
]

rag.index_documents(documents)
print(f"Indexed {len(documents)} documents")
```

### Query System

```python
# Query with semantic search
result = rag.query(
    "What is machine learning?",
    top_k=3,
    method="semantic"
)

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
for i, source in enumerate(result['sources'], 1):
    print(f"  [{i}] {source['text'][:80]}...")
```

### Interactive Mode

```bash
python query.py

# Commands:
# > query <text>          - Ask a question
# > search <text>         - Search without generation
# > index <file>          - Add document
# > stats                 - Show statistics
# > history               - View query history
# > exit                  - Quit
```

---

## 🎯 Success Criteria

Your implementation should:
- ✅ Support document chunking with overlap
- ✅ Generate and store vector embeddings
- ✅ Implement semantic and hybrid search
- ✅ Generate answers with citations
- ✅ Track sources and confidence
- ✅ Provide interactive query interface
- ✅ Handle errors gracefully
- ✅ Keep files under 400 lines each

---

## 🔧 Troubleshooting

**Ollama not running**:
```bash
ollama serve
```

**ChromaDB errors**:
```bash
pip install --upgrade chromadb
```

**Slow embedding generation**:
- Use smaller model (all-MiniLM-L6-v2)
- Batch process documents
- Cache embeddings

**Poor retrieval quality**:
- Adjust chunk size (200-500 tokens)
- Increase chunk overlap (10-20%)
- Use hybrid search instead of semantic only
- Add metadata filtering

**Low quality answers**:
- Increase top_k (retrieve more context)
- Adjust temperature (lower for factual)
- Use reranking to improve relevance
- Provide more context in prompt

---

## 📚 Resources

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama Documentation](https://github.com/ollama/ollama)

---

## 🚀 Extensions (Optional)

1. **Advanced Retrieval**
   - Multi-query retrieval
   - Parent-child chunking
   - Contextual compression
   - Query expansion

2. **Enhanced Generation**
   - Multi-step reasoning
   - Self-consistency
   - Fact verification
   - Answer refinement

3. **Production Features**
   - Caching layer
   - Rate limiting
   - Monitoring and logging
   - A/B testing

4. **Web Interface**
   - FastAPI backend
   - React frontend
   - Real-time streaming
   - Document upload

---

## Tomorrow: Day 85 - Agent Concepts

Learn about AI agents that can reason, plan, and use tools to accomplish complex tasks autonomously.
