#!/bin/bash

echo "=========================================="
echo "RAG System - Test Suite"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi
echo "✅ Python3 found"

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Install from https://ollama.com"
    exit 1
fi
echo "✅ Ollama found"

# Check if Ollama is running
if ! ollama list &> /dev/null; then
    echo "❌ Ollama is not running. Start with: ollama serve"
    exit 1
fi
echo "✅ Ollama is running"

# Check for mistral model
if ! ollama list | grep -q "mistral"; then
    echo "⚠️  Mistral model not found. Pulling..."
    ollama pull mistral
fi
echo "✅ Mistral model available"

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
python3 -c "import chromadb" 2>/dev/null || { echo "❌ chromadb not installed. Run: pip install chromadb"; exit 1; }
echo "✅ chromadb installed"

python3 -c "import sentence_transformers" 2>/dev/null || { echo "❌ sentence-transformers not installed. Run: pip install sentence-transformers"; exit 1; }
echo "✅ sentence-transformers installed"

python3 -c "import ollama" 2>/dev/null || { echo "❌ ollama not installed. Run: pip install ollama"; exit 1; }
echo "✅ ollama installed"

python3 -c "import numpy" 2>/dev/null || { echo "❌ numpy not installed. Run: pip install numpy"; exit 1; }
echo "✅ numpy installed"

echo ""
echo "=========================================="
echo "Running Tests"
echo "=========================================="
echo ""

# Test 1: Document Chunking
echo "Test 1: Document Chunking"
python3 << 'EOF'
from rag_system import DocumentChunker

chunker = DocumentChunker(chunk_size=50, overlap=10)
text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
chunks = chunker.chunk_text(text)

assert len(chunks) > 0, "No chunks created"
assert all('text' in c for c in chunks), "Missing text field"
print(f"✅ Created {len(chunks)} chunks")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Document chunking test failed"
    exit 1
fi

# Test 2: Vector Store
echo ""
echo "Test 2: Vector Store"
python3 << 'EOF'
from rag_system import VectorStore

store = VectorStore(collection_name="test_store")
chunks = [
    {'text': 'Python is a programming language', 'metadata': {}},
    {'text': 'Machine learning uses algorithms', 'metadata': {}}
]
count = store.add_documents(chunks)
assert count == 2, f"Expected 2, got {count}"

results = store.search("programming", top_k=1)
assert len(results) > 0, "No search results"
print(f"✅ Indexed {count} documents, search returned {len(results)} results")

store.clear()
EOF

if [ $? -ne 0 ]; then
    echo "❌ Vector store test failed"
    exit 1
fi

# Test 3: Retrieval
echo ""
echo "Test 3: Retrieval"
python3 << 'EOF'
from rag_system import VectorStore, Retriever

store = VectorStore(collection_name="test_retrieval")
chunks = [
    {'text': 'Python programming language', 'metadata': {}},
    {'text': 'Java programming language', 'metadata': {}},
    {'text': 'Machine learning algorithms', 'metadata': {}}
]
store.add_documents(chunks)

retriever = Retriever(store)
results = retriever.semantic_search("programming", top_k=2)
assert len(results) == 2, f"Expected 2 results, got {len(results)}"
print(f"✅ Semantic search returned {len(results)} results")

hybrid_results = retriever.hybrid_search("programming", top_k=2)
assert len(hybrid_results) > 0, "Hybrid search failed"
print(f"✅ Hybrid search returned {len(hybrid_results)} results")

store.clear()
EOF

if [ $? -ne 0 ]; then
    echo "❌ Retrieval test failed"
    exit 1
fi

# Test 4: Generation
echo ""
echo "Test 4: Generation"
python3 << 'EOF'
from rag_system import Generator

generator = Generator(model="mistral")
chunks = [
    {'text': 'Python is a high-level programming language.', 'metadata': {}},
    {'text': 'Python is known for its simplicity.', 'metadata': {}}
]

result = generator.generate_answer("What is Python?", chunks)
assert 'answer' in result, "Missing answer field"
assert 'sources' in result, "Missing sources field"
assert 'confidence' in result, "Missing confidence field"
print(f"✅ Generated answer with {len(result['sources'])} sources")
print(f"   Confidence: {result['confidence']:.2f}")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Generation test failed"
    exit 1
fi

# Test 5: End-to-End RAG
echo ""
echo "Test 5: End-to-End RAG System"
python3 << 'EOF'
from rag_system import RAGSystem

rag = RAGSystem(collection_name="test_e2e")

docs = [
    "Python is a programming language created by Guido van Rossum.",
    "Machine learning is a subset of artificial intelligence.",
    "RAG combines retrieval with generation for better answers."
]

index_result = rag.index_documents(docs)
assert index_result['indexed'] > 0, "Indexing failed"
print(f"✅ Indexed {index_result['indexed']} chunks")

query_result = rag.query("What is Python?", top_k=2)
assert 'answer' in query_result, "Query failed"
assert len(query_result['sources']) > 0, "No sources returned"
print(f"✅ Query returned answer with {len(query_result['sources'])} sources")

stats = rag.get_stats()
assert stats['count'] > 0, "No documents in collection"
print(f"✅ System stats: {stats['count']} documents")

rag.clear()
EOF

if [ $? -ne 0 ]; then
    echo "❌ End-to-end test failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All Tests Passed! ✅"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Try the interactive interface: python3 query.py"
echo "2. Index your own documents"
echo "3. Experiment with different retrieval methods"
echo ""
