"""Day 84: RAG System Implementation"""

from typing import List, Dict, Tuple, Optional
import re
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np


class DocumentChunker:
    """Intelligent document chunking with overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into overlapping chunks."""
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({'text': chunk_text, 'metadata': metadata or {}})
                overlap_words = self.overlap
                while current_chunk and sum(len(s.split()) for s in current_chunk) > overlap_words:
                    current_chunk.pop(0)
                current_length = sum(len(s.split()) for s in current_chunk)
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append({'text': ' '.join(current_chunk), 'metadata': metadata or {}})
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def load_document(self, filepath: str) -> str:
        """Load document from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


class VectorStore:
    """Vector storage using ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_docs", model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.Client()
        self.model = SentenceTransformer(model_name)
        try:
            self.collection = self.client.create_collection(collection_name)
        except:
            self.collection = self.client.get_collection(collection_name)
    
    def add_documents(self, chunks: List[Dict]) -> int:
        """Add documents with embeddings."""
        if not chunks:
            return 0
        texts = [c['text'] for c in chunks]
        metadatas = [c.get('metadata', {}) for c in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
        return len(chunks)
    
    def search(self, query: str, top_k: int = 5, where: Dict = None) -> List[Dict]:
        """Semantic search."""
        results = self.collection.query(query_texts=[query], n_results=top_k, where=where)
        return self._format_results(results)
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results."""
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        return formatted
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {'count': self.collection.count(), 'name': self.collection.name}
    
    def clear(self):
        """Clear all documents."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(self.collection.name)
        except:
            pass


class Retriever:
    """Advanced retrieval strategies."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Pure vector similarity search."""
        return self.vector_store.search(query, top_k)
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based search."""
        query_terms = set(query.lower().split())
        all_results = self.vector_store.search(query, top_k * 3)
        scored = []
        for result in all_results:
            doc_terms = set(result['text'].lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / len(query_terms) if query_terms else 0
            scored.append((result, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in scored[:top_k]]
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict]:
        """Combine semantic and keyword search."""
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        combined = {}
        for result in semantic_results:
            text = result['text']
            combined[text] = {'result': result, 'semantic': 1.0, 'keyword': 0.0}
        for result in keyword_results:
            text = result['text']
            if text in combined:
                combined[text]['keyword'] = 1.0
            else:
                combined[text] = {'result': result, 'semantic': 0.0, 'keyword': 1.0}
        for text in combined:
            sem_score = combined[text]['semantic']
            key_score = combined[text]['keyword']
            combined[text]['score'] = alpha * sem_score + (1 - alpha) * key_score
        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return [r['result'] for r in sorted_results[:top_k]]
    
    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results by relevance."""
        query_terms = set(query.lower().split())
        for result in results:
            doc_terms = set(result['text'].lower().split())
            overlap = len(query_terms & doc_terms)
            result['relevance_score'] = overlap / max(len(query_terms), 1)
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return results


class Generator:
    """Answer generation with citations."""
    
    def __init__(self, model: str = "mistral"):
        self.model = model
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Generate answer from context."""
        if not context_chunks:
            return {'answer': 'No relevant information found.', 'sources': [], 'confidence': 0.0}
        context = self._format_context(context_chunks)
        prompt = f"""Answer the question based on the context below. Include citations [1], [2], etc. to indicate sources.

Context:
{context}

Question: {query}

Answer with citations:"""
        try:
            response = ollama.generate(model=self.model, prompt=prompt, options={'temperature': 0.3, 'num_predict': 200})
            answer = response['response'].strip()
            citations = self._extract_citations(answer)
            confidence = self._score_confidence(answer, context_chunks)
            return {'answer': answer, 'sources': context_chunks, 'citations': citations, 'confidence': confidence}
        except Exception as e:
            return {'answer': f'Error generating answer: {e}', 'sources': context_chunks, 'citations': [], 'confidence': 0.0}
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format context with source numbers."""
        return '\n\n'.join([f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(chunks)])
    
    def _extract_citations(self, answer: str) -> List[int]:
        """Extract citation numbers from answer."""
        citations = re.findall(r'\[(\d+)\]', answer)
        return [int(c) for c in citations]
    
    def _score_confidence(self, answer: str, chunks: List[Dict]) -> float:
        """Estimate answer confidence."""
        if not answer or 'error' in answer.lower():
            return 0.0
        citations = self._extract_citations(answer)
        if not citations:
            return 0.5
        return min(1.0, len(citations) / len(chunks) + 0.3)


class RAGSystem:
    """Complete RAG system orchestrator."""
    
    def __init__(self, collection_name: str = "rag_docs", model_name: str = "all-MiniLM-L6-v2", llm_model: str = "mistral"):
        self.chunker = DocumentChunker(chunk_size=500, overlap=50)
        self.vector_store = VectorStore(collection_name, model_name)
        self.retriever = Retriever(self.vector_store)
        self.generator = Generator(llm_model)
    
    def index_documents(self, documents: List[str], metadatas: List[Dict] = None) -> Dict:
        """Process and index documents."""
        if metadatas is None:
            metadatas = [{}] * len(documents)
        all_chunks = []
        for doc, metadata in zip(documents, metadatas):
            chunks = self.chunker.chunk_text(doc, metadata)
            all_chunks.extend(chunks)
        count = self.vector_store.add_documents(all_chunks)
        return {'indexed': count, 'documents': len(documents), 'chunks': len(all_chunks)}
    
    def index_file(self, filepath: str, metadata: Dict = None) -> Dict:
        """Index document from file."""
        text = self.chunker.load_document(filepath)
        return self.index_documents([text], [metadata or {}])
    
    def query(self, question: str, top_k: int = 3, method: str = "semantic") -> Dict:
        """Query the RAG system."""
        if method == "semantic":
            results = self.retriever.semantic_search(question, top_k)
        elif method == "keyword":
            results = self.retriever.keyword_search(question, top_k)
        elif method == "hybrid":
            results = self.retriever.hybrid_search(question, top_k)
        else:
            results = self.retriever.semantic_search(question, top_k)
        if results:
            results = self.retriever.rerank(question, results)
        return self.generator.generate_answer(question, results)
    
    def search_only(self, question: str, top_k: int = 5, method: str = "semantic") -> List[Dict]:
        """Search without generation."""
        if method == "semantic":
            return self.retriever.semantic_search(question, top_k)
        elif method == "keyword":
            return self.retriever.keyword_search(question, top_k)
        elif method == "hybrid":
            return self.retriever.hybrid_search(question, top_k)
        return self.retriever.semantic_search(question, top_k)
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return self.vector_store.get_stats()
    
    def clear(self):
        """Clear all indexed documents."""
        self.vector_store.clear()


if __name__ == "__main__":
    print("RAG System - Quick Test\n" + "=" * 50)
    rag = RAGSystem(collection_name="test_rag")
    sample_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with language model generation.",
        "Vector databases store embeddings and enable fast similarity search for semantic retrieval.",
        "LangChain is a framework for building applications with large language models and external data sources."
    ]
    print("\nIndexing documents...")
    result = rag.index_documents(sample_docs)
    print(f"Indexed: {result['indexed']} chunks from {result['documents']} documents")
    print(f"\nStats: {rag.get_stats()}")
    query = "What is RAG?"
    print(f"\nQuery: {query}")
    answer = rag.query(query, top_k=2, method="semantic")
    print(f"\nAnswer:\n{answer['answer']}")
    print(f"\nSources ({len(answer['sources'])}):")
    for i, source in enumerate(answer['sources'], 1):
        print(f"  [{i}] {source['text'][:80]}...")
    print(f"\nConfidence: {answer['confidence']:.2f}")
    print(f"Citations: {answer['citations']}")
