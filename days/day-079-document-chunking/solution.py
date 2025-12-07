"""
Day 79: Document Chunking - Solutions
"""

import re
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


# Sample text for testing
SAMPLE_TEXT = """
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. The process involves feeding data to algorithms that can identify patterns and make decisions.

Deep learning is a specialized form of machine learning that uses neural networks with multiple layers. These networks can automatically learn hierarchical representations of data, making them particularly effective for complex tasks like image recognition and natural language processing.

Natural language processing (NLP) focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a valuable way. Applications include chatbots, translation services, and sentiment analysis.

Computer vision is another important field that enables machines to interpret and understand visual information from the world. It powers technologies like facial recognition, autonomous vehicles, and medical image analysis.
"""


# Exercise 1: Fixed-Size Chunker
class FixedSizeChunker:
    """Fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[Dict]:
        """Split text into fixed-size chunks with metadata."""
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'metadata': {
                    'chunk_id': 0,
                    'char_count': len(text),
                    'start_pos': 0,
                    'end_pos': len(text)
                }
            }]
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'chunk_id': chunk_id,
                    'char_count': len(chunk_text),
                    'start_pos': start,
                    'end_pos': end,
                    'has_overlap': chunk_id > 0
                }
            })
            
            chunk_id += 1
            
            # Move start position with overlap
            start = end - self.overlap
            
            # Avoid infinite loop at end
            if end == len(text):
                break
        
        return chunks


def exercise_1_fixed_size_chunker():
    """Exercise 1: Fixed-Size Chunker"""
    print("Exercise 1: Fixed-Size Chunker")
    print("-" * 40)
    
    chunker = FixedSizeChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk(SAMPLE_TEXT)
    
    print(f"\nTotal chunks: {len(chunks)}")
    for chunk in chunks[:3]:  # Show first 3
        print(f"\nChunk {chunk['metadata']['chunk_id']}:")
        print(f"  Length: {chunk['metadata']['char_count']} chars")
        print(f"  Position: {chunk['metadata']['start_pos']}-{chunk['metadata']['end_pos']}")
        print(f"  Text: {chunk['text'][:80]}...")


# Exercise 2: Semantic Chunker
class SemanticChunker:
    """Semantic chunking based on sentence similarity."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 similarity_threshold: float = 0.7):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str) -> List[Dict]:
        """Create semantic chunks."""
        sentences = self.split_sentences(text)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [{'text': sentences[0], 'sentences': [sentences[0]]}]
        
        # Get embeddings
        embeddings = self.model.encode(sentences)
        
        # Group by similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_embeddings = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            prev_emb = embeddings[i-1]
            curr_emb = embeddings[i]
            
            similarity = np.dot(prev_emb, curr_emb) / (
                np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb)
            )
            
            if similarity >= self.similarity_threshold:
                # Similar - add to current chunk
                current_chunk.append(sentences[i])
                current_embeddings.append(embeddings[i])
            else:
                # Different topic - save current and start new
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'sentences': current_chunk.copy(),
                    'num_sentences': len(current_chunk)
                })
                current_chunk = [sentences[i]]
                current_embeddings = [embeddings[i]]
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'sentences': current_chunk.copy(),
                'num_sentences': len(current_chunk)
            })
        
        return chunks


def exercise_2_semantic_chunker():
    """Exercise 2: Semantic Chunker"""
    print("\nExercise 2: Semantic Chunker")
    print("-" * 40)
    
    chunker = SemanticChunker(similarity_threshold=0.6)
    chunks = chunker.chunk(SAMPLE_TEXT)
    
    print(f"\nTotal semantic chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Sentences: {chunk['num_sentences']}")
        print(f"  Text: {chunk['text'][:100]}...")


# Exercise 3: Recursive Chunker
class RecursiveChunker:
    """Recursive chunking with separator hierarchy."""
    
    def __init__(self, chunk_size: int = 500, 
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.separators = separators or ['\n\n', '\n', '. ', ' ']
    
    def chunk(self, text: str) -> List[str]:
        """Recursively split text."""
        return self._split_text(text, 0)
    
    def _split_text(self, text: str, sep_index: int) -> List[str]:
        """Recursive splitting helper."""
        # Base case: text fits in chunk
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # No more separators: force split
        if sep_index >= len(self.separators):
            chunks = []
            for i in range(0, len(text), self.chunk_size):
                chunks.append(text[i:i + self.chunk_size])
            return chunks
        
        # Try current separator
        separator = self.separators[sep_index]
        parts = text.split(separator)
        
        chunks = []
        current = ""
        
        for part in parts:
            # Try adding part to current chunk
            test_chunk = current + part + separator if current else part
            
            if len(test_chunk) <= self.chunk_size:
                current = test_chunk
            else:
                # Current chunk is full
                if current:
                    chunks.append(current.rstrip(separator))
                
                # Part too large: try next separator
                if len(part) > self.chunk_size:
                    chunks.extend(self._split_text(part, sep_index + 1))
                    current = ""
                else:
                    current = part + separator
        
        # Add remaining
        if current:
            chunks.append(current.rstrip(separator))
        
        return [c for c in chunks if c.strip()]


def exercise_3_recursive_chunker():
    """Exercise 3: Recursive Chunker"""
    print("\nExercise 3: Recursive Chunker")
    print("-" * 40)
    
    chunker = RecursiveChunker(chunk_size=300)
    chunks = chunker.chunk(SAMPLE_TEXT)
    
    print(f"\nTotal recursive chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk)} chars")
        print(f"  Text: {chunk[:80]}...")


# Exercise 4: Chunk Optimizer
class ChunkOptimizer:
    """Find optimal chunk size for dataset."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_chunk_size(self, documents: List[str], 
                           queries: List[str], 
                           chunk_sizes: List[int]) -> Dict:
        """Test different chunk sizes."""
        results = {}
        
        for size in chunk_sizes:
            # Chunk all documents
            all_chunks = []
            for doc in documents:
                chunker = FixedSizeChunker(chunk_size=size, overlap=size // 10)
                chunks = chunker.chunk(doc)
                all_chunks.extend([c['text'] for c in chunks])
            
            # Embed chunks
            chunk_embeddings = self.model.encode(all_chunks)
            
            # Test retrieval quality
            relevance_scores = []
            for query in queries:
                query_emb = self.model.encode([query])[0]
                similarities = np.dot(chunk_embeddings, query_emb)
                top_score = similarities.max() if len(similarities) > 0 else 0
                relevance_scores.append(top_score)
            
            results[size] = {
                'num_chunks': len(all_chunks),
                'avg_chunk_length': np.mean([len(c) for c in all_chunks]),
                'avg_relevance': np.mean(relevance_scores),
                'max_relevance': np.max(relevance_scores) if relevance_scores else 0
            }
        
        return results
    
    def recommend_size(self, results: Dict) -> int:
        """Recommend best chunk size."""
        # Choose size with highest average relevance
        best_size = max(results.items(), key=lambda x: x[1]['avg_relevance'])
        return best_size[0]


def exercise_4_chunk_optimizer():
    """Exercise 4: Chunk Optimizer"""
    print("\nExercise 4: Chunk Optimizer")
    print("-" * 40)
    
    optimizer = ChunkOptimizer()
    
    documents = [SAMPLE_TEXT]
    queries = ["What is machine learning?", "Explain deep learning"]
    chunk_sizes = [200, 400, 600]
    
    print("\nTesting chunk sizes...")
    results = optimizer.evaluate_chunk_size(documents, queries, chunk_sizes)
    
    print("\nResults:")
    for size, metrics in results.items():
        print(f"\nChunk size: {size}")
        print(f"  Num chunks: {metrics['num_chunks']}")
        print(f"  Avg length: {metrics['avg_chunk_length']:.0f} chars")
        print(f"  Avg relevance: {metrics['avg_relevance']:.3f}")
    
    recommended = optimizer.recommend_size(results)
    print(f"\nRecommended chunk size: {recommended}")


# Exercise 5: Metadata Enrichment
class MetadataChunker:
    """Chunker with rich metadata."""
    
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
    
    def chunk(self, text: str, source: str = "unknown") -> List[Dict]:
        """Create chunks with comprehensive metadata."""
        chunker = FixedSizeChunker(chunk_size=self.chunk_size, overlap=50)
        chunks = chunker.chunk(text)
        
        # Enrich with additional metadata
        enriched = []
        for chunk in chunks:
            words = chunk['text'].split()
            
            enriched.append({
                'text': chunk['text'],
                'metadata': {
                    **chunk['metadata'],
                    'source': source,
                    'word_count': len(words),
                    'total_chunks': len(chunks),
                    'is_first': chunk['metadata']['chunk_id'] == 0,
                    'is_last': chunk['metadata']['chunk_id'] == len(chunks) - 1,
                    'avg_word_length': np.mean([len(w) for w in words]) if words else 0
                }
            })
        
        return enriched
    
    def reconstruct(self, chunks: List[Dict]) -> str:
        """Reconstruct original text from chunks."""
        # Sort by chunk_id
        sorted_chunks = sorted(chunks, key=lambda x: x['metadata']['chunk_id'])
        
        # Remove overlaps (simplified - just concatenate)
        texts = [c['text'] for c in sorted_chunks]
        return ' '.join(texts)


def exercise_5_metadata_enrichment():
    """Exercise 5: Metadata Enrichment"""
    print("\nExercise 5: Metadata Enrichment")
    print("-" * 40)
    
    chunker = MetadataChunker(chunk_size=300)
    chunks = chunker.chunk(SAMPLE_TEXT, source="ml_tutorial.txt")
    
    print(f"\nTotal chunks: {len(chunks)}")
    for chunk in chunks[:2]:
        print(f"\nChunk {chunk['metadata']['chunk_id']}:")
        print(f"  Source: {chunk['metadata']['source']}")
        print(f"  Words: {chunk['metadata']['word_count']}")
        print(f"  Position: {chunk['metadata']['start_pos']}-{chunk['metadata']['end_pos']}")
        print(f"  First: {chunk['metadata']['is_first']}, Last: {chunk['metadata']['is_last']}")
        print(f"  Text: {chunk['text'][:60]}...")


if __name__ == "__main__":
    print("Day 79: Document Chunking - Solutions\n")
    print("=" * 60)
    
    try:
        exercise_1_fixed_size_chunker()
        exercise_2_semantic_chunker()
        exercise_3_recursive_chunker()
        exercise_4_chunk_optimizer()
        exercise_5_metadata_enrichment()
        
        print("\n" + "=" * 60)
        print("All exercises completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed:")
        print("pip install sentence-transformers numpy")
