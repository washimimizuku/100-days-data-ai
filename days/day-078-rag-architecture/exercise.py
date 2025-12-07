"""
Day 78: RAG Architecture - Exercises

NOTE: Requires sentence-transformers and ollama
pip install sentence-transformers ollama
"""

from typing import List, Dict, Tuple
import numpy as np


def exercise_1_basic_rag():
    """
    Exercise 1: Basic RAG System
    
    Build a simple RAG system that:
    - Indexes documents with embeddings
    - Retrieves top-k relevant documents for a query
    - Generates answer using retrieved context
    - Returns answer with source documents
    
    TODO: Implement document indexing
    TODO: Implement retrieval function
    TODO: Implement generation with context
    TODO: Test with sample queries
    """
    pass


def exercise_2_hybrid_retrieval():
    """
    Exercise 2: Hybrid Retrieval
    
    Implement hybrid retrieval that:
    - Performs semantic search (dense retrieval)
    - Performs keyword search (sparse retrieval)
    - Combines scores with weighted average
    - Returns top-k documents from combined ranking
    
    TODO: Implement semantic search
    TODO: Implement keyword search (BM25 or simple)
    TODO: Combine scores with alpha parameter
    TODO: Test and compare with single methods
    """
    pass


def exercise_3_query_rewriting():
    """
    Exercise 3: Query Rewriting
    
    Create a query rewriter that:
    - Analyzes the original query
    - Generates expanded/clarified version
    - Adds related terms and context
    - Improves retrieval quality
    
    TODO: Implement query analysis
    TODO: Implement query expansion
    TODO: Test with ambiguous queries
    TODO: Compare retrieval before/after rewriting
    """
    pass


def exercise_4_reranking():
    """
    Exercise 4: Reranking Pipeline
    
    Build a reranker that:
    - Takes initial retrieved documents
    - Scores each document for relevance
    - Filters low-scoring documents
    - Returns reordered list
    
    TODO: Implement relevance scoring
    TODO: Implement filtering logic
    TODO: Test with diverse document sets
    TODO: Measure improvement over initial retrieval
    """
    pass


def exercise_5_citation_system():
    """
    Exercise 5: Citation System
    
    Implement citation generation that:
    - Generates answer from context
    - Identifies which sources support each claim
    - Adds inline citations [1], [2], etc.
    - Provides source list at end
    
    TODO: Implement answer generation
    TODO: Implement citation insertion
    TODO: Format source list
    TODO: Test with multi-source queries
    """
    pass


if __name__ == "__main__":
    print("Day 78: RAG Architecture - Exercises\n")
    print("=" * 60)
    
    # Uncomment to run exercises
    # print("\n1. Basic RAG System")
    # exercise_1_basic_rag()
    
    # print("\n2. Hybrid Retrieval")
    # exercise_2_hybrid_retrieval()
    
    # print("\n3. Query Rewriting")
    # exercise_3_query_rewriting()
    
    # print("\n4. Reranking Pipeline")
    # exercise_4_reranking()
    
    # print("\n5. Citation System")
    # exercise_5_citation_system()
