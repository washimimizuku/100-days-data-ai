"""
Day 81: Vector Databases - Exercises

NOTE: Requires faiss-cpu and chromadb
pip install faiss-cpu chromadb sentence-transformers
"""

from typing import List, Dict
import numpy as np


def exercise_1_faiss_index():
    """
    Exercise 1: FAISS Index
    
    Build a FAISS-based system that:
    - Creates different index types (Flat, IVF, HNSW)
    - Adds vectors to each index
    - Performs similarity search
    - Compares performance across index types
    
    TODO: Implement Flat index
    TODO: Implement IVF index with training
    TODO: Implement HNSW index
    TODO: Compare search performance
    """
    pass


def exercise_2_chroma_database():
    """
    Exercise 2: Chroma Database
    
    Implement CRUD operations with Chroma:
    - Create collection
    - Insert documents with metadata
    - Query with semantic search
    - Update existing documents
    - Delete documents
    - Filter by metadata
    
    TODO: Implement create/insert
    TODO: Implement query with filters
    TODO: Implement update
    TODO: Implement delete
    """
    pass


def exercise_3_index_comparison():
    """
    Exercise 3: Index Comparison
    
    Compare different index types:
    - Measure indexing time
    - Measure query time
    - Calculate accuracy (recall@k)
    - Analyze memory usage
    - Generate comparison report
    
    TODO: Implement performance measurement
    TODO: Calculate accuracy metrics
    TODO: Compare memory usage
    TODO: Generate report
    """
    pass


def exercise_4_metadata_filtering():
    """
    Exercise 4: Metadata Filtering
    
    Build filtered search system:
    - Add documents with rich metadata
    - Implement equality filters
    - Implement range filters
    - Implement compound filters (AND/OR)
    - Test filter performance
    
    TODO: Implement metadata storage
    TODO: Add equality filters
    TODO: Add range filters
    TODO: Add compound filters
    """
    pass


def exercise_5_batch_operations():
    """
    Exercise 5: Batch Operations
    
    Optimize with batch processing:
    - Batch insert large datasets
    - Batch query multiple queries
    - Measure performance improvement
    - Compare with single operations
    
    TODO: Implement batch insert
    TODO: Implement batch query
    TODO: Measure performance
    TODO: Compare with single ops
    """
    pass


if __name__ == "__main__":
    print("Day 81: Vector Databases - Exercises\n")
    print("=" * 60)
    
    # Uncomment to run exercises
    # print("\n1. FAISS Index")
    # exercise_1_faiss_index()
    
    # print("\n2. Chroma Database")
    # exercise_2_chroma_database()
    
    # print("\n3. Index Comparison")
    # exercise_3_index_comparison()
    
    # print("\n4. Metadata Filtering")
    # exercise_4_metadata_filtering()
    
    # print("\n5. Batch Operations")
    # exercise_5_batch_operations()
