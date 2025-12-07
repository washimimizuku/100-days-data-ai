"""
Day 82: Retrieval Strategies - Exercises

NOTE: Requires sentence-transformers, rank-bm25
pip install sentence-transformers rank-bm25 ollama
"""

from typing import List, Dict
import numpy as np


def exercise_1_hybrid_retriever():
    """
    Exercise 1: Hybrid Retriever
    
    Implement hybrid search that:
    - Performs dense retrieval (semantic)
    - Performs sparse retrieval (BM25)
    - Combines scores with weighted average
    - Returns top-k results with both scores
    
    TODO: Implement dense retrieval
    TODO: Implement sparse retrieval
    TODO: Combine scores with alpha parameter
    TODO: Return ranked results
    """
    pass


def exercise_2_query_expander():
    """
    Exercise 2: Query Expander
    
    Build query expansion that:
    - Expands with synonyms
    - Expands with LLM
    - Expands with pseudo-relevance feedback
    - Compares expansion methods
    
    TODO: Implement synonym expansion
    TODO: Implement LLM expansion
    TODO: Implement PRF expansion
    TODO: Compare methods
    """
    pass


def exercise_3_reranker():
    """
    Exercise 3: Reranker
    
    Create reranking system that:
    - Takes initial retrieval results
    - Scores query-document pairs
    - Reorders by relevance
    - Returns top-k reranked results
    
    TODO: Implement cross-encoder scoring
    TODO: Reorder by scores
    TODO: Compare with initial ranking
    TODO: Measure improvement
    """
    pass


def exercise_4_multi_stage_pipeline():
    """
    Exercise 4: Multi-Stage Pipeline
    
    Build complete pipeline that:
    - Stage 1: Query expansion
    - Stage 2: Hybrid retrieval (top 20)
    - Stage 3: Reranking (top 5)
    - Returns final results
    
    TODO: Implement query expansion stage
    TODO: Implement hybrid retrieval stage
    TODO: Implement reranking stage
    TODO: Connect all stages
    """
    pass


def exercise_5_retrieval_evaluator():
    """
    Exercise 5: Retrieval Evaluator
    
    Implement evaluation that:
    - Calculates precision@k
    - Calculates recall@k
    - Calculates MRR
    - Generates evaluation report
    
    TODO: Implement precision@k
    TODO: Implement recall@k
    TODO: Implement MRR
    TODO: Generate report
    """
    pass


if __name__ == "__main__":
    print("Day 82: Retrieval Strategies - Exercises\n")
    print("=" * 60)
    
    # Uncomment to run exercises
    # print("\n1. Hybrid Retriever")
    # exercise_1_hybrid_retriever()
    
    # print("\n2. Query Expander")
    # exercise_2_query_expander()
    
    # print("\n3. Reranker")
    # exercise_3_reranker()
    
    # print("\n4. Multi-Stage Pipeline")
    # exercise_4_multi_stage_pipeline()
    
    # print("\n5. Retrieval Evaluator")
    # exercise_5_retrieval_evaluator()
