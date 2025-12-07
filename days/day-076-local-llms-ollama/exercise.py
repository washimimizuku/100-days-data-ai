"""
Day 76: Local LLMs with Ollama - Exercises

NOTE: These exercises require Ollama to be installed and running.
Install: https://ollama.com/download
Pull a model: ollama pull mistral
"""

import ollama
from typing import List, Dict, Optional
import time


def exercise_1_model_manager():
    """
    Exercise 1: Model Manager
    
    Create a ModelManager class that:
    - Lists all installed models with their sizes
    - Checks if a specific model is installed
    - Provides model recommendations based on task type
    - Estimates memory requirements
    
    TODO: Implement the ModelManager class
    TODO: Add method to list models with details
    TODO: Add method to check model availability
    TODO: Add method to recommend model for task type
    """
    pass


def exercise_2_smart_qa():
    """
    Exercise 2: Smart Q&A System
    
    Build a Q&A system that:
    - Analyzes question complexity (simple vs complex)
    - Selects appropriate model (phi for simple, mistral for complex)
    - Uses lower temperature for factual questions
    - Uses higher temperature for creative questions
    - Returns answer with metadata (model used, tokens, time)
    
    TODO: Implement question complexity analyzer
    TODO: Implement model selector
    TODO: Implement Q&A function with metadata
    TODO: Test with different question types
    """
    pass


def exercise_3_document_processor():
    """
    Exercise 3: Document Processor
    
    Create a DocumentProcessor that:
    - Summarizes documents in specified length
    - Extracts key points (bullet list)
    - Answers questions about the document
    - Handles long documents by chunking
    
    TODO: Implement summarization function
    TODO: Implement key points extraction
    TODO: Implement document Q&A
    TODO: Add chunking for long documents
    """
    pass


def exercise_4_conversational_memory():
    """
    Exercise 4: Conversational Memory
    
    Build a ChatBot with:
    - Conversation history management
    - Context window truncation (keep recent messages)
    - Conversation export/import (save/load)
    - Token counting and limits
    - System prompt customization
    
    TODO: Implement ChatBot class with history
    TODO: Add context truncation logic
    TODO: Add save/load conversation methods
    TODO: Add token estimation
    """
    pass


def exercise_5_semantic_search():
    """
    Exercise 5: Semantic Search
    
    Create a semantic search system that:
    - Generates embeddings for documents
    - Stores documents with their embeddings
    - Finds similar documents using cosine similarity
    - Returns top-k most similar documents
    - Handles new document additions
    
    TODO: Implement embedding generation
    TODO: Implement document storage with embeddings
    TODO: Implement similarity search
    TODO: Add top-k retrieval
    """
    pass


if __name__ == "__main__":
    print("Day 76: Local LLMs with Ollama - Exercises\n")
    print("=" * 60)
    
    # Uncomment to run exercises
    # print("\n1. Model Manager")
    # exercise_1_model_manager()
    
    # print("\n2. Smart Q&A System")
    # exercise_2_smart_qa()
    
    # print("\n3. Document Processor")
    # exercise_3_document_processor()
    
    # print("\n4. Conversational Memory")
    # exercise_4_conversational_memory()
    
    # print("\n5. Semantic Search")
    # exercise_5_semantic_search()
