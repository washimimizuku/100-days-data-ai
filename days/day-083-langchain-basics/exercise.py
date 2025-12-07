"""
Day 83: LangChain Basics - Exercises

NOTE: Requires langchain and ollama
pip install langchain langchain-community ollama
"""

from typing import List, Dict


def exercise_1_prompt_templates():
    """
    Exercise 1: Prompt Templates
    
    Create prompt templates that:
    - Use variable substitution
    - Support multiple input variables
    - Include few-shot examples
    - Format consistently
    
    TODO: Create basic prompt template
    TODO: Create chat prompt template
    TODO: Create few-shot template
    TODO: Test with different inputs
    """
    pass


def exercise_2_sequential_chain():
    """
    Exercise 2: Sequential Chain
    
    Build a chain that:
    - Extracts main topic from text
    - Generates summary of topic
    - Creates questions about topic
    - Chains all steps together
    
    TODO: Create topic extraction chain
    TODO: Create summary chain
    TODO: Create question generation chain
    TODO: Connect chains sequentially
    """
    pass


def exercise_3_conversation_memory():
    """
    Exercise 3: Conversation Memory
    
    Implement chatbot with:
    - Conversation buffer memory
    - Context preservation across turns
    - Memory clearing functionality
    - Conversation history export
    
    TODO: Create conversation chain with memory
    TODO: Implement multi-turn conversation
    TODO: Add memory management
    TODO: Export conversation history
    """
    pass


def exercise_4_rag_system():
    """
    Exercise 4: RAG System
    
    Build RAG system that:
    - Loads and chunks documents
    - Creates vector store
    - Implements retrieval
    - Generates answers with sources
    
    TODO: Load and chunk documents
    TODO: Create vector store with embeddings
    TODO: Implement retrieval chain
    TODO: Add source citations
    """
    pass


def exercise_5_output_parser():
    """
    Exercise 5: Output Parser
    
    Create structured parsing that:
    - Defines output schema
    - Parses LLM responses
    - Validates structure
    - Handles parsing errors
    
    TODO: Define response schema
    TODO: Create output parser
    TODO: Parse LLM responses
    TODO: Add error handling
    """
    pass


if __name__ == "__main__":
    print("Day 83: LangChain Basics - Exercises\n")
    print("=" * 60)
    
    # Uncomment to run exercises
    # print("\n1. Prompt Templates")
    # exercise_1_prompt_templates()
    
    # print("\n2. Sequential Chain")
    # exercise_2_sequential_chain()
    
    # print("\n3. Conversation Memory")
    # exercise_3_conversation_memory()
    
    # print("\n4. RAG System")
    # exercise_4_rag_system()
    
    # print("\n5. Output Parser")
    # exercise_5_output_parser()
