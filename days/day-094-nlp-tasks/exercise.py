"""Day 94: NLP Tasks - Exercises

NOTE: Uses mock implementations for learning without model downloads.
"""

from typing import List, Dict, Any, Tuple
import re


# Exercise 1: Text Classification
class TextClassifier:
    """Classify text into categories."""
    
    def __init__(self, categories: List[str] = None):
        self.categories = categories or ["positive", "negative", "neutral"]
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text.
        
        TODO: Lowercase
        TODO: Remove special characters
        TODO: Remove extra whitespace
        """
        pass
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text sentiment.
        
        TODO: Preprocess text
        TODO: Classify sentiment
        TODO: Return label and confidence
        """
        pass
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple texts.
        
        TODO: Process all texts
        TODO: Return list of results
        """
        pass


# Exercise 2: Named Entity Recognition
class NamedEntityRecognizer:
    """Extract named entities from text."""
    
    def __init__(self):
        self.entity_types = ["PER", "ORG", "LOC", "DATE", "MISC"]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities.
        
        Returns list of entities:
        {
            'text': 'Apple Inc.',
            'type': 'ORG',
            'start': 0,
            'end': 10
        }
        
        TODO: Tokenize text
        TODO: Identify entities
        TODO: Return entity list
        """
        pass
    
    def group_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Group entities by type.
        
        TODO: Group by entity type
        TODO: Return dictionary of lists
        """
        pass


# Exercise 3: Text Generation
class TextGenerator:
    """Generate text continuations."""
    
    def __init__(self, max_length: int = 50):
        self.max_length = max_length
    
    def generate(self, prompt: str, num_tokens: int = 20) -> str:
        """
        Generate text continuation.
        
        TODO: Process prompt
        TODO: Generate continuation
        TODO: Return complete text
        """
        pass
    
    def generate_multiple(self, prompt: str, 
                         num_sequences: int = 3) -> List[str]:
        """
        Generate multiple continuations.
        
        TODO: Generate multiple sequences
        TODO: Return list of texts
        """
        pass


# Exercise 4: Text Summarization
class TextSummarizer:
    """Summarize long text."""
    
    def __init__(self):
        self.max_summary_length = 130
        self.min_summary_length = 30
    
    def extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """
        Create extractive summary.
        
        TODO: Split into sentences
        TODO: Score sentences
        TODO: Select top sentences
        TODO: Return summary
        """
        pass
    
    def abstractive_summary(self, text: str) -> str:
        """
        Create abstractive summary.
        
        TODO: Generate summary
        TODO: Ensure length constraints
        TODO: Return summary
        """
        pass


# Exercise 5: Question Answering
class QuestionAnswerer:
    """Answer questions from context."""
    
    def answer(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer question from context.
        
        Returns:
        {
            'answer': 'Paris',
            'confidence': 0.95,
            'start': 25,
            'end': 30
        }
        
        TODO: Find answer in context
        TODO: Calculate confidence
        TODO: Return answer with position
        """
        pass
    
    def answer_batch(self, questions: List[str], 
                    context: str) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.
        
        TODO: Process all questions
        TODO: Return list of answers
        """
        pass


# Bonus: Text Similarity
class TextSimilarity:
    """Compute text similarity."""
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity score.
        
        TODO: Encode texts
        TODO: Calculate similarity
        TODO: Return score [0, 1]
        """
        pass
    
    def find_similar(self, query: str, documents: List[str], 
                    top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Find most similar documents.
        
        TODO: Calculate similarities
        TODO: Sort by score
        TODO: Return top-k indices and scores
        """
        pass


if __name__ == "__main__":
    print("Day 94: NLP Tasks - Exercises")
    print("=" * 50)
    
    # Test Exercise 1
    print("\nExercise 1: Text Classification")
    classifier = TextClassifier()
    print(f"Classifier created: {classifier is not None}")
    
    # Test Exercise 2
    print("\nExercise 2: Named Entity Recognition")
    ner = NamedEntityRecognizer()
    print(f"NER created: {ner is not None}")
    
    # Test Exercise 3
    print("\nExercise 3: Text Generation")
    generator = TextGenerator()
    print(f"Generator created: {generator is not None}")
    
    # Test Exercise 4
    print("\nExercise 4: Text Summarization")
    summarizer = TextSummarizer()
    print(f"Summarizer created: {summarizer is not None}")
    
    # Test Exercise 5
    print("\nExercise 5: Question Answering")
    qa = QuestionAnswerer()
    print(f"QA system created: {qa is not None}")
    
    print("\n" + "=" * 50)
    print("Complete the TODOs to finish the exercises!")
    print("\nNote: These are mock implementations for learning.")
    print("For real NLP tasks, use Hugging Face Transformers or spaCy.")
