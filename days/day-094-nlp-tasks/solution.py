"""Day 94: NLP Tasks - Solutions

NOTE: Mock implementations for learning without model downloads.
"""

from typing import List, Dict, Any, Tuple
import re
import random


# Exercise 1: Text Classification
class TextClassifier:
    """Classify text into categories."""
    
    def __init__(self, categories: List[str] = None):
        self.categories = categories or ["positive", "negative", "neutral"]
    
    def preprocess(self, text: str) -> str:
        """Preprocess text."""
        # Lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify text sentiment."""
        preprocessed = self.preprocess(text)
        
        # Mock classification based on keywords
        positive_words = ['good', 'great', 'excellent', 'love', 'amazing']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'poor']
        
        words = preprocessed.split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        if pos_count > neg_count:
            label = "positive"
            confidence = 0.85 + random.random() * 0.14
        elif neg_count > pos_count:
            label = "negative"
            confidence = 0.85 + random.random() * 0.14
        else:
            label = "neutral"
            confidence = 0.70 + random.random() * 0.20
        
        return {
            'label': label,
            'confidence': confidence
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]


# Exercise 2: Named Entity Recognition
class NamedEntityRecognizer:
    """Extract named entities from text."""
    
    def __init__(self):
        self.entity_types = ["PER", "ORG", "LOC", "DATE", "MISC"]
        # Mock entity patterns
        self.patterns = {
            "ORG": ["Inc.", "Corp.", "LLC", "Company"],
            "LOC": ["City", "Street", "Avenue"],
            "DATE": [r'\d{4}', r'\d{1,2}/\d{1,2}/\d{4}']
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities."""
        entities = []
        
        # Mock entity extraction
        words = text.split()
        for i, word in enumerate(words):
            # Check for organization markers
            if any(marker in word for marker in self.patterns["ORG"]):
                entities.append({
                    'text': word,
                    'type': 'ORG',
                    'start': text.find(word),
                    'end': text.find(word) + len(word)
                })
            # Check for capitalized words (potential names/places)
            elif word[0].isupper() and len(word) > 1:
                entity_type = random.choice(["PER", "LOC", "ORG"])
                entities.append({
                    'text': word,
                    'type': entity_type,
                    'start': text.find(word),
                    'end': text.find(word) + len(word)
                })
        
        return entities
    
    def group_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group entities by type."""
        grouped = {entity_type: [] for entity_type in self.entity_types}
        
        for entity in entities:
            entity_type = entity['type']
            if entity_type in grouped:
                grouped[entity_type].append(entity['text'])
        
        return {k: v for k, v in grouped.items() if v}


# Exercise 3: Text Generation
class TextGenerator:
    """Generate text continuations."""
    
    def __init__(self, max_length: int = 50):
        self.max_length = max_length
        self.common_words = ['the', 'a', 'is', 'was', 'in', 'on', 'at', 'to', 'for']
    
    def generate(self, prompt: str, num_tokens: int = 20) -> str:
        """Generate text continuation."""
        # Mock generation
        words = prompt.split()
        
        for _ in range(num_tokens):
            # Simple mock: add random common words
            next_word = random.choice(self.common_words + words[-3:])
            words.append(next_word)
        
        return ' '.join(words)
    
    def generate_multiple(self, prompt: str, num_sequences: int = 3) -> List[str]:
        """Generate multiple continuations."""
        return [self.generate(prompt, num_tokens=15) for _ in range(num_sequences)]


# Exercise 4: Text Summarization
class TextSummarizer:
    """Summarize long text."""
    
    def __init__(self):
        self.max_summary_length = 130
        self.min_summary_length = 30
    
    def extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """Create extractive summary."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Score sentences (mock: by length and position)
        scored = []
        for i, sent in enumerate(sentences):
            score = len(sent.split()) * (1 - i / len(sentences))
            scored.append((score, sent))
        
        # Select top sentences
        scored.sort(reverse=True)
        top_sentences = [sent for _, sent in scored[:num_sentences]]
        
        return '. '.join(top_sentences) + '.'
    
    def abstractive_summary(self, text: str) -> str:
        """Create abstractive summary."""
        # Mock: take first few sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        summary = '. '.join(sentences[:2]) + '.'
        
        # Ensure length constraints
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length] + '...'
        
        return summary


# Exercise 5: Question Answering
class QuestionAnswerer:
    """Answer questions from context."""
    
    def answer(self, question: str, context: str) -> Dict[str, Any]:
        """Answer question from context."""
        # Mock: find question words in context
        question_words = question.lower().replace('?', '').split()
        context_lower = context.lower()
        
        # Find potential answer (mock: words after question keywords)
        for word in question_words:
            if word in context_lower:
                idx = context_lower.find(word)
                # Extract nearby words as answer
                start = max(0, idx - 20)
                end = min(len(context), idx + 30)
                answer = context[start:end].strip()
                
                return {
                    'answer': answer,
                    'confidence': 0.75 + random.random() * 0.24,
                    'start': start,
                    'end': end
                }
        
        # Default answer
        words = context.split()
        answer = ' '.join(words[:5])
        return {
            'answer': answer,
            'confidence': 0.50,
            'start': 0,
            'end': len(answer)
        }
    
    def answer_batch(self, questions: List[str], context: str) -> List[Dict[str, Any]]:
        """Answer multiple questions."""
        return [self.answer(q, context) for q in questions]


# Bonus: Text Similarity
class TextSimilarity:
    """Compute text similarity."""
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity score."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def find_similar(self, query: str, documents: List[str], 
                    top_k: int = 3) -> List[Tuple[int, float]]:
        """Find most similar documents."""
        similarities = []
        
        for idx, doc in enumerate(documents):
            sim = self.compute_similarity(query, doc)
            similarities.append((idx, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


def demo_nlp_tasks():
    """Demonstrate NLP tasks."""
    print("Day 94: NLP Tasks - Solutions Demo\n" + "=" * 60)
    
    print("\n1. Text Classification")
    classifier = TextClassifier()
    texts = [
        "I love this product! It's amazing!",
        "This is terrible. Very disappointed.",
        "It's okay, nothing special."
    ]
    for text in texts:
        result = classifier.classify(text)
        print(f"   '{text[:30]}...' → {result['label']} ({result['confidence']:.2%})")
    
    print("\n2. Named Entity Recognition")
    ner = NamedEntityRecognizer()
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    entities = ner.extract_entities(text)
    print(f"   Found {len(entities)} entities:")
    for ent in entities[:3]:
        print(f"     - {ent['text']} ({ent['type']})")
    
    print("\n3. Text Generation")
    generator = TextGenerator()
    prompt = "Once upon a time"
    generated = generator.generate(prompt, num_tokens=10)
    print(f"   Prompt: '{prompt}'")
    print(f"   Generated: '{generated[:50]}...'")
    
    print("\n4. Text Summarization")
    summarizer = TextSummarizer()
    article = "This is a long article. It has many sentences. Each sentence contains information. The article discusses various topics. It provides detailed explanations."
    summary = summarizer.extractive_summary(article, num_sentences=2)
    print(f"   Original: {len(article)} chars")
    print(f"   Summary: {len(summary)} chars")
    print(f"   '{summary}'")
    
    print("\n5. Question Answering")
    qa = QuestionAnswerer()
    context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
    question = "Where is the Eiffel Tower?"
    answer = qa.answer(question, context)
    print(f"   Question: {question}")
    print(f"   Answer: {answer['answer']} (confidence: {answer['confidence']:.2%})")
    
    print("\n6. Text Similarity")
    similarity = TextSimilarity()
    text1 = "I love pizza and pasta"
    text2 = "Pizza is my favorite food"
    score = similarity.compute_similarity(text1, text2)
    print(f"   Text 1: '{text1}'")
    print(f"   Text 2: '{text2}'")
    print(f"   Similarity: {score:.3f}")
    
    print("\n" + "=" * 60)
    print("All NLP tasks demonstrated!")


if __name__ == "__main__":
    demo_nlp_tasks()
