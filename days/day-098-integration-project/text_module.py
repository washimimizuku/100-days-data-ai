"""
Day 98: Integration Project - Text Analysis Module
"""
import re
from collections import Counter
from typing import Dict, List, Optional


class TextAnalyzer:
    """Analyze text content with sentiment, entities, topics, and summarization."""
    
    def __init__(self):
        """Initialize text analyzer with word lists."""
        self.positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "best", "perfect", "awesome", "brilliant", "outstanding",
            "superb", "exceptional", "impressive", "beautiful", "nice"
        }
        
        self.negative_words = {
            "bad", "terrible", "awful", "horrible", "poor", "worst",
            "hate", "disappointing", "useless", "pathetic", "disgusting",
            "annoying", "frustrating", "inferior", "mediocre", "unacceptable"
        }
        
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are",
            "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "should", "could", "may", "might", "must",
            "can", "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "what", "which", "who", "when", "where", "why", "how"
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using word-based approach.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment, confidence, and scores
        """
        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            }
        
        # Tokenize and count sentiment words
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        total = pos_count + neg_count
        
        if total == 0:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            }
        
        # Calculate scores
        pos_score = pos_count / total
        neg_score = neg_count / total
        
        # Determine sentiment
        if pos_score > 0.6:
            sentiment = "positive"
            confidence = pos_score
        elif neg_score > 0.6:
            sentiment = "negative"
            confidence = neg_score
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "scores": {
                "positive": round(pos_score, 3),
                "negative": round(neg_score, 3),
                "neutral": round(1 - pos_score - neg_score, 3)
            }
        }
    
    def extract_entities(self, text: str) -> Dict:
        """
        Extract named entities from text using regex patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with list of entities
        """
        if not text:
            return {"entities": []}
        
        entities = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "EMAIL",
                "start": match.start(),
                "end": match.end()
            })
        
        # URL pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for match in re.finditer(url_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "URL",
                "start": match.start(),
                "end": match.end()
            })
        
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "type": "DATE",
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Phone number pattern
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "PHONE",
                "start": match.start(),
                "end": match.end()
            })
        
        # Money pattern
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        for match in re.finditer(money_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "MONEY",
                "start": match.start(),
                "end": match.end()
            })
        
        return {"entities": entities}
    
    def extract_topics(self, text: str, num_topics: int = 3) -> Dict:
        """
        Extract main topics from text using TF-IDF-like approach.
        
        Args:
            text: Input text to analyze
            num_topics: Number of topics to extract
            
        Returns:
            Dictionary with topics and keywords
        """
        if not text or not text.strip():
            return {"topics": [], "keywords": []}
        
        # Extract words (4+ characters, lowercase)
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Remove stopwords
        words = [w for w in words if w not in self.stopwords]
        
        if not words:
            return {"topics": [], "keywords": []}
        
        # Count word frequencies
        counter = Counter(words)
        
        # Get top topics and keywords
        topics = [word for word, _ in counter.most_common(num_topics)]
        keywords = [word for word, _ in counter.most_common(10)]
        
        return {
            "topics": topics,
            "keywords": keywords
        }
    
    def summarize(self, text: str, max_length: int = 100) -> Dict:
        """
        Generate text summary using sentence scoring.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary in characters
            
        Returns:
            Dictionary with summary and compression ratio
        """
        if not text or not text.strip():
            return {"summary": "", "compression_ratio": 0.0}
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"summary": "", "compression_ratio": 0.0}
        
        if len(sentences) == 1:
            summary = sentences[0][:max_length]
            return {
                "summary": summary,
                "compression_ratio": round(len(summary) / len(text), 3)
            }
        
        # Score sentences by word frequency
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        words = [w for w in words if w not in self.stopwords]
        word_freq = Counter(words)
        
        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            sentence_words = re.findall(r'\b[a-z]{4,}\b', sentence.lower())
            score = sum(word_freq.get(w, 0) for w in sentence_words)
            sentence_scores.append((sentence, score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build summary up to max_length
        summary_parts = []
        current_length = 0
        
        for sentence, _ in sentence_scores:
            if current_length + len(sentence) + 1 <= max_length:
                summary_parts.append(sentence)
                current_length += len(sentence) + 1
            else:
                break
        
        if not summary_parts:
            summary_parts = [sentence_scores[0][0][:max_length]]
        
        summary = ". ".join(summary_parts) + "."
        
        return {
            "summary": summary,
            "compression_ratio": round(len(summary) / len(text), 3)
        }
    
    def analyze(self, text: str) -> Dict:
        """
        Perform complete text analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with all analysis results
        """
        return {
            "sentiment": self.analyze_sentiment(text),
            "entities": self.extract_entities(text),
            "topics": self.extract_topics(text),
            "summary": self.summarize(text),
            "metadata": {
                "length": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(re.split(r'[.!?]+', text))
            }
        }


if __name__ == "__main__":
    print("Day 98: Text Analysis Module\n")
    
    analyzer = TextAnalyzer()
    
    # Test sentiment analysis
    print("=== Sentiment Analysis ===")
    test_texts = [
        "This is an amazing product! I love it!",
        "Terrible experience. Very disappointing.",
        "The weather is nice today."
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']})")
        print(f"Scores: {result['scores']}")
    
    # Test entity extraction
    print("\n\n=== Entity Extraction ===")
    text = "Contact me at john@example.com or visit https://example.com. Call 555-123-4567 on 12/25/2024. Price: $99.99"
    result = analyzer.extract_entities(text)
    print(f"\nText: {text}")
    print(f"Found {len(result['entities'])} entities:")
    for entity in result['entities']:
        print(f"  - {entity['type']}: {entity['text']}")
    
    # Test topic extraction
    print("\n\n=== Topic Extraction ===")
    text = "Machine learning and artificial intelligence are transforming technology. Deep learning models use neural networks for pattern recognition."
    result = analyzer.extract_topics(text)
    print(f"\nText: {text}")
    print(f"Topics: {result['topics']}")
    print(f"Keywords: {result['keywords']}")
    
    # Test summarization
    print("\n\n=== Summarization ===")
    text = "Python is a high-level programming language. It is widely used for web development. Python is also popular for data science. Many companies use Python for their applications."
    result = analyzer.summarize(text, max_length=100)
    print(f"\nOriginal ({len(text)} chars): {text}")
    print(f"Summary ({len(result['summary'])} chars): {result['summary']}")
    print(f"Compression: {result['compression_ratio']}")
    
    # Complete analysis
    print("\n\n=== Complete Analysis ===")
    text = "Great product! Contact support@company.com for help. Visit https://company.com for more info."
    result = analyzer.analyze(text)
    print(f"\nText: {text}")
    print(f"Sentiment: {result['sentiment']['sentiment']}")
    print(f"Entities: {len(result['entities']['entities'])}")
    print(f"Topics: {result['topics']['topics']}")
    print(f"Metadata: {result['metadata']}")
