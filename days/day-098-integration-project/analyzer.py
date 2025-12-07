"""
Day 98: Integration Project - Main Content Analyzer
"""
from typing import Dict, Optional, List
from text_module import TextAnalyzer
from image_module import ImageAnalyzer


class ContentAnalyzer:
    """
    Main content analyzer that integrates text and image analysis.
    Provides unified interface for multi-modal content analysis.
    """
    
    def __init__(self):
        """Initialize text and image analyzers."""
        self.text_analyzer = TextAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.analysis_count = 0
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text content.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with complete text analysis
        """
        if not text or not text.strip():
            return {
                "error": "Empty text provided",
                "sentiment": None,
                "entities": None,
                "topics": None,
                "summary": None,
                "metadata": None
            }
        
        try:
            result = self.text_analyzer.analyze(text)
            self.analysis_count += 1
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze image content.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with complete image analysis
        """
        if not image_path:
            return {
                "error": "No image path provided",
                "classification": None,
                "objects": None,
                "features": None,
                "metadata": None
            }
        
        try:
            result = self.image_analyzer.analyze(image_path)
            self.analysis_count += 1
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_content(
        self, 
        text: Optional[str] = None, 
        image_path: Optional[str] = None
    ) -> Dict:
        """
        Analyze both text and image content, generate insights.
        
        Args:
            text: Optional text to analyze
            image_path: Optional image path to analyze
            
        Returns:
            Dictionary with combined analysis and insights
        """
        if not text and not image_path:
            return {
                "error": "No content provided",
                "text_analysis": None,
                "image_analysis": None,
                "insights": [],
                "overall_sentiment": "neutral",
                "confidence": 0.0
            }
        
        # Analyze text if provided
        text_result = None
        if text:
            text_result = self.analyze_text(text)
        
        # Analyze image if provided
        image_result = None
        if image_path:
            image_result = self.analyze_image(image_path)
        
        # Generate insights
        insights = self.generate_insights(text_result, image_result)
        
        # Calculate overall sentiment and confidence
        overall_sentiment, confidence = self._calculate_overall_sentiment(
            text_result, image_result
        )
        
        return {
            "text_analysis": text_result,
            "image_analysis": image_result,
            "insights": insights,
            "overall_sentiment": overall_sentiment,
            "confidence": round(confidence, 3)
        }
    
    def generate_insights(
        self, 
        text_result: Optional[Dict], 
        image_result: Optional[Dict]
    ) -> List[str]:
        """
        Generate insights from combined text and image analysis.
        
        Args:
            text_result: Text analysis results
            image_result: Image analysis results
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Text insights
        if text_result and "error" not in text_result:
            sentiment_data = text_result.get("sentiment", {})
            sentiment = sentiment_data.get("sentiment")
            confidence = sentiment_data.get("confidence", 0)
            
            if sentiment:
                insights.append(
                    f"Text sentiment is {sentiment} with {confidence:.1%} confidence"
                )
            
            entities = text_result.get("entities", {}).get("entities", [])
            if entities:
                entity_types = set(e["type"] for e in entities)
                insights.append(
                    f"Found {len(entities)} entities: {', '.join(entity_types)}"
                )
            
            topics = text_result.get("topics", {}).get("topics", [])
            if topics:
                insights.append(f"Main topics: {', '.join(topics[:3])}")
            
            metadata = text_result.get("metadata", {})
            word_count = metadata.get("word_count", 0)
            if word_count > 0:
                insights.append(f"Text contains {word_count} words")
        
        # Image insights
        if image_result and "error" not in image_result:
            classification = image_result.get("classification", {})
            category = classification.get("category")
            conf = classification.get("confidence", 0)
            
            if category:
                insights.append(
                    f"Image classified as '{category}' with {conf:.1%} confidence"
                )
            
            features = image_result.get("features", {})
            if features and "error" not in features:
                dims = features.get("dimensions", {})
                width = dims.get("width", 0)
                height = dims.get("height", 0)
                if width and height:
                    insights.append(f"Image dimensions: {width}x{height}")
                
                brightness = features.get("brightness", 0)
                if brightness > 0.7:
                    insights.append("Image is bright")
                elif brightness < 0.3:
                    insights.append("Image is dark")
                
                contrast = features.get("contrast", 0)
                if contrast > 0.5:
                    insights.append("Image has high contrast")
            
            objects = image_result.get("objects", {}).get("objects", [])
            if objects:
                insights.append(f"Detected {len(objects)} objects in image")
        
        # Combined insights
        if text_result and image_result:
            if "error" not in text_result and "error" not in image_result:
                insights.append("Multi-modal analysis complete")
                
                # Check for sentiment-image alignment
                text_sentiment = text_result.get("sentiment", {}).get("sentiment")
                image_brightness = image_result.get("features", {}).get("brightness", 0.5)
                
                if text_sentiment == "positive" and image_brightness > 0.6:
                    insights.append("Positive sentiment aligns with bright imagery")
                elif text_sentiment == "negative" and image_brightness < 0.4:
                    insights.append("Negative sentiment aligns with dark imagery")
        
        # Add default insight if none generated
        if not insights:
            insights.append("Analysis complete")
        
        return insights
    
    def _calculate_overall_sentiment(
        self,
        text_result: Optional[Dict],
        image_result: Optional[Dict]
    ) -> tuple:
        """
        Calculate overall sentiment from text and image analysis.
        
        Args:
            text_result: Text analysis results
            image_result: Image analysis results
            
        Returns:
            Tuple of (sentiment, confidence)
        """
        sentiments = []
        confidences = []
        
        # Text sentiment
        if text_result and "error" not in text_result:
            text_sentiment = text_result.get("sentiment", {})
            sentiment = text_sentiment.get("sentiment")
            confidence = text_sentiment.get("confidence", 0)
            
            if sentiment:
                sentiments.append(sentiment)
                confidences.append(confidence)
        
        # Image-based sentiment (from brightness)
        if image_result and "error" not in image_result:
            features = image_result.get("features", {})
            if features and "error" not in features:
                brightness = features.get("brightness", 0.5)
                
                if brightness > 0.6:
                    sentiments.append("positive")
                    confidences.append(0.6)
                elif brightness < 0.4:
                    sentiments.append("negative")
                    confidences.append(0.6)
                else:
                    sentiments.append("neutral")
                    confidences.append(0.5)
        
        # Calculate overall
        if not sentiments:
            return "neutral", 0.5
        
        # Count sentiment occurrences
        sentiment_counts = {
            "positive": sentiments.count("positive"),
            "negative": sentiments.count("negative"),
            "neutral": sentiments.count("neutral")
        }
        
        # Get dominant sentiment
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return overall_sentiment, avg_confidence
    
    def get_stats(self) -> Dict:
        """
        Get analyzer statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_analyses": self.analysis_count,
            "text_analyzer_ready": self.text_analyzer is not None,
            "image_analyzer_ready": self.image_analyzer is not None
        }


if __name__ == "__main__":
    print("Day 98: Content Analyzer Integration\n")
    
    analyzer = ContentAnalyzer()
    
    # Test text analysis
    print("=== Text Analysis ===")
    text = "This is an amazing product! Contact us at support@example.com or visit https://example.com"
    result = analyzer.analyze_text(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result['sentiment']['sentiment']}")
    print(f"Entities: {len(result['entities']['entities'])}")
    print(f"Topics: {result['topics']['topics']}")
    
    # Test image analysis
    print("\n=== Image Analysis ===")
    from PIL import Image
    import os
    
    # Create test image
    test_image = Image.new('RGB', (800, 600), color=(150, 200, 100))
    test_path = "test_content.jpg"
    test_image.save(test_path)
    
    result = analyzer.analyze_image(test_path)
    print(f"Image: {test_path}")
    if "error" not in result["classification"]:
        print(f"Category: {result['classification']['category']}")
        print(f"Confidence: {result['classification']['confidence']}")
    
    # Test combined analysis
    print("\n=== Combined Analysis ===")
    result = analyzer.analyze_content(
        text="Great experience! Highly recommended.",
        image_path=test_path
    )
    print(f"Overall sentiment: {result['overall_sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print("Insights:")
    for insight in result['insights']:
        print(f"  - {insight}")
    
    # Test with only text
    print("\n=== Text Only ===")
    result = analyzer.analyze_content(text="Terrible product. Very disappointing.")
    print(f"Overall sentiment: {result['overall_sentiment']}")
    print("Insights:")
    for insight in result['insights']:
        print(f"  - {insight}")
    
    # Test with only image
    print("\n=== Image Only ===")
    result = analyzer.analyze_content(image_path=test_path)
    print(f"Overall sentiment: {result['overall_sentiment']}")
    print("Insights:")
    for insight in result['insights']:
        print(f"  - {insight}")
    
    # Get stats
    print("\n=== Statistics ===")
    stats = analyzer.get_stats()
    print(f"Total analyses: {stats['total_analyses']}")
    print(f"Text analyzer ready: {stats['text_analyzer_ready']}")
    print(f"Image analyzer ready: {stats['image_analyzer_ready']}")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"\nCleaned up test image")
