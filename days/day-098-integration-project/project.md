# Day 98: Integration Project - Detailed Specification

## Project Overview

Build an AI Content Analyzer that integrates multiple AI techniques:
- NLP for text analysis
- Computer vision for image analysis
- ML for classification and insights
- FastAPI for deployment

**Time**: 2 hours

---

## System Architecture

```
Input Layer → Processing Layer → Integration Layer → API Layer
   ├─ Text         ├─ Text Module      ├─ Aggregation     └─ FastAPI
   └─ Image        └─ Image Module     └─ Insights
```

---

## Component Specifications

### 1. Text Analysis Module (`text_module.py`)

**Class**: `TextAnalyzer`

**Methods**:
- `analyze_sentiment(text)` - Returns sentiment, confidence, scores
- `extract_entities(text)` - Returns entities with types (EMAIL, URL, DATE, PHONE, MONEY)
- `extract_topics(text, num_topics)` - Returns topics and keywords
- `summarize(text, max_length)` - Returns summary and compression ratio

**Implementation**:
- Word-based sentiment (positive/negative word counts)
- Regex patterns for entity extraction
- TF-IDF for topic extraction
- Sentence scoring for summarization

---

### 2. Image Analysis Module (`image_module.py`)

**Class**: `ImageAnalyzer`

**Methods**:
- `classify_image(image_path)` - Returns category, confidence, predictions
- `detect_objects(image_path)` - Returns objects with bounding boxes
- `extract_features(image_path)` - Returns dimensions, colors, brightness, contrast

**Implementation**:
- PIL for image loading
- Property-based classification
- Mock object detection
- Color histogram analysis

---

### 3. Main Analyzer (`analyzer.py`)

**Class**: `ContentAnalyzer`

**Methods**:
- `analyze_text(text)` - Complete text analysis
- `analyze_image(image_path)` - Complete image analysis
- `analyze_content(text, image_path)` - Combined analysis with insights
- `generate_insights(text_result, image_result)` - Cross-modal insights

**Features**:
- Orchestrates text and image analyzers
- Generates combined insights
- Calculates overall sentiment and confidence

---

### 4. API Layer (`api.py`)

**Endpoints**:
- `POST /analyze/text` - Analyze text content
- `POST /analyze/image` - Analyze uploaded image
- `POST /analyze/combined` - Analyze text and image together
- `GET /health` - Health check
- `GET /stats` - System statistics

**Models**:
- `TextRequest/Response` - Text analysis I/O
- `ImageResponse` - Image analysis output
- `CombinedResponse` - Combined analysis output

---

## Implementation Examples

### Sentiment Analysis
```python
positive_words = {"good", "great", "excellent", "amazing", "love"}
negative_words = {"bad", "terrible", "awful", "hate", "worst"}

words = text.lower().split()
pos_count = sum(1 for w in words if w in positive_words)
neg_count = sum(1 for w in words if w in negative_words)

# Calculate sentiment based on counts
```

### Entity Extraction
```python
# Email: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
# URL: r'http[s]?://...'
# Date: r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
# Phone: r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
# Money: r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
```

### Image Features
```python
img = Image.open(image_path)
pixels = np.array(img)

brightness = pixels.mean() / 255.0
contrast = pixels.std() / 255.0
avg_color = pixels.mean(axis=(0, 1))
```

### Insight Generation
```python
insights = []

# Text insights
if sentiment == "positive":
    insights.append(f"Text sentiment is positive")

# Image insights
if brightness > 0.7:
    insights.append("Image is bright")

# Combined insights
if text_sentiment == "positive" and brightness > 0.6:
    insights.append("Positive sentiment aligns with bright imagery")
```

---

## Testing Strategy

### Unit Tests
```bash
# Test text module
python text_module.py

# Test image module
python image_module.py

# Test integration
python analyzer.py
```

### API Tests
```bash
# Run test script
./test_system.sh

# Manual API tests
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'
```

---

## Performance Requirements

- Text Analysis: < 100ms
- Image Analysis: < 500ms
- Combined Analysis: < 1 second
- API Response: < 2 seconds
- Memory: < 200MB

---

## Error Handling

```python
# Component errors
try:
    result = analyzer.analyze_text(text)
except Exception as e:
    return {"error": str(e)}

# API errors
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})
```

---

## Success Criteria

- ✅ Text analysis works correctly
- ✅ Image analysis processes images
- ✅ Integration generates insights
- ✅ API endpoints respond properly
- ✅ Error handling is robust
- ✅ All files under 400 lines
- ✅ Tests pass successfully

---

## Extensions

1. Use transformers for better NLP
2. Use pre-trained CV models
3. Add database for history
4. Add caching layer
5. Add authentication
6. Deploy to AWS Lambda

---

## Timeline

- 30 min: Text analysis module
- 30 min: Image analysis module
- 30 min: Integration layer
- 30 min: API layer + testing

---

## Deliverables

1. `text_module.py` - Text analysis
2. `image_module.py` - Image analysis
3. `analyzer.py` - Integration
4. `api.py` - FastAPI app
5. `test_system.sh` - Tests
6. `requirements.txt` - Dependencies
