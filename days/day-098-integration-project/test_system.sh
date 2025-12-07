#!/bin/bash

# Day 98: Integration Project - Test Script

echo "=================================="
echo "AI Content Analyzer - Test Suite"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: $2"
        ((TESTS_FAILED++))
    fi
}

# Check Python
echo "=== Checking Dependencies ==="
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} Python 3 found"
else
    echo -e "${RED}✗${NC} Python 3 not found"
    exit 1
fi

# Check required packages
echo ""
echo "=== Checking Python Packages ==="
python3 -c "import PIL" 2>/dev/null
print_result $? "PIL (Pillow) installed"

python3 -c "import numpy" 2>/dev/null
print_result $? "NumPy installed"

python3 -c "import fastapi" 2>/dev/null
print_result $? "FastAPI installed"

python3 -c "import uvicorn" 2>/dev/null
print_result $? "Uvicorn installed"

echo ""
echo "=== Testing Text Analysis Module ==="

# Test sentiment analysis
python3 << 'EOF'
from text_module import TextAnalyzer
analyzer = TextAnalyzer()

# Test positive sentiment
result = analyzer.analyze_sentiment("This is amazing! I love it!")
assert result["sentiment"] == "positive", f"Expected positive, got {result['sentiment']}"
print("✓ Positive sentiment detection works")

# Test negative sentiment
result = analyzer.analyze_sentiment("Terrible product. Very disappointing.")
assert result["sentiment"] == "negative", f"Expected negative, got {result['sentiment']}"
print("✓ Negative sentiment detection works")

# Test entity extraction
result = analyzer.extract_entities("Contact me at test@example.com")
assert len(result["entities"]) > 0, "Should find at least one entity"
print("✓ Entity extraction works")

# Test topic extraction
result = analyzer.extract_topics("machine learning artificial intelligence")
assert len(result["topics"]) > 0, "Should find topics"
print("✓ Topic extraction works")

# Test summarization
result = analyzer.summarize("This is a test. This is another sentence. More text here.")
assert len(result["summary"]) > 0, "Should generate summary"
print("✓ Summarization works")
EOF
print_result $? "Text analysis module"

echo ""
echo "=== Testing Image Analysis Module ==="

# Test image analysis
python3 << 'EOF'
from image_module import ImageAnalyzer
from PIL import Image
import os

analyzer = ImageAnalyzer()

# Create test image
test_image = Image.new('RGB', (800, 600), color=(100, 150, 200))
test_path = "test_img_temp.jpg"
test_image.save(test_path)

try:
    # Test classification
    result = analyzer.classify_image(test_path)
    assert "category" in result, "Should have category"
    assert "confidence" in result, "Should have confidence"
    print("✓ Image classification works")
    
    # Test object detection
    result = analyzer.detect_objects(test_path)
    assert "objects" in result, "Should have objects"
    print("✓ Object detection works")
    
    # Test feature extraction
    result = analyzer.extract_features(test_path)
    assert "dimensions" in result, "Should have dimensions"
    assert "brightness" in result, "Should have brightness"
    print("✓ Feature extraction works")
    
finally:
    if os.path.exists(test_path):
        os.remove(test_path)
EOF
print_result $? "Image analysis module"

echo ""
echo "=== Testing Content Analyzer Integration ==="

# Test integration
python3 << 'EOF'
from analyzer import ContentAnalyzer
from PIL import Image
import os

analyzer = ContentAnalyzer()

# Create test image
test_image = Image.new('RGB', (800, 600), color=(150, 200, 100))
test_path = "test_integration.jpg"
test_image.save(test_path)

try:
    # Test text analysis
    result = analyzer.analyze_text("Great product! Highly recommended.")
    assert "sentiment" in result, "Should have sentiment"
    assert "entities" in result, "Should have entities"
    print("✓ Text analysis integration works")
    
    # Test image analysis
    result = analyzer.analyze_image(test_path)
    assert "classification" in result, "Should have classification"
    assert "features" in result, "Should have features"
    print("✓ Image analysis integration works")
    
    # Test combined analysis
    result = analyzer.analyze_content(
        text="Amazing experience!",
        image_path=test_path
    )
    assert "text_analysis" in result, "Should have text analysis"
    assert "image_analysis" in result, "Should have image analysis"
    assert "insights" in result, "Should have insights"
    assert len(result["insights"]) > 0, "Should generate insights"
    print("✓ Combined analysis works")
    
    # Test insight generation
    assert result["overall_sentiment"] in ["positive", "negative", "neutral"], "Should have valid sentiment"
    assert 0 <= result["confidence"] <= 1, "Confidence should be between 0 and 1"
    print("✓ Insight generation works")
    
finally:
    if os.path.exists(test_path):
        os.remove(test_path)
EOF
print_result $? "Content analyzer integration"

echo ""
echo "=== Testing API Endpoints ==="

# Start API server in background
echo "Starting API server..."
python3 api.py > /dev/null 2>&1 &
API_PID=$!
sleep 3

# Check if server started
if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}✓${NC} API server started (PID: $API_PID)"
    
    # Test health endpoint
    response=$(curl -s http://localhost:8000/health)
    if echo "$response" | grep -q "healthy"; then
        print_result 0 "Health endpoint"
    else
        print_result 1 "Health endpoint"
    fi
    
    # Test stats endpoint
    response=$(curl -s http://localhost:8000/stats)
    if echo "$response" | grep -q "total_analyses"; then
        print_result 0 "Stats endpoint"
    else
        print_result 1 "Stats endpoint"
    fi
    
    # Test text analysis endpoint
    response=$(curl -s -X POST http://localhost:8000/analyze/text \
        -H "Content-Type: application/json" \
        -d '{"text": "Great product!"}')
    if echo "$response" | grep -q "sentiment"; then
        print_result 0 "Text analysis endpoint"
    else
        print_result 1 "Text analysis endpoint"
    fi
    
    # Stop server
    kill $API_PID 2>/dev/null
    wait $API_PID 2>/dev/null
    echo -e "${GREEN}✓${NC} API server stopped"
else
    echo -e "${RED}✗${NC} API server failed to start"
    print_result 1 "API server startup"
fi

echo ""
echo "=== Test Summary ==="
echo "Tests Passed: ${TESTS_PASSED}"
echo "Tests Failed: ${TESTS_FAILED}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
