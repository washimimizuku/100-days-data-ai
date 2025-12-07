#!/bin/bash

echo "=========================================="
echo "Image Classifier Test Suite"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run test
run_test() {
    local test_name=$1
    local command=$2
    
    echo "Running: $test_name"
    if eval $command > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: $test_name"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Test 1: Generate sample data
echo "Test 1: Generate Sample Data"
python classifier.py --generate-data
if [ -d "data/train" ] && [ -d "data/val" ]; then
    echo -e "${GREEN}✓ PASSED${NC}: Sample data generated"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}: Sample data generation"
    ((TESTS_FAILED++))
fi
echo ""

# Test 2: Check data structure
echo "Test 2: Verify Data Structure"
if [ -d "data/train/cat" ] && [ -d "data/train/dog" ] && [ -d "data/train/bird" ]; then
    echo -e "${GREEN}✓ PASSED${NC}: Data structure correct"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}: Data structure incorrect"
    ((TESTS_FAILED++))
fi
echo ""

# Test 3: Train model (2 epochs for quick test)
echo "Test 3: Train Model (2 epochs)"
python train.py --epochs 2 --batch-size 8 > train_output.txt 2>&1
if [ -f "best_model.pth" ]; then
    echo -e "${GREEN}✓ PASSED${NC}: Model training completed"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}: Model training failed"
    ((TESTS_FAILED++))
fi
echo ""

# Test 4: Check training output
echo "Test 4: Verify Training Output"
if grep -q "Epoch 1/2" train_output.txt && grep -q "Epoch 2/2" train_output.txt; then
    echo -e "${GREEN}✓ PASSED${NC}: Training output correct"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}: Training output incorrect"
    ((TESTS_FAILED++))
fi
echo ""

# Test 5: Make prediction
echo "Test 5: Single Image Prediction"
TEST_IMAGE=$(find data/val -name "*.jpg" | head -1)
if [ -n "$TEST_IMAGE" ]; then
    python predict.py --image "$TEST_IMAGE" --num-classes 3 > predict_output.txt 2>&1
    if grep -q "Prediction Results" predict_output.txt; then
        echo -e "${GREEN}✓ PASSED${NC}: Prediction successful"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: Prediction failed"
        ((TESTS_FAILED++))
    fi
else
    echo -e "${RED}✗ FAILED${NC}: No test image found"
    ((TESTS_FAILED++))
fi
echo ""

# Test 6: Verify model file
echo "Test 6: Verify Model Checkpoint"
if [ -f "best_model.pth" ]; then
    SIZE=$(stat -f%z "best_model.pth" 2>/dev/null || stat -c%s "best_model.pth" 2>/dev/null)
    if [ "$SIZE" -gt 1000000 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: Model checkpoint valid (${SIZE} bytes)"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: Model checkpoint too small"
        ((TESTS_FAILED++))
    fi
else
    echo -e "${RED}✗ FAILED${NC}: Model checkpoint not found"
    ((TESTS_FAILED++))
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests Failed: ${RED}${TESTS_FAILED}${NC}"
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo ""

# Cleanup
echo "Cleaning up test files..."
rm -f train_output.txt predict_output.txt
echo "Cleanup complete"
echo ""

# Exit with appropriate code
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
