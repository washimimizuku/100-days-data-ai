#!/bin/bash

# Test script for Prompt Engineering Toolkit

echo "=========================================="
echo "Prompt Engineering Toolkit - Test Suite"
echo "=========================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Error: Ollama is not installed"
    echo "Install from: https://ollama.com/download"
    exit 1
fi

echo "✓ Ollama is installed"

# Check if Ollama is running
if ! ollama list &> /dev/null; then
    echo "❌ Error: Ollama is not running"
    echo "Start with: ollama serve"
    exit 1
fi

echo "✓ Ollama is running"

# Check if mistral model is available
if ! ollama list | grep -q "mistral"; then
    echo "⚠ Warning: mistral model not found"
    echo "Pulling mistral model..."
    ollama pull mistral
    if [ $? -ne 0 ]; then
        echo "❌ Failed to pull mistral model"
        exit 1
    fi
fi

echo "✓ mistral model is available"
echo ""

# Run Python tests
echo "Running toolkit tests..."
echo "=========================================="
echo ""

python3 toolkit.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ All tests passed!"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ Some tests failed"
    echo "=========================================="
    exit 1
fi
