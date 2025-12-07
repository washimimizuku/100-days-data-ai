# Day 77: Mini Project - Prompt Engineering Toolkit

## 🎯 Project Overview

Build a comprehensive prompt engineering toolkit that combines few-shot learning, chain of thought reasoning, and local LLM integration to solve real-world NLP tasks. This project integrates concepts from Days 73-76.

**Time**: 2 hours

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Prompt Engineering Toolkit                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Few-Shot    │  │  Chain of    │  │   Template   │ │
│  │  Manager     │  │  Thought     │  │   Engine     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Task Processors                         │  │
│  │  - Classification  - Extraction  - Generation    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │           LLM Backend (Ollama)                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 Requirements

### Core Features

1. **Few-Shot Learning Manager**
   - Store and retrieve examples by task type
   - Select relevant examples based on similarity
   - Format examples consistently
   - Support multiple example formats

2. **Chain of Thought Engine**
   - Zero-shot CoT with trigger phrases
   - Few-shot CoT with reasoning examples
   - Self-consistency with multiple samples
   - Reasoning extraction and validation

3. **Template System**
   - Reusable prompt templates
   - Variable substitution
   - Template composition
   - Format validation

4. **Task Processors**
   - Text classification (sentiment, topic, intent)
   - Information extraction (entities, relationships)
   - Text generation (summaries, rewrites)
   - Question answering

5. **LLM Integration**
   - Ollama backend support
   - Model selection based on task
   - Response parsing and validation
   - Error handling and retries

---

## 🚀 Getting Started

### Prerequisites

```bash
# Install Ollama
# Visit: https://ollama.com/download

# Pull required models
ollama pull mistral
ollama pull phi

# Install Python dependencies
pip install ollama numpy
```

### Project Structure

```
day-077-mini-project-prompt-engineering/
├── README.md                    # This file
├── project.md                   # Detailed specification
├── toolkit.py                   # Main toolkit implementation
├── examples.py                  # Example data and templates
├── test_toolkit.sh             # Test script
└── requirements.txt            # Dependencies
```

---

## 💻 Implementation Guide

### Step 1: Few-Shot Manager (30 min)

Create a system to manage and select examples:
- Store examples with metadata (task, difficulty, format)
- Compute similarity between query and examples
- Select top-k most relevant examples
- Format examples for prompts

### Step 2: Chain of Thought Engine (30 min)

Implement CoT reasoning:
- Zero-shot CoT with trigger phrases
- Few-shot CoT with reasoning steps
- Self-consistency with voting
- Extract and validate reasoning

### Step 3: Template System (20 min)

Build flexible prompt templates:
- Define templates with placeholders
- Support variable substitution
- Compose templates (system + user + examples)
- Validate template structure

### Step 4: Task Processors (30 min)

Implement specific task handlers:
- Classification: Sentiment, topic, intent
- Extraction: Named entities, key phrases
- Generation: Summaries, rewrites
- Q&A: Answer questions with context

### Step 5: Integration & Testing (10 min)

Connect components and test:
- Integrate with Ollama
- Test each task type
- Validate outputs
- Handle errors gracefully

---

## 🎓 Learning Objectives

By completing this project, you will:
- Apply few-shot learning to real tasks
- Implement chain of thought reasoning
- Design reusable prompt templates
- Integrate multiple prompt engineering techniques
- Build production-ready NLP applications
- Work with local LLMs via Ollama

---

## 🧪 Testing

Run the test script to validate your implementation:

```bash
chmod +x test_toolkit.sh
./test_toolkit.sh
```

The test script will:
1. Check Ollama availability
2. Test few-shot classification
3. Test CoT reasoning
4. Test information extraction
5. Test text generation
6. Validate output formats

---

## 📊 Example Usage

### Text Classification

```python
from toolkit import PromptToolkit

toolkit = PromptToolkit()

# Sentiment analysis with few-shot learning
result = toolkit.classify(
    text="This product exceeded my expectations!",
    task="sentiment",
    labels=["positive", "negative", "neutral"],
    use_few_shot=True,
    num_examples=3
)
print(f"Sentiment: {result['label']} (confidence: {result['confidence']})")
```

### Chain of Thought Reasoning

```python
# Math problem with CoT
result = toolkit.reason(
    problem="If a train travels 120 miles in 2 hours, how far will it travel in 5 hours?",
    use_cot=True,
    method="few-shot"
)
print(f"Answer: {result['answer']}")
print(f"Reasoning: {result['reasoning']}")
```

### Information Extraction

```python
# Extract entities
result = toolkit.extract(
    text="Apple Inc. announced a new iPhone in Cupertino on September 12.",
    entity_types=["organization", "product", "location", "date"]
)
print(f"Entities: {result['entities']}")
```

---

## 🎯 Success Criteria

Your implementation should:
- ✅ Support at least 3 task types (classification, extraction, generation)
- ✅ Implement both zero-shot and few-shot learning
- ✅ Include chain of thought reasoning
- ✅ Use reusable prompt templates
- ✅ Integrate with Ollama for LLM inference
- ✅ Handle errors gracefully
- ✅ Pass all test cases
- ✅ Keep files under 400 lines each

---

## 🔧 Troubleshooting

**Ollama not running**:
```bash
# Start Ollama service
ollama serve
```

**Model not found**:
```bash
# Pull the required model
ollama pull mistral
```

**Slow inference**:
- Use smaller models (phi instead of mistral)
- Reduce num_predict parameter
- Limit few-shot examples to 3-5

**Poor quality outputs**:
- Adjust temperature (lower for factual, higher for creative)
- Add more few-shot examples
- Use chain of thought for complex tasks
- Try different models

---

## 📚 Resources

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Few-Shot Learning Paper](https://arxiv.org/abs/2005.14165)
- [Chain of Thought Paper](https://arxiv.org/abs/2201.11903)

---

## 🚀 Extensions (Optional)

1. **Prompt Optimization**
   - A/B test different prompt formats
   - Automatically tune temperature and parameters
   - Cache successful prompts

2. **Multi-Model Support**
   - Compare outputs from different models
   - Ensemble predictions
   - Model routing based on task

3. **Evaluation Framework**
   - Benchmark against test sets
   - Track accuracy metrics
   - Generate performance reports

4. **Web Interface**
   - FastAPI backend
   - Interactive prompt testing
   - Example management UI

---

## Tomorrow: Day 78 - RAG Architecture

Learn about Retrieval-Augmented Generation (RAG) systems that combine information retrieval with LLM generation for knowledge-intensive tasks.
