# Prompt Engineering Toolkit - Detailed Specification

## Project Goals

Build a production-ready prompt engineering toolkit that combines:
- Few-shot learning for task adaptation
- Chain of thought reasoning for complex problems
- Flexible template system for prompt reuse
- Local LLM integration via Ollama

---

## Component Specifications

### 1. Few-Shot Manager

**Purpose**: Manage and select relevant examples for few-shot learning.

**Data Structure**:
```python
{
    "task": "sentiment",
    "input": "I love this product!",
    "output": "positive",
    "reasoning": "The word 'love' indicates strong positive emotion.",
    "metadata": {
        "difficulty": "easy",
        "domain": "product_review"
    }
}
```

**Methods**:
- `add_example(task, input, output, reasoning=None, metadata=None)`: Store example
- `get_examples(task, k=3)`: Retrieve k examples for task
- `select_similar(query, task, k=3)`: Select k most similar examples
- `format_examples(examples, format="standard")`: Format for prompt

**Selection Strategy**:
- Random selection for simple tasks
- Similarity-based for complex tasks (using embeddings)
- Diversity sampling to cover different patterns

---

### 2. Chain of Thought Engine

**Purpose**: Enable step-by-step reasoning for complex problems.

**CoT Methods**:

**Zero-Shot CoT**:
```python
prompt = f"{problem}\n\nLet's think step by step:"
```

**Few-Shot CoT**:
```python
examples = [
    {
        "problem": "...",
        "reasoning": "Step 1: ... Step 2: ...",
        "answer": "..."
    }
]
```

**Self-Consistency**:
- Generate N reasoning paths (N=3-5)
- Extract final answers
- Return most common answer (voting)

**Methods**:
- `zero_shot_cot(problem)`: Apply zero-shot CoT
- `few_shot_cot(problem, examples)`: Apply few-shot CoT
- `self_consistency(problem, n=3)`: Multiple samples with voting
- `extract_reasoning(response)`: Parse reasoning steps
- `extract_answer(response)`: Extract final answer

---

### 3. Template System

**Purpose**: Create reusable, composable prompt templates.

**Template Format**:
```python
{
    "name": "classification",
    "system": "You are a helpful AI assistant.",
    "template": """Classify the following text as {labels}.

Text: {text}

Classification:""",
    "variables": ["text", "labels"],
    "examples_format": "Input: {input}\nOutput: {output}"
}
```

**Methods**:
- `register_template(name, template_dict)`: Add template
- `get_template(name)`: Retrieve template
- `render(template_name, **kwargs)`: Fill template with values
- `compose(system, user, examples=None)`: Build complete prompt

**Built-in Templates**:
- Classification (sentiment, topic, intent)
- Extraction (entities, relationships)
- Generation (summary, rewrite)
- Q&A (with context)
- Chain of thought

---

### 4. Task Processors

**Purpose**: Implement specific NLP task handlers.

#### Classification Processor

**Input**:
```python
{
    "text": "This movie was amazing!",
    "labels": ["positive", "negative", "neutral"],
    "use_few_shot": True,
    "num_examples": 3
}
```

**Output**:
```python
{
    "label": "positive",
    "confidence": 0.95,
    "reasoning": "The word 'amazing' indicates strong positive sentiment."
}
```

**Supported Tasks**:
- Sentiment analysis
- Topic classification
- Intent detection

#### Extraction Processor

**Input**:
```python
{
    "text": "Apple Inc. released iPhone 15 in September 2023.",
    "entity_types": ["organization", "product", "date"]
}
```

**Output**:
```python
{
    "entities": [
        {"text": "Apple Inc.", "type": "organization"},
        {"text": "iPhone 15", "type": "product"},
        {"text": "September 2023", "type": "date"}
    ]
}
```

#### Generation Processor

**Input**:
```python
{
    "text": "Long article text...",
    "task": "summarize",
    "max_length": 100
}
```

**Output**:
```python
{
    "generated_text": "Summary of the article...",
    "length": 85
}
```

**Supported Tasks**:
- Summarization
- Paraphrasing
- Text expansion

#### Q&A Processor

**Input**:
```python
{
    "question": "What is machine learning?",
    "context": "Machine learning is a subset of AI...",
    "use_cot": True
}
```

**Output**:
```python
{
    "answer": "Machine learning is a subset of AI that...",
    "reasoning": "Based on the context, ML is defined as...",
    "confidence": 0.9
}
```

---

### 5. LLM Backend

**Purpose**: Interface with Ollama for LLM inference.

**Configuration**:
```python
{
    "default_model": "mistral",
    "task_models": {
        "simple": "phi",
        "complex": "mistral",
        "code": "codellama"
    },
    "default_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 200
    }
}
```

**Methods**:
- `generate(prompt, model=None, **params)`: Generate text
- `select_model(task_type, complexity)`: Choose appropriate model
- `parse_response(response, format)`: Extract structured output
- `validate_response(response, schema)`: Check output format

**Error Handling**:
- Retry on connection errors (max 3 attempts)
- Fallback to simpler model on timeout
- Return error details for debugging

---

## Implementation Flow

### Example: Sentiment Classification with Few-Shot

```python
# 1. User calls classify
toolkit.classify(
    text="This product is terrible!",
    task="sentiment",
    labels=["positive", "negative", "neutral"],
    use_few_shot=True,
    num_examples=3
)

# 2. Few-Shot Manager selects examples
examples = few_shot_manager.select_similar(
    query="This product is terrible!",
    task="sentiment",
    k=3
)

# 3. Template System builds prompt
template = template_system.get_template("classification")
prompt = template_system.compose(
    system="You are a sentiment analysis expert.",
    user=template.render(text=text, labels=labels),
    examples=examples
)

# 4. LLM Backend generates response
response = llm_backend.generate(
    prompt=prompt,
    model="mistral",
    temperature=0.3
)

# 5. Parse and return result
result = {
    "label": "negative",
    "confidence": 0.95,
    "reasoning": "The word 'terrible' indicates strong negative sentiment."
}
```

---

## Test Cases

### Test 1: Few-Shot Classification
```python
result = toolkit.classify(
    text="The service was outstanding!",
    task="sentiment",
    labels=["positive", "negative", "neutral"],
    use_few_shot=True
)
assert result["label"] == "positive"
```

### Test 2: Zero-Shot CoT
```python
result = toolkit.reason(
    problem="If 5 apples cost $10, how much do 8 apples cost?",
    use_cot=True,
    method="zero-shot"
)
assert "$16" in result["answer"]
```

### Test 3: Entity Extraction
```python
result = toolkit.extract(
    text="Microsoft CEO Satya Nadella spoke at the conference.",
    entity_types=["organization", "person", "event"]
)
assert len(result["entities"]) >= 2
```

### Test 4: Text Summarization
```python
result = toolkit.generate(
    text="Long article about AI...",
    task="summarize",
    max_length=50
)
assert len(result["generated_text"].split()) <= 60
```

---

## Performance Targets

- **Latency**: < 5 seconds per request (with mistral)
- **Accuracy**: > 80% on classification tasks
- **Reliability**: Handle 95% of requests without errors
- **Scalability**: Support 10+ concurrent requests

---

## Code Quality Standards

- Type hints for all functions
- Docstrings with examples
- Error handling for all external calls
- Logging for debugging
- Unit tests for core functions
- Keep files under 400 lines

---

## Example Output Format

All task processors should return consistent format:

```python
{
    "success": True,
    "result": {
        # Task-specific output
    },
    "metadata": {
        "model": "mistral",
        "temperature": 0.7,
        "tokens": 150,
        "time_seconds": 2.3
    },
    "error": None  # or error message if failed
}
```
