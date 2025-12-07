# Day 76: Local LLMs with Ollama

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Understand benefits and trade-offs of running LLMs locally
- Learn Ollama architecture and model management
- Master the Ollama API for programmatic access
- Implement local LLM applications with Python
- Compare local vs cloud LLM deployment strategies

---

## Why Run LLMs Locally?

**Benefits**: Privacy (data stays local), Cost (no per-token fees, unlimited inference), Customization (fine-tune freely), Offline (no internet needed)

**Trade-offs**: Resources (GPU/CPU/memory requirements), Maintenance (updates, monitoring, optimization), Speed (slower than cloud APIs)

---

## Ollama Architecture

### What is Ollama?
Ollama is a tool for running LLMs locally with a simple API. It handles:
- Model downloading and management
- Efficient model loading and caching
- REST API for inference
- Multi-model support

### Key Components
```
┌─────────────────────────────────────┐
│         Application Layer           │
│  (Python, CLI, Web Interface)       │
└──────────────┬──────────────────────┘
               │ HTTP/REST API
┌──────────────▼──────────────────────┐
│         Ollama Server               │
│  - Model Management                 │
│  - Request Handling                 │
│  - Context Management               │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         Model Runtime               │
│  - llama.cpp (optimized inference)  │
│  - GPU/CPU acceleration             │
│  - Memory management                │
└─────────────────────────────────────┘
```

### Model Storage
```
~/.ollama/models/
├── manifests/          # Model metadata
├── blobs/             # Model weights
└── registry/          # Model registry info
```

---

## Getting Started with Ollama

### Installation
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

### Basic Commands
```bash
# Pull a model
ollama pull llama2

# List installed models
ollama list

# Run interactive chat
ollama run llama2

# Remove a model
ollama rm llama2

# Show model info
ollama show llama2
```

### Available Models
- **llama2** (7B, 13B, 70B): Meta's open model
- **mistral** (7B): High-performance small model
- **mixtral** (8x7B): Mixture of experts model
- **codellama** (7B, 13B, 34B): Code-specialized
- **phi** (2.7B): Microsoft's efficient model
- **gemma** (2B, 7B): Google's open model

---

## Ollama Python API

### Installation
```python
pip install ollama
```

### Basic Usage
```python
import ollama

# Simple generation
response = ollama.generate(
    model='llama2',
    prompt='Explain quantum computing in simple terms'
)
print(response['response'])

# Chat interface
messages = [
    {'role': 'user', 'content': 'What is machine learning?'}
]
response = ollama.chat(model='llama2', messages=messages)
print(response['message']['content'])
```

### Streaming Responses
```python
# Stream tokens as they're generated
stream = ollama.generate(
    model='llama2',
    prompt='Write a short story',
    stream=True
)

for chunk in stream:
    print(chunk['response'], end='', flush=True)
```

### Model Parameters
```python
response = ollama.generate(
    model='llama2',
    prompt='Generate creative ideas',
    options={
        'temperature': 0.8,      # Creativity (0.0-2.0)
        'top_p': 0.9,           # Nucleus sampling
        'top_k': 40,            # Top-k sampling
        'num_predict': 100,     # Max tokens
        'stop': ['\n\n'],       # Stop sequences
    }
)
```

---

## Model Management

```python
import ollama

# Pull models
ollama.pull('mistral')
for progress in ollama.pull('llama2', stream=True):
    print(f"Status: {progress.get('status')} - Progress: {progress.get('completed')}/{progress.get('total')}")

# List models
models = ollama.list()
for model in models['models']:
    print(f"{model['name']}: {model['size'] / 1e9:.2f} GB, Modified: {model['modified_at']}")

# Model info
info = ollama.show('llama2')
print(f"Parameters: {info['details']['parameter_size']}, Quantization: {info['details']['quantization_level']}")
```

---

## Building Applications

```python
import ollama

# Simple Q&A
def ask_question(question: str, model: str = 'llama2') -> str:
    return ollama.generate(model=model, prompt=f"Question: {question}\nAnswer:", options={'temperature': 0.3})['response'].strip()

# Conversational Agent
class ChatBot:
    def __init__(self, model: str = 'llama2', system_prompt: str = None):
        self.model, self.messages = model, []
        if system_prompt: self.messages.append({'role': 'system', 'content': system_prompt})
    
    def chat(self, user_message: str) -> str:
        self.messages.append({'role': 'user', 'content': user_message})
        response = ollama.chat(model=self.model, messages=self.messages)
        assistant_message = response['message']['content']
        self.messages.append({'role': 'assistant', 'content': assistant_message})
        return assistant_message
    
    def reset(self): self.messages = []

# Document Summarization
def summarize_document(text: str, max_length: int = 200) -> str:
    return ollama.generate(model='mistral', prompt=f"Summarize in {max_length} words:\n{text}\nSummary:", 
                          options={'temperature': 0.3, 'num_predict': max_length * 2})['response'].strip()
```

---

## Advanced Features

```python
# Embeddings for semantic search
from numpy import dot
from numpy.linalg import norm

def get_embedding(text: str, model: str = 'llama2') -> list:
    return ollama.embeddings(model=model, prompt=text)['embedding']

def cosine_similarity(a, b): return dot(a, b) / (norm(a) * norm(b))

# Custom Modelfile
"""
FROM llama2
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are a Python expert who provides concise, practical code examples.
"""
# Create: subprocess.run(['ollama', 'create', 'python-expert', '-f', 'Modelfile'])
```

---

## Performance Optimization

```python
# Model selection: fast (phi 2.7B), balanced (mistral 7B), quality (llama2:13b), code (codellama)
MODELS = {'fast': 'phi', 'balanced': 'mistral', 'quality': 'llama2:13b', 'code': 'codellama'}

# Context management (1 token ≈ 4 chars)
def truncate_context(messages: list, max_tokens: int = 2048) -> list:
    total_chars = sum(len(m['content']) for m in messages)
    if total_chars > max_tokens * 4:
        system_msgs = [m for m in messages if m['role'] == 'system']
        return system_msgs + messages[-(max_tokens // 100):]
    return messages

# Batch processing
import concurrent.futures
def process_batch(texts: list, model: str = 'mistral') -> list:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda t: ollama.generate(model=model, prompt=t), texts))
    return [r['response'] for r in results]
```

---

## 💻 Exercises (40 min)

### Exercise 1: Model Manager
Create a utility to manage Ollama models with download, list, and cleanup functions.

### Exercise 2: Smart Q&A System
Build a Q&A system that selects the appropriate model based on question complexity.

### Exercise 3: Document Processor
Implement a document processor that summarizes, extracts key points, and answers questions.

### Exercise 4: Conversational Memory
Create a chatbot with conversation history management and context truncation.

### Exercise 5: Semantic Search
Build a semantic search system using Ollama embeddings to find similar documents.

---

## ✅ Quiz

Test your understanding in `quiz.md`

---

## 🎯 Key Takeaways

- Ollama simplifies running LLMs locally with model management and REST API
- Local LLMs provide privacy, cost savings, and offline capability
- Trade-offs include resource requirements and maintenance overhead
- Python API supports generation, chat, streaming, and embeddings
- Model selection depends on task requirements (speed vs quality)
- Context management is crucial for long conversations
- Custom Modelfiles enable behavior customization
- Batch processing and parallel execution improve throughput

---

## 📚 Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Python Library](https://github.com/ollama/ollama-python)
- [Available Models](https://ollama.com/library)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Model Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)

---

## Tomorrow: Day 77 - Mini Project: Prompt Engineering

Build a comprehensive prompt engineering toolkit that combines few-shot learning, chain of thought, and local LLMs to solve real-world tasks.
