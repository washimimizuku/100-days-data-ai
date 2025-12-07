# Day 92: GenAI Checkpoint - Solutions & Guidance

## Overview

This document provides reference solutions and guidance for the checkpoint exercises. Use this to verify your understanding and identify areas for improvement.

---

## Section 1: APIs & Testing (15 points)

### Exercise 1: FastAPI Endpoint (5 points)

**Solution Approach:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

class DataRequest(BaseModel):
    values: List[float]
    operation: str

app = FastAPI()

@app.post("/process")
async def process_data(request: DataRequest):
    if request.operation == "mean":
        result = sum(request.values) / len(request.values)
    elif request.operation == "sum":
        result = sum(request.values)
    else:
        result = None
    
    return {"result": result, "operation": request.operation}
```

**Key Points:**
- Use Pydantic for validation
- Define clear request/response models
- Handle different operations
- Return structured responses

### Exercise 2: Async Operations (5 points)

**Solution Approach:**
```python
import asyncio

async def fetch_data(source: str):
    await asyncio.sleep(0.1)  # Simulate I/O
    return f"Data from {source}"

async def process_concurrent():
    sources = ["api1", "api2", "api3"]
    tasks = [fetch_data(s) for s in sources]
    results = await asyncio.gather(*tasks)
    return results
```

**Key Points:**
- Use asyncio.gather for concurrent execution
- Async is beneficial for I/O-bound operations
- Reduces total execution time
- Maintains code readability

### Exercise 3: Testing (5 points)

**Solution Approach:**
```python
import pytest

def test_data_validation():
    data = {"value": 10}
    assert validate_data(data) == True

def test_transformation():
    input_data = [1, 2, 3]
    result = transform(input_data)
    assert result == [2, 4, 6]

def test_error_handling():
    with pytest.raises(ValueError):
        process_invalid_data(None)
```

**Key Points:**
- Test validation logic
- Test transformations
- Test error cases
- Use fixtures for setup

---

## Section 2: Machine Learning (20 points)

### Exercise 4: Feature Engineering (5 points)

**Solution Approach:**
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

def engineer_features(df):
    # Numerical transformations
    df['log_value'] = np.log1p(df['value'])
    
    # Categorical encoding
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    
    # Scaling
    scaler = StandardScaler()
    df[['value_scaled']] = scaler.fit_transform(df[['value']])
    
    return df
```

**Key Points:**
- Log transform for skewed data
- Encode categorical variables
- Scale numerical features
- Create interaction features

### Exercise 5: Model Training (5 points)

**Solution Approach:**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

**Key Points:**
- Split data properly
- Choose appropriate model
- Train on training set
- Evaluate on test set

### Exercise 6: Cross-Validation (5 points)

**Solution Approach:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Key Points:**
- Use k-fold CV (typically 5 or 10)
- Calculate mean and std
- Helps detect overfitting
- More reliable than single split

### Exercise 7: Hyperparameter Tuning (5 points)

**Solution Approach:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
```

**Key Points:**
- Define parameter grid
- Use GridSearchCV or RandomizedSearchCV
- Evaluate with CV
- Select best parameters

---

## Section 3: Deep Learning (15 points)

### Exercise 8: PyTorch Model (5 points)

**Solution Approach:**
```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

**Key Points:**
- Inherit from nn.Module
- Define layers in __init__
- Implement forward pass
- Use appropriate activations

### Exercise 9: Training Loop (5 points)

**Solution Approach:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Key Points:**
- Forward pass
- Calculate loss
- Zero gradients
- Backward pass
- Update weights

### Exercise 10: Transfer Learning (5 points)

**Solution Approach:**
```python
from torchvision import models

model = models.resnet18(pretrained=True)

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

**Key Points:**
- Load pre-trained model
- Freeze early layers
- Replace final layer
- Fine-tune on new data

---

## Section 4: GenAI (25 points)

### Exercise 11: Prompt Engineering (5 points)

**Solution Examples:**

**Zero-shot:**
```
Classify the sentiment of this text: "I love this product!"
```

**Few-shot:**
```
Classify sentiment:
Text: "Great service!" → Positive
Text: "Terrible experience" → Negative
Text: "It's okay" → Neutral
Text: "Amazing quality!" → ?
```

**Chain of Thought:**
```
Question: What is 25% of 80?
Let's think step by step:
1. Convert 25% to decimal: 0.25
2. Multiply: 80 * 0.25
3. Result: 20
```

**Key Points:**
- Clear instructions
- Provide examples for few-shot
- Break down reasoning for CoT
- Specify output format

### Exercise 12: Document Chunking (5 points)

**Solution Approach:**
```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks
```

**Key Points:**
- Fixed or semantic chunking
- Overlap preserves context
- Consider token limits
- Maintain coherence

### Exercise 13: Vector Embeddings (5 points)

**Solution Approach:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = ["Hello world", "Hi there"]
embeddings = model.encode(texts)

# Cosine similarity
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
```

**Key Points:**
- Use embedding models
- Calculate similarity
- Find nearest neighbors
- Store in vector database

### Exercise 14: RAG Pipeline (5 points)

**Solution Approach:**
```python
def rag_pipeline(query: str, documents: List[str]):
    # 1. Retrieve
    embeddings = embed_documents(documents)
    query_embedding = embed_query(query)
    relevant_docs = find_similar(query_embedding, embeddings, top_k=3)
    
    # 2. Generate
    context = "\n".join(relevant_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    answer = llm.generate(prompt)
    
    return answer
```

**Key Points:**
- Retrieve relevant documents
- Create context
- Generate answer
- Return with sources

### Exercise 15: LangChain Chain (5 points)

**Solution Approach:**
```python
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory

template = "Context: {context}\nQuestion: {question}\nAnswer:"
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

memory = ConversationBufferMemory()
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

**Key Points:**
- Define prompt templates
- Add memory for context
- Chain components
- Handle state

---

## Section 5: Agentic AI (25 points)

### Exercise 16: Agent Architecture (5 points)

**Solution Design:**
```
Components:
- ReAct Engine (reasoning)
- Tool Registry (actions)
- Memory (state)
- Workflow Manager (orchestration)

Interactions:
- Engine selects tools
- Tools execute actions
- Memory stores history
- Manager coordinates flow

Workflow:
1. Receive query
2. Think about approach
3. Select and execute tool
4. Observe result
5. Repeat or finish
```

**Key Points:**
- Clear component separation
- Well-defined interfaces
- State management
- Error handling

### Exercise 17: ReAct Loop (5 points)

**Solution Approach:**
```python
def react_loop(query: str, max_iterations: int = 5):
    state = {"query": query, "thoughts": [], "actions": [], "observations": []}
    
    for i in range(max_iterations):
        # Think
        thought = generate_thought(state)
        state["thoughts"].append(thought)
        
        # Act
        action = select_action(thought)
        result = execute_action(action)
        state["actions"].append(action)
        
        # Observe
        observation = process_result(result)
        state["observations"].append(observation)
        
        if is_complete(state):
            break
    
    return extract_answer(state)
```

**Key Points:**
- Thought-action-observation cycle
- State tracking
- Iteration limit
- Completion check

### Exercise 18: Tool Registry (5 points)

**Solution Approach:**
```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, func: callable, schema: dict):
        self.tools[name] = {"func": func, "schema": schema}
    
    def execute(self, name: str, **kwargs):
        tool = self.tools[name]
        return tool["func"](**kwargs)
```

**Key Points:**
- Tool registration
- Schema definition
- Parameter validation
- Error handling

### Exercise 19: LangGraph Workflow (5 points)

**Solution Approach:**
```python
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)

workflow.add_node("think", think_node)
workflow.add_node("act", act_node)
workflow.add_node("observe", observe_node)

workflow.add_edge("think", "act")
workflow.add_edge("act", "observe")
workflow.add_conditional_edges("observe", should_continue)

app = workflow.compile()
```

**Key Points:**
- Define state structure
- Create node functions
- Add edges
- Conditional routing

### Exercise 20: AWS Deployment (5 points)

**Solution Plan:**
```
Services:
- S3: Model and data storage
- Lambda: Inference endpoint
- API Gateway: REST API
- DynamoDB: Results storage
- CloudWatch: Monitoring

Architecture:
API Gateway → Lambda → Model (from S3) → DynamoDB

Estimated Cost:
- Lambda: $5/month (1M requests)
- S3: $1/month (10GB)
- API Gateway: $3.50/month
- DynamoDB: $2/month
Total: ~$12/month
```

**Key Points:**
- Choose appropriate services
- Design scalable architecture
- Estimate costs
- Plan monitoring

---

## Scoring Guide

### Excellent (90-100 points)
- Deep understanding of all concepts
- Can implement solutions independently
- Explains trade-offs clearly
- Ready for advanced topics

### Strong (80-89 points)
- Good understanding of most concepts
- Can implement with minimal reference
- Understands key principles
- Minor gaps in knowledge

### Proficient (70-79 points)
- Solid understanding of core concepts
- Can implement with reference
- Grasps main ideas
- Some areas need practice

### Needs Review (60-69 points)
- Basic understanding
- Struggles with implementation
- Needs more practice
- Review key concepts

### Revisit Material (Below 60)
- Significant gaps in understanding
- Cannot implement solutions
- Needs comprehensive review
- Revisit course material

---

## Next Steps Based on Score

### 90-100: Excellent
- Move to advanced topics
- Build complex projects
- Mentor others
- Explore cutting-edge research

### 80-89: Strong
- Practice weak areas
- Build more projects
- Deepen understanding
- Continue to Week 14

### 70-79: Proficient
- Review challenging topics
- Complete more exercises
- Build simple projects
- Strengthen fundamentals

### 60-69: Needs Review
- Revisit course material
- Work through examples
- Practice coding
- Seek help on unclear topics

### Below 60: Revisit Material
- Review all material
- Complete all exercises
- Build understanding gradually
- Don't rush ahead

---

## Additional Resources

### APIs & Testing
- FastAPI documentation and tutorials
- Async programming guides
- pytest best practices

### Machine Learning
- Scikit-learn examples
- Feature engineering techniques
- Model evaluation guides

### Deep Learning
- PyTorch tutorials
- Neural network architectures
- Transfer learning examples

### GenAI
- Prompt engineering guides
- RAG implementation examples
- LangChain documentation

### Agentic AI
- Agent design patterns
- ReAct paper and examples
- LangGraph tutorials

---

## Conclusion

Use these solutions as a guide, not a crutch. The goal is understanding, not memorization. If you struggled with any section, that's okay—it shows you where to focus your learning efforts.

Remember: Building AI systems is a journey. Each checkpoint is an opportunity to assess progress and adjust your learning path.

Good luck with the remaining days!
