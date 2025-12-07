# 100 Days of Data and AI - Quick Reference Cheatsheet

## Table of Contents
- [Python Essentials](#python-essentials)
- [SQL Basics](#sql-basics)
- [Data Formats](#data-formats)
- [Pandas & NumPy](#pandas--numpy)
- [Apache Spark (PySpark)](#apache-spark-pyspark)
- [Apache Kafka](#apache-kafka)
- [Apache Airflow](#apache-airflow)
- [FastAPI](#fastapi)
- [Docker](#docker)
- [Testing (pytest)](#testing-pytest)
- [Machine Learning (scikit-learn)](#machine-learning-scikit-learn)
- [PyTorch](#pytorch)
- [LLMs & Prompt Engineering](#llms--prompt-engineering)
- [LangChain](#langchain)
- [Vector Databases](#vector-databases)
- [Git](#git)
- [AWS Basics](#aws-basics)

---

## Python Essentials

### Data Types & Variables
```python
# Basic types
name = "Alice"          # str
age = 25                # int
height = 1.75           # float
is_active = True        # bool

# Collections
numbers = [1, 2, 3]     # list (mutable)
coords = (10, 20)       # tuple (immutable)
unique = {1, 2, 3}      # set (unique values)
person = {"name": "Alice", "age": 25}  # dict
```

### List Comprehensions
```python
# Basic
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]

# Dict comprehension
squared = {x: x**2 for x in range(5)}
```

### Functions
```python
# Basic function
def greet(name: str) -> str:
    return f"Hello, {name}!"

# With default args
def process(data, threshold=0.5):
    return [x for x in data if x > threshold]

# Lambda
square = lambda x: x**2
```

### File I/O
```python
# Read file
with open("data.txt", "r") as f:
    content = f.read()

# Write file
with open("output.txt", "w") as f:
    f.write("Hello, World!")

# Read lines
with open("data.txt", "r") as f:
    lines = f.readlines()
```

### Error Handling
```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    print("Cleanup")
```

---

## SQL Basics

### SELECT Queries
```sql
-- Basic select
SELECT column1, column2 FROM table_name;

-- With WHERE
SELECT * FROM users WHERE age > 25;

-- Aggregations
SELECT COUNT(*), AVG(salary), MAX(age)
FROM employees
GROUP BY department
HAVING COUNT(*) > 5;

-- Sorting
SELECT * FROM products
ORDER BY price DESC, name ASC;
```

### JOINs
```sql
-- INNER JOIN
SELECT u.name, o.order_id
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN
SELECT u.name, o.order_id
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- Multiple joins
SELECT u.name, o.order_id, p.product_name
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN products p ON o.product_id = p.id;
```

### Window Functions
```sql
-- Row number
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- Running total
SELECT date, amount,
       SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions;
```

---

## Data Formats

### CSV
```python
import pandas as pd

# Read CSV
df = pd.read_csv("data.csv")

# Write CSV
df.to_csv("output.csv", index=False)
```

### JSON
```python
import json

# Read JSON
with open("data.json", "r") as f:
    data = json.load(f)

# Write JSON
with open("output.json", "w") as f:
    json.dump(data, f, indent=2)

# Pandas JSON
df = pd.read_json("data.json")
df.to_json("output.json", orient="records")
```

### Parquet
```python
import pandas as pd

# Read Parquet
df = pd.read_parquet("data.parquet")

# Write Parquet
df.to_parquet("output.parquet", compression="snappy")

# With PyArrow
import pyarrow.parquet as pq
table = pq.read_table("data.parquet")
pq.write_table(table, "output.parquet")
```

### Avro
```python
from fastavro import writer, reader

# Write Avro
schema = {
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int"}
    ]
}

records = [{"name": "Alice", "age": 25}]
with open("data.avro", "wb") as f:
    writer(f, schema, records)

# Read Avro
with open("data.avro", "rb") as f:
    for record in reader(f):
        print(record)
```

---

## Pandas & NumPy

### Pandas Basics
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["NYC", "LA", "SF"]
})

# Read data
df = pd.read_csv("data.csv")
df = pd.read_parquet("data.parquet")
df = pd.read_json("data.json")

# Inspect
df.head()           # First 5 rows
df.info()           # Column types and nulls
df.describe()       # Statistics
df.shape            # (rows, columns)
```

### Data Selection
```python
# Select columns
df["name"]                    # Single column (Series)
df[["name", "age"]]          # Multiple columns (DataFrame)

# Select rows
df.iloc[0]                   # By position
df.loc[0]                    # By label
df.iloc[0:5]                 # Slice by position

# Filter
df[df["age"] > 25]           # Single condition
df[(df["age"] > 25) & (df["city"] == "NYC")]  # Multiple
```

### Data Manipulation
```python
# Add column
df["age_group"] = df["age"].apply(lambda x: "young" if x < 30 else "old")

# Drop columns
df.drop(columns=["city"], inplace=True)

# Rename
df.rename(columns={"name": "full_name"}, inplace=True)

# Sort
df.sort_values("age", ascending=False)

# Group by
df.groupby("city").agg({"age": ["mean", "count"]})

# Merge
pd.merge(df1, df2, on="id", how="inner")
```

### Handling Missing Data
```python
# Check for nulls
df.isnull().sum()

# Drop nulls
df.dropna()                  # Drop rows with any null
df.dropna(subset=["age"])    # Drop if age is null

# Fill nulls
df.fillna(0)                 # Fill with value
df.fillna(df.mean())         # Fill with mean
df.fillna(method="ffill")    # Forward fill
```

### NumPy Basics
```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
random = np.random.rand(3, 3)

# Operations
arr + 10                     # Add to all
arr * 2                      # Multiply all
arr.mean()                   # Average
arr.sum()                    # Sum
arr.max()                    # Maximum
arr.std()                    # Standard deviation

# Indexing
arr[0]                       # First element
arr[1:4]                     # Slice
arr[arr > 3]                 # Boolean indexing
```

---

## Apache Spark (PySpark)

### Initialize Spark
```python
from pyspark.sql import SparkSession

# Create session
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```

### Read Data
```python
# CSV
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# Parquet
df = spark.read.parquet("data.parquet")

# JSON
df = spark.read.json("data.json")

# From Pandas
pandas_df = pd.DataFrame({"a": [1, 2, 3]})
spark_df = spark.createDataFrame(pandas_df)
```

### DataFrame Operations
```python
# Show data
df.show(5)                   # Show first 5 rows
df.printSchema()             # Show schema
df.count()                   # Count rows

# Select
df.select("name", "age")
df.select(df.age + 1)

# Filter
df.filter(df.age > 25)
df.where(df.city == "NYC")

# Group by
df.groupBy("city").count()
df.groupBy("city").agg({"age": "mean"})
```

### Transformations
```python
from pyspark.sql.functions import col, when, lit

# Add column
df = df.withColumn("age_group", 
    when(col("age") < 30, "young")
    .otherwise("old"))

# Rename
df = df.withColumnRenamed("name", "full_name")

# Drop
df = df.drop("city")

# Cast type
df = df.withColumn("age", col("age").cast("integer"))
```

### SQL Queries
```python
# Register as temp view
df.createOrReplaceTempView("users")

# Run SQL
result = spark.sql("""
    SELECT city, AVG(age) as avg_age
    FROM users
    GROUP BY city
    HAVING AVG(age) > 25
""")
```

### Write Data
```python
# Parquet
df.write.parquet("output.parquet", mode="overwrite")

# CSV
df.write.csv("output.csv", header=True, mode="overwrite")

# Partitioned
df.write.partitionBy("year", "month").parquet("output")
```

---

## Apache Kafka

### Producer
```python
from kafka import KafkaProducer
import json

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send message
producer.send('my-topic', {'key': 'value'})
producer.flush()
```

### Consumer
```python
from kafka import KafkaConsumer
import json

# Create consumer
consumer = KafkaConsumer(
    'my-topic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='my-group'
)

# Consume messages
for message in consumer:
    print(message.value)
```

### Spark Structured Streaming with Kafka
```python
# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "my-topic") \
    .load()

# Process
df = df.selectExpr("CAST(value AS STRING)")

# Write to console
query = df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
```

---

## Apache Airflow

### Basic DAG
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_dag',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

# Define tasks
def extract():
    print("Extracting data")

def transform():
    print("Transforming data")

def load():
    print("Loading data")

extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform,
    dag=dag
)

load_task = PythonOperator(
    task_id='load',
    python_callable=load,
    dag=dag
)

# Set dependencies
extract_task >> transform_task >> load_task
```

### Task Dependencies
```python
# Linear
task1 >> task2 >> task3

# Parallel
task1 >> [task2, task3] >> task4

# Multiple dependencies
[task1, task2] >> task3
```

---

## FastAPI

### Basic API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Simple endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Path parameter
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

# Query parameter
@app.get("/search")
def search(q: str, limit: int = 10):
    return {"query": q, "limit": limit}
```

### Request Body with Pydantic
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

@app.post("/users")
def create_user(user: User):
    return {"user": user, "status": "created"}
```

### Async Endpoints
```python
import asyncio

@app.get("/async-data")
async def get_async_data():
    await asyncio.sleep(1)  # Simulate I/O
    return {"data": "async result"}
```

### Run Server
```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Docker

### Basic Commands
```bash
# Build image
docker build -t myapp:latest .

# Run container
docker run -p 8000:8000 myapp:latest

# Run in background
docker run -d -p 8000:8000 myapp:latest

# List containers
docker ps                    # Running
docker ps -a                 # All

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>

# Remove image
docker rmi <image_id>
```

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on:
      - db
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down
```

---

## Testing (pytest)

### Basic Tests
```python
# test_calculator.py
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_add_floats():
    assert add(1.5, 2.5) == 4.0
```

### Fixtures
```python
import pytest

@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15

def test_len(sample_data):
    assert len(sample_data) == 5
```

### Parametrize
```python
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (10, 20, 30),
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected
```

### Run Tests
```bash
# Run all tests
pytest

# Run specific file
pytest test_calculator.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=myapp
```

---

## Machine Learning (scikit-learn)

### Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Classification
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

---

## PyTorch

### Tensors
```python
import torch

# Create tensors
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 3)
z = torch.randn(2, 4)

# Operations
a = x + 10
b = x * 2
c = torch.matmul(x, y)

# Move to GPU
if torch.cuda.is_available():
    x = x.cuda()
```

### Simple Neural Network
```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
```

### Training Loop
```python
import torch.optim as optim

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

---

## LLMs & Prompt Engineering

### Basic Prompting
```python
# Zero-shot
prompt = "Classify the sentiment: 'I love this product!'"

# Few-shot
prompt = """
Classify sentiment as positive, negative, or neutral:

Text: "Great service!"
Sentiment: positive

Text: "Terrible experience"
Sentiment: negative

Text: "It was okay"
Sentiment: neutral

Text: "Amazing quality!"
Sentiment:
"""

# Chain-of-thought
prompt = """
Question: If a store has 15 apples and sells 7, then receives 12 more, how many apples does it have?

Let's think step by step:
1. Start with 15 apples
2. Sell 7: 15 - 7 = 8 apples
3. Receive 12: 8 + 12 = 20 apples

Answer: 20 apples
"""
```

### Using Ollama (Local LLMs)
```python
import ollama

# Generate response
response = ollama.generate(
    model='llama2',
    prompt='Explain machine learning in simple terms'
)
print(response['response'])

# Chat
messages = [
    {'role': 'user', 'content': 'What is Python?'}
]
response = ollama.chat(model='llama2', messages=messages)
print(response['message']['content'])
```

---

## LangChain

### Basic Chain
```python
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = Ollama(model="llama2")

# Create prompt template
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate(template=template, input_variables=["product"])

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run
result = chain.run(product="eco-friendly water bottles")
print(result)
```

### RAG (Retrieval-Augmented Generation)
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="llama2")
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Query
result = qa_chain("What is the main topic of the document?")
print(result['result'])
```

---

## Vector Databases

### ChromaDB
```python
import chromadb

# Create client
client = chromadb.Client()

# Create collection
collection = client.create_collection("my_collection")

# Add documents
collection.add(
    documents=["This is document 1", "This is document 2"],
    ids=["id1", "id2"],
    metadatas=[{"source": "web"}, {"source": "book"}]
)

# Query
results = collection.query(
    query_texts=["document about topic"],
    n_results=2
)
print(results)
```

### FAISS
```python
import faiss
import numpy as np

# Create index
dimension = 128
index = faiss.IndexFlatL2(dimension)

# Add vectors
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

# Search
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)
```

---

## Git

### Basic Commands
```bash
# Initialize repo
git init

# Clone repo
git clone <url>

# Check status
git status

# Add files
git add file.py
git add .                    # Add all

# Commit
git commit -m "Add feature"

# Push
git push origin main

# Pull
git pull origin main
```

### Branching
```bash
# Create branch
git branch feature-branch

# Switch branch
git checkout feature-branch

# Create and switch
git checkout -b feature-branch

# Merge
git checkout main
git merge feature-branch

# Delete branch
git branch -d feature-branch
```

---

## AWS Basics

### S3 (boto3)
```python
import boto3

# Create client
s3 = boto3.client('s3')

# Upload file
s3.upload_file('local.txt', 'my-bucket', 'remote.txt')

# Download file
s3.download_file('my-bucket', 'remote.txt', 'local.txt')

# List objects
response = s3.list_objects_v2(Bucket='my-bucket')
for obj in response['Contents']:
    print(obj['Key'])
```

### Lambda Function
```python
def lambda_handler(event, context):
    # Process event
    name = event.get('name', 'World')
    
    return {
        'statusCode': 200,
        'body': f'Hello, {name}!'
    }
```

---

## Tips & Best Practices

### Data Engineering
- Use Parquet for analytical workloads (columnar, compressed)
- Partition large datasets by date or category
- Use Spark for distributed processing (>1GB data)
- Implement data quality checks early
- Version your data schemas

### Machine Learning
- Always split data before any preprocessing
- Use cross-validation for model evaluation
- Track experiments with MLflow
- Monitor model performance in production
- Start simple, then add complexity

### GenAI/LLMs
- Be specific and clear in prompts
- Use few-shot examples for better results
- Implement RAG for domain-specific knowledge
- Chunk documents appropriately (500-1000 tokens)
- Monitor token usage and costs

### Development
- Write tests for critical functions
- Use type hints for clarity
- Follow PEP 8 style guide
- Version control everything
- Document your code

---

**Quick Links:**
- [Python Docs](https://docs.python.org/3/)
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Spark Docs](https://spark.apache.org/docs/latest/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangChain Docs](https://python.langchain.com/)
- [scikit-learn Docs](https://scikit-learn.org/)
- [PyTorch Docs](https://pytorch.org/docs/)
