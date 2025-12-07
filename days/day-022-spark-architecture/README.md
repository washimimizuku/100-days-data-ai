# Day 22: Spark Architecture

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand Spark's distributed architecture
- Learn driver, executor, and cluster manager roles
- Understand jobs, stages, and tasks
- Explore Spark components (Core, SQL, Streaming, MLlib)
- Read Spark UI execution plans

---

## Theory

### What is Apache Spark?

A **unified analytics engine** for large-scale data processing with built-in modules for SQL, streaming, machine learning, and graph processing.

**Created by**: UC Berkeley AMPLab (2009)
**Open Source**: Apache Foundation (2013)
**Language**: Scala (with APIs for Python, Java, R, SQL)

**Key Features**:
- In-memory computing (100x faster than MapReduce)
- Lazy evaluation
- Fault tolerance
- Unified API

---

### Spark Architecture

```
┌─────────────────────────────────────────────┐
│              Driver Program                 │
│  ┌─────────────────────────────────────┐   │
│  │        SparkContext                 │   │
│  │  - Job scheduling                   │   │
│  │  - Task distribution                │   │
│  │  - Result collection                │   │
│  └─────────────────────────────────────┘   │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────────┐
│         Cluster Manager                      │
│    (Standalone / YARN / Mesos / K8s)        │
└──────────────┬───────────────────────────────┘
               │
       ┌───────┴───────┬───────────┐
       ↓               ↓           ↓
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Executor 1 │ │  Executor 2 │ │  Executor 3 │
│  ┌────────┐ │ │  ┌────────┐ │ │  ┌────────┐ │
│  │ Task 1 │ │ │  │ Task 3 │ │ │  │ Task 5 │ │
│  │ Task 2 │ │ │  │ Task 4 │ │ │  │ Task 6 │ │
│  └────────┘ │ │  └────────┘ │ │  └────────┘ │
│   Cache     │ │   Cache     │ │   Cache     │
└─────────────┘ └─────────────┘ └─────────────┘
```

---

### Core Components

#### 1. Driver

**Role**: Orchestrates Spark application

**Responsibilities**:
- Runs main() function
- Creates SparkContext
- Converts code to execution plan
- Schedules tasks
- Collects results

**Code**:
```python
from pyspark.sql import SparkSession

# Driver creates SparkSession
spark = SparkSession.builder \
    .appName("MyApp") \
    .master("local[*]") \
    .getOrCreate()

# Driver code runs here
df = spark.read.csv("data.csv")
result = df.count()  # Driver collects result
```

---

#### 2. Executors

**Role**: Execute tasks and store data

**Responsibilities**:
- Run tasks assigned by driver
- Store data in memory/disk
- Return results to driver
- Report status

**Configuration**:
```python
spark = SparkSession.builder \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "10") \
    .getOrCreate()
```

---

#### 3. Cluster Manager

**Role**: Allocate resources across applications

**Types**:
- **Standalone**: Spark's built-in manager
- **YARN**: Hadoop resource manager
- **Mesos**: General cluster manager
- **Kubernetes**: Container orchestration

---

### Execution Hierarchy

```
Application
    ↓
Job (triggered by action)
    ↓
Stage (separated by shuffle)
    ↓
Task (one per partition)
```

**Example**:
```python
# Application
df = spark.read.csv("data.csv")           # Transformation (lazy)
df = df.filter(col("age") > 25)           # Transformation (lazy)
df = df.groupBy("city").count()           # Transformation (lazy)
result = df.collect()                     # Action (triggers job)

# This creates:
# - 1 Job (triggered by collect)
# - 2 Stages (separated by groupBy shuffle)
# - N Tasks (one per partition)
```

---

### Spark Components

#### Spark Core
- RDD API (low-level)
- Task scheduling
- Memory management
- Fault recovery

#### Spark SQL
- DataFrame/Dataset API
- SQL queries
- Catalyst optimizer
- Data sources (CSV, JSON, Parquet, Delta)

#### Spark Streaming
- Structured Streaming (micro-batches)
- DStreams (legacy)
- Real-time processing

#### MLlib
- Machine learning algorithms
- Feature engineering
- Model training/evaluation

#### GraphX
- Graph processing
- Graph algorithms

---

### Lazy Evaluation

**Concept**: Transformations are not executed until an action is called

**Transformations** (lazy):
```python
df = spark.read.csv("data.csv")      # Not executed
df = df.filter(col("age") > 25)      # Not executed
df = df.select("name", "age")        # Not executed
```

**Actions** (trigger execution):
```python
df.show()           # Executes all transformations
df.count()          # Executes all transformations
df.collect()        # Executes all transformations
df.write.parquet()  # Executes all transformations
```

**Benefits**:
- Optimization opportunities
- Avoid unnecessary computation
- Efficient execution plans

---

### Memory Management

```
Executor Memory
├── Execution Memory (60%)
│   └── Shuffles, joins, sorts
├── Storage Memory (40%)
│   └── Cached data, broadcasts
└── User Memory
    └── User data structures
```

**Configuration**:
```python
spark.conf.set("spark.memory.fraction", "0.6")
spark.conf.set("spark.memory.storageFraction", "0.5")
```

---

### Deployment Modes

#### Local Mode
```python
spark = SparkSession.builder.master("local[*]").getOrCreate()
# local[*] = use all cores
# local[4] = use 4 cores
```

#### Cluster Mode
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 10 \
  --executor-memory 4g \
  --executor-cores 2 \
  my_script.py
```

---

### Spark UI

**Access**: http://localhost:4040 (when Spark is running)

**Tabs**:
- **Jobs**: Job execution timeline
- **Stages**: Stage details and tasks
- **Storage**: Cached RDDs/DataFrames
- **Environment**: Configuration
- **Executors**: Executor metrics
- **SQL**: Query execution plans

**Key Metrics**:
- Duration
- Shuffle read/write
- GC time
- Task distribution

---

### Example Application

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Initialize (Driver)
spark = SparkSession.builder \
    .appName("SalesAnalysis") \
    .master("local[4]") \
    .getOrCreate()

# Read data (Transformation - lazy)
sales = spark.read.csv("sales.csv", header=True, inferSchema=True)

# Transform (Transformation - lazy)
result = sales \
    .filter(col("amount") > 100) \
    .groupBy("category") \
    .agg(
        sum("amount").alias("total_sales"),
        count("*").alias("order_count")
    ) \
    .orderBy(desc("total_sales"))

# Action (triggers execution)
result.show()

# Execution creates:
# - 1 Job (show action)
# - 2 Stages (groupBy causes shuffle)
# - Tasks distributed across 4 cores
```

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Initialize Spark
- Create SparkSession with configuration
- Set app name and master
- Verify Spark UI access

### Exercise 2: Understand Execution
- Create DataFrame with transformations
- Trigger action and observe job
- Check Spark UI for stages/tasks

### Exercise 3: Lazy Evaluation
- Chain multiple transformations
- Verify no execution until action
- Compare execution plans

### Exercise 4: Memory Configuration
- Configure executor memory
- Cache DataFrame
- Check storage in Spark UI

### Exercise 5: Deployment Modes
- Run in local mode
- Simulate cluster configuration
- Compare resource allocation

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is the role of the driver?
2. What is the role of executors?
3. What is lazy evaluation?
4. What triggers job execution?
5. What is the difference between job and stage?
6. What are Spark's main components?
7. What is the Spark UI used for?
8. What is a cluster manager?

---

## 🎯 Key Takeaways

- **Driver** - Orchestrates application, schedules tasks
- **Executors** - Execute tasks, store data
- **Cluster manager** - Allocates resources
- **Lazy evaluation** - Transformations delayed until action
- **Job → Stage → Task** - Execution hierarchy
- **Spark UI** - Monitor and debug applications
- **Components** - Core, SQL, Streaming, MLlib, GraphX
- **In-memory** - 100x faster than MapReduce

---

## 📚 Additional Resources

- [Spark Architecture](https://spark.apache.org/docs/latest/cluster-overview.html)
- [Spark Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html)
- [Spark UI Guide](https://spark.apache.org/docs/latest/web-ui.html)

---

## Tomorrow: Day 23 - Spark DataFrames

We'll dive into the DataFrame API for structured data processing.
