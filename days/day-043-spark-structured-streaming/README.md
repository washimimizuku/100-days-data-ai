# Day 43: Spark Structured Streaming Fundamentals

## 📖 Learning Objectives

By the end of this session, you will:
- Understand Spark Structured Streaming architecture
- Create streaming DataFrames from various sources
- Apply transformations to streaming data
- Write streaming queries with different output modes
- Handle streaming triggers and checkpointing
- Monitor streaming queries

**Time**: 1 hour

---

## What is Spark Structured Streaming?

Spark Structured Streaming is a **scalable and fault-tolerant stream processing engine** built on Spark SQL. It treats streaming data as an unbounded table that continuously grows.

### Key Concepts

**1. Unbounded Table Model**
```
Stream = Unbounded Table

Time    Data
----    ----
t1      [row1]
t2      [row1, row2]
t3      [row1, row2, row3]
...
```

**2. Micro-Batch Processing**
- Processes data in small batches (default 500ms)
- Provides exactly-once semantics
- Balances latency and throughput

**3. Continuous Processing** (experimental)
- True streaming with ~1ms latency
- At-least-once semantics

---

## Architecture

```
Input Sources → Streaming DataFrame → Transformations → Output Sink
    ↓                                                         ↓
  Kafka                                                   Console
  Files                                                   Files
  Sockets                                                 Kafka
  Rate                                                    Memory
```

### Components

1. **Source**: Where data comes from (Kafka, files, sockets)
2. **Streaming DataFrame**: Unbounded table representation
3. **Query**: Transformations applied to streaming data
4. **Sink**: Where results are written
5. **Checkpoint**: Fault tolerance and recovery

---

## Creating Streaming DataFrames

### Source Examples

```python
# Socket (testing)
lines = spark.readStream.format("socket") \
    .option("host", "localhost").option("port", 9999).load()

# Files (CSV)
df = spark.readStream.format("csv").schema(schema) \
    .option("header", True).load("data/input/")

# Kafka
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events").load()

# Rate (testing)
df = spark.readStream.format("rate") \
    .option("rowsPerSecond", 10).load()
```

---

## Transformations

Streaming DataFrames support most DataFrame operations:

### Transformation Examples

```python
# Basic operations
df.select("user_id", "action")
df.filter(df.action == "purchase")
df.withColumn("processed_at", current_timestamp())

# Aggregations
df.groupBy("action").count()
df.groupBy("user_id").agg(count("*"), sum("amount"))

# Windowed aggregations
df.groupBy(window("timestamp", "10 minutes"), "action").count()
df.groupBy(window("timestamp", "10 minutes", "5 minutes"), "action").count()
```

---

## Output Modes

**1. Append Mode**: Only new rows written (default for non-aggregated queries)

**2. Complete Mode**: Entire result table written every trigger (for aggregations)

**3. Update Mode**: Only updated rows written (for aggregations with watermark)

```python
# Append
df.writeStream.outputMode("append").format("console").start()

# Complete
df.groupBy("action").count().writeStream.outputMode("complete").format("console").start()

# Update
df.groupBy("user_id").count().writeStream.outputMode("update").format("console").start()
```

---

## Output Sinks

### Sink Examples

```python
# Console (testing)
df.writeStream.format("console").option("truncate", False).start()

# File (parquet)
df.writeStream.format("parquet") \
    .option("path", "output/data") \
    .option("checkpointLocation", "output/checkpoint").start()

# Kafka
df.selectExpr("CAST(user_id AS STRING) AS key", "to_json(struct(*)) AS value") \
    .writeStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "output").start()

# Memory (testing)
df.writeStream.format("memory").queryName("my_table").start()
spark.sql("SELECT * FROM my_table").show()
```

---

## Triggers

Control when streaming query processes data:

```python
# Default (micro-batch as soon as possible)
df.writeStream.trigger(processingTime="0 seconds").format("console").start()

# Fixed interval (every 10 seconds)
df.writeStream.trigger(processingTime="10 seconds").format("console").start()

# One-time (process all available data once)
df.writeStream.trigger(once=True).format("console").start()

# Continuous (experimental, ~1ms latency)
df.writeStream.trigger(continuous="1 second").format("console").start()
```

---

## Checkpointing

Checkpoints enable fault tolerance and exactly-once semantics.

```python
query = df.writeStream \
    .format("parquet") \
    .option("path", "output/data") \
    .option("checkpointLocation", "output/checkpoint") \
    .start()
```

**Checkpoint stores**:
- Offsets processed
- Query metadata
- State information

**Important**: Don't delete checkpoint directory while query is running!

---

## Query Management

```python
# Start query
query = df.writeStream.format("console").start()

# Monitor
print(query.status)
print(query.lastProgress)
print(query.isActive)

# Stop
query.stop()
spark.streams.stopAll()
query.awaitTermination(timeout=60)

# List active queries
for q in spark.streams.active:
    print(f"Query: {q.name}, Status: {q.status}")
```

---

## 💻 Exercises

### Exercise 1: Socket Stream Word Count
Create a streaming word count from socket input.

### Exercise 2: File Stream Processing
Monitor a directory and process new CSV files.

### Exercise 3: Rate Source Aggregation
Use rate source to generate data and calculate statistics.

### Exercise 4: Multiple Output Modes
Compare append, complete, and update modes.

### Exercise 5: Checkpoint Recovery
Test fault tolerance with checkpointing.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Structured Streaming treats streams as unbounded tables
- Supports most DataFrame operations
- Three output modes: append, complete, update
- Checkpointing enables fault tolerance
- Triggers control processing frequency
- Use console/memory sinks for testing, Kafka/files for production

---

## 📚 Resources

- [Structured Streaming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- [Streaming Sources and Sinks](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#input-sources)
- [Output Modes](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#output-modes)

---

## Tomorrow: Day 44 - Streaming Joins and Aggregations

Learn advanced streaming operations including stream-to-static and stream-to-stream joins.
