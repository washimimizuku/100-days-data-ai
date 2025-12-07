# Day 29: Streaming Concepts

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Understand batch vs streaming processing paradigms
- Learn key streaming concepts (latency, throughput, windowing)
- Identify when to use streaming vs batch
- Understand streaming architectures and patterns

---

## Theory

### What is Stream Processing?

**Stream processing** handles data continuously as it arrives, rather than waiting to collect a batch.

**Key Difference**:
- **Batch**: Process data in large chunks at scheduled intervals
- **Streaming**: Process data continuously as events occur

```python
# Batch mindset
data = read_all_data()  # Wait for all data
results = process(data)  # Process everything
write_results(results)   # Write once

# Streaming mindset
for event in stream:     # Process as data arrives
    result = process(event)
    write_result(result)  # Write immediately
```

### Batch vs Streaming Comparison

| Aspect | Batch Processing | Stream Processing |
|--------|-----------------|-------------------|
| **Latency** | Minutes to hours | Milliseconds to seconds |
| **Data Volume** | Large, bounded datasets | Continuous, unbounded data |
| **Use Case** | Historical analysis, reports | Real-time alerts, monitoring |
| **Complexity** | Simpler | More complex |
| **Cost** | Lower (scheduled) | Higher (always running) |
| **Examples** | Daily ETL, monthly reports | Fraud detection, IoT sensors |

### When to Use Each

**Use Batch When**:
- Data arrives in predictable intervals
- Latency requirements are relaxed (hours/days OK)
- Processing large historical datasets
- Cost optimization is priority
- Examples: Daily sales reports, monthly analytics

**Use Streaming When**:
- Need real-time or near-real-time results
- Data arrives continuously
- Time-sensitive decisions required
- Examples: Fraud detection, stock trading, IoT monitoring

### Key Streaming Concepts

#### 1. Event Time vs Processing Time

```python
# Event time: When event actually occurred
event = {
    "user_id": 123,
    "action": "purchase",
    "event_time": "2024-01-15 10:30:00",  # When it happened
    "processing_time": "2024-01-15 10:30:05"  # When we processed it
}

# Late arrival: Event arrives after delay
late_event = {
    "event_time": "2024-01-15 10:29:00",  # Happened earlier
    "processing_time": "2024-01-15 10:31:00"  # Arrived later
}
```

#### 2. Windowing

Group streaming data into finite chunks for aggregation.

```python
# Tumbling window: Fixed, non-overlapping intervals
# [0-5min] [5-10min] [10-15min]

# Sliding window: Overlapping intervals
# [0-5min] [1-6min] [2-7min]

# Session window: Based on activity gaps
# [user active] [gap] [user active again]
```

#### 3. Watermarks

Handle late-arriving data by defining "how late is too late".

```python
# Watermark: "Process all events up to T-5 minutes"
watermark = current_time - timedelta(minutes=5)

if event.event_time < watermark:
    # Too late, discard or handle separately
    handle_late_event(event)
else:
    # Process normally
    process_event(event)
```

### Streaming Architecture Patterns

#### 1. Lambda Architecture

Combines batch and streaming for accuracy + speed.

```
Raw Data → Batch Layer (accurate, slow)
        → Speed Layer (fast, approximate)
        → Serving Layer (merge results)
```

#### 2. Kappa Architecture

Streaming-only, simpler than Lambda.

```
Raw Data → Stream Processing → Results
        ↓
    Replayable Log (for reprocessing)
```

### Common Streaming Use Cases

**1. Real-Time Analytics**
```python
# Count website clicks per minute
stream.window(minutes=1).count()
```

**2. Fraud Detection**
```python
# Alert on suspicious transactions
if transaction.amount > 10000 and transaction.location != user.home_country:
    send_alert()
```

**3. IoT Monitoring**
```python
# Alert on sensor anomalies
if sensor.temperature > threshold:
    trigger_alarm()
```

**4. Log Processing**
```python
# Real-time error tracking
if log.level == "ERROR":
    notify_team()
```

### Streaming Challenges

**1. State Management**
- Maintaining state across events (counters, aggregations)
- Handling failures without losing state

**2. Exactly-Once Processing**
- Ensuring each event is processed exactly once
- Avoiding duplicates or data loss

**3. Late Data**
- Handling events that arrive out of order
- Deciding when to close windows

**4. Backpressure**
- Handling when data arrives faster than processing
- Preventing system overload

### Streaming Technologies

**Message Queues**:
- Apache Kafka (most popular)
- Apache Pulsar
- AWS Kinesis
- RabbitMQ

**Stream Processing**:
- Apache Flink
- Apache Spark Structured Streaming
- Apache Storm
- Kafka Streams

**Example: Simple Streaming Pipeline**
```python
# Conceptual streaming pipeline
source = kafka_topic("user_events")
stream = source.map(parse_json)
filtered = stream.filter(lambda e: e.type == "purchase")
aggregated = filtered.window(minutes=5).sum("amount")
aggregated.sink(database)
```

---

## 💻 Exercises (40 min)

### Exercise 1: Batch vs Streaming Decision
Analyze scenarios and decide batch or streaming.

### Exercise 2: Event Time Simulation
Simulate events with timestamps and late arrivals.

### Exercise 3: Windowing Implementation
Implement tumbling and sliding windows.

### Exercise 4: Watermark Logic
Handle late-arriving events with watermarks.

### Exercise 5: Simple Stream Processor
Build a basic stream processor with state.

### Exercise 6: Use Case Design
Design a streaming architecture for a use case.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- **Batch** processes data in chunks; **streaming** processes continuously
- **Event time** (when it happened) vs **processing time** (when we saw it)
- **Windowing** groups unbounded streams into finite chunks
- **Watermarks** handle late-arriving data
- Use streaming for real-time needs; batch for cost-effective historical analysis
- Common patterns: Lambda (batch + stream) and Kappa (stream-only)
- Key challenges: state management, exactly-once, late data, backpressure

---

## 📚 Resources

- [Streaming 101 (O'Reilly)](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/)
- [Streaming Systems Book](https://www.oreilly.com/library/view/streaming-systems/9781491983867/)
- [Apache Kafka Docs](https://kafka.apache.org/documentation/)
- [Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)

---

## Tomorrow: Day 30 - Kafka Fundamentals

Learn Apache Kafka, the most popular streaming platform, including topics, partitions, and basic architecture.
