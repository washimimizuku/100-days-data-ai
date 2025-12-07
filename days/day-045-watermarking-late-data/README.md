# Day 45: Watermarking and Late Data Handling

## 📖 Learning Objectives

By the end of this session, you will:
- Understand watermarking concepts and mechanics
- Handle late-arriving data effectively
- Configure watermark delays appropriately
- Manage state with watermarks
- Implement event-time processing
- Handle out-of-order events

**Time**: 1 hour

---

## What is Watermarking?

**Watermark** is a threshold that defines how late data can arrive before being considered "too late" and dropped.

### The Problem

In real-world streaming:
- Events arrive out of order
- Network delays cause late arrivals
- Clock skew between systems
- Infinite state growth without bounds

### The Solution

Watermarks tell Spark:
- When to finalize windows
- When to drop old state
- How long to wait for late data

---

## Event Time vs Processing Time

**Event Time**: When event actually occurred (embedded in data)  
**Processing Time**: When event is processed by Spark

```python
# Event created at 10:00:00, arrives at 10:05:00 (5 minutes late)
# Event time: 10:00:00, Processing time: 10:05:00, Lateness: 5 minutes

# Correct: Use event time for business logic
df.groupBy(window("event_time", "10 minutes")).count()  # ✓

# Wrong: Processing time gives incorrect results
df.groupBy(window(current_timestamp(), "10 minutes")).count()  # ✗
```

---

## Watermark Basics

### Watermark Definition

```python
# Define 10-minute watermark: "Wait up to 10 minutes for late data"
df_with_watermark = df.withWatermark("event_time", "10 minutes")
```

**Calculation**: `Watermark = Max(event_time) - Watermark_Delay`

**Example**:
- Latest event: 10:30:00
- Delay: 10 minutes
- Watermark: 10:20:00
- Events with event_time < 10:20:00 are dropped

---

## Watermark with Aggregations

### Watermark with Output Modes

```python
# Define watermark and aggregate
df_watermarked = df.withWatermark("event_time", "10 minutes")
windowed = df_watermarked.groupBy(window("event_time", "5 minutes")).count()

# Append mode: Output only finalized windows (once)
windowed.writeStream.outputMode("append").format("console").start()

# Update mode: Output windows as they update (multiple times)
windowed.writeStream.outputMode("update").format("console").start()
```

**Window finalized when**: Window end time < Current watermark

---

## Choosing Watermark Delay

### Factors to Consider

1. **Data Lateness**: How late can data arrive?
2. **Business Requirements**: Can you tolerate data loss?
3. **State Size**: Longer delay = more state
4. **Output Latency**: Longer delay = slower results

### Delay Examples & Trade-offs

```python
df.withWatermark("event_time", "1 minute")   # Strict: Less state, more drops
df.withWatermark("event_time", "10 minutes") # Moderate: Balanced
df.withWatermark("event_time", "1 hour")     # Lenient: More state, fewer drops
```

**Trade-off**: Short delay = faster output + more drops, Long delay = slower output + fewer drops

---

## Late Data Handling Strategies

### Late Data Strategies

```python
# Strategy 1: Drop late data (automatic with watermark)
df.withWatermark("event_time", "10 minutes").groupBy(window("event_time", "5 minutes")).count()

# Strategy 2: Separate late data stream
df_with_lateness = df.withColumn("lateness", expr("unix_timestamp(current_timestamp()) - unix_timestamp(event_time)"))
on_time = df_with_lateness.filter(col("lateness") <= 600)
late = df_with_lateness.filter(col("lateness") > 600)

# Strategy 3: Side output for monitoring
main_stream.writeStream.format("kafka").option("topic", "results").start()
late_stream.writeStream.format("kafka").option("topic", "late_data").start()
```

---

## Watermark with Joins

### Joins with Watermark

```python
# Both streams need watermarks
orders = orders_df.withWatermark("order_time", "10 minutes")
payments = payments_df.withWatermark("payment_time", "10 minutes")

# Inner join with time constraint
joined = orders.join(payments, expr("""
    orders.order_id = payments.order_id AND
    payments.payment_time >= orders.order_time AND
    payments.payment_time <= orders.order_time + interval 15 minutes
"""))

# Left outer join (outputs unmatched orders after watermark passes)
left_joined = orders.join(payments, expr("..."), "left_outer")
```

**Watermark enables**: State cleanup, memory management, outer joins

---

## State Management

### State Growth Without Watermark

```python
# No watermark = infinite state growth
df.groupBy(
    window("event_time", "5 minutes")
).count()

# Spark keeps ALL windows in memory forever!
# Eventually: OutOfMemoryError
```

### State Cleanup With Watermark

```python
# With watermark = bounded state
df.withWatermark("event_time", "10 minutes") \
    .groupBy(window("event_time", "5 minutes")) \
    .count()

# Spark drops state for windows older than watermark
# Memory usage bounded
```

### Monitoring State

```python
query = df_watermarked.groupBy(
    window("event_time", "5 minutes")
).count().writeStream \
    .format("console") \
    .start()

# Check state metrics
progress = query.lastProgress
if progress:
    print(f"State rows: {progress['stateOperators'][0]['numRowsTotal']}")
    print(f"State memory: {progress['stateOperators'][0]['memoryUsedBytes']}")
```

---

## Advanced Patterns

### Multiple Watermarks

```python
# Different watermarks for different operations
df1 = df.withWatermark("event_time", "5 minutes")
df2 = df.withWatermark("event_time", "10 minutes")

# Use appropriate watermark for each operation
```

### Watermark Propagation

```python
# Watermark propagates through transformations
df_watermarked = df.withWatermark("event_time", "10 minutes")

# Watermark preserved
filtered = df_watermarked.filter(col("amount") > 100)
mapped = filtered.select("event_time", "amount")

# Still has watermark
aggregated = mapped.groupBy(
    window("event_time", "5 minutes")
).count()
```

### Custom Late Data Handling

```python
from pyspark.sql.functions import when, lit

# Tag late data
df_tagged = df.withColumn(
    "is_late",
    when(
        col("event_time") < expr("current_timestamp() - interval 10 minutes"),
        lit(True)
    ).otherwise(lit(False))
)

# Process with tag
result = df_tagged.groupBy(
    window("event_time", "5 minutes"),
    "is_late"
).count()
```

---

## 💻 Exercises

### Exercise 1: Basic Watermark
Create streaming query with 5-minute watermark.

### Exercise 2: Late Data Detection
Identify and count late-arriving events.

### Exercise 3: Watermark with Aggregation
Implement windowed aggregation with watermark.

### Exercise 4: State Management
Monitor state size with and without watermark.

### Exercise 5: Join with Watermark
Implement stream-to-stream join with watermarks.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Watermarks define how long to wait for late data
- Calculated as: Max(event_time) - delay
- Essential for bounded state in aggregations and joins
- Trade-off: longer delay = more complete data but more state
- Use event time for business logic, not processing time
- Always set watermarks for production streaming jobs

---

## 📚 Resources

- [Watermarking Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#handling-late-data-and-watermarking)
- [Event Time Processing](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#event-time-and-late-data)

---

## Tomorrow: Day 46 - Stream-to-Stream Joins (Advanced)

Deep dive into advanced stream-to-stream join patterns and optimizations.
