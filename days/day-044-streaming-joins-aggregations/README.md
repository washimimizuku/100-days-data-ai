# Day 44: Streaming Joins and Aggregations

## 📖 Learning Objectives

By the end of this session, you will:
- Perform stream-to-static joins
- Implement stream-to-stream joins
- Use windowed aggregations effectively
- Handle multiple aggregation types
- Optimize join performance
- Understand join limitations and best practices

**Time**: 1 hour

---

## Streaming Joins Overview

Spark Structured Streaming supports two types of joins:
1. **Stream-to-Static**: Join streaming data with static DataFrame
2. **Stream-to-Stream**: Join two streaming DataFrames

### Join Types Supported

| Join Type | Stream-to-Static | Stream-to-Stream |
|-----------|------------------|------------------|
| Inner     | ✅ Yes           | ✅ Yes           |
| Left Outer| ✅ Yes           | ✅ Yes (with watermark) |
| Right Outer| ✅ Yes          | ✅ Yes (with watermark) |
| Full Outer| ❌ No            | ✅ Yes (with watermark) |

---

## Stream-to-Static Joins

Join streaming data with static reference data (dimension tables).

### Example: Stream-to-Static Join

```python
# Read streaming events
events = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events").load()

# Parse JSON
parsed = events.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

# Read static products
products = spark.read.format("csv").option("header", True).load("data/products.csv")

# Join (broadcast small table)
enriched = parsed.join(broadcast(products), "product_id")

# Write
enriched.writeStream.format("console").start()
```

**Use Cases**: Enrichment, filtering, validation, lookup tables

**Performance**: Use `broadcast()` for small tables, cache static data

---

## Stream-to-Stream Joins

Join two streaming DataFrames based on keys and time windows.

### Examples: Stream-to-Stream Joins

```python
# Read streams
orders = spark.readStream.format("kafka").option("subscribe", "orders").load()
payments = spark.readStream.format("kafka").option("subscribe", "payments").load()

# Inner join
joined = orders.join(payments, orders.order_id == payments.order_id, "inner")

# Time-bounded join (limit state)
joined = orders.join(payments, expr("""
    orders.order_id = payments.order_id AND
    payments.timestamp >= orders.timestamp AND
    payments.timestamp <= orders.timestamp + interval 10 minutes
"""))

# Outer join with watermark
orders_wm = orders.withWatermark("timestamp", "10 minutes")
payments_wm = payments.withWatermark("timestamp", "10 minutes")
left_joined = orders_wm.join(payments_wm, expr("..."), "left_outer")
```

**Key**: Use time constraints and watermarks to prevent unbounded state growth

---

## Windowed Aggregations

### Window Types

```python
# Tumbling (fixed-size, non-overlapping)
events.groupBy(window("timestamp", "10 minutes"), "product_id") \
    .agg(count("*"), sum("amount"), avg("amount"))

# Sliding (overlapping)
events.groupBy(window("timestamp", "10 minutes", "5 minutes"), "product_id") \
    .agg(count("*"))

# Session (dynamic, based on inactivity gaps)
events.groupBy(session_window("timestamp", "5 minutes"), "user_id") \
    .agg(count("*"))
```

---

## Multiple Aggregations

### Aggregation Patterns

```python
# Multiple columns
events.groupBy(window("timestamp", "10 minutes"), "product_id", "region") \
    .agg(count("*"), sum("amount"), avg("amount"))

# Multiple functions
events.groupBy(window("timestamp", "10 minutes"), "product_id") \
    .agg(count("*"), sum("amount"), avg("amount"), min("amount"), max("amount"), stddev("amount"))

# Conditional aggregations
events.groupBy(window("timestamp", "10 minutes")) \
    .agg(
        sum(when(col("amount") > 100, 1).otherwise(0)).alias("high_value"),
        sum(when(col("amount") <= 100, 1).otherwise(0)).alias("low_value")
    )
```

---

## Advanced Patterns

### Advanced Patterns

```python
# Chained aggregations
user_stats = events.groupBy(window("timestamp", "10 minutes"), "user_id") \
    .agg(count("*"), sum("amount"))
summary = user_stats.groupBy("window").agg(count("*"), avg("event_count"))

# Join after aggregation
order_stats = orders.groupBy(window("timestamp", "10 minutes"), "product_id").agg(count("*"))
enriched_stats = order_stats.join(broadcast(products), "product_id")

# Self-join (detect duplicates within 1 minute)
duplicates = events.alias("e1").join(events.alias("e2"), expr("""
    e1.user_id = e2.user_id AND e1.event_id != e2.event_id AND
    e2.timestamp >= e1.timestamp AND e2.timestamp <= e1.timestamp + interval 1 minute
"""))
```

---

## Performance Optimization

**1. Broadcast Small Tables**: Use `broadcast()` for static dimension tables

**2. Partition Appropriately**: Repartition streams by join key before joining

**3. Use Watermarks**: Always use watermarks for stream-to-stream joins to limit state

**4. Choose Window Sizes**: Balance latency (smaller windows) vs state (larger windows)

---

## 💻 Exercises

### Exercise 1: Stream-to-Static Enrichment
Join streaming orders with static product catalog.

### Exercise 2: Stream-to-Stream Inner Join
Join order and payment streams.

### Exercise 3: Windowed Aggregations
Calculate metrics per 5-minute window.

### Exercise 4: Multiple Aggregations
Compute various statistics by product and region.

### Exercise 5: Time-Bounded Join
Implement join with time constraints.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Stream-to-static joins enrich streaming data with reference tables
- Stream-to-stream joins require time constraints and watermarks
- Windowed aggregations enable time-based analytics
- Use broadcast for small static tables
- Watermarks prevent unbounded state growth
- Choose window sizes based on latency and state requirements

---

## 📚 Resources

- [Structured Streaming Joins](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#stream-stream-joins)
- [Window Operations](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#window-operations-on-event-time)
- [Watermarking](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#handling-late-data-and-watermarking)

---

## Tomorrow: Day 45 - Watermarking and Late Data Handling

Learn how to handle late-arriving data and manage state with watermarks.
