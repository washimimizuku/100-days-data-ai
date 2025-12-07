# Day 46: Stream-to-Stream Joins (Advanced)

## 📖 Learning Objectives

By the end of this session, you will:
- Master advanced stream-to-stream join patterns
- Implement all join types (inner, left, right, full outer)
- Optimize join performance
- Handle unmatched records effectively
- Implement complex join conditions
- Monitor and debug join operations

**Time**: 1 hour

---

## Stream-to-Stream Join Fundamentals

### Why Stream-to-Stream Joins?

Real-world scenarios require joining multiple event streams:
- Orders + Payments
- Clicks + Impressions
- Sensor readings + Alerts
- User actions + System events

### Key Challenges

1. **Out-of-order arrival**: Events from different streams arrive at different times
2. **State management**: Must buffer events waiting for matches
3. **Memory constraints**: Can't keep all events forever
4. **Time synchronization**: Events from different sources have different timestamps

---

## Join Types

### Join Type Examples

```python
orders = orders_stream.withWatermark("order_time", "10 minutes")
payments = payments_stream.withWatermark("payment_time", "10 minutes")

time_constraint = expr("""
    orders.order_id = payments.order_id AND
    payments.payment_time >= orders.order_time AND
    payments.payment_time <= orders.order_time + interval 15 minutes
""")

# Inner: Only matched records
orders.join(payments, time_constraint, "inner")

# Left Outer: All orders + matched payments (find unpaid orders)
orders.join(payments, time_constraint, "left_outer")

# Right Outer: All payments + matched orders (detect orphaned payments)
orders.join(payments, time_constraint, "right_outer")

# Full Outer: All records from both sides (complete reconciliation)
orders.join(payments, time_constraint, "full_outer")
```

---

## Time Constraints

### Time Constraint Patterns

```python
# BAD: No time constraint (infinite state growth)
orders.join(payments, "order_id")  # ❌

# GOOD: Time constraints limit state
# Future-only: Payment after order
expr("orders.order_id = payments.order_id AND payments.payment_time >= orders.order_time")

# Bounded window: Payment within 15 minutes
expr("... AND payments.payment_time <= orders.order_time + interval 15 minutes")

# Symmetric: Events within 5 minutes of each other
expr("... AND timestamp >= base - interval 5 minutes AND timestamp <= base + interval 5 minutes")
```

---

## Watermarks in Joins

### Watermark Requirements

| Join Type | Watermark Required? |
|-----------|---------------------|
| Inner | Optional (recommended) |
| Left Outer | Required on both |
| Right Outer | Required on both |
| Full Outer | Required on both |

### Watermarks Enable State Cleanup

```python
# Join watermark = min(left_watermark, right_watermark)
orders.withWatermark("order_time", "10 minutes")  # 10 min
payments.withWatermark("payment_time", "5 minutes")  # 5 min
# Effective watermark: 5 minutes

# Without watermark: Memory leak
orders.join(payments, "order_id")  # ❌

# With watermark: Bounded state
orders.withWatermark("order_time", "10 minutes") \
    .join(payments.withWatermark("payment_time", "10 minutes"), expr("..."))  # ✅
```

---

## Advanced Join Patterns

### Advanced Join Patterns

```python
# Multi-key join
stream1.join(stream2, expr("s1.user_id = s2.user_id AND s1.session_id = s2.session_id AND ..."))

# Conditional join (with additional conditions)
orders.join(payments, expr("orders.order_id = payments.order_id AND payments.amount >= orders.amount AND ..."))

# Self-join (detect duplicates)
events.alias("e1").join(events.alias("e2"), expr("e1.user_id = e2.user_id AND e1.event_id != e2.event_id AND ..."))

# Chain of joins
orders.join(payments, ...).join(shipments, ...)
```

---

## Handling Unmatched Records

### Detecting Unmatched Records

```python
from pyspark.sql.functions import col, when, lit

# Left outer join
left_joined = orders.join(payments, ..., "left_outer")

# Identify unmatched orders
unmatched = left_joined.filter(col("payment_id").isNull())

# Tag matched vs unmatched
tagged = left_joined.withColumn(
    "status",
    when(col("payment_id").isNull(), "unpaid").otherwise("paid")
)
```

### Separate Processing

```python
# Split into matched and unmatched
matched = left_joined.filter(col("payment_id").isNotNull())
unmatched = left_joined.filter(col("payment_id").isNull())

# Process separately
matched_query = matched.writeStream.format("kafka").option("topic", "matched").start()
unmatched_query = unmatched.writeStream.format("kafka").option("topic", "unmatched").start()
```

### Retry Logic

```python
# Write unmatched to retry topic
unmatched.writeStream \
    .format("kafka") \
    .option("topic", "retry_queue") \
    .start()

# Separate process retries matching
```

---

## Performance Optimization

### 1. Appropriate Watermarks

```python
# Too short: Drops valid data
orders.withWatermark("order_time", "1 minute")  # ❌

# Too long: Excessive state
orders.withWatermark("order_time", "1 day")  # ❌

# Just right: Based on actual latency
orders.withWatermark("order_time", "10 minutes")  # ✅
```

### 2. Narrow Time Windows

```python
# Wide window: More state
expr("... AND timestamp <= base + interval 1 hour")  # ❌

# Narrow window: Less state
expr("... AND timestamp <= base + interval 10 minutes")  # ✅
```

### 3. Partition Alignment

```python
# Partition both streams by join key
orders = orders_stream.repartition("order_id")
payments = payments_stream.repartition("order_id")

joined = orders.join(payments, ...)
```

### 4. State Monitoring

```python
query = joined.writeStream.format("console").start()

# Monitor state size
progress = query.lastProgress
if progress and 'stateOperators' in progress:
    for op in progress['stateOperators']:
        print(f"State rows: {op.get('numRowsTotal', 0)}")
        print(f"State memory: {op.get('memoryUsedBytes', 0)}")
```

---

## Debugging Join Issues

### Common Issues

**1. No output from join**
- Check watermarks are set
- Verify time constraints are correct
- Ensure data is arriving in both streams

**2. Excessive state growth**
- Add/adjust watermarks
- Narrow time constraints
- Check for data skew

**3. Missing matches**
- Watermark too aggressive
- Time constraint too narrow
- Clock skew between sources

### Debugging Techniques

```python
# 1. Check input streams separately
orders.writeStream.format("console").start()
payments.writeStream.format("console").start()

# 2. Monitor watermarks
query = joined.writeStream.format("console").start()
print(query.lastProgress['sources'])

# 3. Add debug columns
debug_joined = orders.join(payments, ...).withColumn(
    "join_latency",
    unix_timestamp(col("payment_time")) - unix_timestamp(col("order_time"))
)
```

---

## 💻 Exercises

### Exercise 1: Inner Join with Time Constraint
Join orders and payments with 15-minute window.

### Exercise 2: Left Outer Join
Find unpaid orders using left outer join.

### Exercise 3: Full Outer Join
Complete reconciliation with full outer join.

### Exercise 4: Multi-Stream Join
Join three streams: orders, payments, shipments.

### Exercise 5: Self-Join for Duplicates
Detect duplicate events within 1-minute window.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Stream-to-stream joins require time constraints to limit state
- Outer joins require watermarks on both streams
- Join watermark = min(left_watermark, right_watermark)
- Use narrow time windows for better performance
- Monitor state size to detect issues
- Handle unmatched records explicitly
- Always test with realistic data latency

---

## 📚 Resources

- [Stream-Stream Joins](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#stream-stream-joins)
- [Join Operations](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#join-operations)

---

## Tomorrow: Day 47 - Stateful Stream Processing

Learn about custom stateful operations and advanced state management.
