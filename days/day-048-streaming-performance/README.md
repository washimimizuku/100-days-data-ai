# Day 48: Streaming Performance Optimization

## 📖 Learning Objectives

By the end of this session, you will:
- Optimize streaming query performance
- Tune resource allocation effectively
- Reduce latency and increase throughput
- Monitor and debug performance issues
- Understand Spark vs Flink trade-offs
- Apply best practices for production

**Time**: 1 hour

---

## Performance Fundamentals

**Key Metrics**: Throughput (events/sec), Latency (arrival to output), State Size (memory), Resource Usage (CPU/memory/network)

**Trade-offs**: High throughput → More batching → Higher latency | Low latency → Smaller batches → Lower throughput | More state → Better accuracy → More memory

---

## Trigger Optimization

```python
# Trigger types
.trigger(processingTime="0 seconds")  # Default micro-batch (low latency, high overhead)
.trigger(processingTime="10 seconds")  # Fixed interval (balanced)
.trigger(once=True)  # One-time (testing/batch-like)
.trigger(continuous="1 second")  # Continuous (experimental, ultra-low latency)

# Choosing interval
.trigger(processingTime="100 milliseconds")  # ❌ Too frequent (high overhead)
.trigger(processingTime="5 minutes")  # ❌ Too infrequent (high latency)
.trigger(processingTime="5 seconds")  # ✅ Balanced
```

---

## Partitioning Optimization

```python
# Repartition by key for better distribution
df.repartition(100, "user_id")  # Distribute by key
df.coalesce(50)  # Reduce partitions

# Partition count: Rule of thumb = 2-3x number of cores
df.repartition(2)  # ❌ Too few (underutilized)
df.repartition(10000)  # ❌ Too many (high overhead)
df.repartition(num_cores * 2)  # ✅ 64 partitions for 32 cores

# Skew handling: Add salt to skewed keys
df_salted = df.withColumn("salted_key", concat(col("user_id"), lit("_"), (rand() * 10).cast("int")))
result = df_salted.groupBy("salted_key").agg(...)
```

---

## State Management

```python
# Minimize state size - keep only recent data
def update_state_bounded(key, values, state):
    history = state.get if state.exists else []
    for value in values:
        history.append(value)
    if len(history) > 1000:  # ✅ Bounded
        history = history[-1000:]
    state.update(history)
    return result

# Aggressive watermarks for smaller state
df.withWatermark("timestamp", "1 hour")  # ❌ Large state
df.withWatermark("timestamp", "5 minutes")  # ✅ Smaller state

# Always set timeouts and cleanup
state.setTimeoutDuration("10 minutes")
if state.hasTimedOut:
    state.remove()
```

---

## Resource Allocation

```python
# Executor configuration
spark = SparkSession.builder.appName("StreamingApp") \
    .config("spark.executor.instances", "10") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.default.parallelism", "100") \
    .getOrCreate()
```

---

## Checkpoint Optimization

```python
# Checkpoint location
.option("checkpointLocation", "file:///tmp/checkpoint")  # ❌ Local (testing only)
.option("checkpointLocation", "s3://bucket/checkpoint")  # ✅ HDFS/S3 (production)

# Checkpoint interval
.trigger(processingTime="1 second")  # ❌ Too frequent (high I/O)
.trigger(processingTime="10 minutes")  # ❌ Too infrequent (data loss risk)
.trigger(processingTime="30 seconds")  # ✅ Balanced
```

---

## Query Optimization

```python
# Select early (projection pushdown)
df = spark.readStream.format("kafka").load().select("key", "value")  # ✅ Only needed columns

# Filter early (predicate pushdown)
df.filter(col("value") > 100)  # ✅ Reduce data volume

# Avoid expensive operations
df.groupBy("key").agg(collect_list("value"))  # ❌ High memory
df.groupBy("key").agg(count("*"), sum("value"))  # ✅ Low memory
```

---

## Monitoring and Debugging

```python
# Query progress metrics
progress = query.lastProgress
print(f"Input rows: {progress['numInputRows']}, Rate: {progress['processedRowsPerSecond']}")
for op in progress['stateOperators']:
    print(f"State rows: {op['numRowsTotal']}, Memory: {op['memoryUsedBytes']}")

# Spark UI (http://localhost:4040): Streaming, Stages, Storage, Executors tabs

# Custom logging
spark.sparkContext.setLogLevel("INFO")
logger = logging.getLogger(__name__)
logger.info(f"Batch {batch_id}: {count} rows")
```

---

## Spark vs Flink: When to Use What

| Feature | Spark Streaming | Flink |
|---------|----------------|-------|
| **Latency** | Seconds | Milliseconds |
| **Throughput** | High | Very High |
| **API** | Unified batch/stream | Separate APIs |
| **State Management** | Good | Excellent |
| **Ecosystem** | Rich (MLlib, GraphX) | Growing |
| **Learning Curve** | Moderate | Steep |
| **Use Case** | Analytics | Event processing |

**Choose Spark if**: Already using Spark, need batch + streaming, complex analytics, latency > 1s OK

**Choose Flink if**: Need < 1s latency, simple transformations, high throughput critical, team has expertise

**For most data engineering teams: Start with Spark Streaming**

---

## Best Practices

1. **Start Simple**: Begin with basic query, add complexity gradually (watermarks, stateful ops, joins)
2. **Monitor Everything**: Track processing rate, batch duration, state size, resource usage
3. **Test with Production Data**: Use realistic data volume, distribution, and latency patterns
4. **Plan for Failures**: Enable checkpointing, ensure idempotent outputs, set up monitoring/alerting
5. **Optimize Iteratively**: Measure baseline → Identify bottleneck → Apply optimization → Measure improvement → Repeat

---

## 💻 Exercises

### Exercise 1: Trigger Tuning
Compare different trigger intervals and measure impact.

### Exercise 2: Partition Optimization
Optimize partition count for your workload.

### Exercise 3: State Size Reduction
Implement bounded state with cleanup.

### Exercise 4: Resource Allocation
Tune executor configuration for performance.

### Exercise 5: End-to-End Optimization
Apply all optimizations to a complete pipeline.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Balance throughput, latency, and resources
- Optimize triggers, partitions, and state
- Monitor continuously in production
- Start with Spark Streaming for most use cases
- Consider Flink for ultra-low latency needs
- Apply optimizations iteratively
- Test with realistic production data

---

## 📚 Resources

- [Spark Streaming Tuning](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#performance-tuning)
- [Flink Documentation](https://flink.apache.org/docs/stable/)
- [Spark vs Flink Comparison](https://www.ververica.com/blog/apache-flink-vs-apache-spark)

---

## Tomorrow: Day 49 - Mini Project: Real-time Analytics (Spark + Kafka)

Build a complete real-time analytics pipeline combining everything learned this week.
