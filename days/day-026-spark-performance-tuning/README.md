# Day 26: Spark Performance Tuning

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Optimize Spark configurations for better performance
- Use Catalyst optimizer effectively
- Implement caching strategies for reused data
- Tune memory and parallelism settings
- Debug performance issues with Spark UI
- Minimize shuffles and optimize joins

---

## Theory

### What is Performance Tuning?

**Performance tuning** is the process of optimizing Spark applications to run faster, use less memory, and handle larger datasets efficiently.

**Why It Matters**:
- Spark jobs can be slow without proper tuning
- Poor configuration wastes cluster resources
- Inefficient code causes unnecessary shuffles
- Memory issues lead to job failures
- Proper tuning can improve performance 10-100x

**Common Performance Problems**:
- Jobs taking hours instead of minutes
- Out of memory errors
- Excessive shuffle operations
- Unbalanced task distribution (data skew)
- Too many small files

---

### Performance Tuning Areas

**7 Key Areas to Optimize**:

1. **Configuration** - Executor memory, cores, parallelism settings
2. **Caching** - Persist frequently used DataFrames in memory
3. **Partitioning** - Right-size partitions for optimal parallelism
4. **Shuffles** - Minimize expensive data movement across network
5. **Joins** - Choose optimal join strategy (broadcast, sort-merge, bucket)
6. **Serialization** - Use efficient data serialization (Kryo vs Java)
7. **Memory** - Balance execution memory vs storage memory

---

### 1. Configuration Tuning

```python
from pyspark.sql import SparkSession

# Optimal configuration
spark = SparkSession.builder \
    .appName("TunedApp") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.memory", "2g") \
    .config("spark.default.parallelism", "200") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()
```

**Guidelines**: 4-8 GB memory, 4-5 cores per executor, 2-3x cores for parallelism, 100-200 MB per partition

---

### 2. Caching Strategies

#### When to Cache

**Cache DataFrames that are**:
- Used multiple times in your code
- Expensive to recompute
- Result of complex transformations
- Intermediate results in iterative algorithms

**Don't cache if**:
- DataFrame is used only once
- Data is too large for memory
- Simple transformations (cheap to recompute)

```python
from pyspark.storagelevel import StorageLevel

# Cache DataFrame used multiple times
df_filtered = df.filter(col("age") > 25).cache()
df_filtered.count()  # Materialize cache

# Subsequent operations are fast
result1 = df_filtered.groupBy("city").count()
result2 = df_filtered.groupBy("gender").avg("salary")

# Storage levels
df.persist(StorageLevel.MEMORY_ONLY)  # Default, fastest
df.persist(StorageLevel.MEMORY_AND_DISK)  # Spill to disk if needed
df.persist(StorageLevel.MEMORY_ONLY_SER)  # Serialized, less memory

# Free memory when done
df_filtered.unpersist()
```

---

### 3. Join Optimization

**Broadcast Join**: For small table (<10 MB) + large table. Copies small table to all executors, no shuffle needed.

```python
from pyspark.sql.functions import broadcast

# Broadcast small table
result = df_large.join(broadcast(df_small), "id")

# Auto-broadcast threshold
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10 MB
```

**Sort-Merge Join**: Default for large tables. Sorts both tables, then merges. Requires shuffle.

---

### 4. Catalyst Optimizer

**Catalyst** is Spark's query optimizer that automatically improves your queries.

**Automatic Optimizations**: Predicate pushdown (filter early), column pruning (read only needed columns), constant folding, join reordering.

```python
# View execution plan
df.explain(True)  # Shows logical and physical plans
```

---

### 5. Serialization

**Kryo** is 10x faster and more compact than Java serialization.

```python
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

---

### 6. Shuffle Optimization

**Shuffle** redistributes data across executors (expensive: disk I/O + network transfer).

**Minimize Shuffles**:
```python
# ❌ Bad: Multiple shuffles
df.groupBy("city").count().groupBy("country").sum("count")

# ✅ Good: Single shuffle
df.groupBy("country", "city").count()

# ✅ Filter before shuffle
df.filter(col("valid") == True).groupBy("city").count()
```

**Tune Partitions**: 100-200 MB per partition. Small data: 50-100, Medium: 200-500, Large: 500-2000.

```python
spark.conf.set("spark.sql.shuffle.partitions", "200")
```

---

### 7. Memory Tuning

```python
# ❌ Bad: Collect large DataFrame
data = large_df.collect()  # OOM!

# ✅ Good: Keep distributed or limit
large_df.write.parquet("output")
sample = large_df.limit(1000).collect()
```

---

### 8. Adaptive Query Execution (AQE)

**AQE** (Spark 3.0+) dynamically optimizes at runtime: coalesces partitions, handles skewed joins, optimizes join strategies.

```python
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

---

### Common Performance Issues and Solutions

```python
# Issue 1: Too many small files
df.coalesce(10).write.parquet("output")

# Issue 2: Data skew - salt keys
df = df.withColumn("salt", (rand() * 10).cast("int"))
df.groupBy(concat(col("user_id"), col("salt"))).count()

# Issue 3: Excessive shuffles - combine operations
df.groupBy("city").agg(count("*"), sum("amount"), avg("price"))
```

---

### Monitoring with Spark UI

**Spark UI**: http://localhost:4040

**Key Tabs**: Jobs (overall progress), Stages (shuffle metrics), Storage (cache usage), Executors (resources)

**Watch For**: High shuffle read/write, GC time >10%, skewed task duration, disk spills

---

### Performance Tuning Checklist

**Essential Configurations**:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("OptimizedApp") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "200") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
    .getOrCreate()
```

**Before/After Example**:

```python
# ❌ Unoptimized (10 minutes)
df = spark.read.parquet("data.parquet")
df_filtered = df.filter(col("age") > 25)
df_grouped = df_filtered.groupBy("city").count()
df_joined = df_grouped.join(df_lookup, "city")
result = df_joined.collect()

# ✅ Optimized (1 minute)
df = spark.read.parquet("data.parquet")
df_filtered = df.filter(col("age") > 25).cache()  # Cache reused data
df_filtered.count()  # Materialize cache
df_grouped = df_filtered.groupBy("city").count()
df_joined = df_grouped.join(broadcast(df_lookup), "city")  # Broadcast small table
df_joined.write.parquet("output")  # Don't collect large data
```

---

### Best Practices Summary

**Configuration**:
- ✅ Set executor memory to 4-8 GB
- ✅ Use 4-5 cores per executor
- ✅ Set parallelism to 2-3x total cores
- ✅ Enable AQE for Spark 3.0+
- ✅ Use Kryo serialization

**Caching**:
- ✅ Cache DataFrames used multiple times
- ✅ Materialize cache with an action
- ✅ Unpersist when done
- ❌ Don't cache DataFrames used once

**Joins**:
- ✅ Broadcast small tables (<10 MB)
- ✅ Filter before joining
- ✅ Use appropriate join strategy
- ❌ Don't join without filtering first

**Shuffles**:
- ✅ Combine multiple aggregations
- ✅ Filter early to reduce data
- ✅ Tune shuffle partitions
- ❌ Don't create unnecessary shuffles

**Memory**:
- ✅ Keep data distributed
- ✅ Write to files instead of collect
- ✅ Limit data before collecting
- ❌ Don't collect large DataFrames

**Monitoring**:
- ✅ Check Spark UI regularly
- ✅ Watch for data skew
- ✅ Monitor GC time
- ✅ Identify shuffle bottlenecks

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Configuration Tuning
Configure Spark with optimal settings:
- Set executor memory to 4 GB
- Set executor cores to 4
- Enable AQE
- Enable Kryo serialization
- Set shuffle partitions based on data size

### Exercise 2: Caching Strategy
Implement caching for reused DataFrames:
- Read a large dataset
- Apply transformations
- Cache the result
- Use cached data multiple times
- Measure performance with/without cache
- Unpersist when done

### Exercise 3: Join Optimization
Optimize joins using broadcast:
- Create large and small DataFrames
- Join without broadcast (measure time)
- Join with broadcast (measure time)
- Compare performance
- Adjust broadcast threshold

### Exercise 4: Shuffle Reduction
Minimize shuffles in aggregations:
- Identify operations that cause shuffles
- Combine multiple aggregations into one
- Filter before grouping
- Measure shuffle read/write
- Compare optimized vs unoptimized

### Exercise 5: End-to-End Optimization
Optimize a complete pipeline:
- Read data
- Apply all tuning techniques
- Cache appropriately
- Broadcast small tables
- Minimize shuffles
- Measure total improvement
- Check Spark UI for bottlenecks

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`:

1. What is the Catalyst optimizer and what does it do?
2. When should you cache a DataFrame?
3. What is a broadcast join and when should you use it?
4. How can you reduce shuffles in Spark?
5. What is Adaptive Query Execution (AQE)?
6. What is the optimal number of cores per executor?
7. Why is Kryo serialization better than Java serialization?
8. How do you monitor Spark performance?
9. What causes data skew and how do you fix it?
10. What is the recommended partition size?

---

## 🎯 Key Takeaways

- **Configuration** - Tune memory, cores, parallelism
- **Caching** - Persist reused DataFrames
- **Broadcast** - Optimize small table joins
- **Partitioning** - 100-200 MB per partition
- **Shuffles** - Minimize data movement
- **AQE** - Enable for automatic optimization
- **Kryo** - Faster serialization
- **Spark UI** - Monitor and debug

---

## 📚 Resources

- [Spark Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)
- [Spark Configuration](https://spark.apache.org/docs/latest/configuration.html)

---

## Tomorrow: Day 27 - PySpark Exercises

We'll practice with real-world Spark scenarios.
