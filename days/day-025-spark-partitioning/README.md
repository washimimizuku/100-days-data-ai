# Day 25: Spark Partitioning

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand partitioning concepts
- Use repartition vs coalesce
- Handle data skew
- Optimize partition sizes
- Implement custom partitioners

---

## Theory

### What is Partitioning?

**Partition**: A logical chunk of data distributed across executors

```
DataFrame (1M rows)
├── Partition 1 (250K rows) → Executor 1
├── Partition 2 (250K rows) → Executor 2
├── Partition 3 (250K rows) → Executor 3
└── Partition 4 (250K rows) → Executor 4
```

**Key Points**:
- Each partition processed by one task
- More partitions = more parallelism
- Too many = overhead, too few = underutilization

---

### Default Partitioning

```python
# Reading files
df = spark.read.csv("data.csv")  # Partitions = number of input files

# Creating DataFrames
df = spark.range(1000000)  # Default: spark.default.parallelism (usually 200)

# Check partitions
print(df.rdd.getNumPartitions())
```

**Configuration**:
```python
spark.conf.set("spark.default.parallelism", "100")
spark.conf.set("spark.sql.shuffle.partitions", "200")  # For shuffles
```

---

### Repartition vs Coalesce

#### repartition()

**Purpose**: Increase or decrease partitions with full shuffle

```python
# Increase partitions (for more parallelism)
df = df.repartition(100)

# Partition by column (for better locality)
df = df.repartition("city")
df = df.repartition(50, "city")  # 50 partitions by city
```

**Characteristics**:
- Full shuffle (expensive)
- Can increase or decrease partitions
- Evenly distributes data

#### coalesce()

**Purpose**: Decrease partitions without full shuffle

```python
# Reduce partitions (for fewer output files)
df = df.coalesce(10)
df = df.coalesce(1)  # Single partition (single output file)
```

**Characteristics**:
- No full shuffle (cheaper)
- Only decreases partitions
- May create uneven partitions

**When to use**:
- `repartition()`: Need more partitions or even distribution
- `coalesce()`: Reduce partitions before writing

---

### Partition Size Guidelines

**Optimal size**: 100-200 MB per partition

```python
# Calculate partitions
data_size_gb = 100
partition_size_mb = 128
num_partitions = (data_size_gb * 1024) / partition_size_mb
# = 800 partitions

df = df.repartition(800)
```

**Too many partitions**:
- High task scheduling overhead
- Many small files on write

**Too few partitions**:
- Underutilized cluster
- Memory pressure
- Slow processing

---

### Data Skew

**Problem**: Uneven data distribution across partitions

```python
# Skewed data example
# Partition 1: 1M rows (NYC)
# Partition 2: 100 rows (Boston)
# Partition 3: 50 rows (Seattle)
# → Partition 1 becomes bottleneck
```

**Detection**:
```python
# Check partition sizes
df.rdd.glom().map(len).collect()  # [1000000, 100, 50]

# In Spark UI: Look for tasks with much longer duration
```

**Solutions**:

1. **Salt keys** (add randomness)
```python
from pyspark.sql.functions import rand, concat

# Add salt to skewed key
df = df.withColumn("salted_key", concat(col("city"), lit("_"), (rand() * 10).cast("int")))
df = df.repartition("salted_key")
```

2. **Broadcast small table**
```python
# If one side is small
df_large.join(broadcast(df_small), "key")
```

3. **Increase partitions**
```python
df = df.repartition(1000)  # More partitions = smaller chunks
```

---

### Partition Pruning

**Concept**: Skip reading unnecessary partitions

```python
# Write with partitioning
df.write.partitionBy("year", "month").parquet("output")

# Creates structure:
# output/
#   year=2024/
#     month=01/
#     month=02/
#   year=2023/
#     month=12/

# Read with filter (only reads relevant partitions)
df = spark.read.parquet("output")
df = df.filter(col("year") == 2024)  # Only reads year=2024 partitions
```

**Benefits**:
- Faster reads
- Less data scanned
- Lower costs (cloud storage)

---

### Custom Partitioners

```python
# Partition by column
df.repartition("city")

# Partition by expression
df.repartition(col("city"), col("state"))

# Partition by range
df.repartitionByRange("age")  # Sorts and partitions by range
```

---

### Partition Operations

#### mapPartitions

**Process entire partition at once**:
```python
def process_partition(iterator):
    # Setup (once per partition)
    connection = create_connection()
    
    # Process rows
    for row in iterator:
        yield process_row(row, connection)
    
    # Cleanup
    connection.close()

df.rdd.mapPartitions(process_partition)
```

**Benefits**:
- Amortize setup cost
- Batch processing
- Connection pooling

#### foreachPartition

**Side effects per partition**:
```python
def write_partition(iterator):
    connection = create_connection()
    for row in iterator:
        connection.write(row)
    connection.close()

df.foreachPartition(write_partition)
```

---

### Shuffle Partitions

**Configuration**:
```python
# Default: 200 shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "100")

# Adaptive Query Execution (Spark 3.0+)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

**AQE Benefits**:
- Automatically adjusts partition count
- Combines small partitions
- Handles skew

---

### Best Practices

1. **Right-size partitions**
```python
# Target: 100-200 MB per partition
num_partitions = data_size_gb * 1024 / 128
```

2. **Partition by frequently filtered columns**
```python
df.write.partitionBy("date").parquet("output")
```

3. **Coalesce before writing**
```python
df.coalesce(10).write.parquet("output")  # 10 output files
```

4. **Avoid too many small files**
```python
# Bad: 10,000 small files
df.write.parquet("output")

# Good: Coalesce first
df.coalesce(100).write.parquet("output")
```

5. **Monitor in Spark UI**
- Check task duration distribution
- Look for stragglers (slow tasks)
- Verify even data distribution

---

### Example: Optimizing Partitions

```python
# Initial state
df = spark.read.csv("large_data.csv")  # 1000 partitions, 10 GB
print(f"Partitions: {df.rdd.getNumPartitions()}")

# Problem: Too many small partitions
# Solution: Repartition to optimal size
optimal_partitions = (10 * 1024) / 128  # ~80 partitions
df = df.repartition(80)

# Process data
df = df.filter(col("amount") > 100) \
       .groupBy("city") \
       .agg(sum("amount"))

# Before writing: Reduce partitions
df = df.coalesce(10)

# Write
df.write.parquet("output")  # 10 output files
```

---

## 💻 Exercises (40 min)

### Exercise 1: Partition Analysis
- Check default partitions
- Calculate optimal partition count
- Repartition DataFrame

### Exercise 2: Repartition vs Coalesce
- Compare performance
- Test with different sizes
- Measure shuffle impact

### Exercise 3: Data Skew
- Create skewed data
- Detect skew
- Apply salting solution

### Exercise 4: Partition Pruning
- Write partitioned data
- Read with filters
- Verify partition pruning

### Exercise 5: Optimization
- Optimize end-to-end pipeline
- Right-size partitions
- Minimize output files

---

## ✅ Quiz (5 min)

1. What is a partition?
2. Repartition vs coalesce?
3. What is data skew?
4. How to detect skew?
5. What is partition pruning?
6. Optimal partition size?
7. When to use repartition?
8. What is AQE?

---

## 🎯 Key Takeaways

- **Partition** - Logical chunk of data
- **Repartition** - Full shuffle, increase/decrease
- **Coalesce** - No shuffle, decrease only
- **Optimal size** - 100-200 MB per partition
- **Data skew** - Uneven distribution, use salting
- **Partition pruning** - Skip unnecessary partitions
- **AQE** - Adaptive query execution (Spark 3.0+)
- **Monitor** - Use Spark UI to verify distribution

---

## 📚 Resources

- [Spark Partitioning](https://spark.apache.org/docs/latest/rdd-programming-guide.html#partitions)
- [AQE Guide](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution)

---

## Tomorrow: Day 26 - Spark Performance Tuning

We'll explore advanced optimization techniques.
