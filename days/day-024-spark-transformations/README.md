# Day 24: Spark Transformations

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand narrow vs wide transformations
- Differentiate transformations from actions
- Master common transformations (map, filter, join)
- Optimize transformation chains
- Understand shuffle operations

---

## Theory

### Transformations vs Actions

**Transformations**: Lazy operations that define a new DataFrame
**Actions**: Eager operations that trigger execution

```python
# Transformations (lazy - not executed)
df = df.filter(col("age") > 25)    # Transformation
df = df.select("name", "age")      # Transformation
df = df.groupBy("city").count()    # Transformation

# Action (triggers execution of all above)
df.show()  # Action - NOW everything executes
```

---

### Narrow vs Wide Transformations

#### Narrow Transformations

**Definition**: Each input partition contributes to only one output partition (no shuffle)

**Examples**:
- `filter()`, `select()`, `map()`, `withColumn()`
- `union()`, `drop()`, `cast()`

```python
# Narrow - no data movement between partitions
df.filter(col("age") > 25)        # Each partition filtered independently
df.select("name", "age")          # Each partition processed independently
df.withColumn("age2", col("age") * 2)  # No shuffle needed
```

**Characteristics**:
- Fast (no network I/O)
- No shuffle
- Pipelined execution

#### Wide Transformations

**Definition**: Input partitions contribute to multiple output partitions (requires shuffle)

**Examples**:
- `groupBy()`, `join()`, `distinct()`
- `repartition()`, `sortBy()`, `reduceByKey()`

```python
# Wide - requires shuffle
df.groupBy("city").count()        # Data must be shuffled by city
df.join(df2, "id")                # Data shuffled to match join keys
df.distinct()                     # All data compared across partitions
df.repartition(10)                # Data redistributed
```

**Characteristics**:
- Slower (network I/O)
- Requires shuffle
- Creates stage boundaries

---

### Common Transformations

#### map / select

```python
# map (RDD-style)
rdd.map(lambda x: x * 2)

# select (DataFrame-style)
df.select(col("age") * 2)
df.select("name", "age")
```

#### filter / where

```python
df.filter(col("age") > 25)
df.where((col("age") > 25) & (col("city") == "NYC"))
```

#### flatMap

```python
# Explode one row into multiple
rdd.flatMap(lambda x: x.split(" "))

# DataFrame equivalent
df.select(explode(split(col("text"), " ")))
```

#### distinct

```python
df.distinct()  # Remove duplicates (wide - requires shuffle)
df.dropDuplicates(["id"])  # Remove duplicates by column
```

#### union

```python
df1.union(df2)  # Combine DataFrames (narrow - no shuffle)
```

---

### Join Transformations

```python
# Inner join (wide)
df1.join(df2, "id")
df1.join(df2, df1.id == df2.id, "inner")

# Left join
df1.join(df2, "id", "left")

# Broadcast join (optimized for small tables)
from pyspark.sql.functions import broadcast
df1.join(broadcast(df2), "id")
```

**Join Types**:
- `inner`, `left`, `right`, `outer`
- `left_semi`, `left_anti`, `cross`

---

### Aggregation Transformations

```python
# groupBy (wide - requires shuffle)
df.groupBy("city").count()
df.groupBy("city").agg(
    sum("amount").alias("total"),
    avg("amount").alias("average"),
    max("amount").alias("max_amount")
)

# Multiple grouping columns
df.groupBy("city", "category").count()
```

---

### Sorting Transformations

```python
# orderBy / sort (wide - requires shuffle)
df.orderBy("age")
df.orderBy(col("age").desc())
df.orderBy("city", desc("age"))

# sortWithinPartitions (narrow - sorts within each partition)
df.sortWithinPartitions("age")
```

---

### Repartitioning

```python
# repartition (wide - full shuffle)
df = df.repartition(10)  # Increase partitions
df = df.repartition("city")  # Partition by column

# coalesce (narrow - reduce partitions without full shuffle)
df = df.coalesce(1)  # Reduce to 1 partition (for single output file)
```

---

### Actions (Trigger Execution)

```python
# Collect results
df.show()           # Display rows
df.count()          # Count rows
df.collect()        # Return all rows to driver (careful!)
df.take(5)          # Return first 5 rows
df.first()          # Return first row

# Write data
df.write.parquet("output")
df.write.csv("output")

# Aggregate actions
df.agg(sum("amount")).collect()
```

---

### Transformation Chaining

```python
# Chain transformations (all lazy until action)
result = df \
    .filter(col("age") > 18) \
    .select("name", "age", "city") \
    .groupBy("city") \
    .agg(avg("age").alias("avg_age")) \
    .orderBy(desc("avg_age")) \
    .limit(10)

# Nothing executed yet!

# Trigger execution
result.show()  # NOW everything executes
```

---

### Shuffle Operations

**What is Shuffle?**
- Redistribution of data across partitions
- Expensive (disk I/O, network I/O, serialization)
- Creates stage boundaries

**Operations that cause shuffle**:
- `groupBy`, `join`, `distinct`
- `repartition`, `sortBy`
- `reduceByKey`, `aggregateByKey`

**Minimize shuffles**:
```python
# Bad - multiple shuffles
df.groupBy("city").count() \
  .join(df2, "city") \
  .groupBy("region").sum()

# Better - reduce shuffles
df.join(df2, "city") \
  .groupBy("city", "region") \
  .agg(count("*"), sum("amount"))
```

---

### Optimization Tips

1. **Filter early**
```python
# Good - filter before expensive operations
df.filter(col("date") == "2024-01-01") \
  .join(df2, "id") \
  .groupBy("city").count()
```

2. **Use broadcast for small tables**
```python
df_large.join(broadcast(df_small), "id")
```

3. **Avoid collect() on large data**
```python
# Bad - brings all data to driver
data = df.collect()  # OOM risk!

# Good - process in distributed manner
df.write.parquet("output")
```

4. **Cache intermediate results**
```python
df_filtered = df.filter(col("age") > 25).cache()
df_filtered.count()  # Materialize cache
# Reuse df_filtered multiple times
```

5. **Coalesce before writing**
```python
df.coalesce(1).write.csv("output")  # Single output file
```

---

### Execution Plan

```python
# View logical plan
df.explain(True)

# View physical plan
df.explain()

# Example output:
# == Physical Plan ==
# *(2) HashAggregate
# +- Exchange hashpartitioning  <- SHUFFLE!
#    +- *(1) HashAggregate
#       +- *(1) Filter
```

---

## 💻 Exercises (40 min)

### Exercise 1: Identify Transformations
- List narrow transformations
- List wide transformations
- Explain shuffle impact

### Exercise 2: Transformation Chains
- Build multi-step pipeline
- Optimize transformation order
- Measure performance

### Exercise 3: Join Optimization
- Compare regular vs broadcast join
- Test different join types
- Analyze execution plans

### Exercise 4: Shuffle Analysis
- Identify shuffle operations
- Count shuffles in pipeline
- Minimize shuffles

### Exercise 5: Performance Tuning
- Cache intermediate results
- Filter early
- Compare execution times

---

## ✅ Quiz (5 min)

1. What is a transformation?
2. What is an action?
3. What is a narrow transformation?
4. What is a wide transformation?
5. What causes a shuffle?
6. How to optimize joins?
7. When to use cache?
8. What does explain() show?

---

## 🎯 Key Takeaways

- **Transformations** - Lazy, define new DataFrame
- **Actions** - Eager, trigger execution
- **Narrow** - No shuffle, fast
- **Wide** - Shuffle required, slower
- **Shuffle** - Expensive, minimize when possible
- **Filter early** - Reduce data before expensive ops
- **Broadcast** - Optimize small table joins
- **Cache** - Reuse intermediate results

---

## 📚 Resources

- [Spark Transformations](https://spark.apache.org/docs/latest/rdd-programming-guide.html#transformations)
- [Spark Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)

---

## Tomorrow: Day 25 - Spark Partitioning

We'll explore partitioning strategies and optimization.
