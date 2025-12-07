# Day 3: Apache Arrow

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand what Apache Arrow is
- Learn the difference between Arrow and Parquet
- Use Arrow for in-memory data processing
- Convert between Arrow, Pandas, and Parquet
- Understand zero-copy reads

---

## Theory

### What is Apache Arrow?

Apache Arrow is an **in-memory columnar data format** for fast analytics and data exchange.

**Key Difference:**
- **Parquet**: On-disk storage (files)
- **Arrow**: In-memory format (RAM)

**Think of it as:**
- Parquet = How data is stored
- Arrow = How data is processed

### Why Arrow Matters

**Problem**: Different systems use different memory formats
- Pandas, Spark, R all use different formats
- Converting between them is slow

**Solution**: Arrow provides a standard
- All systems use Arrow format
- Zero-copy data sharing
- No conversion overhead

### Arrow Features

1. **Columnar Memory** - Data organized by columns in RAM
2. **Zero-Copy** - No data copying between systems
3. **Language Agnostic** - Same format in Python, R, Java, C++
4. **Fast** - Optimized for modern CPUs

### Creating Arrow Tables

```python
import pyarrow as pa

# Method 1: From Python dict
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 28],
    'salary': [75000, 85000, 70000]
}
table = pa.table(data)

# Method 2: From Pandas
import pandas as pd
df = pd.DataFrame(data)
table = pa.Table.from_pandas(df)

# Method 3: Column by column
names = pa.array(['Alice', 'Bob', 'Charlie'])
ages = pa.array([25, 30, 28])
table = pa.table({'name': names, 'age': ages})
```

### Arrow Data Types

```python
import pyarrow as pa

# Numeric types
int_array = pa.array([1, 2, 3], type=pa.int64())
float_array = pa.array([1.1, 2.2], type=pa.float64())

# String type
string_array = pa.array(['a', 'b'], type=pa.string())

# Nested types
list_array = pa.array([[1, 2], [3, 4]], type=pa.list_(pa.int64()))
```

### Reading Parquet with Arrow

```python
import pyarrow.parquet as pq

# Read Parquet into Arrow Table (zero-copy)
table = pq.read_table('data.parquet')

# Much faster than pandas for large files
# Convert to Pandas when needed
df = table.to_pandas()

# Read specific columns only
table = pq.read_table('data.parquet', columns=['name', 'salary'])
```

### Arrow Table Operations

```python
import pyarrow as pa
import pyarrow.compute as pc

table = pa.table({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 28],
    'salary': [75000, 85000, 70000]
})

# Select columns
subset = table.select(['name', 'age'])

# Filter rows
filtered = table.filter(pc.greater(table['age'], 28))

# Sort
sorted_table = table.sort_by([('salary', 'descending')])

# Compute statistics
mean_salary = pc.mean(table['salary'])
print(f"Mean salary: {mean_salary.as_py()}")
```

### Zero-Copy Example

```python
import pyarrow as pa
import pandas as pd

# Create large dataset
df = pd.DataFrame({
    'id': range(1000000),
    'value': range(1000000)
})

# Convert to Arrow (minimal copy)
table = pa.Table.from_pandas(df)

# Access column (zero-copy - no data copied!)
id_column = table['id']

# Convert back to Pandas (zero-copy when possible)
df2 = table.to_pandas(zero_copy_only=True)
```

### Arrow vs Pandas

```python
import pyarrow as pa
import pandas as pd
import sys

# Create data
data = {'col': list(range(1000000))}

# Pandas
df = pd.DataFrame(data)
pandas_memory = df.memory_usage(deep=True).sum()

# Arrow
table = pa.table(data)
arrow_memory = table.nbytes

print(f"Pandas: {pandas_memory / 1024 / 1024:.2f} MB")
print(f"Arrow: {arrow_memory / 1024 / 1024:.2f} MB")
# Arrow is typically more memory efficient
```

### Real-World: ETL Pipeline

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Extract: Read from Parquet (zero-copy)
table = pq.read_table('raw_data.parquet')

# Transform: Use Arrow compute
table = table.filter(pc.greater(table['amount'], 0))

# Add computed column
table = table.append_column(
    'amount_usd',
    pc.multiply(table['amount'], 1.1)
)

# Load: Write to Parquet
pq.write_table(table, 'processed_data.parquet')
```

### Arrow + Multiple Systems

```python
# Read with Arrow
table = pq.read_table('data.parquet')

# Process with Arrow compute
import pyarrow.compute as pc
filtered = table.filter(pc.greater(table['value'], 100))

# Convert to Pandas for analysis
df = filtered.to_pandas()

# Or write back to Parquet
pq.write_table(filtered, 'output.parquet')
```

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Create Arrow Table
- Create Arrow table from Python dict
- Print schema and data
- Convert to Pandas DataFrame

### Exercise 2: Arrow vs Pandas Memory
- Create dataset with 1M rows
- Compare memory usage (Arrow vs Pandas)
- Calculate memory savings

### Exercise 3: Parquet with Arrow
- Read Parquet file using Arrow
- Select specific columns
- Filter rows using Arrow compute
- Write filtered data back

### Exercise 4: Arrow Compute
- Use Arrow compute for filtering
- Calculate statistics (mean, max, min)
- Sort data
- Compare speed with Pandas

### Exercise 5: Zero-Copy Demo
- Create large Pandas DataFrame
- Convert to Arrow
- Access columns (zero-copy)
- Measure performance difference

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is the main difference between Arrow and Parquet?
2. What does "zero-copy" mean?
3. Why is Arrow faster than traditional formats?
4. Is Arrow a storage format or memory format?
5. Can Arrow handle nested data?
6. How does Arrow improve interoperability?
7. What are Arrow compute functions?
8. When would you use Arrow vs Pandas?

---

## 🎯 Key Takeaways

- **Arrow is in-memory**, Parquet is on-disk
- **Zero-copy** operations = extremely fast
- **Columnar layout** = optimized for analytics
- **Language agnostic** = works across Python, R, Java, C++
- **More efficient** than Pandas for large datasets
- **Arrow + Parquet** = powerful combination
- **Use Arrow** for data exchange between systems

---

## 📚 Additional Resources

- [Apache Arrow Documentation](https://arrow.apache.org/docs/python/)
- [Arrow Compute Functions](https://arrow.apache.org/docs/python/compute.html)
- [Arrow vs Pandas](https://arrow.apache.org/docs/python/pandas.html)

---

## Tomorrow: Day 4 - Avro

We'll learn about Apache Avro, a row-based format with schema evolution for streaming.
