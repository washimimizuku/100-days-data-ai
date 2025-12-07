# Day 2: Parquet Format

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand what Parquet is and why it's used
- Learn columnar vs row-based storage
- Read and write Parquet files in Python
- Compare Parquet performance with CSV
- Use compression with Parquet

---

## Theory

### What is Parquet?

Apache Parquet is a **columnar storage format** optimized for analytics workloads. Unlike CSV (row-based), Parquet stores data by columns, making it extremely efficient for reading specific columns.

**Key Features:**
- Columnar storage (read only needed columns)
- Built-in compression (smaller files)
- Schema embedded in file
- Supports complex nested data
- Optimized for analytics queries

### Row-Based vs Columnar Storage

**Row-Based (CSV)**:
```
Row 1: Alice, 25, Engineer, 75000
Row 2: Bob, 30, Sales, 85000
Row 3: Charlie, 28, Marketing, 70000
```
- Reads entire row even if you need one column
- Good for transactional workloads (OLTP)
- Simple structure

**Columnar (Parquet)**:
```
Column name: [Alice, Bob, Charlie]
Column age: [25, 30, 28]
Column dept: [Engineer, Sales, Marketing]
Column salary: [75000, 85000, 70000]
```
- Reads only needed columns
- Better compression (similar values together)
- Ideal for analytics (OLAP)

### When to Use Parquet

✅ **Use Parquet when:**
- Running analytics queries (SELECT specific columns)
- Working with large datasets (>1GB)
- Need efficient compression
- Using data lakes (S3, HDFS)
- Working with Spark, Athena, BigQuery
- Schema is stable

❌ **Don't use Parquet when:**
- Need human-readable format
- Frequent schema changes
- Small datasets (<10MB)
- Need to edit files manually
- Streaming real-time data

### Reading Parquet in Python

```python
import pandas as pd
import pyarrow.parquet as pq

# Method 1: Using pandas
df = pd.read_parquet('data.parquet')
print(df.head())

# Method 2: Using pyarrow (more control)
table = pq.read_table('data.parquet')
df = table.to_pandas()

# Read specific columns only
df = pd.read_parquet('data.parquet', columns=['name', 'salary'])
```

### Writing Parquet in Python

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 28],
    'salary': [75000, 85000, 70000]
})

# Write to Parquet
df.to_parquet('output.parquet', compression='snappy')

# Different compression options
df.to_parquet('output_gzip.parquet', compression='gzip')
df.to_parquet('output_none.parquet', compression=None)
```

### Compression Options

| Compression | Speed | Ratio | Use Case |
|-------------|-------|-------|----------|
| **snappy** | Fast | Good | Default, balanced |
| **gzip** | Slow | Best | Maximum compression |
| **zstd** | Medium | Better | Modern, recommended |
| **lz4** | Fastest | Fair | Speed priority |
| **none** | Instant | None | Already compressed data |

### Parquet Metadata

```python
import pyarrow.parquet as pq

# Read metadata without loading data
parquet_file = pq.ParquetFile('data.parquet')

# Schema information
print(parquet_file.schema)

# File metadata
print(f"Rows: {parquet_file.metadata.num_rows}")
print(f"Columns: {parquet_file.metadata.num_columns}")
print(f"Size: {parquet_file.metadata.serialized_size} bytes")

# Column statistics
for i in range(parquet_file.metadata.num_row_groups):
    row_group = parquet_file.metadata.row_group(i)
    for j in range(row_group.num_columns):
        column = row_group.column(j)
        print(f"Column {j}: {column.path_in_schema}")
        print(f"  Min: {column.statistics.min}")
        print(f"  Max: {column.statistics.max}")
```

### Performance Comparison

```python
import pandas as pd
import time

# Create large dataset
df = pd.DataFrame({
    'id': range(1000000),
    'value': range(1000000),
    'category': ['A', 'B', 'C', 'D'] * 250000
})

# Write CSV
start = time.time()
df.to_csv('large.csv', index=False)
csv_write_time = time.time() - start

# Write Parquet
start = time.time()
df.to_parquet('large.parquet', compression='snappy')
parquet_write_time = time.time() - start

# Read CSV
start = time.time()
df_csv = pd.read_csv('large.csv')
csv_read_time = time.time() - start

# Read Parquet
start = time.time()
df_parquet = pd.read_parquet('large.parquet')
parquet_read_time = time.time() - start

print(f"CSV write: {csv_write_time:.2f}s")
print(f"Parquet write: {parquet_write_time:.2f}s")
print(f"CSV read: {csv_read_time:.2f}s")
print(f"Parquet read: {parquet_read_time:.2f}s")
```

**Typical Results:**
- Parquet is 5-10x faster to read
- Parquet files are 50-80% smaller
- Write times are similar

### Partitioning

Parquet supports partitioning for better query performance:

```python
# Write partitioned Parquet
df.to_parquet(
    'partitioned_data',
    partition_cols=['year', 'month'],
    compression='snappy'
)

# Creates structure:
# partitioned_data/
#   year=2024/
#     month=01/
#       data.parquet
#     month=02/
#       data.parquet

# Read specific partition
df = pd.read_parquet('partitioned_data/year=2024/month=01')
```

### Real-World Example: Data Lake

```python
import pandas as pd
from datetime import datetime

# Read CSV from source
df = pd.read_csv('daily_sales.csv')

# Add metadata
df['ingestion_date'] = datetime.now()
df['year'] = df['date'].str[:4]
df['month'] = df['date'].str[5:7]

# Write to data lake with partitioning
df.to_parquet(
    's3://my-datalake/sales/',
    partition_cols=['year', 'month'],
    compression='snappy',
    engine='pyarrow'
)

# Query specific month (only reads that partition)
df_jan = pd.read_parquet('s3://my-datalake/sales/year=2024/month=01')
```

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Basic Parquet Operations
- Create a DataFrame with 1000 rows
- Write to Parquet with snappy compression
- Read it back and verify data

### Exercise 2: Compression Comparison
- Write same data with different compressions (snappy, gzip, none)
- Compare file sizes
- Measure read times

### Exercise 3: Column Selection
- Create a wide DataFrame (20 columns)
- Write to Parquet
- Read only 3 specific columns
- Compare time vs reading all columns

### Exercise 4: CSV to Parquet Conversion
- Read a large CSV file
- Convert to Parquet
- Compare file sizes and read performance

### Exercise 5: Metadata Inspection
- Read Parquet file metadata
- Print schema, row count, column stats
- Don't load the actual data

### Exercise 6: Partitioned Data
- Create sales data with date column
- Write as partitioned Parquet (by year/month)
- Read specific partition

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is the main advantage of columnar storage?
2. Which compression is fastest: snappy, gzip, or zstd?
3. When should you NOT use Parquet?
4. How do you read only specific columns from Parquet?
5. What is partitioning and why is it useful?
6. Is Parquet human-readable like CSV?
7. Which is typically smaller: CSV or Parquet?
8. What metadata is stored in Parquet files?

---

## 🎯 Key Takeaways

- **Parquet is columnar** - stores data by columns, not rows
- **Much faster for analytics** - read only needed columns
- **Smaller files** - built-in compression (50-80% smaller than CSV)
- **Schema included** - self-describing format
- **Partitioning** - organize data for faster queries
- **Not human-readable** - binary format, need tools to read
- **Industry standard** - used by Spark, Athena, BigQuery, Snowflake

---

## 📚 Additional Resources

- [Apache Parquet Documentation](https://parquet.apache.org/docs/)
- [PyArrow Parquet](https://arrow.apache.org/docs/python/parquet.html)
- [Pandas Parquet](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html)
- [Parquet Format Explained](https://www.databricks.com/glossary/what-is-parquet)

---

## 🔗 Data Files

Sample data files in `/data/day-002/`:
- `sample_data.csv` - Source data
- `sample_data.parquet` - Parquet version

---

## Tomorrow: Day 3 - Apache Arrow

We'll learn about Apache Arrow, an in-memory columnar format for fast data processing.
