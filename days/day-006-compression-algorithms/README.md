# Day 6: Compression Algorithms

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand different compression algorithms (Snappy, ZSTD, LZ4, Gzip)
- Compare compression ratios and speeds
- Choose the right compression for your use case
- Use compression with Parquet files
- Understand compression trade-offs

---

## Theory

### What is Data Compression?

Compression reduces file size by encoding data more efficiently. This saves:
- **Storage space** (disk/cloud costs)
- **Network bandwidth** (faster transfers)
- **I/O time** (less data to read/write)

### The Compression Trade-off

```
Fast Compression ←→ High Compression Ratio
     LZ4              Snappy         ZSTD          Gzip
     ↑                  ↑              ↑             ↑
   Fastest          Balanced      Better        Best Ratio
   Lower Ratio                    Compression   Slowest
```

### The 4 Main Algorithms

#### 1. Snappy (Default for Parquet)

**Characteristics:**
- Fast compression and decompression
- Moderate compression ratio (2-3x)
- Low CPU usage
- Developed by Google

**Use When:**
- Need balanced performance
- Default choice for most cases
- Real-time processing
- CPU is limited

**Example:**
```python
df.to_parquet('data.parquet', compression='snappy')
```

#### 2. ZSTD (Zstandard)

**Characteristics:**
- Better compression than Snappy
- Still reasonably fast
- Adjustable compression levels (1-22)
- Modern algorithm (Facebook)

**Use When:**
- Storage cost is important
- Can afford slightly slower writes
- Want best balance of speed and ratio
- Modern systems

**Example:**
```python
df.to_parquet('data.parquet', compression='zstd')
```

#### 3. LZ4

**Characteristics:**
- Fastest compression/decompression
- Lower compression ratio than Snappy
- Extremely low CPU usage
- Good for real-time systems

**Use When:**
- Speed is critical
- CPU is bottleneck
- Real-time streaming
- Temporary files

**Example:**
```python
# LZ4 not directly supported in pandas, use pyarrow
import pyarrow.parquet as pq
pq.write_table(table, 'data.parquet', compression='lz4')
```

#### 4. Gzip

**Characteristics:**
- Best compression ratio (4-5x)
- Slowest compression/decompression
- High CPU usage
- Universal compatibility

**Use When:**
- Storage is expensive
- Data rarely accessed (cold storage)
- Network transfer cost is high
- Compatibility is critical

**Example:**
```python
df.to_parquet('data.parquet', compression='gzip')
```

### Compression Comparison

| Algorithm | Speed | Ratio | CPU | Use Case |
|-----------|-------|-------|-----|----------|
| **LZ4** | Fastest | 2x | Low | Real-time, streaming |
| **Snappy** | Fast | 2-3x | Low | Default, balanced |
| **ZSTD** | Medium | 3-4x | Medium | Storage optimization |
| **Gzip** | Slow | 4-5x | High | Cold storage, archives |

### Benchmark Example

```python
import pandas as pd
import time
import os

# Create test data
df = pd.DataFrame({
    'id': range(100000),
    'text': ['Sample text ' * 50 for _ in range(100000)]
})

compressions = ['snappy', 'gzip', 'zstd', None]

for comp in compressions:
    comp_name = comp if comp else 'none'
    
    # Write
    start = time.time()
    df.to_parquet(f'test_{comp_name}.parquet', compression=comp)
    write_time = time.time() - start
    
    # Size
    size = os.path.getsize(f'test_{comp_name}.parquet')
    
    # Read
    start = time.time()
    pd.read_parquet(f'test_{comp_name}.parquet')
    read_time = time.time() - start
    
    print(f"{comp_name}:")
    print(f"  Write: {write_time:.2f}s")
    print(f"  Size: {size / 1024 / 1024:.2f} MB")
    print(f"  Read: {read_time:.2f}s")
```

**Typical Results:**
```
none:
  Write: 0.15s, Size: 50 MB, Read: 0.10s
snappy:
  Write: 0.25s, Size: 20 MB, Read: 0.15s
zstd:
  Write: 0.40s, Size: 15 MB, Read: 0.20s
gzip:
  Write: 1.20s, Size: 12 MB, Read: 0.50s
```

### ZSTD Compression Levels

ZSTD supports levels 1-22:

```python
# Level 1: Fastest, lower compression
df.to_parquet('data.parquet', compression='zstd', compression_level=1)

# Level 3: Default, balanced
df.to_parquet('data.parquet', compression='zstd', compression_level=3)

# Level 10: Better compression, slower
df.to_parquet('data.parquet', compression='zstd', compression_level=10)

# Level 22: Maximum compression, very slow
df.to_parquet('data.parquet', compression='zstd', compression_level=22)
```

### When to Use Each

**Snappy (Default):**
- General purpose
- Don't know what to choose
- Balanced workloads
- Most Parquet files

**ZSTD:**
- Storage costs matter
- Modern infrastructure
- Long-term storage
- Can afford slower writes

**LZ4:**
- Real-time processing
- Streaming data
- CPU constrained
- Temporary files

**Gzip:**
- Cold storage
- Archival data
- Maximum compression needed
- Legacy systems

**None (No compression):**
- Data already compressed (images, videos)
- Extremely fast writes needed
- Temporary processing

### Real-World Example: Data Lake

```python
import pandas as pd

# Hot data (frequently accessed) - use Snappy
df_recent = pd.read_csv('recent_data.csv')
df_recent.to_parquet(
    's3://datalake/hot/data.parquet',
    compression='snappy'
)

# Warm data (occasionally accessed) - use ZSTD
df_monthly = pd.read_csv('monthly_data.csv')
df_monthly.to_parquet(
    's3://datalake/warm/data.parquet',
    compression='zstd',
    compression_level=5
)

# Cold data (rarely accessed) - use Gzip
df_archive = pd.read_csv('archive_data.csv')
df_archive.to_parquet(
    's3://datalake/cold/data.parquet',
    compression='gzip'
)
```

### Cost Savings Example

**Scenario:** 1TB of data in S3

```
No compression: 1000 GB × $0.023/GB = $23/month
Snappy (2.5x):   400 GB × $0.023/GB = $9.20/month
ZSTD (3.5x):     286 GB × $0.023/GB = $6.58/month
Gzip (4.5x):     222 GB × $0.023/GB = $5.11/month

Annual savings with ZSTD: ($23 - $6.58) × 12 = $197/year
```

### Compression with Different Formats

```python
# Parquet
df.to_parquet('data.parquet', compression='snappy')

# CSV (gzip only)
df.to_csv('data.csv.gz', compression='gzip')

# JSON (gzip only)
df.to_json('data.json.gz', compression='gzip')

# Pickle
df.to_pickle('data.pkl.gz', compression='gzip')
```

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Compression Benchmark
- Create dataset with 50K rows
- Write with all 4 compressions
- Measure write times
- Compare file sizes

### Exercise 2: Read Performance
- Read files from Exercise 1
- Measure read times
- Calculate compression ratios
- Determine best overall

### Exercise 3: ZSTD Levels
- Test ZSTD levels 1, 3, 10, 22
- Compare sizes and times
- Find optimal level

### Exercise 4: Data Type Impact
- Test with numeric data
- Test with text data
- Compare compression effectiveness

### Exercise 5: Cost Calculation
- Calculate storage costs for 1TB
- Compare all compressions
- Calculate annual savings

### Exercise 6: Real-World Scenario
- Implement hot/warm/cold storage
- Use appropriate compression for each
- Measure total storage savings

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. Which compression is fastest?
2. Which compression has best ratio?
3. What is the default compression for Parquet?
4. When should you use Gzip?
5. What is ZSTD's advantage?
6. Can you use LZ4 with pandas directly?
7. What compression level is ZSTD default?
8. When should you use no compression?

---

## 🎯 Key Takeaways

- **Snappy** - Default choice, balanced performance
- **ZSTD** - Best modern option, adjustable levels
- **LZ4** - Fastest, use for real-time
- **Gzip** - Best ratio, use for cold storage
- **Trade-off** - Speed vs compression ratio
- **Cost savings** - Compression reduces storage costs significantly
- **Choose based on** - Access patterns and requirements

---

## 📚 Additional Resources

- [Snappy Documentation](https://github.com/google/snappy)
- [ZSTD Documentation](https://facebook.github.io/zstd/)
- [LZ4 Documentation](https://lz4.github.io/lz4/)
- [Compression Benchmarks](https://github.com/inikep/lzbench)

---

## Tomorrow: Day 7 - Mini Project: Format Converter

We'll build a CLI tool that converts between formats with compression options.
