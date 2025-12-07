# Day 8: Introduction to Table Formats

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand what table formats are and why they exist
- Learn the difference between file formats and table formats
- Compare Iceberg, Delta Lake, and Hudi
- Understand ACID transactions in data lakes
- Know when to use table formats

---

## Theory

### What are Table Formats?

**File Formats** (Parquet, Avro, ORC):
- How individual files are stored
- Single file operations
- No transaction support
- No schema evolution across files

**Table Formats** (Iceberg, Delta Lake, Hudi):
- How collections of files work together as a table
- Multi-file operations
- ACID transactions
- Schema evolution
- Time travel
- Metadata management

**Analogy:**
- File format = Individual pages in a book
- Table format = The book's table of contents and index

### The Problem Table Formats Solve

**Without Table Formats:**
```
data/
  part-001.parquet
  part-002.parquet
  part-003.parquet
  
Problems:
- No atomicity (partial writes visible)
- No consistency (concurrent writes conflict)
- No isolation (readers see incomplete data)
- No durability guarantees
- No time travel
- No schema evolution
```

**With Table Formats:**
```
data/
  metadata/
    v1.json (transaction log)
    v2.json
  data/
    part-001.parquet
    part-002.parquet
    
Benefits:
✅ ACID transactions
✅ Time travel
✅ Schema evolution
✅ Concurrent reads/writes
✅ Metadata management
```

### ACID Transactions

**Atomicity**: All or nothing
```python
# Either all files are added or none
table.add_files([file1, file2, file3])  # Atomic operation
```

**Consistency**: Valid state always
```python
# Schema is always consistent
table.evolve_schema(add_column='new_col')  # Validated
```

**Isolation**: Concurrent operations don't interfere
```python
# Writer 1 and Writer 2 can write simultaneously
# Readers always see consistent snapshots
```

**Durability**: Changes are permanent
```python
# Once committed, data is safe
table.commit()  # Durable
```

### The Big Three Table Formats

#### 1. Apache Iceberg

**Created by**: Netflix (2017)
**Open Source**: Apache Foundation

**Key Features:**
- Hidden partitioning (automatic)
- Time travel and rollback
- Schema evolution
- Partition evolution
- Multiple engine support

**Best For:**
- Multi-engine environments (Spark, Trino, Flink)
- Complex schema evolution
- Large-scale data lakes

**Example:**
```python
from pyiceberg.catalog import load_catalog

catalog = load_catalog("default")
table = catalog.load_table("db.table")

# Time travel
snapshot = table.scan().use_snapshot(snapshot_id).to_arrow()

# Schema evolution
table.update_schema().add_column("new_col", "string").commit()
```

#### 2. Delta Lake

**Created by**: Databricks (2019)
**Open Source**: Linux Foundation

**Key Features:**
- Transaction log (JSON)
- Time travel
- ACID transactions
- Schema enforcement
- Optimized for Spark

**Best For:**
- Databricks environments
- Spark-heavy workloads
- Streaming + batch

**Example:**
```python
from delta import DeltaTable

# Read Delta table
df = spark.read.format("delta").load("path/to/table")

# Time travel
df = spark.read.format("delta").option("versionAsOf", 0).load("path")

# Update
deltaTable = DeltaTable.forPath(spark, "path/to/table")
deltaTable.update(condition="id = 5", set={"value": "100"})
```

#### 3. Apache Hudi

**Created by**: Uber (2016)
**Open Source**: Apache Foundation

**Key Features:**
- Upserts and deletes
- Incremental processing
- Copy-on-write and merge-on-read
- Record-level operations

**Best For:**
- CDC workloads
- Frequent updates/deletes
- Incremental ETL

**Example:**
```python
# Write with Hudi
df.write.format("hudi") \
  .option("hoodie.table.name", "table_name") \
  .option("hoodie.datasource.write.operation", "upsert") \
  .save("path/to/table")
```

### Comparison Table

| Feature | Iceberg | Delta Lake | Hudi |
|---------|---------|------------|------|
| **Creator** | Netflix | Databricks | Uber |
| **Year** | 2017 | 2019 | 2016 |
| **ACID** | ✅ | ✅ | ✅ |
| **Time Travel** | ✅ | ✅ | ✅ |
| **Schema Evolution** | ✅✅ | ✅ | ✅ |
| **Partition Evolution** | ✅✅ | ❌ | ❌ |
| **Hidden Partitioning** | ✅ | ❌ | ❌ |
| **Upserts** | ✅ | ✅ | ✅✅ |
| **Multi-Engine** | ✅✅ | ✅ | ✅ |
| **Spark Integration** | ✅ | ✅✅ | ✅✅ |
| **Streaming** | ✅ | ✅✅ | ✅✅ |
| **Maturity** | High | High | High |

### When to Use Each

**Use Iceberg When:**
- Need multi-engine support (Spark, Trino, Flink, Presto)
- Complex partition evolution needed
- Large-scale data lake
- Cloud-agnostic solution

**Use Delta Lake When:**
- Using Databricks
- Spark-centric environment
- Need streaming + batch
- Want simplicity

**Use Hudi When:**
- Heavy CDC workloads
- Frequent updates/deletes
- Incremental processing critical
- Record-level operations needed

### Time Travel Example

```python
# Iceberg
table.scan().use_snapshot(snapshot_id).to_arrow()

# Delta Lake
spark.read.format("delta").option("versionAsOf", 5).load("path")

# Hudi
spark.read.format("hudi") \
  .option("as.of.instant", "20230101000000") \
  .load("path")
```

### Schema Evolution Example

```python
# Iceberg
table.update_schema() \
  .add_column("new_col", "string") \
  .commit()

# Delta Lake
deltaTable.update_schema() \
  .add_column("new_col", StringType()) \
  .commit()

# Hudi (automatic on write)
df_with_new_col.write.format("hudi").save("path")
```

### Real-World Use Cases

**Netflix (Iceberg):**
- 100+ PB data lake
- Multiple query engines
- Complex schema evolution
- Partition evolution

**Uber (Hudi):**
- Real-time CDC
- Incremental ETL
- Frequent updates
- Record-level changes

**Databricks (Delta Lake):**
- Unified batch + streaming
- ACID transactions
- ML feature stores
- Data warehousing

### Architecture Comparison

**Iceberg:**
```
Table
├── Metadata (JSON)
│   ├── Snapshots
│   ├── Manifests
│   └── Schema
└── Data (Parquet/ORC/Avro)
```

**Delta Lake:**
```
Table
├── _delta_log/
│   ├── 00000.json
│   ├── 00001.json
│   └── 00002.json
└── Data (Parquet)
```

**Hudi:**
```
Table
├── .hoodie/
│   ├── Timeline
│   └── Metadata
└── Data (Parquet)
```

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Concept Understanding
- Explain difference between file format and table format
- List 3 problems table formats solve
- Name the 3 main table formats

### Exercise 2: Feature Comparison
- Create comparison matrix
- Identify best format for given scenarios
- Justify your choices

### Exercise 3: ACID Transactions
- Explain each ACID property
- Give examples of each
- Describe why ACID matters

### Exercise 4: Use Case Analysis
- Analyze 5 scenarios
- Choose appropriate table format
- Explain reasoning

### Exercise 5: Time Travel
- Explain time travel concept
- Describe use cases
- Compare implementations

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is the difference between file format and table format?
2. What does ACID stand for?
3. Which table format has hidden partitioning?
4. Which table format is best for CDC workloads?
5. What is time travel?
6. Can you use Iceberg with Trino?
7. Which format is optimized for Databricks?
8. What problem do table formats solve?

---

## 🎯 Key Takeaways

- **Table formats** manage collections of files as tables
- **ACID transactions** ensure data consistency
- **Iceberg** - Multi-engine, partition evolution
- **Delta Lake** - Databricks, Spark-optimized
- **Hudi** - CDC, frequent updates
- **Time travel** - Query historical data
- **Schema evolution** - Change schemas safely
- **Choose based on** - Environment and use case

---

## 📚 Additional Resources

- [Apache Iceberg](https://iceberg.apache.org/)
- [Delta Lake](https://delta.io/)
- [Apache Hudi](https://hudi.apache.org/)
- [Table Format Comparison](https://www.onehouse.ai/blog/apache-hudi-vs-delta-lake-vs-apache-iceberg-lakehouse-feature-comparison)

---

## Tomorrow: Day 9 - Apache Iceberg

We'll dive deep into Apache Iceberg with hands-on examples.
