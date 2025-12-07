# Day 13: Iceberg vs Delta Lake

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Compare Iceberg and Delta Lake features
- Understand architectural differences
- Choose the right format for your use case
- Evaluate performance characteristics
- Assess ecosystem compatibility

---

## Theory

### Overview

Both are **open table formats** adding ACID transactions to data lakes, but with different approaches.

| Aspect | Apache Iceberg | Delta Lake |
|--------|----------------|------------|
| **Created by** | Netflix (2017) | Databricks (2019) |
| **Foundation** | Apache | Linux Foundation |
| **Primary Engine** | Multi-engine first | Spark-centric |
| **Metadata** | Avro/JSON | JSON |
| **Data Format** | Parquet/ORC/Avro | Parquet only |

### Architecture Comparison

#### Iceberg Architecture

```
Metadata Layer (3 levels)
├── Metadata File (JSON) → Schema, Snapshots
├── Manifest List (Avro) → List of Manifests
└── Manifest Files (Avro) → Data File Locations

Data Layer
└── Parquet/ORC/Avro Files
```

#### Delta Lake Architecture

```
Transaction Log (_delta_log/)
├── 00000.json → Add/Remove Operations
├── 00001.json → Metadata Changes
└── 00010.checkpoint.parquet → Aggregated State

Data Layer
└── Parquet Files
```

**Key Difference**: Iceberg has 3-level metadata; Delta has flat transaction log with checkpoints.

### Feature Comparison

#### Hidden Partitioning

**Iceberg**: ✅ Built-in
```python
# Users don't specify partition filters
SELECT * FROM table WHERE date = '2024-01-15'
# Iceberg automatically prunes partitions
```

**Delta Lake**: ❌ Manual
```python
# Users must know partition structure
SELECT * FROM table WHERE year = 2024 AND month = 1
```

#### Partition Evolution

**Iceberg**: ✅ Without rewrite
```sql
ALTER TABLE events 
REPLACE PARTITION FIELD days(ts) WITH months(ts)
```

**Delta Lake**: ❌ Requires rewrite
```python
# Must recreate table with new partitioning
```

#### Schema Evolution

**Iceberg**: ✅ Full support
- Add, drop, rename, reorder columns
- Change column types (compatible)
- Nested schema changes

**Delta Lake**: ✅ Good support
- Add columns (with mergeSchema)
- Limited rename/reorder
- Type changes restricted

#### Time Travel

**Iceberg**:
```python
# By snapshot ID
df = spark.read.option("snapshot-id", 123).table("table")

# By timestamp
df = spark.read.option("as-of-timestamp", "2024-01-01").table("table")
```

**Delta Lake**:
```python
# By version
df = spark.read.option("versionAsOf", 2).format("delta").load(path)

# By timestamp
df = spark.read.option("timestampAsOf", "2024-01-01").format("delta").load(path)
```

**Both support time travel**, but Iceberg's snapshot model is more granular.

#### ACID Transactions

**Iceberg**: ✅ Serializable isolation
- Optimistic concurrency
- Snapshot isolation

**Delta Lake**: ✅ Serializable isolation
- Optimistic concurrency
- Transaction log based

**Both provide full ACID**, implementation differs.

### Engine Support

#### Iceberg
- ✅ Spark
- ✅ Trino
- ✅ Flink
- ✅ Presto
- ✅ Hive
- ✅ Dremio
- ✅ Athena

**Multi-engine by design**

#### Delta Lake
- ✅ Spark (native)
- ✅ Trino (via connector)
- ✅ Flink (via connector)
- ✅ Presto (limited)
- ⚠️ Others (varying support)

**Spark-first, others via connectors**

### Performance Characteristics

#### Metadata Operations

**Iceberg**: Faster for large tables
- 3-level metadata = efficient pruning
- Manifest files cache file stats

**Delta Lake**: Faster for small tables
- Flat log = simpler for small datasets
- Checkpoints help at scale

#### Query Performance

**Both comparable** for most workloads:
- Columnar Parquet storage
- Predicate pushdown
- Partition pruning

**Iceberg edge**: Hidden partitioning reduces user errors

#### Write Performance

**Delta Lake**: Slightly faster writes
- Simpler transaction log
- Less metadata overhead

**Iceberg**: Better for concurrent writes
- Finer-grained conflict detection

### Use Case Recommendations

#### Choose Iceberg When:
- Multi-engine environment (Spark + Trino + Flink)
- Need partition evolution
- Large tables (billions of rows)
- Hidden partitioning desired
- Vendor-neutral solution required

#### Choose Delta Lake When:
- Spark-centric environment
- Databricks platform
- Simpler architecture preferred
- Strong Spark integration needed
- Existing Delta Lake expertise

### Code Comparison

#### Creating Tables

**Iceberg**:
```python
spark.sql("""
    CREATE TABLE catalog.db.users (
        id INT, name STRING
    ) USING iceberg
    PARTITIONED BY (days(created_at))
""")
```

**Delta Lake**:
```python
spark.sql("""
    CREATE TABLE users (
        id INT, name STRING
    ) USING DELTA
    PARTITIONED BY (DATE(created_at))
""")
```

#### Updates

**Iceberg**:
```python
spark.sql("UPDATE catalog.db.users SET name = 'New' WHERE id = 1")
```

**Delta Lake**:
```python
spark.sql("UPDATE users SET name = 'New' WHERE id = 1")
```

**Similar syntax**, both support DML.

#### Maintenance

**Iceberg**:
```python
# Expire snapshots
spark.sql("CALL catalog.system.expire_snapshots('db.users', TIMESTAMP '2024-01-01')")

# Remove orphans
spark.sql("CALL catalog.system.remove_orphan_files('db.users')")

# Rewrite files
spark.sql("CALL catalog.system.rewrite_data_files('db.users')")
```

**Delta Lake**:
```python
# Optimize
spark.sql("OPTIMIZE users")

# Vacuum
spark.sql("VACUUM users RETAIN 168 HOURS")

# Z-order
spark.sql("OPTIMIZE users ZORDER BY (user_id)")
```

**Different commands**, similar goals.

### Migration Considerations

#### Iceberg → Delta Lake
- Export data to Parquet
- Recreate as Delta table
- Lose Iceberg-specific features

#### Delta Lake → Iceberg
- Use Delta Standalone Reader
- Convert via Spark
- Preserve data, rebuild metadata

**No direct conversion** - requires data copy.

### Ecosystem Integration

#### Cloud Platforms

**Iceberg**:
- AWS: Athena, EMR, Glue
- Azure: Synapse (preview)
- GCP: BigQuery (preview)

**Delta Lake**:
- AWS: EMR, Glue (via connector)
- Azure: Synapse (native), Databricks
- GCP: Dataproc, Databricks

#### Streaming

**Iceberg**:
```python
# Flink streaming
tableEnv.executeSql("""
    INSERT INTO iceberg_table 
    SELECT * FROM kafka_source
""")
```

**Delta Lake**:
```python
# Spark streaming
stream.writeStream.format("delta").start(path)
```

**Both support streaming**, Delta Lake more mature in Spark.

### Decision Matrix

| Requirement | Iceberg | Delta Lake |
|-------------|---------|------------|
| Multi-engine | ⭐⭐⭐ | ⭐⭐ |
| Spark-only | ⭐⭐ | ⭐⭐⭐ |
| Partition evolution | ⭐⭐⭐ | ⭐ |
| Hidden partitioning | ⭐⭐⭐ | ⭐ |
| Simplicity | ⭐⭐ | ⭐⭐⭐ |
| Maturity | ⭐⭐⭐ | ⭐⭐⭐ |
| Community | ⭐⭐⭐ | ⭐⭐⭐ |
| Cloud support | ⭐⭐⭐ | ⭐⭐ |

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Feature Comparison
- Create same table in both formats
- Compare metadata structure
- Analyze differences

### Exercise 2: Time Travel
- Implement time travel in both
- Compare query syntax
- Measure performance

### Exercise 3: Schema Evolution
- Add column in both formats
- Test compatibility
- Compare ease of use

### Exercise 4: Partition Handling
- Create partitioned tables
- Test hidden partitioning (Iceberg)
- Compare query patterns

### Exercise 5: Maintenance Operations
- Run optimization in both
- Compare commands
- Analyze results

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is the main architectural difference?
2. Which supports hidden partitioning?
3. Which is better for multi-engine environments?
4. Can you convert between formats easily?
5. Which has simpler metadata structure?
6. Do both support ACID transactions?
7. Which supports partition evolution?
8. When should you choose Iceberg?

---

## 🎯 Key Takeaways

- **Both provide ACID** - Reliable transactions guaranteed
- **Iceberg: Multi-engine** - Better cross-platform support
- **Delta: Spark-centric** - Deeper Spark integration
- **Hidden partitioning** - Iceberg advantage
- **Partition evolution** - Iceberg only
- **Simpler architecture** - Delta Lake advantage
- **No direct migration** - Requires data copy
- **Choose based on ecosystem** - Not features alone

---

## 📚 Additional Resources

- [Iceberg vs Delta Comparison](https://www.onehouse.ai/blog/apache-iceberg-vs-delta-lake-vs-apache-hudi)
- [Iceberg Documentation](https://iceberg.apache.org/)
- [Delta Lake Documentation](https://docs.delta.io/)

---

## Tomorrow: Day 14 - Mini Project: Iceberg Table Manager

We'll build a practical tool to manage Iceberg tables.
