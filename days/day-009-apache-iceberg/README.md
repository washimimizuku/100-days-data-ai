# Day 9: Apache Iceberg

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand Iceberg architecture and metadata
- Create and query Iceberg tables
- Use hidden partitioning and schema evolution
- Query table metadata and snapshots

---

## Theory

### What is Apache Iceberg?

Apache Iceberg is an **open table format** for huge analytic datasets with:
- ACID transactions
- Hidden partitioning
- Schema evolution
- Partition evolution
- Time travel
- Multi-engine support

**Created by**: Netflix (2017) | **Used by**: Netflix, Apple, Adobe, LinkedIn, Airbnb

### Iceberg Architecture

```
Iceberg Table
│
├── Metadata Layer (JSON/Avro)
│   ├── Metadata File → Schema, Partition Spec, Snapshots
│   ├── Manifest List → List of Manifest Files
│   └── Manifest Files → List of Data Files
│
└── Data Layer (Parquet/ORC/Avro)
```

### Key Concepts

#### 1. Hidden Partitioning

**Traditional (Hive):**
```sql
SELECT * FROM table WHERE year = 2024 AND month = 1
```

**Iceberg:**
```sql
SELECT * FROM table WHERE date = '2024-01-15'  -- Auto partition pruning
```

**Benefits**: No partition filters needed, change partitioning without rewrite

#### 2. Snapshots

Each write creates immutable snapshot:
```
Snapshot 1: [file1, file2]
Snapshot 2: [file1, file2, file3]
Snapshot 3: [file1, file3, file4]
```

### Creating Tables

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "warehouse") \
    .getOrCreate()

# Create table
spark.sql("""
    CREATE TABLE local.db.users (
        id INT,
        name STRING,
        age INT
    ) USING iceberg
""")
```

### Writing and Reading

```python
# Write
df = spark.createDataFrame([(1, "Alice", 25)], ["id", "name", "age"])
df.writeTo("local.db.users").append()

# Read current
df = spark.table("local.db.users")

# Read specific snapshot
df = spark.read.option("snapshot-id", 123456).table("local.db.users")

# Read as of timestamp
df = spark.read.option("as-of-timestamp", "2024-01-01 00:00:00").table("local.db.users")
```

### Schema Evolution

```python
# Add column
spark.sql("ALTER TABLE local.db.users ADD COLUMN email STRING")

# Rename column
spark.sql("ALTER TABLE local.db.users RENAME COLUMN age TO user_age")

# Drop column
spark.sql("ALTER TABLE local.db.users DROP COLUMN email")
```

### Partition Evolution

```python
# Create with daily partitions
spark.sql("""
    CREATE TABLE local.db.events (
        id INT,
        event_time TIMESTAMP,
        data STRING
    ) USING iceberg
    PARTITIONED BY (days(event_time))
""")

# Change to monthly (no rewrite!)
spark.sql("""
    ALTER TABLE local.db.events 
    REPLACE PARTITION FIELD days(event_time) WITH months(event_time)
""")
```

### Time Travel

```python
# View snapshots
spark.sql("SELECT * FROM local.db.users.snapshots").show()

# Rollback
spark.sql("CALL system.rollback_to_snapshot('local.db.users', 123456)")
```

### Metadata Tables

```python
# Available metadata tables
spark.sql("SELECT * FROM local.db.users.snapshots").show()
spark.sql("SELECT * FROM local.db.users.files").show()
spark.sql("SELECT * FROM local.db.users.history").show()
spark.sql("SELECT * FROM local.db.users.manifests").show()
spark.sql("SELECT * FROM local.db.users.partitions").show()
```

### Maintenance Operations

```python
# Expire old snapshots
spark.sql("""
    CALL system.expire_snapshots(
        table => 'local.db.users',
        older_than => TIMESTAMP '2024-01-01 00:00:00',
        retain_last => 5
    )
""")

# Remove orphan files
spark.sql("""
    CALL system.remove_orphan_files(
        table => 'local.db.users',
        older_than => TIMESTAMP '2024-01-01 00:00:00'
    )
""")

# Compact small files
spark.sql("""
    CALL system.rewrite_data_files(
        table => 'local.db.users',
        strategy => 'binpack'
    )
""")
```

### Multi-Engine Support

```python
# Spark
df = spark.table("db.users")

# Trino
# SELECT * FROM iceberg.db.users;

# Flink
# SELECT * FROM iceberg_catalog.db.users;

# Presto
# SELECT * FROM iceberg.db.users;
```

### Real-World Example

```python
# Create orders table
spark.sql("""
    CREATE TABLE local.db.orders (
        order_id BIGINT,
        customer_id BIGINT,
        amount DECIMAL(10,2),
        order_date DATE,
        status STRING
    ) USING iceberg
    PARTITIONED BY (months(order_date))
""")

# Insert data
orders = spark.createDataFrame([
    (1, 101, 99.99, "2024-01-15", "completed"),
    (2, 102, 149.99, "2024-01-20", "pending")
], ["order_id", "customer_id", "amount", "order_date", "status"])

orders.writeTo("local.db.orders").append()

# Query with auto partition pruning
result = spark.sql("""
    SELECT * FROM local.db.orders
    WHERE order_date >= '2024-01-01' AND order_date < '2024-02-01'
""")

# Schema evolution
spark.sql("ALTER TABLE local.db.orders ADD COLUMN shipping_address STRING")

# View snapshots
spark.sql("SELECT * FROM local.db.orders.snapshots").show()
```

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Create Iceberg Table
- Set up Spark with Iceberg
- Create table with schema
- Insert sample data

### Exercise 2: Hidden Partitioning
- Create partitioned table
- Query without partition filters
- Compare with traditional partitioning

### Exercise 3: Schema Evolution
- Add new column
- Rename column
- Query old and new data

### Exercise 4: Time Travel
- Create multiple snapshots
- Query specific snapshot
- Rollback to previous version

### Exercise 5: Metadata Exploration
- Query snapshots table
- Query files table
- Analyze table history

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is hidden partitioning in Iceberg?
2. What is a snapshot?
3. Can you change partitioning without rewriting data?
4. What engines support Iceberg?
5. How do you query a specific snapshot?
6. What is partition evolution?
7. What metadata tables does Iceberg provide?
8. How do you expire old snapshots?

---

## 🎯 Key Takeaways

- **Hidden partitioning** - Users don't specify partition filters
- **Snapshots** - Each write creates immutable snapshot
- **Schema evolution** - Add/rename/drop columns safely
- **Partition evolution** - Change partitioning without rewrite
- **Time travel** - Query any historical snapshot
- **Multi-engine** - Works with Spark, Trino, Flink, Presto
- **Metadata tables** - Query table metadata directly
- **ACID transactions** - Reliable concurrent operations

---

## 📚 Additional Resources

- [Apache Iceberg Documentation](https://iceberg.apache.org/docs/latest/)
- [PyIceberg](https://py.iceberg.apache.org/)
- [Iceberg Spark Integration](https://iceberg.apache.org/docs/latest/spark-getting-started/)

---

## Tomorrow: Day 10 - Delta Lake

We'll explore Delta Lake, another popular table format.
