# Day 11: Delta Lake

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand Delta Lake architecture and features
- Create and manage Delta tables
- Perform CRUD operations with ACID guarantees
- Use Delta Lake time travel
- Optimize Delta tables

---

## Theory

### What is Delta Lake?

Delta Lake is an **open-source storage layer** that brings ACID transactions to Apache Spark and big data workloads.

**Created by**: Databricks (2019)
**Open Source**: Linux Foundation
**Used by**: Databricks, AWS, Azure, GCP customers

**Key Features:**
- ACID transactions
- Time travel (data versioning)
- Schema enforcement and evolution
- Unified batch and streaming
- Audit history
- DML operations (UPDATE, DELETE, MERGE)

### Delta Lake Architecture

```
Delta Table
│
├── Transaction Log (_delta_log/)
│   ├── 00000000000000000000.json  # Version 0
│   ├── 00000000000000000001.json  # Version 1
│   └── 00000000000000000002.json  # Version 2
│
└── Data Files (Parquet)
    ├── part-00000.parquet
    ├── part-00001.parquet
    └── part-00002.parquet
```

**Transaction Log**: JSON files tracking all changes (add/remove files, metadata)
**Data Files**: Parquet files containing actual data

### Creating Delta Tables

```python
from pyspark.sql import SparkSession
from delta import *

# Initialize Spark with Delta
builder = SparkSession.builder \
    .appName("DeltaLake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Create from DataFrame
df = spark.createDataFrame([
    (1, "Alice", 25),
    (2, "Bob", 30)
], ["id", "name", "age"])

df.write.format("delta").save("/tmp/delta-table")

# Create with SQL
spark.sql("""
    CREATE TABLE users (
        id INT,
        name STRING,
        age INT
    ) USING DELTA
    LOCATION '/tmp/users'
""")
```

### Reading Delta Tables

```python
# Read as DataFrame
df = spark.read.format("delta").load("/tmp/delta-table")

# Read with SQL
df = spark.sql("SELECT * FROM delta.`/tmp/delta-table`")

# Read registered table
df = spark.table("users")
```

### CRUD Operations

#### INSERT

```python
# Append
new_data = spark.createDataFrame([(3, "Charlie", 35)], ["id", "name", "age"])
new_data.write.format("delta").mode("append").save("/tmp/delta-table")

# SQL
spark.sql("INSERT INTO users VALUES (4, 'David', 40)")
```

#### UPDATE

```python
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/tmp/delta-table")

# Update with condition
delta_table.update(
    condition = "age < 30",
    set = {"age": "age + 1"}
)

# SQL
spark.sql("UPDATE users SET age = age + 1 WHERE age < 30")
```

#### DELETE

```python
# Delete with condition
delta_table.delete("age > 50")

# SQL
spark.sql("DELETE FROM users WHERE age > 50")
```

#### MERGE (UPSERT)

```python
updates = spark.createDataFrame([
    (1, "Alice Updated", 26),
    (5, "Eve", 28)
], ["id", "name", "age"])

delta_table.alias("target").merge(
    updates.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set = {
    "name": "source.name",
    "age": "source.age"
}).whenNotMatchedInsert(values = {
    "id": "source.id",
    "name": "source.name",
    "age": "source.age"
}).execute()
```

### Time Travel

```python
# Read by version
df_v0 = spark.read.format("delta").option("versionAsOf", 0).load("/tmp/delta-table")

# Read by timestamp
df_ts = spark.read.format("delta").option("timestampAsOf", "2024-01-01").load("/tmp/delta-table")

# SQL
spark.sql("SELECT * FROM users VERSION AS OF 2")
spark.sql("SELECT * FROM users TIMESTAMP AS OF '2024-01-01'")
```

### Schema Enforcement

```python
# Schema validation on write
try:
    bad_data = spark.createDataFrame([(1, "Test")], ["id", "name"])  # Missing 'age'
    bad_data.write.format("delta").mode("append").save("/tmp/delta-table")
except Exception as e:
    print(f"Schema mismatch: {e}")
```

### Schema Evolution

```python
# Add new column
new_data = spark.createDataFrame([
    (6, "Frank", 45, "frank@example.com")
], ["id", "name", "age", "email"])

new_data.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("/tmp/delta-table")
```

### Table History

```python
# View history
delta_table = DeltaTable.forPath(spark, "/tmp/delta-table")
delta_table.history().show()

# SQL
spark.sql("DESCRIBE HISTORY users").show()
```

### Optimization

#### OPTIMIZE (Compaction)

```python
# Compact small files
delta_table.optimize().executeCompaction()

# SQL
spark.sql("OPTIMIZE users")

# With Z-ordering
spark.sql("OPTIMIZE users ZORDER BY (age)")
```

#### VACUUM (Cleanup)

```python
# Remove old files (default: 7 days retention)
delta_table.vacuum()

# Custom retention
delta_table.vacuum(168)  # 7 days in hours

# SQL
spark.sql("VACUUM users RETAIN 168 HOURS")
```

### Real-World Example

```python
# Create orders table
spark.sql("""
    CREATE TABLE orders (
        order_id INT,
        customer_id INT,
        amount DECIMAL(10,2),
        status STRING,
        created_at TIMESTAMP
    ) USING DELTA
    PARTITIONED BY (DATE(created_at))
""")

# Insert initial data
orders = spark.createDataFrame([
    (1, 101, 99.99, "pending", "2024-01-01 10:00:00"),
    (2, 102, 149.99, "completed", "2024-01-01 11:00:00")
], ["order_id", "customer_id", "amount", "status", "created_at"])

orders.write.format("delta").mode("append").saveAsTable("orders")

# Update order status
spark.sql("UPDATE orders SET status = 'completed' WHERE order_id = 1")

# Merge new orders
new_orders = spark.createDataFrame([
    (2, 102, 149.99, "refunded", "2024-01-01 11:00:00"),  # Update
    (3, 103, 79.99, "pending", "2024-01-02 09:00:00")     # Insert
], ["order_id", "customer_id", "amount", "status", "created_at"])

DeltaTable.forName(spark, "orders").alias("target").merge(
    new_orders.alias("source"),
    "target.order_id = source.order_id"
).whenMatchedUpdate(set = {
    "status": "source.status"
}).whenNotMatchedInsertAll().execute()

# View history
spark.sql("DESCRIBE HISTORY orders").show()

# Time travel
df_v0 = spark.sql("SELECT * FROM orders VERSION AS OF 0")
print(f"Version 0 count: {df_v0.count()}")

# Optimize
spark.sql("OPTIMIZE orders")
```

### Delta vs Parquet

| Feature | Parquet | Delta Lake |
|---------|---------|------------|
| ACID | ❌ | ✅ |
| Time Travel | ❌ | ✅ |
| Schema Evolution | ❌ | ✅ |
| UPDATE/DELETE | ❌ | ✅ |
| Streaming | Limited | ✅ |
| Audit History | ❌ | ✅ |

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Create Delta Table
- Initialize Spark with Delta
- Create table from DataFrame
- Read and verify data

### Exercise 2: CRUD Operations
- INSERT new records
- UPDATE existing records
- DELETE records with condition
- MERGE (upsert) data

### Exercise 3: Time Travel
- Create multiple versions
- Query by version number
- Query by timestamp
- View table history

### Exercise 4: Schema Management
- Test schema enforcement
- Add new column with mergeSchema
- Verify schema evolution

### Exercise 5: Optimization
- Create table with many small files
- Run OPTIMIZE
- Run VACUUM
- Compare file counts

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is the Delta Lake transaction log?
2. What file format does Delta Lake use for data?
3. How do you perform an UPDATE in Delta Lake?
4. What is the difference between OPTIMIZE and VACUUM?
5. How do you query a previous version?
6. What is schema enforcement?
7. What does MERGE do?
8. How long does VACUUM retain old files by default?

---

## 🎯 Key Takeaways

- **Transaction log** - JSON files tracking all table changes
- **ACID transactions** - Reliable concurrent reads/writes
- **Time travel** - Query historical versions
- **CRUD operations** - UPDATE, DELETE, MERGE support
- **Schema enforcement** - Prevents bad data writes
- **Schema evolution** - Add columns with mergeSchema
- **OPTIMIZE** - Compact small files for performance
- **VACUUM** - Remove old files to save storage

---

## 📚 Additional Resources

- [Delta Lake Documentation](https://docs.delta.io/)
- [Delta Lake GitHub](https://github.com/delta-io/delta)
- [Databricks Delta Guide](https://docs.databricks.com/delta/)

---

## Tomorrow: Day 12 - Delta Lake ACID

We'll dive deeper into ACID transactions and concurrent operations.
