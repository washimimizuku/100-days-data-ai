# Day 10: Iceberg Time Travel & Snapshots

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Master time travel queries in Iceberg
- Understand snapshot lifecycle and management
- Perform rollbacks and data recovery
- Optimize snapshot retention

---

## Theory

### What is Time Travel?

Time travel allows querying historical versions of data by accessing previous snapshots. Each write operation creates a new snapshot, preserving the complete history.

**Use Cases:**
- Audit and compliance
- Data recovery
- Debugging data issues
- A/B testing with historical data
- Reproducing ML training datasets

### Snapshot Anatomy

```
Snapshot
├── snapshot_id: 123456789
├── parent_snapshot_id: 123456788
├── timestamp_ms: 1704067200000
├── operation: append|overwrite|delete
├── summary: {added-files: 5, deleted-files: 2}
└── manifest_list: path/to/manifest-list.avro
```

### Querying Snapshots

#### Current Data (Latest Snapshot)

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "warehouse") \
    .getOrCreate()

# Current data
df = spark.table("local.db.orders")
```

#### Query by Snapshot ID

```python
# Get snapshot ID
snapshots = spark.sql("SELECT * FROM local.db.orders.snapshots")
snapshot_id = snapshots.select("snapshot_id").first()[0]

# Query specific snapshot
df_snapshot = spark.read \
    .option("snapshot-id", snapshot_id) \
    .table("local.db.orders")
```

#### Query by Timestamp

```python
# As of specific timestamp
df_historical = spark.read \
    .option("as-of-timestamp", "2024-01-01 00:00:00") \
    .table("local.db.orders")

# Using SQL
spark.sql("""
    SELECT * FROM local.db.orders
    TIMESTAMP AS OF '2024-01-01 00:00:00'
""")
```

### Snapshot Metadata

```python
# View all snapshots
spark.sql("SELECT * FROM local.db.orders.snapshots").show()

# Snapshot details
spark.sql("""
    SELECT 
        snapshot_id,
        parent_id,
        committed_at,
        operation,
        summary
    FROM local.db.orders.snapshots
    ORDER BY committed_at DESC
""").show()

# History table
spark.sql("SELECT * FROM local.db.orders.history").show()
```

### Rollback Operations

#### Rollback to Snapshot

```python
# Rollback to specific snapshot
spark.sql(f"""
    CALL local.system.rollback_to_snapshot('db.orders', {snapshot_id})
""")

# Rollback to timestamp
spark.sql("""
    CALL local.system.rollback_to_timestamp(
        'db.orders', 
        TIMESTAMP '2024-01-01 00:00:00'
    )
""")
```

#### Set Current Snapshot

```python
# Set current snapshot (doesn't delete newer snapshots)
spark.sql(f"""
    CALL local.system.set_current_snapshot('db.orders', {snapshot_id})
""")
```

### Snapshot Expiration

Old snapshots consume storage. Expire them regularly:

```python
# Expire snapshots older than date, keep at least 5
spark.sql("""
    CALL local.system.expire_snapshots(
        table => 'db.orders',
        older_than => TIMESTAMP '2024-01-01 00:00:00',
        retain_last => 5
    )
""")

# Expire by age (7 days)
spark.sql("""
    CALL local.system.expire_snapshots(
        table => 'db.orders',
        max_snapshot_age_ms => 604800000,
        retain_last => 10
    )
""")
```

### Orphan File Cleanup

Remove files not referenced by any snapshot:

```python
spark.sql("""
    CALL local.system.remove_orphan_files(
        table => 'db.orders',
        older_than => TIMESTAMP '2024-01-01 00:00:00'
    )
""")
```

### Incremental Reads

Read only changes between snapshots:

```python
# Read changes between two snapshots
df_incremental = spark.read \
    .format("iceberg") \
    .option("start-snapshot-id", start_snapshot_id) \
    .option("end-snapshot-id", end_snapshot_id) \
    .load("local.db.orders")

# Read changes since timestamp
df_changes = spark.read \
    .format("iceberg") \
    .option("start-timestamp", "2024-01-01 00:00:00") \
    .load("local.db.orders")
```

### Real-World Example

```python
# Create table and insert data
spark.sql("""
    CREATE TABLE local.db.transactions (
        id BIGINT,
        user_id BIGINT,
        amount DECIMAL(10,2),
        created_at TIMESTAMP
    ) USING iceberg
""")

# Insert v1
df1 = spark.createDataFrame([
    (1, 101, 100.00, "2024-01-01 10:00:00"),
    (2, 102, 200.00, "2024-01-01 11:00:00")
], ["id", "user_id", "amount", "created_at"])
df1.writeTo("local.db.transactions").append()

# Insert v2
df2 = spark.createDataFrame([
    (3, 103, 300.00, "2024-01-01 12:00:00")
], ["id", "user_id", "amount", "created_at"])
df2.writeTo("local.db.transactions").append()

# View snapshots
snapshots = spark.sql("SELECT * FROM local.db.transactions.snapshots")
snapshots.show(truncate=False)

# Get first snapshot ID
snapshot_v1 = snapshots.orderBy("committed_at").first()["snapshot_id"]

# Query v1 (only 2 records)
df_v1 = spark.read.option("snapshot-id", snapshot_v1).table("local.db.transactions")
print(f"V1 count: {df_v1.count()}")  # 2

# Query current (3 records)
df_current = spark.table("local.db.transactions")
print(f"Current count: {df_current.count()}")  # 3

# Rollback to v1
spark.sql(f"""
    CALL local.system.rollback_to_snapshot('db.transactions', {snapshot_v1})
""")

# Verify rollback
df_after_rollback = spark.table("local.db.transactions")
print(f"After rollback: {df_after_rollback.count()}")  # 2
```

### Best Practices

1. **Retention Policy**: Keep snapshots for compliance period (7-90 days)
2. **Regular Cleanup**: Schedule snapshot expiration jobs
3. **Monitor Storage**: Track snapshot count and storage usage
4. **Document Rollbacks**: Log all rollback operations
5. **Test Recovery**: Regularly test time travel queries
6. **Incremental Processing**: Use incremental reads for efficiency

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Time Travel Queries
- Create table with multiple versions
- Query by snapshot ID
- Query by timestamp

### Exercise 2: Snapshot Analysis
- View snapshot metadata
- Analyze snapshot history
- Calculate data growth

### Exercise 3: Rollback Operations
- Perform rollback to previous snapshot
- Verify data after rollback
- Restore to specific timestamp

### Exercise 4: Snapshot Management
- Expire old snapshots
- Remove orphan files
- Optimize storage

### Exercise 5: Incremental Processing
- Read changes between snapshots
- Process incremental data
- Build change data capture (CDC)

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What creates a new snapshot in Iceberg?
2. How do you query data as of a specific timestamp?
3. What's the difference between rollback and set_current_snapshot?
4. Why should you expire old snapshots?
5. How do you read only changes between two snapshots?
6. What are orphan files?
7. What metadata does a snapshot contain?
8. How long should you retain snapshots?

---

## 🎯 Key Takeaways

- **Snapshots** - Immutable versions created on each write
- **Time travel** - Query any historical snapshot by ID or timestamp
- **Rollback** - Restore table to previous state
- **Expiration** - Remove old snapshots to save storage
- **Incremental reads** - Process only changes between snapshots
- **Orphan cleanup** - Remove unreferenced files
- **Audit trail** - Complete history for compliance
- **Data recovery** - Restore from accidental deletes/updates

---

## 📚 Additional Resources

- [Iceberg Time Travel](https://iceberg.apache.org/docs/latest/spark-queries/#time-travel)
- [Snapshot Management](https://iceberg.apache.org/docs/latest/maintenance/)
- [Incremental Reads](https://iceberg.apache.org/docs/latest/spark-structured-streaming/)

---

## Tomorrow: Day 11 - Delta Lake

We'll explore Delta Lake, another popular table format with time travel capabilities.
