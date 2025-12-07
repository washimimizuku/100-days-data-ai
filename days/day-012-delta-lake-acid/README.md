# Day 12: Delta Lake ACID Transactions

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand ACID properties in Delta Lake
- Handle concurrent reads and writes
- Implement transaction isolation
- Resolve conflicts and handle failures
- Use optimistic concurrency control

---

## Theory

### ACID Properties

**A**tomicity - **C**onsistency - **I**solation - **D**urability

#### Atomicity
All operations in a transaction succeed or fail together.

```python
# Either all records are written or none
df.write.format("delta").mode("append").save(path)
# If this fails, no partial data is written
```

#### Consistency
Data remains in a valid state before and after transactions.

```python
# Schema enforcement ensures consistency
# Invalid data is rejected, maintaining data integrity
```

#### Isolation
Concurrent transactions don't interfere with each other.

```python
# Reader sees consistent snapshot while writer modifies data
# No dirty reads, phantom reads, or lost updates
```

#### Durability
Committed changes persist even after system failures.

```python
# Once transaction log is written, changes are permanent
# Recovery possible from transaction log
```

### Transaction Log

Delta Lake achieves ACID through a transaction log:

```
_delta_log/
├── 00000000000000000000.json  # Transaction 0
├── 00000000000000000001.json  # Transaction 1
├── 00000000000000000002.json  # Transaction 2
└── 00000000000000000010.checkpoint.parquet  # Checkpoint
```

**Each transaction file contains:**
- Add/Remove file operations
- Metadata changes
- Protocol version
- Commit timestamp

### Optimistic Concurrency Control

Delta Lake uses optimistic concurrency:

1. **Read** - Transaction reads current version
2. **Modify** - Transaction prepares changes
3. **Validate** - Check if base version changed
4. **Commit** - Write new transaction log entry

**If conflict detected**: Transaction retries or fails

### Concurrent Operations

#### Concurrent Reads
Multiple readers can query simultaneously without blocking:

```python
# Reader 1
df1 = spark.read.format("delta").load(path)

# Reader 2 (concurrent)
df2 = spark.read.format("delta").load(path)

# Both see consistent snapshot
```

#### Concurrent Writes
Multiple writers handled with conflict resolution:

```python
from delta.tables import DeltaTable
import threading

def writer1():
    DeltaTable.forPath(spark, path).update(
        condition = "id < 100",
        set = {"status": "processed"}
    )

def writer2():
    DeltaTable.forPath(spark, path).update(
        condition = "id >= 100",
        set = {"status": "processed"}
    )

# Both can succeed if no conflicts
t1 = threading.Thread(target=writer1)
t2 = threading.Thread(target=writer2)
t1.start()
t2.start()
```

### Conflict Resolution

**Compatible operations** (succeed):
- Appends to different partitions
- Updates to different rows
- Reads during writes

**Conflicting operations** (retry/fail):
- Concurrent schema changes
- Updates to same rows
- DELETE + UPDATE on same data

```python
from pyspark.sql.utils import AnalysisException

try:
    # This might conflict with concurrent operation
    delta_table.update(set = {"price": "price * 1.1"})
except AnalysisException as e:
    if "ConcurrentAppendException" in str(e):
        print("Conflict detected, retrying...")
        # Retry logic here
```

### Isolation Levels

Delta Lake provides **Serializable** isolation:

```python
# Writer commits transaction
df.write.format("delta").mode("append").save(path)

# Reader sees either:
# - All new data (if read after commit)
# - None of new data (if read before commit)
# Never partial data
```

### Transaction Guarantees

#### Write Guarantees

```python
# Atomic write - all or nothing
df = spark.createDataFrame([...])
df.write.format("delta").mode("append").save(path)
# Either all rows written or none
```

#### Read Guarantees

```python
# Consistent snapshot
df = spark.read.format("delta").load(path)
# Sees complete state at specific version
# Not affected by concurrent writes
```

### Handling Failures

#### Write Failure Recovery

```python
# If write fails, no partial data
try:
    df.write.format("delta").mode("append").save(path)
except Exception as e:
    print(f"Write failed: {e}")
    # Table remains in previous consistent state
    # No cleanup needed
```

#### Transaction Log Recovery

```python
# Delta Lake can recover from transaction log
from delta.tables import DeltaTable

# Repair table if needed
spark.sql(f"MSCK REPAIR TABLE delta.`{path}`")

# Or use DeltaTable API
DeltaTable.forPath(spark, path).generate("symlink_format_manifest")
```

### Real-World Example: Bank Transfers

```python
from delta.tables import DeltaTable
from pyspark.sql.functions import col

# Create accounts table
spark.sql("""
    CREATE TABLE accounts (
        account_id INT,
        balance DECIMAL(10,2)
    ) USING DELTA
""")

# Initial balances
accounts = spark.createDataFrame([
    (1, 1000.00),
    (2, 500.00)
], ["account_id", "balance"])
accounts.write.format("delta").mode("append").saveAsTable("accounts")

# Transfer $100 from account 1 to account 2
# ACID ensures both operations succeed or both fail
def transfer(from_id, to_id, amount):
    delta_table = DeltaTable.forName(spark, "accounts")
    
    # Check sufficient balance
    balance = spark.sql(f"SELECT balance FROM accounts WHERE account_id = {from_id}").first()[0]
    if balance < amount:
        raise ValueError("Insufficient funds")
    
    # Debit from account
    delta_table.update(
        condition = f"account_id = {from_id}",
        set = {"balance": f"balance - {amount}"}
    )
    
    # Credit to account
    delta_table.update(
        condition = f"account_id = {to_id}",
        set = {"balance": f"balance + {amount}"}
    )
    
    print(f"Transferred ${amount} from {from_id} to {to_id}")

# Execute transfer
transfer(1, 2, 100.00)

# Verify balances
spark.sql("SELECT * FROM accounts").show()
```

### Streaming with ACID

```python
# Streaming write with ACID guarantees
stream = spark.readStream.format("delta").load(source_path)

query = stream.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", checkpoint_path) \
    .start(target_path)

# Each micro-batch is atomic
# Exactly-once processing guaranteed
```

### Best Practices

1. **Partition wisely** - Reduces conflicts
2. **Batch operations** - Fewer transactions = less overhead
3. **Handle retries** - Implement retry logic for conflicts
4. **Monitor conflicts** - Track ConcurrentAppendException
5. **Use checkpoints** - For streaming exactly-once
6. **Avoid long transactions** - Increases conflict probability
7. **Test concurrency** - Simulate concurrent workloads

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Atomic Operations
- Create table and test atomic writes
- Simulate write failure
- Verify no partial data

### Exercise 2: Concurrent Reads
- Start multiple concurrent readers
- Verify consistent snapshots
- Check isolation

### Exercise 3: Concurrent Writes
- Implement concurrent updates
- Test non-conflicting operations
- Handle conflicts

### Exercise 4: Transaction Isolation
- Write data while reading
- Verify readers see consistent state
- Test serializable isolation

### Exercise 5: Failure Recovery
- Simulate transaction failures
- Verify table consistency
- Test recovery mechanisms

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What does ACID stand for?
2. How does Delta Lake achieve atomicity?
3. What is optimistic concurrency control?
4. What happens when two writers conflict?
5. What isolation level does Delta Lake provide?
6. Can readers block writers?
7. What is a transaction log checkpoint?
8. How does Delta Lake handle write failures?

---

## 🎯 Key Takeaways

- **ACID** - Atomicity, Consistency, Isolation, Durability
- **Transaction log** - Enables ACID guarantees
- **Optimistic concurrency** - Read-validate-commit pattern
- **Serializable isolation** - Strongest isolation level
- **Concurrent reads** - Never block, always consistent
- **Conflict resolution** - Automatic retry or fail
- **Atomic writes** - All or nothing, no partial data
- **Failure recovery** - Table remains consistent

---

## 📚 Additional Resources

- [Delta Lake Concurrency Control](https://docs.delta.io/latest/concurrency-control.html)
- [ACID Guarantees](https://docs.delta.io/latest/delta-batch.html#acid-guarantees)
- [Transaction Log Protocol](https://github.com/delta-io/delta/blob/master/PROTOCOL.md)

---

## Tomorrow: Day 13 - Iceberg vs Delta Lake

We'll compare Iceberg and Delta Lake features and use cases.
