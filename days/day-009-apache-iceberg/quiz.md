# Day 9 Quiz: Apache Iceberg

## Questions

1. **What is hidden partitioning in Apache Iceberg?**
   - A) Partitions are encrypted for security
   - B) Users query without specifying partition filters; Iceberg handles it automatically
   - C) Partitions are stored in a hidden directory
   - D) Partitioning is disabled by default

2. **What happens when you write data to an Iceberg table?**
   - A) Data is immediately committed to disk
   - B) A new immutable snapshot is created
   - C) Old data is overwritten
   - D) Metadata is deleted

3. **Can you change partitioning strategy without rewriting data in Iceberg?**
   - A) No, you must rewrite all data
   - B) Yes, through partition evolution
   - C) Only if the table is empty
   - D) Only for small tables

4. **Which query engines support Apache Iceberg?**
   - A) Only Apache Spark
   - B) Spark and Trino only
   - C) Spark, Trino, Flink, Presto, and others
   - D) Only Hadoop-based engines

5. **How do you query a specific snapshot in Iceberg using Spark?**
   - A) `spark.read.snapshot(123456).table("table")`
   - B) `spark.read.option("snapshot-id", 123456).table("table")`
   - C) `spark.table("table").version(123456)`
   - D) `spark.sql("SELECT * FROM table@123456")`

6. **What is partition evolution in Iceberg?**
   - A) Automatic partition creation
   - B) Ability to change partition strategy without rewriting data
   - C) Partitions that grow over time
   - D) Deleting old partitions

7. **Which metadata tables does Iceberg provide?**
   - A) Only snapshots
   - B) snapshots, files, history, manifests, partitions
   - C) Only files and partitions
   - D) No metadata tables available

8. **How do you expire old snapshots in Iceberg?**
   - A) `DELETE FROM table.snapshots`
   - B) `CALL system.expire_snapshots(table => 'table', older_than => TIMESTAMP)`
   - C) Snapshots expire automatically after 24 hours
   - D) `ALTER TABLE table DROP SNAPSHOTS`

9. **What layers make up Iceberg architecture?**
   - A) Data layer only
   - B) Metadata layer (JSON/Avro) and Data layer (Parquet/ORC/Avro)
   - C) Storage layer and compute layer
   - D) Schema layer and partition layer

10. **What is the main advantage of Iceberg's schema evolution?**
    - A) Faster queries
    - B) Add/rename/drop columns without breaking existing queries
    - C) Automatic data compression
    - D) Reduced storage costs

---

## Answers

1. **B** - Hidden partitioning means users query without specifying partition filters; Iceberg automatically handles partition pruning based on query predicates.

2. **B** - Each write creates a new immutable snapshot, enabling time travel and ACID transactions.

3. **B** - Partition evolution allows changing partition strategy (e.g., daily to monthly) without rewriting existing data.

4. **C** - Iceberg supports multiple engines: Spark, Trino, Flink, Presto, Hive, and more.

5. **B** - Use `.option("snapshot-id", 123456)` to query a specific snapshot.

6. **B** - Partition evolution lets you change how data is partitioned without rewriting existing files.

7. **B** - Iceberg provides snapshots, files, history, manifests, and partitions metadata tables.

8. **B** - Use `CALL system.expire_snapshots()` procedure to remove old snapshots.

9. **B** - Iceberg has a metadata layer (JSON/Avro files) and data layer (Parquet/ORC/Avro files).

10. **B** - Schema evolution allows safe column operations without breaking existing queries or requiring data rewrites.

---

## Scoring

- **9-10 correct**: Excellent! You understand Iceberg architecture.
- **7-8 correct**: Good! Review hidden partitioning and metadata tables.
- **5-6 correct**: Fair. Revisit snapshots and partition evolution.
- **Below 5**: Review the README and try the exercises again.
