# Day 8 Quiz: Table Formats Introduction

## Questions

1. **What is the difference between file format and table format?**
   - A) They are the same thing
   - B) File formats (Parquet, Avro) define individual files; table formats (Iceberg, Delta) manage collections with ACID
   - C) Table formats are older
   - D) File formats are faster

2. **What does ACID stand for?**
   - A) Automatic, Complete, Independent, Durable
   - B) Atomicity, Consistency, Isolation, Durability
   - C) Available, Consistent, Isolated, Distributed
   - D) Atomic, Centralized, Integrated, Distributed

3. **Which table format has hidden partitioning?**
   - A) Delta Lake
   - B) Apache Hudi
   - C) Apache Iceberg with automatic partition handling
   - D) All table formats

4. **Which table format is best for CDC workloads?**
   - A) Delta Lake
   - B) Apache Iceberg
   - C) Apache Hudi optimized for updates, deletes, upserts
   - D) Parquet

5. **What is time travel?**
   - A) A backup feature
   - B) Ability to query historical versions of a table
   - C) A performance optimization
   - D) A security feature

6. **Can you use Iceberg with Trino?**
   - A) No, only with Spark
   - B) Yes, Iceberg has excellent multi-engine support
   - C) Only with special plugins
   - D) Only in cloud environments

7. **Which format is optimized for Databricks?**
   - A) Apache Iceberg
   - B) Apache Hudi
   - C) Delta Lake created by Databricks
   - D) Parquet

8. **What problem do table formats solve?**
   - A) Storage capacity
   - B) Managing file collections as tables with ACID, preventing partial writes and conflicts
   - C) Network speed
   - D) Data compression

9. **What is partition evolution?**
   - A) Automatic partition creation
   - B) Ability to change partition scheme without rewriting data (Iceberg only)
   - C) Deleting old partitions
   - D) Partition compression

10. **Why do we need ACID transactions in data lakes?**
    - A) For faster queries
    - B) Ensure consistency, prevent partial writes, enable concurrent operations
    - C) To reduce storage costs
    - D) For better compression

---

## Answers

1. **B** - File formats (Parquet, Avro) define how individual files are stored. Table formats (Iceberg, Delta, Hudi) define how collections of files work together as a table with ACID transactions, schema evolution, and metadata management.
2. **B** - ACID stands for Atomicity, Consistency, Isolation, Durability - properties that guarantee reliable database transactions.
3. **C** - Apache Iceberg has hidden partitioning, which means users don't need to specify partition filters in queries - the system handles it automatically.
4. **C** - Apache Hudi is best for CDC (Change Data Capture) workloads because it's optimized for frequent updates, deletes, and upserts with excellent incremental processing.
5. **B** - Time travel is the ability to query historical versions of a table. Each write creates a snapshot, and you can read data as it existed at any previous point in time.
6. **B** - Yes, Iceberg has excellent multi-engine support and works seamlessly with Trino, Spark, Flink, Presto, and other query engines.
7. **C** - Delta Lake is optimized for Databricks (created by Databricks) and provides the best performance and integration on that platform.
8. **B** - Table formats solve the problem of managing collections of files as tables with ACID transactions, preventing issues like partial writes, concurrent write conflicts, and lack of schema evolution.
9. **B** - Partition evolution is the ability to change how a table is partitioned without rewriting all data. Only Iceberg supports this feature, allowing you to change partition schemes as data patterns evolve.
10. **B** - ACID transactions ensure data consistency and reliability in data lakes, preventing partial writes, enabling concurrent operations, and guaranteeing that committed data is durable and consistent.

---

## Scoring

- **9-10 correct**: Excellent! You understand table formats well.
- **7-8 correct**: Good! Review the areas you missed.
- **5-6 correct**: Fair. Re-read the theory section.
- **Below 5**: Review the material and try again.
