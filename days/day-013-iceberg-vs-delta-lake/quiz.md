# Day 13 Quiz: Iceberg vs Delta Lake

## Questions

1. **What is the main architectural difference between Iceberg and Delta Lake?**
   - A) Iceberg uses Parquet, Delta uses Avro
   - B) Iceberg has 3-level metadata; Delta has flat transaction log
   - C) Iceberg is faster
   - D) Delta Lake doesn't support ACID

2. **Which format supports hidden partitioning?**
   - A) Both equally
   - B) Delta Lake only
   - C) Apache Iceberg only
   - D) Neither

3. **Which is better for multi-engine environments (Spark, Trino, Flink)?**
   - A) Delta Lake
   - B) Apache Iceberg
   - C) Both are equal
   - D) Neither supports multiple engines

4. **Can you easily convert between Iceberg and Delta Lake?**
   - A) Yes, with a simple command
   - B) No, requires data copy and metadata rebuild
   - C) Only Iceberg to Delta
   - D) Only Delta to Iceberg

5. **Which has a simpler metadata structure?**
   - A) Iceberg (3-level hierarchy)
   - B) Delta Lake (flat transaction log)
   - C) Both are equally complex
   - D) Neither has metadata

6. **Do both formats support ACID transactions?**
   - A) No, only Delta Lake
   - B) No, only Iceberg
   - C) Yes, both provide full ACID
   - D) Neither supports ACID

7. **Which format supports partition evolution without rewriting data?**
   - A) Both equally
   - B) Delta Lake only
   - C) Apache Iceberg only
   - D) Neither

8. **When should you choose Apache Iceberg?**
   - A) When using only Spark
   - B) When you need multi-engine support and partition evolution
   - C) When you want simpler architecture
   - D) When using Databricks

9. **What data formats does Delta Lake support?**
   - A) Parquet, ORC, Avro
   - B) Parquet only
   - C) Any format
   - D) JSON only

10. **Which format was created first?**
    - A) Delta Lake (2019)
    - B) Apache Iceberg (2017)
    - C) Both in 2017
    - D) Both in 2019

---

## Answers

1. **B** - Iceberg uses a 3-level metadata hierarchy (metadata file → manifest list → manifest files), while Delta Lake uses a flat transaction log with periodic checkpoints.

2. **C** - Apache Iceberg supports hidden partitioning where users don't need to specify partition filters. Delta Lake requires explicit partition awareness.

3. **B** - Apache Iceberg was designed for multi-engine support from the start. Delta Lake is Spark-centric with varying support for other engines.

4. **B** - No direct conversion exists. Migration requires exporting data and rebuilding metadata in the target format.

5. **B** - Delta Lake's flat transaction log with checkpoints is simpler than Iceberg's 3-level metadata hierarchy.

6. **C** - Both formats provide full ACID guarantees (Atomicity, Consistency, Isolation, Durability) with serializable isolation.

7. **C** - Only Apache Iceberg supports partition evolution (changing partition strategy) without rewriting existing data files.

8. **B** - Choose Iceberg for multi-engine environments, partition evolution needs, hidden partitioning, and vendor-neutral solutions.

9. **B** - Delta Lake stores data exclusively in Parquet format. Iceberg supports Parquet, ORC, and Avro.

10. **B** - Apache Iceberg was created by Netflix in 2017. Delta Lake was created by Databricks in 2019.

---

## Scoring

- **9-10 correct**: Excellent! You understand the key differences.
- **7-8 correct**: Good! Review architectural differences and use cases.
- **5-6 correct**: Fair. Revisit feature comparison and engine support.
- **Below 5**: Review the README and comparison tables again.
