# Day 25 Quiz: Spark Partitioning

## Questions

1. **What is partitioning in Spark?**
   - A) Storing data in multiple files
   - B) Dividing data into logical chunks distributed across executors
   - C) Compressing data
   - D) Sorting data

2. **What is the default partitioning strategy in Spark?**
   - A) Range partitioning
   - B) Hash partitioning
   - C) Random partitioning
   - D) Round-robin partitioning

3. **What's the difference between repartition() and coalesce()?**
   - A) No difference
   - B) repartition() can increase/decrease partitions with shuffle; coalesce() only decreases without shuffle
   - C) coalesce() is faster always
   - D) repartition() is deprecated

4. **When should you use repartition()?**
   - A) Always
   - B) When increasing partitions or need even distribution (accepts shuffle cost)
   - C) Never, use coalesce instead
   - D) Only for small datasets

5. **What is partition skew?**
   - A) Partitions stored on disk
   - B) Uneven data distribution causing some partitions much larger than others
   - C) Too many partitions
   - D) Corrupted partitions

6. **How do you fix partition skew?**
   - A) Use more executors
   - B) Repartition by different key, add salt to skewed keys, or use AQE
   - C) Increase memory
   - D) Use coalesce

7. **What is the optimal number of partitions?**
   - A) Always 200
   - B) 2-3x number of CPU cores, with 128 MB - 1 GB per partition
   - C) As many as possible
   - D) 1 partition per executor

8. **What is partition pruning?**
   - A) Removing empty partitions
   - B) Skipping irrelevant partitions during read based on filters
   - C) Compressing partitions
   - D) Sorting partitions

9. **How do you partition data when writing?**
   - A) `df.write.partition("column")`
   - B) `df.write.partitionBy("column").parquet("path")`
   - C) `df.partition().write()`
   - D) Not possible

10. **What happens with too many small partitions?**
    - A) Better performance
    - B) Task scheduling overhead exceeds processing time
    - C) More parallelism
    - D) Lower memory usage

---

## Answers

1. **B** - Partitioning divides data into logical chunks (partitions) distributed across executors for parallel processing. Each partition is processed independently.

2. **B** - Spark uses hash partitioning by default. Data is distributed based on hash of the key, ensuring same keys go to same partition.

3. **B** - repartition() can increase or decrease partitions and always causes shuffle for even distribution. coalesce() only decreases partitions and avoids shuffle by combining adjacent partitions (faster but may cause skew).

4. **B** - Use repartition() when you need to increase partitions or need even distribution across partitions. Accept the shuffle cost for better parallelism or to fix skew.

5. **B** - Partition skew occurs when data is unevenly distributed, causing some partitions to be much larger than others. This leads to slow tasks (stragglers) that delay the entire job.

6. **B** - Fix skew by: 1) Repartitioning by a different key with better distribution, 2) Adding salt (random suffix) to skewed keys to spread them across partitions, 3) Using Adaptive Query Execution (AQE) to automatically handle skew.

7. **B** - Optimal partition count is 2-3x the number of CPU cores in your cluster. Each partition should be 128 MB - 1 GB. Too few partitions = underutilized resources. Too many = scheduling overhead.

8. **B** - Partition pruning is an optimization where Spark skips reading irrelevant partitions based on filter predicates. For example, filtering `year=2024` only reads the 2024 partition directory.

9. **B** - Use `df.write.partitionBy("column").parquet("path")` to partition data when writing. This creates subdirectories for each partition value (e.g., `year=2024/month=01/`).

10. **B** - Too many small partitions cause task scheduling overhead to exceed actual processing time. Each task has ~10ms overhead. If tasks complete in <50ms, you're spending more time scheduling than processing.

---

## Scoring

- **9-10 correct**: Excellent! You understand partitioning deeply.
- **7-8 correct**: Good! Review repartition vs coalesce and skew handling.
- **5-6 correct**: Fair. Revisit optimal partition sizing and pruning.
- **Below 5**: Review the README and partitioning examples again.
