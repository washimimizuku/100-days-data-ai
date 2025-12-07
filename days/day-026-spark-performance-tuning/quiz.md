# Day 26 Quiz: Spark Performance Tuning

## Questions

1. **What is the Catalyst optimizer in Spark?**
   - A) A hardware accelerator for Spark
   - B) A query optimization engine that applies logical and physical optimizations
   - C) A caching mechanism
   - D) A serialization library

2. **When should you cache a DataFrame?**
   - A) Always, for all DataFrames
   - B) When the DataFrame is reused multiple times in your pipeline
   - C) Only for small DataFrames
   - D) Never, it slows down performance

3. **What is a broadcast join best used for?**
   - A) Joining two large tables
   - B) Joining a large table with a small table (< 10 MB)
   - C) Joining tables with skewed data
   - D) Avoiding all shuffle operations

4. **How can you reduce shuffle operations?**
   - A) Increase the number of partitions
   - B) Combine multiple operations into a single aggregation
   - C) Use more executors
   - D) Disable AQE

5. **What is Adaptive Query Execution (AQE)?**
   - A) A feature that dynamically optimizes queries at runtime
   - B) A manual tuning tool
   - C) A caching strategy
   - D) A join algorithm

6. **What is the recommended number of cores per executor?**
   - A) 1-2 cores
   - B) 4-5 cores (diminishing returns after 5)
   - C) 10+ cores for maximum parallelism
   - D) As many as available

7. **What is Kryo serialization?**
   - A) A slower but more compatible serializer
   - B) A faster, more compact serializer than Java serialization
   - C) A compression algorithm
   - D) A caching mechanism

8. **How do you monitor Spark performance?**
   - A) Only through application logs
   - B) Using the Spark UI to view jobs, stages, and executor metrics
   - C) By measuring wall-clock time only
   - D) Performance cannot be monitored

9. **What is the ideal partition size for Spark?**
   - A) 1-10 MB
   - B) 100-200 MB
   - C) 1-2 GB
   - D) As large as possible

10. **What happens when you call collect() on a large DataFrame?**
    - A) Data is processed in parallel efficiently
    - B) All data is brought to the driver, potentially causing OOM
    - C) Data is automatically cached
    - D) Nothing, it's a lazy operation

---

## Answers

1. **B** - Catalyst is Spark's query optimization engine that applies logical optimizations (predicate pushdown, column pruning) and physical optimizations (join reordering, code generation).

2. **B** - Cache DataFrames that are reused multiple times to avoid recomputation. Caching everything wastes memory, and caching unused DataFrames provides no benefit.

3. **B** - Broadcast joins are optimal when joining a large table with a small table (typically < 10 MB) to avoid shuffling the large table.

4. **B** - Combining operations (e.g., multiple aggregations in one groupBy) reduces the number of shuffle stages. More partitions or executors don't reduce shuffles.

5. **A** - AQE (Spark 3.0+) dynamically optimizes queries at runtime by coalescing partitions, handling skewed joins, and switching join strategies.

6. **B** - 4-5 cores per executor is recommended. More cores show diminishing returns and can cause resource contention.

7. **B** - Kryo is a faster, more compact serializer than Java's default serialization, providing ~10x performance improvement.

8. **B** - The Spark UI provides detailed metrics on jobs, stages, tasks, shuffles, storage, and executor utilization for performance analysis.

9. **B** - 100-200 MB per partition is ideal. Smaller partitions increase overhead; larger partitions reduce parallelism and can cause memory issues.

10. **B** - collect() brings all data to the driver node, which can cause out-of-memory errors for large DataFrames. Use distributed operations instead.

---

## Scoring

- **10 correct**: Spark Performance Expert! 🚀
- **8-9 correct**: Strong tuning skills
- **6-7 correct**: Good foundation, practice more
- **4-5 correct**: Review caching and optimization concepts
- **0-3 correct**: Revisit the material and examples
