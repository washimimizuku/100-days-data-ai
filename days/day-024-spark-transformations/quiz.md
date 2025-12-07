# Day 24 Quiz: Spark Transformations

## Questions

1. **What is a transformation in Spark?**
   - A) An action that triggers execution
   - B) A lazy operation that defines a new DataFrame without executing
   - C) A function that writes data to disk
   - D) A method to read data from files

2. **What is an action in Spark?**
   - A) A lazy operation
   - B) An eager operation that triggers execution and returns results
   - C) A transformation
   - D) A configuration setting

3. **What is a narrow transformation?**
   - A) A transformation that requires shuffle
   - B) Each input partition contributes to one output partition (no shuffle)
   - C) A transformation that reduces data size
   - D) A transformation that joins tables

4. **What is a wide transformation?**
   - A) A transformation without shuffle
   - B) Input partitions contribute to multiple output partitions (requires shuffle)
   - C) A transformation that increases data size
   - D) A transformation that filters data

5. **Which operations cause a shuffle?**
   - A) select, filter, map
   - B) groupBy, join, distinct, repartition, sortBy
   - C) withColumn, drop, cast
   - D) show, count, collect

6. **How do you optimize joins with small tables?**
   - A) Use more partitions
   - B) Use broadcast(df_small) to avoid shuffling large table
   - C) Use coalesce
   - D) Use cache on large table

7. **When should you use cache()?**
   - A) For all DataFrames
   - B) When a DataFrame is reused multiple times in the pipeline
   - C) Only for small DataFrames
   - D) Never, it slows performance

8. **What does explain() show?**
   - A) DataFrame schema
   - B) Physical and logical execution plan with shuffle points
   - C) Data statistics
   - D) Error messages

9. **Which transformations are faster?**
   - A) Narrow transformations (no shuffle, no network I/O)
   - B) Wide transformations
   - C) All transformations have same speed
   - D) Actions are faster

10. **What is the best practice for filters?**
    - A) Apply filters at the end
    - B) Apply filters early before expensive operations to reduce data volume
    - C) Never use filters
    - D) Use filters only with actions

---

## Answers

1. **B** - Transformations are lazy operations that define a new DataFrame without executing. They build up a logical execution plan that runs when an action is called.

2. **B** - Actions are eager operations that trigger execution of all transformations in the plan. Examples: show(), count(), collect(), write(). They return results to the driver or write to storage.

3. **B** - Narrow transformations process each partition independently without data movement across partitions. Examples: select, filter, map, withColumn. Each input partition contributes to exactly one output partition.

4. **B** - Wide transformations require shuffling data across partitions because input partitions contribute to multiple output partitions. Examples: groupBy, join, distinct, repartition. They involve network I/O.

5. **B** - Shuffle occurs with groupBy (aggregation), join (combining tables), distinct (deduplication), repartition (redistribution), and sortBy (ordering). These require data movement across executors.

6. **B** - Use broadcast(df_small) to send the small DataFrame to all executors, avoiding shuffle of the large table. Works best when small table is < 10 MB. Significantly improves join performance.

7. **B** - Cache DataFrames that are reused multiple times to avoid recomputation. Caching stores data in memory after first computation. Don't cache everything as it wastes memory.

8. **B** - explain() shows the physical and logical execution plan, including which operations cause shuffles (Exchange nodes). Use explain(True) for detailed plan with all optimization stages.

9. **A** - Narrow transformations are faster because they don't require shuffle or network I/O. Each partition is processed independently in parallel without data movement.

10. **B** - Filter early (predicate pushdown) to reduce data volume before expensive operations like joins or aggregations. This minimizes data processed and shuffled, improving performance significantly.

---

## Scoring

- **9-10 correct**: Excellent! You understand transformations deeply.
- **7-8 correct**: Good! Review narrow vs wide transformations.
- **5-6 correct**: Fair. Revisit shuffle operations and optimization.
- **Below 5**: Review the README and examples again.
