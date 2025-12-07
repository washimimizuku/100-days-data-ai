# Day 23 Quiz: Spark DataFrames

## Questions

1. **What is a DataFrame in Spark?**
   - A) A file format
   - B) A distributed collection of data organized into named columns
   - C) A database table
   - D) A Python library

2. **How do you create a DataFrame from a list?**
   - A) `spark.list(data)`
   - B) `spark.createDataFrame(data, schema)`
   - C) `DataFrame(data)`
   - D) `spark.read.list(data)`

3. **What's the difference between select and filter?**
   - A) No difference
   - B) select chooses columns; filter chooses rows
   - C) select is faster
   - D) filter is deprecated

4. **How do you add a new column?**
   - A) `df.addColumn()`
   - B) `df.withColumn("name", expression)`
   - C) `df.insert()`
   - D) `df.append()`

5. **What does groupBy do?**
   - A) Sorts data
   - B) Groups rows by column values for aggregation
   - C) Filters data
   - D) Joins tables

6. **How do you execute SQL on a DataFrame?**
   - A) `df.sql(query)`
   - B) `df.createOrReplaceTempView("name")` then `spark.sql(query)`
   - C) `df.query(sql)`
   - D) Not possible

7. **What is the difference between cache and persist?**
   - A) No difference
   - B) persist allows specifying storage level; cache uses default
   - C) cache is faster
   - D) persist is deprecated

8. **How do you join two DataFrames?**
   - A) `df1.merge(df2)`
   - B) `df1.join(df2, "key")`
   - C) `df1.concat(df2)`
   - D) `df1.union(df2)`

9. **What does show() do?**
   - A) Saves DataFrame
   - B) Displays DataFrame rows in console
   - C) Counts rows
   - D) Exports data

10. **Which is a transformation?**
    - A) show()
    - B) count()
    - C) filter()
    - D) collect()

---

## Answers

1. **B** - DataFrame is a distributed collection of data organized into named columns with schema.
2. **B** - Use `spark.createDataFrame(data, schema)` to create from list.
3. **B** - select chooses which columns to keep; filter chooses which rows to keep.
4. **B** - Use `withColumn("name", expression)` to add or modify columns.
5. **B** - groupBy groups rows by column values, typically followed by aggregation.
6. **B** - Register DataFrame as temp view, then use `spark.sql(query)`.
7. **B** - persist allows specifying storage level (MEMORY_ONLY, DISK_ONLY, etc.); cache uses MEMORY_ONLY.
8. **B** - Use `df1.join(df2, "key")` or `df1.join(df2, condition)`.
9. **B** - show() displays DataFrame rows in console (default 20 rows).
10. **C** - filter() is a transformation (lazy); show(), count(), collect() are actions.

---

## Scoring

- **9-10 correct**: Excellent! You understand DataFrames.
- **7-8 correct**: Good! Review transformations and SQL.
- **5-6 correct**: Fair. Revisit basic operations.
- **Below 5**: Review the README and examples again.
