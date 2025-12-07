# Day 27 Quiz: PySpark Exercises

## Questions

1. **How do you remove duplicate rows in PySpark?**
   - A) df.distinct()
   - B) df.dropDuplicates()
   - C) df.unique()
   - D) df.removeDuplicates()

2. **What is a window function used for?**
   - A) Opening new windows in the UI
   - B) Performing calculations across rows related to the current row
   - C) Filtering data
   - D) Joining tables

3. **When should you use a broadcast join?**
   - A) When both tables are large
   - B) When joining a large table with a small table (< 10 MB)
   - C) Never, it's deprecated
   - D) Only for outer joins

4. **How do you handle null values in PySpark?**
   - A) df.fillna() or df.dropna()
   - B) df.removeNulls()
   - C) df.cleanNulls()
   - D) Nulls cannot be handled

5. **What is RFM analysis?**
   - A) Random Forest Model
   - B) Recency, Frequency, Monetary - customer segmentation technique
   - C) Regression Fitting Method
   - D) Real-time Feature Monitoring

6. **How do you optimize joins in Spark?**
   - A) Use broadcast for small tables, partition on join keys
   - B) Always use sort-merge join
   - C) Avoid joins completely
   - D) Use nested loops

7. **What does the lag() function do?**
   - A) Delays query execution
   - B) Returns the value from a previous row in a window
   - C) Slows down performance
   - D) Creates a time delay

8. **How do you pivot data in PySpark?**
   - A) df.pivot("column")
   - B) df.groupBy("col1").pivot("col2").agg(sum("col3"))
   - C) df.transpose()
   - D) df.rotate()

9. **When should you cache a DataFrame?**
   - A) Always
   - B) When it's reused multiple times in your pipeline
   - C) Never
   - D) Only for small DataFrames

10. **How do you validate data quality in PySpark?**
    - A) Manual inspection only
    - B) Count nulls, check ranges, detect outliers, validate constraints
    - C) Data quality cannot be validated
    - D) Use only external tools

---

## Answers

1. **B** - `df.dropDuplicates()` removes duplicate rows. You can optionally specify columns: `df.dropDuplicates(["col1", "col2"])`.

2. **B** - Window functions perform calculations across rows related to the current row, such as running totals, rankings, and moving averages.

3. **B** - Broadcast joins are optimal when joining a large table with a small table (typically < 10 MB) to avoid shuffling the large table.

4. **A** - Use `df.fillna(value)` to replace nulls with a value, or `df.dropna()` to remove rows with nulls. Can specify columns and strategies.

5. **B** - RFM (Recency, Frequency, Monetary) is a customer segmentation technique that analyzes when customers last purchased, how often, and how much they spent.

6. **A** - Optimize joins by broadcasting small tables, partitioning on join keys, using appropriate join types, and handling data skew.

7. **B** - `lag(column, offset)` returns the value from a previous row in a window partition, useful for time series analysis and calculating differences.

8. **B** - Pivot syntax: `df.groupBy("row_col").pivot("pivot_col").agg(aggregation)` transforms rows into columns.

9. **B** - Cache DataFrames that are reused multiple times to avoid recomputation. Caching everything wastes memory.

10. **B** - Validate data quality by counting nulls, checking value ranges, detecting outliers (statistical methods), and validating business constraints.

---

## Scoring

- **10 correct**: PySpark Expert! 🚀
- **8-9 correct**: Strong practical skills
- **6-7 correct**: Good foundation, practice more
- **4-5 correct**: Review window functions and joins
- **0-3 correct**: Revisit exercises and solutions
