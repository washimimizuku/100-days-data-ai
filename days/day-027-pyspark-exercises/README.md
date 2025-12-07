# Day 27: PySpark Exercises

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Apply Spark concepts to real-world scenarios
- Practice data transformations and aggregations
- Implement joins and window functions
- Optimize queries with best practices
- Build end-to-end data pipelines
- Master common data engineering patterns

---

## Theory

### What is Practice Day?

**Practice Day** is dedicated to hands-on exercises that reinforce concepts from Days 22-26. Instead of learning new theory, you'll apply what you've learned to realistic scenarios.

**Why Practice Matters**:
- Reinforces theoretical knowledge
- Builds muscle memory for common patterns
- Exposes edge cases and gotchas
- Prepares you for real-world projects
- Identifies knowledge gaps

**Skills You'll Practice**:
- DataFrame operations (select, filter, transform)
- Aggregations (groupBy, window functions)
- Joins (broadcast, sort-merge, optimization)
- Performance tuning (caching, partitioning)
- Data quality (validation, cleaning)
- ETL patterns (extract, transform, load)

---

### Exercise Structure

Each exercise follows this pattern:

1. **Scenario**: Real-world business problem
2. **Data Description**: What the data looks like
3. **Tasks**: Specific steps to complete
4. **Hints**: Guidance on approach
5. **Expected Output**: What success looks like

**Time Allocation**:
- 10 exercises × 4 minutes each = 40 minutes
- Focus on completing tasks, not perfection
- Use hints if stuck
- Compare with solutions afterward

---

### Exercise 1: Data Cleaning Pipeline
Clean customer dataset with quality issues:
- Remove duplicate records by customer_id
- Fill null ages with median
- Fill null cities with "Unknown"
- Remove invalid email formats
- Filter out negative ages
- Standardize city names (title case)
- Count records before/after cleaning

---

### Exercise 2: Sales Analysis
Analyze quarterly sales data:
- Calculate total sales by region
- Find top 3 products by revenue
- Compute running total by region over time
- Calculate month-over-month growth rate
- Identify regions with declining sales
- Create summary report with all metrics

---

### Exercise 3: Customer Segmentation (RFM Analysis)
Segment customers using RFM analysis:
- Calculate Recency (days since last purchase)
- Calculate Frequency (number of orders)
- Calculate Monetary (total spend)
- Score each metric (1-5 scale)
- Create segments: Champions, Loyal, At Risk, Lost
- Count customers per segment
- Calculate average value per segment

---

### Exercise 4: Join Optimization
Optimize joins for large datasets:
- Join orders with customers (large-large)
- Join result with products (large-small)
- Use broadcast for products table
- Measure performance without broadcast
- Measure performance with broadcast
- Calculate performance improvement
- Handle null joins appropriately

---

### Exercise 5: Window Functions (Running Totals)
Calculate running totals and moving averages:
- Calculate running total by region
- Compute 7-day moving average
- Rank products by sales within region
- Calculate percent of total for each product

---

### Exercise 6: Data Quality Validation
Implement comprehensive data quality checks:
- Check null counts for each column
- Validate value ranges (age 0-120, amount > 0)
- Detect outliers using z-score
- Check for duplicate keys
- Validate foreign key relationships
- Generate quality report with pass/fail status

---

### Exercise 7: Performance Optimization Challenge
Optimize slow query (30 min → 5 min):
- Identify performance issues
- Combine multiple aggregations into one
- Cache intermediate results if reused
- Tune partition count
- Measure before/after performance

---

### Exercise 8: ETL Pipeline with Error Handling
Build production-ready ETL pipeline:
- Extract from CSV, JSON, and Parquet sources
- Validate data quality at each stage
- Transform with business rules
- Handle bad records (quarantine)
- Load to Delta Lake with partitioning
- Log metrics (records processed, errors, duration)

---

### Exercise 9: Time Series Features
Create time-based features for ML:
- Calculate lag features (previous 1, 7, 30 days)
- Compute rolling statistics (mean, std, min, max)
- Extract date features (day of week, month, quarter)
- Calculate time since last event
- Create binary flags (weekend, holiday, peak hour)

---

### Exercise 10: Pivot and Unpivot
Transform data between wide and long formats:
- Pivot sales data (region × product matrix)
- Calculate row and column totals
- Unpivot back to long format
- Create percentage of total calculations
- Generate summary statistics by dimension

---

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete all 10 exercises. Each exercise should take approximately 4 minutes.

**Exercise Files**:
- `exercise.py` - TODO-based exercises with starter code
- `solution.py` - Complete implementations for reference
- Sample data files in `/data/day-027/`

**Tips for Success**:
1. Read the scenario carefully
2. Break down tasks into small steps
3. Use hints if stuck
4. Test your code incrementally
5. Compare with solutions afterward
6. Focus on completing tasks, not perfection

**Common Patterns You'll Use**:
- `dropDuplicates()`, `fillna()`, `filter()`
- `groupBy()`, `agg()`, `sum()`, `count()`, `avg()`
- `Window.partitionBy()`, `orderBy()`, `rowsBetween()`
- `join()`, `broadcast()`, `cache()`
- `lag()`, `lead()`, `rank()`, `dense_rank()`
- `pivot()`, `when()`, `otherwise()`

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`:

1. How do you remove duplicate records in Spark?
2. What is a window function and when would you use it?
3. When should you use a broadcast join?
4. What are the different ways to handle null values?
5. What is RFM analysis and why is it useful?
6. How can you optimize joins in Spark?
7. What does the lag() function do?
8. How do you pivot data from long to wide format?
9. When should you cache a DataFrame?
10. What are key data quality checks to implement?

---

## 🎯 Key Takeaways

- **Data Cleaning** - Remove duplicates, handle nulls, validate formats
- **Aggregations** - Use groupBy() with multiple agg() functions
- **Window Functions** - Calculate running totals, moving averages, rankings
- **Join Optimization** - Broadcast small tables, minimize shuffles
- **RFM Analysis** - Segment customers by recency, frequency, monetary value
- **Performance** - Cache reused DataFrames, tune partitions, combine operations
- **Data Quality** - Validate nulls, ranges, duplicates, relationships
- **ETL Patterns** - Extract, transform, load with error handling
- **Time Series** - Create lag features, rolling statistics, date features
- **Pivot/Unpivot** - Transform between wide and long formats
- **Best Practices** - Filter early, broadcast small tables, avoid collect()
- **Monitoring** - Check Spark UI for bottlenecks and optimization opportunities

---

## 📚 Resources

- [PySpark API](https://spark.apache.org/docs/latest/api/python/)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)

---

## Tomorrow: Day 28 - Mini Project: Spark ETL

We'll build a complete ETL pipeline with Spark.
