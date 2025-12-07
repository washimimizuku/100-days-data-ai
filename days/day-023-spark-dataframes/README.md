# Day 23: Spark DataFrames

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand DataFrame API basics
- Create DataFrames from various sources
- Perform selections, filters, and transformations
- Use column operations and expressions
- Execute SQL queries on DataFrames

---

## Theory

### What are DataFrames?

A **distributed collection of data organized into named columns**, similar to a table in a database or pandas DataFrame.

**Benefits over RDDs**:
- Schema enforcement
- Catalyst optimizer
- Easier to use
- Better performance
- Language-agnostic

---

### Creating DataFrames

#### From Files

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DataFrames").getOrCreate()

# CSV
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# JSON
df = spark.read.json("data.json")

# Parquet
df = spark.read.parquet("data.parquet")

# Delta
df = spark.read.format("delta").load("path/to/delta")
```

#### From Data

```python
# From list
data = [(1, "Alice", 25), (2, "Bob", 30)]
df = spark.createDataFrame(data, ["id", "name", "age"])

# From dict
data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
df = spark.createDataFrame(data)

# From pandas
import pandas as pd
pdf = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
df = spark.createDataFrame(pdf)
```

#### From SQL

```python
df.createOrReplaceTempView("users")
df2 = spark.sql("SELECT * FROM users WHERE age > 25")
```

---

### Basic Operations

#### Show Data

```python
df.show()           # Show 20 rows
df.show(5)          # Show 5 rows
df.show(truncate=False)  # Don't truncate columns
```

#### Schema

```python
df.printSchema()    # Print schema tree
df.schema           # Get schema object
df.columns          # Get column names
df.dtypes           # Get column types
```

#### Info

```python
df.count()          # Row count
df.describe().show()  # Summary statistics
df.summary().show()   # Extended statistics
```

---

### Selecting Columns

```python
from pyspark.sql.functions import col

# Select single column
df.select("name").show()
df.select(col("name")).show()

# Select multiple columns
df.select("name", "age").show()
df.select(col("name"), col("age")).show()

# Select with expressions
df.select(col("name"), (col("age") + 1).alias("next_age")).show()

# Select all columns
df.select("*").show()
```

---

### Filtering Rows

```python
# Simple filter
df.filter(col("age") > 25).show()
df.where(col("age") > 25).show()  # Same as filter

# Multiple conditions
df.filter((col("age") > 25) & (col("name") == "Alice")).show()
df.filter((col("age") < 20) | (col("age") > 60)).show()

# String operations
df.filter(col("name").startswith("A")).show()
df.filter(col("name").contains("li")).show()

# Null checks
df.filter(col("age").isNotNull()).show()
df.filter(col("age").isNull()).show()
```

---

### Column Operations

```python
from pyspark.sql.functions import *

# Add column
df = df.withColumn("age_plus_10", col("age") + 10)

# Rename column
df = df.withColumnRenamed("age", "years")

# Drop column
df = df.drop("age")

# Cast type
df = df.withColumn("age", col("age").cast("integer"))

# Conditional column
df = df.withColumn("category", 
    when(col("age") < 18, "minor")
    .when(col("age") < 65, "adult")
    .otherwise("senior")
)
```

---

### Aggregations

```python
# Simple aggregations
df.agg(count("*")).show()
df.agg(sum("age"), avg("age"), max("age"), min("age")).show()

# Group by
df.groupBy("city").count().show()
df.groupBy("city").agg(avg("age"), max("age")).show()

# Multiple groups
df.groupBy("city", "gender").count().show()
```

---

### Sorting

```python
# Ascending
df.orderBy("age").show()
df.sort("age").show()  # Same as orderBy

# Descending
df.orderBy(col("age").desc()).show()
df.orderBy(desc("age")).show()

# Multiple columns
df.orderBy("city", desc("age")).show()
```

---

### Joins

```python
# Inner join
df1.join(df2, "id").show()
df1.join(df2, df1.id == df2.id).show()

# Left join
df1.join(df2, "id", "left").show()

# Other joins
df1.join(df2, "id", "right").show()
df1.join(df2, "id", "outer").show()
df1.join(df2, "id", "left_anti").show()  # Left anti join
```

---

### SQL Queries

```python
# Register as temp view
df.createOrReplaceTempView("users")

# Run SQL
result = spark.sql("""
    SELECT city, AVG(age) as avg_age
    FROM users
    WHERE age > 18
    GROUP BY city
    ORDER BY avg_age DESC
""")
result.show()

# Global temp view (across sessions)
df.createGlobalTempView("global_users")
spark.sql("SELECT * FROM global_temp.global_users").show()
```

---

### Common Functions

```python
from pyspark.sql.functions import *

# String: upper, lower, length, concat
# Date: current_date, year, month, dayofmonth
# Math: round, ceil, floor
# Null: coalesce, df.na.drop(), df.na.fill(0)
```

---

### Window Functions

```python
from pyspark.sql.window import Window

# Define window
window = Window.partitionBy("city").orderBy("age")

# Ranking
df.withColumn("rank", rank().over(window)).show()
df.withColumn("row_num", row_number().over(window)).show()

# Aggregations
df.withColumn("avg_age", avg("age").over(window)).show()
```

---

### Writing DataFrames

```python
# CSV
df.write.csv("output.csv", header=True, mode="overwrite")

# Parquet
df.write.parquet("output.parquet", mode="overwrite")

# Delta
df.write.format("delta").mode("overwrite").save("output")

# Partitioned
df.write.partitionBy("year", "month").parquet("output")

# Modes: overwrite, append, ignore, error
```

---

### Performance Tips

```python
# Cache frequently used DataFrames
df.cache()
df.persist()

# Repartition for better parallelism
df = df.repartition(10)

# Coalesce to reduce partitions
df = df.coalesce(1)

# Broadcast small DataFrames in joins
from pyspark.sql.functions import broadcast
df1.join(broadcast(df2), "id").show()
```

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Create DataFrames
- Create from list, dict, pandas
- Read from CSV/JSON
- Print schema

### Exercise 2: Select and Filter
- Select specific columns
- Filter with conditions
- Chain operations

### Exercise 3: Transformations
- Add/rename/drop columns
- Use when/otherwise
- Apply functions

### Exercise 4: Aggregations
- Group by and aggregate
- Use window functions
- Calculate statistics

### Exercise 5: SQL Queries
- Register temp view
- Write SQL queries
- Join tables

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is a DataFrame?
2. How to create DataFrame from list?
3. What's the difference between select and filter?
4. How to add a new column?
5. What does groupBy do?
6. How to execute SQL on DataFrame?
7. What is the difference between cache and persist?
8. How to join two DataFrames?

---

## 🎯 Key Takeaways

- **DataFrame** - Distributed table with schema
- **Select** - Choose columns
- **Filter/Where** - Filter rows
- **withColumn** - Add/modify columns
- **groupBy** - Aggregate data
- **SQL** - Query with SQL syntax
- **Joins** - Combine DataFrames
- **Cache** - Store in memory for reuse

---

## 📚 Additional Resources

- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [DataFrame API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)
- [SQL Functions](https://spark.apache.org/docs/latest/api/sql/index.html)

---

## Tomorrow: Day 24 - Spark Transformations

We'll explore narrow vs wide transformations and optimization.
