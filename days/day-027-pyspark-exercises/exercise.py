"""
Day 27: PySpark Exercises
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as _sum, avg, max as _max, min as _min
from pyspark.sql.functions import when, lag, lead, row_number, rank, dense_rank
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("Day27").getOrCreate()

# Exercise 1: Data Cleaning
def exercise_1():
    """Clean customer data"""
    # TODO: Create sample data with nulls and duplicates
    # TODO: Remove duplicates by customer_id
    # TODO: Fill nulls: age=0, city="Unknown"
    # TODO: Filter invalid ages (< 0 or > 120)
    pass

# Exercise 2: Sales Analysis
def exercise_2():
    """Analyze sales by region and product"""
    # TODO: Create sales data
    # TODO: Calculate total sales per region
    # TODO: Find top 3 products by sales
    # TODO: Compute running total by date
    pass

# Exercise 3: Customer Segmentation
def exercise_3():
    """Create RFM customer segments"""
    # TODO: Create order data
    # TODO: Calculate Recency (days since last order)
    # TODO: Calculate Frequency (number of orders)
    # TODO: Calculate Monetary (total spent)
    # TODO: Create segments based on RFM scores
    pass

# Exercise 4: Join Optimization
def exercise_4():
    """Join multiple datasets efficiently"""
    # TODO: Create customers, orders, products tables
    # TODO: Join all three tables
    # TODO: Use broadcast for small tables
    # TODO: Compare performance with/without broadcast
    pass

# Exercise 5: Real-Time Metrics
def exercise_5():
    """Calculate streaming metrics"""
    # TODO: Create event data with timestamps
    # TODO: Group by 1-hour windows
    # TODO: Calculate count and average per window
    # TODO: Compute moving average (3-hour window)
    pass

# Exercise 6: Data Quality Checks
def exercise_6():
    """Validate data quality"""
    # TODO: Create data with quality issues
    # TODO: Count nulls per column
    # TODO: Check value ranges
    # TODO: Detect outliers (> 3 std dev)
    # TODO: Generate quality report
    pass

# Exercise 7: Performance Tuning
def exercise_7():
    """Optimize slow query"""
    # TODO: Create large dataset
    # TODO: Run query without optimization
    # TODO: Apply caching
    # TODO: Tune partitions
    # TODO: Measure time difference
    pass

# Exercise 8: ETL Pipeline
def exercise_8():
    """Build ETL pipeline"""
    # TODO: Extract from multiple sources
    # TODO: Transform: clean, enrich, aggregate
    # TODO: Load to parquet
    # TODO: Add error handling
    pass

# Exercise 9: Time Series Analysis
def exercise_9():
    """Analyze time series data"""
    # TODO: Create time series data
    # TODO: Calculate lag features (previous value)
    # TODO: Calculate lead features (next value)
    # TODO: Compute rolling average (7-day window)
    pass

# Exercise 10: Complex Aggregations
def exercise_10():
    """Multi-dimensional aggregations"""
    # TODO: Create sales data with region, product, date
    # TODO: Group by region and product
    # TODO: Calculate sum, avg, min, max
    # TODO: Pivot by product
    # TODO: Calculate percentiles
    pass

if __name__ == "__main__":
    print("Day 27: PySpark Exercises")
    print("\nUncomment exercises to run:")
    print("# exercise_1()")
    print("# exercise_2()")
    print("# exercise_3()")
    print("# exercise_4()")
    print("# exercise_5()")
    print("# exercise_6()")
    print("# exercise_7()")
    print("# exercise_8()")
    print("# exercise_9()")
    print("# exercise_10()")
