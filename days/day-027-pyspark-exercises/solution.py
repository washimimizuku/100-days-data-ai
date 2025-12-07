"""
Day 27: PySpark Exercises - Solutions
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as _sum, avg, max as _max, min as _min
from pyspark.sql.functions import when, lag, lead, row_number, broadcast, window, datediff, current_date
from pyspark.sql.window import Window
from datetime import datetime, timedelta

spark = SparkSession.builder.appName("Day27-Solutions").getOrCreate()

def exercise_1():
    """Clean customer data"""
    data = [
        (1, "Alice", 25, "NYC"),
        (1, "Alice", 25, "NYC"),  # Duplicate
        (2, "Bob", None, "LA"),
        (3, "Charlie", -5, None),
        (4, "David", 150, "Chicago")
    ]
    df = spark.createDataFrame(data, ["customer_id", "name", "age", "city"])
    
    df_clean = df.dropDuplicates(["customer_id"]) \
        .fillna({"age": 0, "city": "Unknown"}) \
        .filter((col("age") >= 0) & (col("age") <= 120))
    
    df_clean.show()

def exercise_2():
    """Analyze sales by region and product"""
    data = [
        ("East", "A", "2024-01-01", 100),
        ("East", "B", "2024-01-02", 150),
        ("West", "A", "2024-01-01", 200),
        ("West", "B", "2024-01-02", 120)
    ]
    df = spark.createDataFrame(data, ["region", "product", "date", "sales"])
    
    # Total by region
    df.groupBy("region").agg(_sum("sales").alias("total")).show()
    
    # Top 3 products
    df.groupBy("product").agg(_sum("sales").alias("total")).orderBy(col("total").desc()).limit(3).show()
    
    # Running total
    window = Window.partitionBy("region").orderBy("date")
    df.withColumn("running_total", _sum("sales").over(window)).show()

def exercise_3():
    """Create RFM customer segments"""
    data = [
        (1, "2024-01-15", 100),
        (1, "2024-02-01", 150),
        (2, "2024-01-10", 200),
        (3, "2024-02-20", 50)
    ]
    df = spark.createDataFrame(data, ["customer_id", "order_date", "amount"])
    
    rfm = df.groupBy("customer_id").agg(
        datediff(current_date(), _max("order_date")).alias("recency"),
        count("*").alias("frequency"),
        _sum("amount").alias("monetary")
    )
    
    rfm.withColumn("segment", 
        when((col("frequency") > 1) & (col("monetary") > 150), "High Value")
        .otherwise("Regular")
    ).show()

def exercise_4():
    """Join multiple datasets efficiently"""
    customers = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "name"])
    orders = spark.createDataFrame([(1, 1, 100), (2, 2, 200)], ["order_id", "customer_id", "amount"])
    products = spark.createDataFrame([(1, "Widget"), (2, "Gadget")], ["id", "product"])
    
    # Regular join
    result = orders.join(customers, orders.customer_id == customers.id) \
        .join(products, orders.order_id == products.id)
    
    # Broadcast join (for small tables)
    result_opt = orders.join(broadcast(customers), orders.customer_id == customers.id)
    
    result_opt.show()

def exercise_5():
    """Calculate streaming metrics"""
    data = [
        ("2024-01-01 10:00:00", 100),
        ("2024-01-01 10:30:00", 150),
        ("2024-01-01 11:00:00", 200),
        ("2024-01-01 11:30:00", 120)
    ]
    df = spark.createDataFrame(data, ["timestamp", "value"])
    df = df.withColumn("timestamp", col("timestamp").cast("timestamp"))
    
    # Hourly aggregation
    df.groupBy(window("timestamp", "1 hour")).agg(
        count("*").alias("events"),
        avg("value").alias("avg_value")
    ).show(truncate=False)

def exercise_6():
    """Validate data quality"""
    data = [(1, 25, 100), (2, None, 200), (3, 30, None), (4, 150, 50)]
    df = spark.createDataFrame(data, ["id", "age", "amount"])
    
    # Null counts
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    
    # Range validation
    df.select(
        count(when((col("age") < 0) | (col("age") > 120), "age")).alias("invalid_age")
    ).show()

def exercise_7():
    """Optimize slow query"""
    import time
    
    df = spark.range(1000000).withColumn("value", col("id") % 100)
    
    # Without optimization
    start = time.time()
    df.groupBy("value").count().collect()
    no_opt = time.time() - start
    
    # With optimization
    df_cached = df.repartition(50, "value").cache()
    df_cached.count()
    start = time.time()
    df_cached.groupBy("value").count().collect()
    opt = time.time() - start
    
    print(f"Without optimization: {no_opt:.2f}s")
    print(f"With optimization: {opt:.2f}s")
    df_cached.unpersist()

def exercise_8():
    """Build ETL pipeline"""
    # Extract
    df = spark.range(100).withColumn("value", col("id") * 2)
    
    # Transform
    df_clean = df.filter(col("value") > 0)
    df_enriched = df_clean.withColumn("category", when(col("value") > 100, "High").otherwise("Low"))
    df_agg = df_enriched.groupBy("category").agg(count("*").alias("count"))
    
    # Load
    df_agg.write.mode("overwrite").parquet("/tmp/etl_output")
    print("ETL pipeline completed")

def exercise_9():
    """Analyze time series data"""
    data = [(1, 100), (2, 150), (3, 200), (4, 120), (5, 180)]
    df = spark.createDataFrame(data, ["day", "value"])
    
    window = Window.orderBy("day")
    df.withColumn("prev_value", lag("value", 1).over(window)) \
      .withColumn("next_value", lead("value", 1).over(window)) \
      .show()

def exercise_10():
    """Multi-dimensional aggregations"""
    data = [
        ("East", "A", 100),
        ("East", "B", 150),
        ("West", "A", 200),
        ("West", "B", 120)
    ]
    df = spark.createDataFrame(data, ["region", "product", "sales"])
    
    # Multi-dimensional aggregation
    df.groupBy("region", "product").agg(
        _sum("sales").alias("total"),
        avg("sales").alias("average")
    ).show()
    
    # Pivot
    df.groupBy("region").pivot("product").agg(_sum("sales")).show()

if __name__ == "__main__":
    print("Day 27: PySpark Exercises - Solutions\n")
    
    print("Exercise 1: Data Cleaning")
    exercise_1()
    
    print("\nExercise 2: Sales Analysis")
    exercise_2()
    
    print("\nExercise 3: Customer Segmentation")
    exercise_3()
    
    print("\nExercise 4: Join Optimization")
    exercise_4()
    
    print("\nExercise 5: Real-Time Metrics")
    exercise_5()
    
    print("\nExercise 6: Data Quality")
    exercise_6()
    
    print("\nExercise 7: Performance Tuning")
    exercise_7()
    
    print("\nExercise 8: ETL Pipeline")
    exercise_8()
    
    print("\nExercise 9: Time Series")
    exercise_9()
    
    print("\nExercise 10: Complex Aggregations")
    exercise_10()
    
    spark.stop()
