"""
Day 44: Streaming Joins and Aggregations - Solutions
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import time
import os


def exercise_1_stream_static_join():
    """Stream-to-static enrichment"""
    spark = SparkSession.builder \
        .appName("StreamStaticJoin") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Create streaming orders
    orders = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value AS STRING)")) \
        .withColumn("product_id", expr("CAST(value % 5 AS STRING)")) \
        .withColumn("amount", expr("10 + (value % 100)"))
    
    # Create static products
    products_data = [
        ("0", "Laptop", "Electronics", 999.99),
        ("1", "Phone", "Electronics", 699.99),
        ("2", "Tablet", "Electronics", 499.99),
        ("3", "Monitor", "Electronics", 299.99),
        ("4", "Keyboard", "Accessories", 79.99)
    ]
    
    products_schema = StructType([
        StructField("product_id", StringType(), True),
        StructField("product_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", DoubleType(), True)
    ])
    
    products = spark.createDataFrame(products_data, products_schema)
    
    # Join with broadcast
    enriched = orders.join(
        broadcast(products),
        "product_id",
        "inner"
    ).select(
        "order_id",
        "product_name",
        "category",
        "amount",
        "price"
    )
    
    query = enriched.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Stream-to-Static Join Running...")
    query.awaitTermination(timeout=20)
    query.stop()
    spark.stop()


def exercise_2_stream_stream_join():
    """Stream-to-stream inner join"""
    spark = SparkSession.builder \
        .appName("StreamStreamJoin") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Orders stream
    orders = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 3) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("amount", expr("50 + (value % 100)")) \
        .withColumnRenamed("timestamp", "order_time")
    
    # Payments stream (slightly delayed)
    payments = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 3) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("payment_method", expr("CASE WHEN value % 2 = 0 THEN 'credit' ELSE 'debit' END")) \
        .withColumnRenamed("timestamp", "payment_time")
    
    # Join with time constraint
    joined = orders.join(
        payments,
        expr("""
            orders.order_id = payments.order_id AND
            payments.payment_time >= orders.order_time AND
            payments.payment_time <= orders.order_time + interval 1 minute
        """)
    ).select(
        "order_id",
        "amount",
        "payment_method",
        "order_time",
        "payment_time"
    )
    
    query = joined.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Stream-to-Stream Join Running...")
    query.awaitTermination(timeout=20)
    query.stop()
    spark.stop()


def exercise_3_windowed_aggregations():
    """Windowed aggregations"""
    spark = SparkSession.builder \
        .appName("WindowedAggregations") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Create stream with product data
    events = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("product_id", expr("CAST(value % 5 AS STRING)")) \
        .withColumn("amount", expr("10 + (value % 100)"))
    
    # 5-minute tumbling window aggregation
    windowed = events.groupBy(
        window("timestamp", "5 minutes"),
        "product_id"
    ).agg(
        count("*").alias("event_count"),
        sum("amount").alias("total_amount"),
        avg("amount").alias("avg_amount")
    )
    
    query = windowed.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    print("Windowed Aggregations Running...")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_4_multiple_aggregations():
    """Multiple aggregations"""
    spark = SparkSession.builder \
        .appName("MultipleAggregations") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Create stream with multiple dimensions
    events = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("product_id", expr("CAST(value % 3 AS STRING)")) \
        .withColumn("region", expr("CASE WHEN value % 2 = 0 THEN 'US' ELSE 'EU' END")) \
        .withColumn("amount", expr("10 + (value % 100)"))
    
    # Multiple aggregations
    stats = events.groupBy(
        window("timestamp", "10 minutes"),
        "product_id",
        "region"
    ).agg(
        count("*").alias("count"),
        sum("amount").alias("total"),
        avg("amount").alias("average"),
        min("amount").alias("min_amount"),
        max("amount").alias("max_amount"),
        stddev("amount").alias("stddev")
    )
    
    query = stats.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    print("Multiple Aggregations Running...")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_5_time_bounded_join():
    """Time-bounded join with watermarks"""
    spark = SparkSession.builder \
        .appName("TimeBoundedJoin") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # First stream with watermark
    stream1 = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("event_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("data1", expr("value * 2")) \
        .withWatermark("timestamp", "2 minutes")
    
    # Second stream with watermark
    stream2 = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("event_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("data2", expr("value * 3")) \
        .withWatermark("timestamp", "2 minutes")
    
    # Join with time constraint
    joined = stream1.alias("s1").join(
        stream2.alias("s2"),
        expr("""
            s1.event_id = s2.event_id AND
            s2.timestamp >= s1.timestamp AND
            s2.timestamp <= s1.timestamp + interval 2 minutes
        """)
    ).select(
        col("s1.event_id"),
        col("s1.timestamp").alias("time1"),
        col("s2.timestamp").alias("time2"),
        col("s1.data1"),
        col("s2.data2")
    )
    
    query = joined.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Time-Bounded Join Running...")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def demo_sliding_windows():
    """Demonstrate sliding windows"""
    spark = SparkSession.builder \
        .appName("SlidingWindows") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*60)
    print("Sliding Windows Demo")
    print("="*60)
    
    events = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("value_mod", expr("value % 3"))
    
    # 10-minute window, 5-minute slide
    sliding = events.groupBy(
        window("timestamp", "10 minutes", "5 minutes"),
        "value_mod"
    ).count()
    
    query = sliding.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    print("Each event appears in 2 overlapping windows")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def demo_conditional_aggregations():
    """Demonstrate conditional aggregations"""
    spark = SparkSession.builder \
        .appName("ConditionalAggregations") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*60)
    print("Conditional Aggregations Demo")
    print("="*60)
    
    events = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("amount", expr("10 + (value % 100)"))
    
    # Conditional aggregations
    conditional = events.groupBy(
        window("timestamp", "10 minutes")
    ).agg(
        sum(when(col("amount") > 60, 1).otherwise(0)).alias("high_value_count"),
        sum(when(col("amount") <= 60, 1).otherwise(0)).alias("low_value_count"),
        sum("amount").alias("total_amount"),
        avg("amount").alias("avg_amount")
    )
    
    query = conditional.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


if __name__ == "__main__":
    print("Day 44: Streaming Joins and Aggregations Solutions\n")
    
    # Exercise 1: Stream-to-static join
    print("\n" + "="*60)
    print("Exercise 1: Stream-to-Static Join")
    print("="*60)
    exercise_1_stream_static_join()
    
    # Exercise 2: Stream-to-stream join
    print("\n" + "="*60)
    print("Exercise 2: Stream-to-Stream Join")
    print("="*60)
    exercise_2_stream_stream_join()
    
    # Exercise 3: Windowed aggregations
    print("\n" + "="*60)
    print("Exercise 3: Windowed Aggregations")
    print("="*60)
    exercise_3_windowed_aggregations()
    
    # Exercise 4: Multiple aggregations
    print("\n" + "="*60)
    print("Exercise 4: Multiple Aggregations")
    print("="*60)
    exercise_4_multiple_aggregations()
    
    # Exercise 5: Time-bounded join
    print("\n" + "="*60)
    print("Exercise 5: Time-Bounded Join")
    print("="*60)
    exercise_5_time_bounded_join()
    
    # Demos
    demo_sliding_windows()
    demo_conditional_aggregations()
