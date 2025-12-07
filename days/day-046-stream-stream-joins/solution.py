"""
Day 46: Stream-to-Stream Joins - Solutions
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import time


def exercise_1_inner_join():
    """Inner join with time constraint"""
    spark = SparkSession.builder \
        .appName("InnerJoin") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Orders stream
    orders = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("amount", expr("50 + (value % 100)")) \
        .withWatermark("timestamp", "10 minutes") \
        .withColumnRenamed("timestamp", "order_time")
    
    # Payments stream
    payments = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("payment_method", expr("CASE WHEN value % 2 = 0 THEN 'credit' ELSE 'debit' END")) \
        .withWatermark("timestamp", "10 minutes") \
        .withColumnRenamed("timestamp", "payment_time")
    
    # Inner join with 15-minute time constraint
    joined = orders.join(
        payments,
        expr("""
            orders.order_id = payments.order_id AND
            payments.payment_time >= orders.order_time AND
            payments.payment_time <= orders.order_time + interval 15 minutes
        """),
        "inner"
    ).select(
        col("order_id"),
        col("amount"),
        col("payment_method"),
        col("order_time"),
        col("payment_time")
    )
    
    query = joined.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Inner join: Only matched orders and payments")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_2_left_outer_join():
    """Left outer join to find unpaid orders"""
    spark = SparkSession.builder \
        .appName("LeftOuterJoin") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Orders stream
    orders = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("amount", expr("50 + (value % 100)")) \
        .withWatermark("timestamp", "10 minutes") \
        .withColumnRenamed("timestamp", "order_time")
    
    # Payments stream (fewer events to create unpaid orders)
    payments = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 3) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("payment_id", expr("CONCAT('PAY-', value)")) \
        .withWatermark("timestamp", "10 minutes") \
        .withColumnRenamed("timestamp", "payment_time")
    
    # Left outer join
    left_joined = orders.join(
        payments,
        expr("""
            orders.order_id = payments.order_id AND
            payments.payment_time >= orders.order_time AND
            payments.payment_time <= orders.order_time + interval 15 minutes
        """),
        "left_outer"
    )
    
    # Filter for unpaid orders
    unpaid = left_joined.filter(col("payment_id").isNull()) \
        .select("order_id", "amount", "order_time")
    
    query = unpaid.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Left outer join: Finding unpaid orders")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_3_full_outer_join():
    """Full outer join for complete reconciliation"""
    spark = SparkSession.builder \
        .appName("FullOuterJoin") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Orders stream
    orders = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("amount", expr("50 + (value % 100)")) \
        .withWatermark("timestamp", "10 minutes") \
        .withColumnRenamed("timestamp", "order_time")
    
    # Payments stream
    payments = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("payment_id", expr("CONCAT('PAY-', value)")) \
        .withWatermark("timestamp", "10 minutes") \
        .withColumnRenamed("timestamp", "payment_time")
    
    # Full outer join
    full_joined = orders.join(
        payments,
        expr("""
            orders.order_id = payments.order_id AND
            payments.payment_time >= orders.order_time AND
            payments.payment_time <= orders.order_time + interval 15 minutes
        """),
        "full_outer"
    )
    
    # Tag records
    tagged = full_joined.withColumn(
        "status",
        when(col("order_time").isNotNull() & col("payment_time").isNotNull(), "matched")
        .when(col("order_time").isNotNull() & col("payment_time").isNull(), "order_only")
        .otherwise("payment_only")
    ).select("order_id", "amount", "payment_id", "status")
    
    query = tagged.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Full outer join: Complete reconciliation")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_4_multi_stream_join():
    """Join three streams"""
    spark = SparkSession.builder \
        .appName("MultiStreamJoin") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Orders stream
    orders = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("amount", expr("50 + (value % 100)")) \
        .withWatermark("timestamp", "10 minutes") \
        .withColumnRenamed("timestamp", "order_time")
    
    # Payments stream
    payments = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("payment_method", lit("credit")) \
        .withWatermark("timestamp", "10 minutes") \
        .withColumnRenamed("timestamp", "payment_time")
    
    # Shipments stream
    shipments = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("tracking_id", expr("CONCAT('TRACK-', value)")) \
        .withWatermark("timestamp", "10 minutes") \
        .withColumnRenamed("timestamp", "ship_time")
    
    # Join orders + payments
    orders_payments = orders.join(
        payments,
        expr("""
            orders.order_id = payments.order_id AND
            payments.payment_time >= orders.order_time AND
            payments.payment_time <= orders.order_time + interval 15 minutes
        """)
    )
    
    # Join with shipments
    full_join = orders_payments.join(
        shipments,
        expr("""
            orders_payments.order_id = shipments.order_id AND
            shipments.ship_time >= orders_payments.payment_time AND
            shipments.ship_time <= orders_payments.payment_time + interval 20 minutes
        """)
    ).select(
        col("order_id"),
        col("amount"),
        col("payment_method"),
        col("tracking_id"),
        col("order_time"),
        col("payment_time"),
        col("ship_time")
    )
    
    query = full_join.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Multi-stream join: Orders -> Payments -> Shipments")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_5_self_join_duplicates():
    """Self-join to detect duplicates"""
    spark = SparkSession.builder \
        .appName("SelfJoinDuplicates") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Events stream
    events = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("event_id", expr("CAST(value AS STRING)")) \
        .withColumn("user_id", expr("CAST(value % 5 AS STRING)")) \
        .withWatermark("timestamp", "5 minutes")
    
    # Self-join to find duplicates
    duplicates = events.alias("e1").join(
        events.alias("e2"),
        expr("""
            e1.user_id = e2.user_id AND
            e1.event_id != e2.event_id AND
            e2.timestamp >= e1.timestamp AND
            e2.timestamp <= e1.timestamp + interval 1 minute
        """)
    ).select(
        col("e1.event_id").alias("event1_id"),
        col("e2.event_id").alias("event2_id"),
        col("e1.user_id"),
        col("e1.timestamp").alias("time1"),
        col("e2.timestamp").alias("time2")
    )
    
    query = duplicates.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Self-join: Detecting duplicate events within 1 minute")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def demo_state_monitoring():
    """Monitor state size in joins"""
    spark = SparkSession.builder \
        .appName("StateMonitoring") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*60)
    print("State Monitoring Demo")
    print("="*60)
    
    # Create two streams
    stream1 = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("key", expr("CAST(value % 5 AS STRING)")) \
        .withWatermark("timestamp", "5 minutes")
    
    stream2 = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("key", expr("CAST(value % 5 AS STRING)")) \
        .withWatermark("timestamp", "5 minutes")
    
    # Join
    joined = stream1.alias("s1").join(
        stream2.alias("s2"),
        expr("""
            s1.key = s2.key AND
            s2.timestamp >= s1.timestamp AND
            s2.timestamp <= s1.timestamp + interval 10 minutes
        """)
    )
    
    query = joined.writeStream \
        .outputMode("append") \
        .format("memory") \
        .queryName("joined_data") \
        .trigger(processingTime="5 seconds") \
        .start()
    
    # Monitor state
    for i in range(5):
        time.sleep(5)
        progress = query.lastProgress
        if progress and 'stateOperators' in progress:
            print(f"\nBatch {i+1}:")
            for op in progress['stateOperators']:
                print(f"  State rows: {op.get('numRowsTotal', 0)}")
                print(f"  State memory: {op.get('memoryUsedBytes', 0)} bytes")
    
    query.stop()
    spark.stop()


if __name__ == "__main__":
    print("Day 46: Stream-to-Stream Joins Solutions\n")
    
    # Run all exercises
    exercise_1_inner_join()
    exercise_2_left_outer_join()
    exercise_3_full_outer_join()
    exercise_4_multi_stream_join()
    exercise_5_self_join_duplicates()
    
    # Demo
    demo_state_monitoring()
