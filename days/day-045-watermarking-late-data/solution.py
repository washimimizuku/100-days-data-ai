"""
Day 45: Watermarking and Late Data - Solutions
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import time


def exercise_1_basic_watermark():
    """Basic watermark implementation"""
    spark = SparkSession.builder \
        .appName("BasicWatermark") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load()
    
    # Add 5-minute watermark
    df_watermarked = df.withWatermark("timestamp", "5 minutes")
    
    # 10-minute tumbling window
    windowed = df_watermarked.groupBy(
        window("timestamp", "10 minutes")
    ).count()
    
    # Append mode - only finalized windows
    query = windowed.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    print("Watermark: 5 minutes")
    print("Window: 10 minutes")
    print("Mode: Append (only finalized windows)")
    
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_2_late_data_detection():
    """Detect late-arriving events"""
    spark = SparkSession.builder \
        .appName("LateDataDetection") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load()
    
    # Calculate lateness
    df_with_lateness = df.withColumn(
        "lateness_seconds",
        unix_timestamp(current_timestamp()) - unix_timestamp(col("timestamp"))
    )
    
    # Tag as late or on-time (30 second threshold)
    df_tagged = df_with_lateness.withColumn(
        "status",
        when(col("lateness_seconds") > 30, "late").otherwise("on-time")
    )
    
    # Count by status
    counts = df_tagged.groupBy("status").count()
    
    query = counts.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="5 seconds") \
        .start()
    
    print("Detecting late data (threshold: 30 seconds)")
    
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_3_watermark_aggregation():
    """Watermark with aggregation - compare modes"""
    spark = SparkSession.builder \
        .appName("WatermarkAggregation") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("product_id", expr("CAST(value % 3 AS STRING)")) \
        .withColumn("amount", expr("10 + (value % 100)"))
    
    # Add 10-minute watermark
    df_watermarked = df.withWatermark("timestamp", "10 minutes")
    
    # 5-minute windowed aggregation
    windowed = df_watermarked.groupBy(
        window("timestamp", "5 minutes"),
        "product_id"
    ).agg(
        count("*").alias("count"),
        sum("amount").alias("total")
    )
    
    # Append mode
    print("\n=== Append Mode ===")
    print("Only outputs finalized windows (once)")
    
    query_append = windowed.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    time.sleep(20)
    query_append.stop()
    
    # Update mode
    print("\n=== Update Mode ===")
    print("Outputs windows as they update (multiple times)")
    
    query_update = windowed.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    time.sleep(20)
    query_update.stop()
    
    spark.stop()


def exercise_4_state_management():
    """Monitor state with and without watermark"""
    spark = SparkSession.builder \
        .appName("StateManagement") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load()
    
    # WITHOUT watermark
    print("\n=== Without Watermark ===")
    print("State grows indefinitely")
    
    windowed_no_wm = df.groupBy(
        window("timestamp", "5 minutes")
    ).count()
    
    query_no_wm = windowed_no_wm.writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("no_watermark") \
        .trigger(processingTime="5 seconds") \
        .start()
    
    for i in range(3):
        time.sleep(5)
        progress = query_no_wm.lastProgress
        if progress and 'stateOperators' in progress:
            state_rows = progress['stateOperators'][0].get('numRowsTotal', 0)
            print(f"Batch {i+1}: State rows = {state_rows}")
    
    query_no_wm.stop()
    
    # WITH watermark
    print("\n=== With Watermark ===")
    print("State is bounded and cleaned up")
    
    df_watermarked = df.withWatermark("timestamp", "10 minutes")
    
    windowed_wm = df_watermarked.groupBy(
        window("timestamp", "5 minutes")
    ).count()
    
    query_wm = windowed_wm.writeStream \
        .outputMode("update") \
        .format("memory") \
        .queryName("with_watermark") \
        .trigger(processingTime="5 seconds") \
        .start()
    
    for i in range(3):
        time.sleep(5)
        progress = query_wm.lastProgress
        if progress and 'stateOperators' in progress:
            state_rows = progress['stateOperators'][0].get('numRowsTotal', 0)
            print(f"Batch {i+1}: State rows = {state_rows}")
    
    query_wm.stop()
    spark.stop()


def exercise_5_join_with_watermark():
    """Stream-to-stream join with watermarks"""
    spark = SparkSession.builder \
        .appName("JoinWithWatermark") \
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
        .withWatermark("timestamp", "5 minutes") \
        .withColumnRenamed("timestamp", "order_time")
    
    # Shipments stream
    shipments = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("order_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("tracking_id", expr("CONCAT('TRACK-', value)")) \
        .withWatermark("timestamp", "5 minutes") \
        .withColumnRenamed("timestamp", "ship_time")
    
    # Join with time constraint
    joined = orders.join(
        shipments,
        expr("""
            orders.order_id = shipments.order_id AND
            shipments.ship_time >= orders.order_time AND
            shipments.ship_time <= orders.order_time + interval 10 minutes
        """)
    ).select(
        col("order_id"),
        col("amount"),
        col("tracking_id"),
        col("order_time"),
        col("ship_time")
    )
    
    query = joined.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Stream-to-stream join with watermarks")
    print("Watermark: 5 minutes on both streams")
    print("Time constraint: shipment within 10 minutes of order")
    
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def demo_watermark_propagation():
    """Demonstrate watermark propagation"""
    spark = SparkSession.builder \
        .appName("WatermarkPropagation") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*60)
    print("Watermark Propagation Demo")
    print("="*60)
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load()
    
    # Add watermark
    df_watermarked = df.withWatermark("timestamp", "5 minutes")
    
    # Watermark propagates through transformations
    filtered = df_watermarked.filter(col("value") % 2 == 0)
    mapped = filtered.withColumn("doubled", col("value") * 2)
    
    # Still has watermark
    windowed = mapped.groupBy(
        window("timestamp", "10 minutes")
    ).count()
    
    query = windowed.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    print("Watermark propagates through filter and map operations")
    
    query.awaitTermination(timeout=20)
    query.stop()
    spark.stop()


def demo_late_data_side_output():
    """Separate processing for late data"""
    spark = SparkSession.builder \
        .appName("LateDataSideOutput") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*60)
    print("Late Data Side Output Demo")
    print("="*60)
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load()
    
    # Calculate lateness
    df_with_lateness = df.withColumn(
        "lateness_seconds",
        unix_timestamp(current_timestamp()) - unix_timestamp(col("timestamp"))
    )
    
    # Split into on-time and late
    on_time = df_with_lateness.filter(col("lateness_seconds") <= 30)
    late = df_with_lateness.filter(col("lateness_seconds") > 30)
    
    # Process on-time data
    on_time_agg = on_time.groupBy(
        window("timestamp", "10 minutes")
    ).count()
    
    query_on_time = on_time_agg.writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("on_time_data") \
        .start()
    
    # Monitor late data
    late_count = late.groupBy(
        window("timestamp", "10 minutes")
    ).count()
    
    query_late = late_count.writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("late_data") \
        .start()
    
    time.sleep(20)
    
    print("\nOn-time data:")
    spark.sql("SELECT * FROM on_time_data").show()
    
    print("\nLate data:")
    spark.sql("SELECT * FROM late_data").show()
    
    query_on_time.stop()
    query_late.stop()
    spark.stop()


if __name__ == "__main__":
    print("Day 45: Watermarking and Late Data Solutions\n")
    
    # Run all exercises
    exercise_1_basic_watermark()
    exercise_2_late_data_detection()
    exercise_3_watermark_aggregation()
    exercise_4_state_management()
    exercise_5_join_with_watermark()
    
    # Demos
    demo_watermark_propagation()
    demo_late_data_side_output()
