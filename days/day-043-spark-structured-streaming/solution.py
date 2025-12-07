"""
Day 43: Spark Structured Streaming - Solutions
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, window, count, avg, current_timestamp, expr
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
import time
import os
import shutil


def exercise_1_socket_word_count():
    """Socket stream word count"""
    spark = SparkSession.builder \
        .appName("SocketWordCount") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    lines = spark.readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()
    
    words = lines.select(
        explode(split(col("value"), " ")).alias("word")
    )
    
    word_counts = words.groupBy("word").count()
    
    query = word_counts.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Listening on localhost:9999")
    print("Type words in socket terminal (nc -lk 9999)")
    
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_2_file_stream():
    """File stream processing"""
    spark = SparkSession.builder \
        .appName("FileStream") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    schema = StructType([
        StructField("timestamp", TimestampType(), True),
        StructField("user_id", StringType(), True),
        StructField("action", StringType(), True),
        StructField("amount", DoubleType(), True)
    ])
    
    os.makedirs("data/input", exist_ok=True)
    
    df = spark.readStream \
        .format("csv") \
        .schema(schema) \
        .option("header", True) \
        .load("data/input/")
    
    purchases = df.filter(col("action") == "purchase")
    
    user_totals = purchases.groupBy("user_id") \
        .agg(
            count("*").alias("purchase_count"),
            avg("amount").alias("avg_amount")
        )
    
    query = user_totals.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Monitoring data/input/ for CSV files")
    print("Add CSV files to see results")
    
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_3_rate_source():
    """Rate source aggregation"""
    spark = SparkSession.builder \
        .appName("RateSource") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load()
    
    df_with_time = df.withColumn("processed_at", current_timestamp())
    
    windowed = df_with_time.groupBy(
        window("timestamp", "10 seconds")
    ).agg(
        count("*").alias("event_count"),
        avg("value").alias("avg_value")
    )
    
    query = windowed.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Generating 10 events/second")
    print("Aggregating per 10-second window")
    
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_4_output_modes():
    """Compare output modes"""
    spark = SparkSession.builder \
        .appName("OutputModes") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load()
    
    df_grouped = df.withColumn("group", expr("value % 10"))
    
    counts = df_grouped.groupBy("group").count()
    
    # Complete mode
    print("\n=== Complete Mode ===")
    print("Shows entire result table each time")
    
    query = counts.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="5 seconds") \
        .start()
    
    time.sleep(15)
    query.stop()
    
    # Update mode
    print("\n=== Update Mode ===")
    print("Shows only updated rows")
    
    query = counts.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="5 seconds") \
        .start()
    
    time.sleep(15)
    query.stop()
    
    spark.stop()


def exercise_5_checkpoint_recovery():
    """Checkpoint and recovery"""
    spark = SparkSession.builder \
        .appName("CheckpointRecovery") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Clean up previous run
    if os.path.exists("output/streaming"):
        shutil.rmtree("output/streaming")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load()
    
    # First run
    print("\n=== First Run ===")
    query = df.writeStream \
        .format("parquet") \
        .option("path", "output/streaming/data") \
        .option("checkpointLocation", "output/streaming/checkpoint") \
        .trigger(processingTime="2 seconds") \
        .start()
    
    print("Running for 10 seconds...")
    time.sleep(10)
    
    print(f"Status: {query.status}")
    print(f"Last progress: {query.lastProgress}")
    
    query.stop()
    
    # Count records from first run
    first_count = spark.read.parquet("output/streaming/data").count()
    print(f"Records after first run: {first_count}")
    
    # Second run (recovery)
    print("\n=== Second Run (Recovery) ===")
    query = df.writeStream \
        .format("parquet") \
        .option("path", "output/streaming/data") \
        .option("checkpointLocation", "output/streaming/checkpoint") \
        .trigger(processingTime="2 seconds") \
        .start()
    
    print("Running for 10 seconds...")
    time.sleep(10)
    
    query.stop()
    
    # Count records after recovery
    second_count = spark.read.parquet("output/streaming/data").count()
    print(f"Records after second run: {second_count}")
    print(f"New records: {second_count - first_count}")
    
    # Verify no duplicates
    df_result = spark.read.parquet("output/streaming/data")
    duplicate_count = df_result.count() - df_result.dropDuplicates(["value"]).count()
    print(f"Duplicate records: {duplicate_count}")
    
    if duplicate_count == 0:
        print("✓ No duplicates - checkpoint recovery successful!")
    else:
        print("✗ Duplicates found - checkpoint issue")
    
    spark.stop()


def demo_all_features():
    """Demonstrate all streaming features"""
    spark = SparkSession.builder \
        .appName("StreamingDemo") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*60)
    print("Spark Structured Streaming Demo")
    print("="*60)
    
    # Create rate source
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load()
    
    # Add transformations
    transformed = df \
        .withColumn("group", expr("value % 3")) \
        .withColumn("processed_at", current_timestamp())
    
    # Aggregation
    aggregated = transformed.groupBy("group").agg(
        count("*").alias("count"),
        avg("value").alias("avg_value")
    )
    
    # Write to memory sink
    query = aggregated.writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("streaming_demo") \
        .trigger(processingTime="3 seconds") \
        .start()
    
    # Monitor query
    for i in range(5):
        time.sleep(3)
        print(f"\n--- Batch {i+1} ---")
        print(f"Active: {query.isActive}")
        print(f"Status: {query.status}")
        
        # Query results
        result = spark.sql("SELECT * FROM streaming_demo ORDER BY group")
        result.show()
    
    query.stop()
    spark.stop()


if __name__ == "__main__":
    print("Day 43: Spark Structured Streaming Solutions\n")
    
    # Uncomment to run specific exercises
    
    # Exercise 1: Socket word count (requires: nc -lk 9999)
    # print("\n" + "="*60)
    # print("Exercise 1: Socket Word Count")
    # print("="*60)
    # exercise_1_socket_word_count()
    
    # Exercise 2: File stream
    # print("\n" + "="*60)
    # print("Exercise 2: File Stream")
    # print("="*60)
    # exercise_2_file_stream()
    
    # Exercise 3: Rate source
    print("\n" + "="*60)
    print("Exercise 3: Rate Source")
    print("="*60)
    exercise_3_rate_source()
    
    # Exercise 4: Output modes
    print("\n" + "="*60)
    print("Exercise 4: Output Modes")
    print("="*60)
    exercise_4_output_modes()
    
    # Exercise 5: Checkpoint recovery
    print("\n" + "="*60)
    print("Exercise 5: Checkpoint Recovery")
    print("="*60)
    exercise_5_checkpoint_recovery()
    
    # Demo all features
    print("\n" + "="*60)
    print("Demo: All Features")
    print("="*60)
    demo_all_features()
