"""
Day 48: Streaming Performance Optimization - Solutions
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.streaming import GroupState, GroupStateTimeout
from pyspark.sql.types import *
import time


def exercise_1_trigger_tuning():
    """Compare trigger intervals"""
    spark = SparkSession.builder \
        .appName("TriggerTuning") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 100) \
        .load()
    
    # Test 1: 1-second trigger
    print("\n=== 1-Second Trigger ===")
    query1 = df.groupBy(window("timestamp", "10 seconds")).count() \
        .writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("trigger_1s") \
        .trigger(processingTime="1 second") \
        .start()
    
    time.sleep(15)
    progress1 = query1.lastProgress
    if progress1:
        print(f"Processing rate: {progress1.get('processedRowsPerSecond', 0):.2f} rows/sec")
        print(f"Batch duration: {progress1.get('batchDuration', 0)} ms")
    query1.stop()
    
    # Test 2: 10-second trigger
    print("\n=== 10-Second Trigger ===")
    query2 = df.groupBy(window("timestamp", "10 seconds")).count() \
        .writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("trigger_10s") \
        .trigger(processingTime="10 seconds") \
        .start()
    
    time.sleep(25)
    progress2 = query2.lastProgress
    if progress2:
        print(f"Processing rate: {progress2.get('processedRowsPerSecond', 0):.2f} rows/sec")
        print(f"Batch duration: {progress2.get('batchDuration', 0)} ms")
    query2.stop()
    
    print("\nConclusion: Longer trigger = higher throughput, higher latency")
    spark.stop()


def exercise_2_partition_optimization():
    """Optimize partition count"""
    spark = SparkSession.builder \
        .appName("PartitionOptimization") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 100) \
        .load() \
        .withColumn("key", expr("value % 100"))
    
    # Test 1: 10 partitions
    print("\n=== 10 Partitions ===")
    df_10 = df.repartition(10, "key")
    query1 = df_10.groupBy("key").count() \
        .writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("part_10") \
        .start()
    
    time.sleep(15)
    progress1 = query1.lastProgress
    if progress1:
        print(f"Processing rate: {progress1.get('processedRowsPerSecond', 0):.2f} rows/sec")
    query1.stop()
    
    # Test 2: 50 partitions
    print("\n=== 50 Partitions ===")
    df_50 = df.repartition(50, "key")
    query2 = df_50.groupBy("key").count() \
        .writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("part_50") \
        .start()
    
    time.sleep(15)
    progress2 = query2.lastProgress
    if progress2:
        print(f"Processing rate: {progress2.get('processedRowsPerSecond', 0):.2f} rows/sec")
    query2.stop()
    
    print("\nConclusion: Optimal partitions = 2-3x cores")
    spark.stop()


def exercise_3_state_size_reduction():
    """Bounded vs unbounded state"""
    spark = SparkSession.builder \
        .appName("StateSizeReduction") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 50) \
        .load() \
        .withColumn("user_id", expr("CAST(value % 5 AS STRING)"))
    
    # Unbounded state (bad)
    print("\n=== Unbounded State ===")
    
    state_schema = ArrayType(LongType())
    output_schema = StructType([
        StructField("user_id", StringType()),
        StructField("count", IntegerType())
    ])
    
    def unbounded_state(user_id, values, state):
        if state.exists:
            history = list(state.get)
        else:
            history = []
        
        for value in values:
            history.append(value.value)  # Keeps growing!
        
        state.update(history)
        return iter([Row(user_id=user_id, count=len(history))])
    
    query1 = df.groupByKey(lambda x: x.user_id) \
        .flatMapGroupsWithState(
            unbounded_state,
            output_schema,
            state_schema,
            GroupStateTimeout.NoTimeout,
            outputMode="update"
        ) \
        .writeStream \
        .outputMode("update") \
        .format("memory") \
        .queryName("unbounded") \
        .start()
    
    time.sleep(10)
    progress1 = query1.lastProgress
    if progress1 and 'stateOperators' in progress1:
        for op in progress1['stateOperators']:
            print(f"State rows: {op.get('numRowsTotal', 0)}")
            print(f"State memory: {op.get('memoryUsedBytes', 0)} bytes")
    query1.stop()
    
    # Bounded state (good)
    print("\n=== Bounded State ===")
    
    def bounded_state(user_id, values, state):
        if state.exists:
            history = list(state.get)
        else:
            history = []
        
        for value in values:
            history.append(value.value)
        
        # Keep only last 100
        if len(history) > 100:
            history = history[-100:]
        
        state.update(history)
        return iter([Row(user_id=user_id, count=len(history))])
    
    query2 = df.groupByKey(lambda x: x.user_id) \
        .flatMapGroupsWithState(
            bounded_state,
            output_schema,
            state_schema,
            GroupStateTimeout.NoTimeout,
            outputMode="update"
        ) \
        .writeStream \
        .outputMode("update") \
        .format("memory") \
        .queryName("bounded") \
        .start()
    
    time.sleep(10)
    progress2 = query2.lastProgress
    if progress2 and 'stateOperators' in progress2:
        for op in progress2['stateOperators']:
            print(f"State rows: {op.get('numRowsTotal', 0)}")
            print(f"State memory: {op.get('memoryUsedBytes', 0)} bytes")
    query2.stop()
    
    print("\nConclusion: Always bound state size!")
    spark.stop()


def exercise_4_resource_allocation():
    """Tune executor configuration"""
    print("\n=== Resource Allocation ===")
    
    # Default config
    print("\n--- Default Configuration ---")
    spark1 = SparkSession.builder \
        .appName("DefaultConfig") \
        .master("local[*]") \
        .getOrCreate()
    
    df1 = spark1.readStream \
        .format("rate") \
        .option("rowsPerSecond", 100) \
        .load()
    
    query1 = df1.groupBy(window("timestamp", "10 seconds")).count() \
        .writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("default") \
        .start()
    
    time.sleep(15)
    progress1 = query1.lastProgress
    if progress1:
        print(f"Processing rate: {progress1.get('processedRowsPerSecond', 0):.2f} rows/sec")
    query1.stop()
    spark1.stop()
    
    # Tuned config
    print("\n--- Tuned Configuration ---")
    spark2 = SparkSession.builder \
        .appName("TunedConfig") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.default.parallelism", "50") \
        .getOrCreate()
    
    df2 = spark2.readStream \
        .format("rate") \
        .option("rowsPerSecond", 100) \
        .load()
    
    query2 = df2.groupBy(window("timestamp", "10 seconds")).count() \
        .writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("tuned") \
        .start()
    
    time.sleep(15)
    progress2 = query2.lastProgress
    if progress2:
        print(f"Processing rate: {progress2.get('processedRowsPerSecond', 0):.2f} rows/sec")
    query2.stop()
    spark2.stop()
    
    print("\nConclusion: Tune shuffle partitions for workload")


def exercise_5_end_to_end_optimization():
    """Complete pipeline optimization"""
    print("\n=== End-to-End Optimization ===")
    
    spark = SparkSession.builder \
        .appName("OptimizedPipeline") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Optimized pipeline
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 100) \
        .load() \
        .withColumn("user_id", expr("CAST(value % 10 AS STRING)")) \
        .withColumn("amount", expr("10 + (value % 50)")) \
        .repartition(50, "user_id")  # Optimize partitions
    
    # Add watermark
    df_watermarked = df.withWatermark("timestamp", "5 minutes")
    
    # Windowed aggregation
    result = df_watermarked.groupBy(
        window("timestamp", "10 seconds"),
        "user_id"
    ).agg(
        count("*").alias("count"),
        sum("amount").alias("total")
    )
    
    # Optimized trigger
    query = result.writeStream \
        .outputMode("append") \
        .format("memory") \
        .queryName("optimized") \
        .trigger(processingTime="5 seconds") \
        .start()
    
    # Monitor
    for i in range(3):
        time.sleep(10)
        progress = query.lastProgress
        if progress:
            print(f"\nBatch {i+1}:")
            print(f"  Input rows: {progress.get('numInputRows', 0)}")
            print(f"  Processing rate: {progress.get('processedRowsPerSecond', 0):.2f} rows/sec")
            print(f"  Batch duration: {progress.get('batchDuration', 0)} ms")
            
            if 'stateOperators' in progress:
                for op in progress['stateOperators']:
                    print(f"  State rows: {op.get('numRowsTotal', 0)}")
    
    query.stop()
    spark.stop()
    
    print("\n✓ Applied all optimizations:")
    print("  - Tuned shuffle partitions")
    print("  - Optimized partition count")
    print("  - Added watermark")
    print("  - Balanced trigger interval")


if __name__ == "__main__":
    print("Day 48: Streaming Performance Optimization Solutions\n")
    
    exercise_1_trigger_tuning()
    exercise_2_partition_optimization()
    exercise_3_state_size_reduction()
    exercise_4_resource_allocation()
    exercise_5_end_to_end_optimization()
