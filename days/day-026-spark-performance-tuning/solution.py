"""
Day 26: Spark Performance Tuning - Solutions
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast, rand, concat, lit, count, sum as _sum, avg, max as _max
from pyspark import StorageLevel
import time

def exercise_1():
    """Set optimal Spark configurations"""
    spark = SparkSession.builder \
        .appName("Ex1-Config") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    print("Spark configured with optimal settings")
    print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"AQE enabled: {spark.conf.get('spark.sql.adaptive.enabled')}")
    spark.stop()

def exercise_2():
    """Implement caching for reused DataFrames"""
    spark = SparkSession.builder.appName("Ex2-Cache").getOrCreate()
    df = spark.range(1000000).withColumn("value", rand())
    
    # Without cache
    start = time.time()
    df.count()
    df.agg(avg("value")).show()
    df.agg(_max("value")).show()
    no_cache_time = time.time() - start
    
    # With cache
    df_cached = df.cache()
    start = time.time()
    df_cached.count()
    df_cached.agg(avg("value")).show()
    df_cached.agg(_max("value")).show()
    cache_time = time.time() - start
    
    print(f"Without cache: {no_cache_time:.2f}s")
    print(f"With cache: {cache_time:.2f}s")
    df_cached.unpersist()
    spark.stop()

def exercise_3():
    """Optimize join with broadcast"""
    spark = SparkSession.builder.appName("Ex3-Broadcast").getOrCreate()
    large_df = spark.range(1000000).withColumn("value", rand())
    small_df = spark.range(100).withColumn("name", col("id").cast("string"))
    
    # Regular join
    result1 = large_df.join(small_df, "id")
    print("Regular join plan:")
    result1.explain()
    
    # Broadcast join
    result2 = large_df.join(broadcast(small_df), "id")
    print("\nBroadcast join plan:")
    result2.explain()
    spark.stop()

def exercise_4():
    """Minimize shuffle operations"""
    spark = SparkSession.builder.appName("Ex4-Shuffle").getOrCreate()
    df = spark.range(10000).withColumn("group", (col("id") % 10).cast("int"))
    
    # Single aggregation (1 shuffle)
    result = df.groupBy("group").agg(count("*").alias("cnt"), _sum("id").alias("total"))
    print("Optimized aggregation:")
    result.explain()
    result.show()
    spark.stop()

def exercise_5():
    """Apply all tuning techniques"""
    spark = SparkSession.builder \
        .appName("Ex5-Complete") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    # Create data
    large_df = spark.range(100000).withColumn("value", rand()).withColumn("group", (col("id") % 100).cast("int"))
    small_df = spark.range(10).withColumn("category", col("id").cast("string"))
    
    # Cache reused DataFrame
    large_cached = large_df.cache()
    large_cached.count()
    
    # Broadcast join
    joined = large_cached.join(broadcast(small_df), large_cached.group == small_df.id)
    
    # Aggregation
    result = joined.groupBy("category").agg(count("*").alias("cnt"), avg("value").alias("avg_val"))
    
    # Write efficiently
    result.coalesce(1).write.mode("overwrite").parquet("/tmp/spark_tuned_output")
    
    print("Pipeline completed with all optimizations")
    large_cached.unpersist()
    spark.stop()

if __name__ == "__main__":
    print("Day 26: Spark Performance Tuning Solutions\n")
    exercise_1()
    print("\n" + "="*50 + "\n")
    exercise_2()
    print("\n" + "="*50 + "\n")
    exercise_3()
    print("\n" + "="*50 + "\n")
    exercise_4()
    print("\n" + "="*50 + "\n")
    exercise_5()
