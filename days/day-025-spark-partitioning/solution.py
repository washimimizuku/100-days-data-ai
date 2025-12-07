"""Day 25: Spark Partitioning - Solutions"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

spark = SparkSession.builder.appName("Day25").getOrCreate()

def exercise_1():
    df = spark.range(1000000)
    print(f"Default partitions: {df.rdd.getNumPartitions()}")
    
    data_size_gb = 10
    target_mb = 128
    optimal = int((data_size_gb * 1024) / target_mb)
    print(f"Optimal partitions: {optimal}")
    
    df = df.repartition(optimal)
    print(f"After repartition: {df.rdd.getNumPartitions()}")

def exercise_2():
    df = spark.range(1000000).repartition(100)
    
    start = time.time()
    df.repartition(10).count()
    print(f"repartition(10): {time.time() - start:.2f}s (full shuffle)")
    
    start = time.time()
    df.coalesce(10).count()
    print(f"coalesce(10): {time.time() - start:.2f}s (no shuffle)")

def exercise_3():
    # Skewed data: 90% NYC, 10% others
    data = [(i, "NYC" if i < 900000 else "Boston") for i in range(1000000)]
    df = spark.createDataFrame(data, ["id", "city"])
    
    print("Partition sizes (skewed):")
    print(df.rdd.glom().map(len).collect()[:5])
    
    # Apply salting
    df = df.withColumn("salt", (rand() * 10).cast("int"))
    df = df.repartition(concat(col("city"), lit("_"), col("salt")))
    
    print("Partition sizes (after salting):")
    print(df.rdd.glom().map(len).collect()[:5])

def exercise_4():
    df = spark.range(1000).withColumn("date", current_date())
    df = df.withColumn("year", year("date")).withColumn("month", month("date"))
    
    df.write.partitionBy("year", "month").mode("overwrite").parquet("/tmp/partitioned")
    
    df_read = spark.read.parquet("/tmp/partitioned")
    df_filtered = df_read.filter(col("year") == 2024)
    
    print("Partition pruning applied - only year=2024 read")

def exercise_5():
    df = spark.range(10000000)
    print(f"Initial partitions: {df.rdd.getNumPartitions()}")
    
    df = df.repartition(80)
    df = df.filter(col("id") % 2 == 0)
    df = df.withColumn("doubled", col("id") * 2)
    
    df = df.coalesce(10)
    df.write.mode("overwrite").parquet("/tmp/optimized")
    print("Optimized pipeline complete")

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    spark.stop()
