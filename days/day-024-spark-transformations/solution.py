"""Day 24: Spark Transformations - Solutions"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

spark = SparkSession.builder.appName("Day24").getOrCreate()

def exercise_1():
    print("Narrow: filter, select, map, withColumn, union")
    print("Wide: groupBy, join, distinct, repartition, sortBy")
    print("groupBy requires shuffle to group data by key across partitions")

def exercise_2():
    df = spark.range(1000000).withColumn("city", lit("NYC"))
    
    # Unoptimized
    result1 = df.groupBy("city").count().filter(col("count") > 100)
    
    # Optimized - filter early
    result2 = df.filter(col("id") > 100).groupBy("city").count()
    
    result2.explain()

def exercise_3():
    df_large = spark.range(1000000).withColumn("value", col("id") * 2)
    df_small = spark.range(100).withColumn("category", lit("A"))
    
    # Regular join
    start = time.time()
    df_large.join(df_small, "id").count()
    print(f"Regular join: {time.time() - start:.2f}s")
    
    # Broadcast join
    start = time.time()
    df_large.join(broadcast(df_small), "id").count()
    print(f"Broadcast join: {time.time() - start:.2f}s")

def exercise_4():
    df = spark.range(1000).withColumn("city", lit("NYC"))
    result = df.groupBy("city").count().orderBy("count")
    
    print("Execution plan:")
    result.explain()
    print("Shuffles: 2 (groupBy and orderBy)")

def exercise_5():
    df = spark.range(1000000)
    
    # Without cache
    start = time.time()
    df.filter(col("id") > 100).count()
    df.filter(col("id") > 100).count()
    print(f"Without cache: {time.time() - start:.2f}s")
    
    # With cache
    df_cached = df.filter(col("id") > 100).cache()
    start = time.time()
    df_cached.count()
    df_cached.count()
    print(f"With cache: {time.time() - start:.2f}s")

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    spark.stop()
