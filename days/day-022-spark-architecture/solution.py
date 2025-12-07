"""
Day 22: Spark Architecture - Solutions
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def exercise_1():
    spark = SparkSession.builder \
        .appName("Day22") \
        .master("local[4]") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    
    print(f"Spark version: {spark.version}")
    print("Spark UI: http://localhost:4040")
    return spark

def exercise_2():
    spark = exercise_1()
    df = spark.range(1000000)
    df = df.filter(col("id") % 2 == 0)
    df = df.withColumn("doubled", col("id") * 2)
    
    count = df.count()
    print(f"Count: {count}")
    print("Check Spark UI Jobs tab for execution details")

def exercise_3():
    spark = exercise_1()
    df = spark.range(1000)
    df = df.filter(col("id") > 100)
    df = df.withColumn("squared", col("id") * col("id"))
    df = df.filter(col("squared") < 10000)
    df = df.select("id", "squared")
    
    print("Transformations defined (not executed yet)")
    print("Check Spark UI - no jobs yet")
    
    result = df.count()
    print(f"Action triggered, result: {result}")
    print("Now check Spark UI - job executed")

def exercise_4():
    spark = exercise_1()
    df = spark.range(1000000)
    df.cache()
    
    df.count()
    print("DataFrame cached")
    print("Check Spark UI Storage tab")

def exercise_5():
    print("Local[*]: Use all available cores")
    print("Local[4]: Use 4 cores")
    print("\nCluster config example:")
    print("  --num-executors 10")
    print("  --executor-memory 4g")
    print("  --executor-cores 2")

if __name__ == "__main__":
    print("Day 22: Spark Architecture - Solutions\n")
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
