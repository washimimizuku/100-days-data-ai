"""Day 25: Spark Partitioning - Exercises"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Day25").getOrCreate()

def exercise_1():
    """Partition analysis"""
    # TODO: Create DataFrame with 1M rows
    # TODO: Check default partition count
    # TODO: Calculate optimal partitions (assume 10GB data, 128MB target)
    # TODO: Repartition to optimal count
    pass

def exercise_2():
    """Repartition vs coalesce"""
    # TODO: Create DataFrame with 100 partitions
    # TODO: Use repartition(10) - measure time
    # TODO: Use coalesce(10) - measure time
    # TODO: Compare results
    pass

def exercise_3():
    """Handle data skew"""
    # TODO: Create skewed data (90% in one city)
    # TODO: Check partition sizes
    # TODO: Apply salting to distribute evenly
    # TODO: Verify improvement
    pass

def exercise_4():
    """Partition pruning"""
    # TODO: Create DataFrame with date column
    # TODO: Write partitioned by year, month
    # TODO: Read with date filter
    # TODO: Verify only relevant partitions read
    pass

def exercise_5():
    """End-to-end optimization"""
    # TODO: Read large CSV
    # TODO: Repartition optimally
    # TODO: Apply transformations
    # TODO: Coalesce before write
    # TODO: Write to parquet
    pass

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    spark.stop()
