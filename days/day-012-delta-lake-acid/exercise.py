"""
Day 12: Delta Lake ACID Transactions - Exercises
"""

from pyspark.sql import SparkSession
from delta import *
import threading

builder = SparkSession.builder \
    .appName("Day12-DeltaACID") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()


# Exercise 1: Atomic Operations
def exercise_1():
    """Test atomic writes and failure handling"""
    # TODO: Create orders table
    # TODO: Write 5 orders atomically
    # TODO: Simulate failure (try invalid data)
    # TODO: Verify table has either 0 or 5 records (not partial)
    pass


# Exercise 2: Concurrent Reads
def exercise_2():
    """Test concurrent readers with consistent snapshots"""
    # TODO: Create products table with 10 products
    # TODO: Define reader function that counts records
    # TODO: Start 3 concurrent reader threads
    # TODO: Verify all readers see same count
    pass


# Exercise 3: Concurrent Writes
def exercise_3():
    """Test concurrent writes and conflict handling"""
    # TODO: Create inventory table
    # TODO: Define writer1 (updates items 1-5)
    # TODO: Define writer2 (updates items 6-10)
    # TODO: Run both writers concurrently
    # TODO: Verify both succeeded (non-conflicting)
    pass


# Exercise 4: Transaction Isolation
def exercise_4():
    """Test isolation between readers and writers"""
    # TODO: Create events table with initial data
    # TODO: Start reader thread (reads continuously)
    # TODO: Start writer thread (adds data)
    # TODO: Verify reader sees consistent snapshots
    # TODO: Check reader never sees partial writes
    pass


# Exercise 5: Failure Recovery
def exercise_5():
    """Test recovery from transaction failures"""
    # TODO: Create logs table
    # TODO: Write batch 1 successfully
    # TODO: Attempt batch 2 with invalid schema (should fail)
    # TODO: Verify table still has batch 1 data
    # TODO: Write batch 3 successfully
    # TODO: Verify table has batch 1 + batch 3
    pass


if __name__ == "__main__":
    print("Day 12: Delta Lake ACID Transactions\n")
    
    print("Exercise 1: Atomic Operations")
    exercise_1()
    
    print("\nExercise 2: Concurrent Reads")
    exercise_2()
    
    print("\nExercise 3: Concurrent Writes")
    exercise_3()
    
    print("\nExercise 4: Transaction Isolation")
    exercise_4()
    
    print("\nExercise 5: Failure Recovery")
    exercise_5()
    
    spark.stop()
