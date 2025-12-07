"""
Day 10: Iceberg Time Travel & Snapshots - Exercises
"""

from pyspark.sql import SparkSession
from datetime import datetime

# Initialize Spark with Iceberg
spark = SparkSession.builder \
    .appName("Day10-Iceberg-TimeTravel") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "warehouse") \
    .getOrCreate()


# Exercise 1: Time Travel Queries
def exercise_1():
    """Create table with multiple versions and query historical data"""
    # TODO: Create products table
    # TODO: Insert version 1 (3 products)
    # TODO: Insert version 2 (2 more products)
    # TODO: Update version 3 (modify prices)
    # TODO: Query each version by snapshot ID
    # TODO: Query version 2 by timestamp
    pass


# Exercise 2: Snapshot Analysis
def exercise_2():
    """Analyze snapshot metadata and history"""
    # TODO: Query snapshots table
    # TODO: Show snapshot details (id, timestamp, operation)
    # TODO: Query history table
    # TODO: Calculate data growth between snapshots
    pass


# Exercise 3: Rollback Operations
def exercise_3():
    """Perform rollback and verify results"""
    # TODO: Create orders table with 3 versions
    # TODO: Show current count
    # TODO: Rollback to first snapshot
    # TODO: Verify count after rollback
    # TODO: Restore to second snapshot using set_current_snapshot
    pass


# Exercise 4: Snapshot Management
def exercise_4():
    """Manage snapshots and optimize storage"""
    # TODO: Create table with 5+ snapshots
    # TODO: View all snapshots
    # TODO: Expire snapshots older than 30 days, keep last 3
    # TODO: Remove orphan files
    # TODO: Show remaining snapshots
    pass


# Exercise 5: Incremental Processing
def exercise_5():
    """Read and process incremental changes"""
    # TODO: Create events table
    # TODO: Insert batch 1 (10 events)
    # TODO: Get snapshot ID
    # TODO: Insert batch 2 (10 more events)
    # TODO: Read only changes between snapshots
    # TODO: Show incremental data
    pass


if __name__ == "__main__":
    print("Day 10: Iceberg Time Travel & Snapshots\n")
    
    print("Exercise 1: Time Travel Queries")
    exercise_1()
    
    print("\nExercise 2: Snapshot Analysis")
    exercise_2()
    
    print("\nExercise 3: Rollback Operations")
    exercise_3()
    
    print("\nExercise 4: Snapshot Management")
    exercise_4()
    
    print("\nExercise 5: Incremental Processing")
    exercise_5()
    
    spark.stop()
