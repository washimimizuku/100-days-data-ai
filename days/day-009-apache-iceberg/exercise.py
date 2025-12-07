"""
Day 9: Apache Iceberg - Exercises
Complete each exercise below
"""

# Note: These exercises use PySpark with Iceberg
# Install: pip install pyspark pyiceberg

from pyspark.sql import SparkSession

# Exercise 1: Create Iceberg Table
# TODO: Initialize Spark with Iceberg extensions
# TODO: Create database
# TODO: Create table with schema (id, name, age, city)
# TODO: Insert 5 records

print("Exercise 1: Create Iceberg Table")
# Your code here


# Exercise 2: Hidden Partitioning
# TODO: Create table partitioned by date (using days())
# TODO: Insert data with different dates
# TODO: Query without specifying partition filter
# TODO: Verify partition pruning works

print("\nExercise 2: Hidden Partitioning")
# Your code here


# Exercise 3: Schema Evolution
# TODO: Add new column to existing table
# TODO: Insert data with new column
# TODO: Query old data (should have null for new column)
# TODO: Rename a column

print("\nExercise 3: Schema Evolution")
# Your code here


# Exercise 4: Time Travel
# TODO: Create table and insert data (snapshot 1)
# TODO: Update data (snapshot 2)
# TODO: Delete some rows (snapshot 3)
# TODO: Query each snapshot
# TODO: Compare results

print("\nExercise 4: Time Travel")
# Your code here


# Exercise 5: Metadata Exploration
# TODO: Query snapshots table
# TODO: Query files table
# TODO: Query history table
# TODO: Analyze table statistics

print("\nExercise 5: Metadata Exploration")
# Your code here


# Bonus Challenge
# TODO: Implement partition evolution
# TODO: Start with daily partitions
# TODO: Change to monthly partitions
# TODO: Query data across both partition schemes

print("\nBonus Challenge: Partition Evolution")
# Your code here
