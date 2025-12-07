"""
Day 5: Data Serialization Comparison - Exercises
Complete each exercise below
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import os
import json

# Exercise 1: Format Benchmark
# TODO: Create dataset with 100K rows (id, name, value, category)
# TODO: Write to CSV, JSON, Parquet
# TODO: Measure write times
# TODO: Compare file sizes

print("Exercise 1: Format Benchmark")
# Your code here


# Exercise 2: Read Performance
# TODO: Read each format (CSV, JSON, Parquet)
# TODO: Measure read times
# TODO: Read specific columns from Parquet
# TODO: Compare performance

print("\nExercise 2: Read Performance")
# Your code here


# Exercise 3: Format Conversion
# TODO: Read employees.csv from data/day-001/
# TODO: Convert to Parquet
# TODO: Convert to JSON
# TODO: Verify data integrity
# TODO: Compare sizes

print("\nExercise 3: Format Conversion")
# Your code here


# Exercise 4: Decision Matrix
# TODO: Given 5 scenarios, choose best format
# Scenario 1: E-commerce analytics (10M rows, read-heavy)
# Scenario 2: Real-time events (streaming, schema changes)
# Scenario 3: REST API (nested data, web clients)
# Scenario 4: Database export (1000 rows, for Excel)
# Scenario 5: ML pipeline (in-memory, Python to Spark)

print("\nExercise 4: Decision Matrix")
scenarios = {
    "E-commerce analytics": "?",
    "Real-time events": "?",
    "REST API": "?",
    "Database export": "?",
    "ML pipeline": "?"
}
# Fill in your answers


# Exercise 5: Compression Test
# TODO: Write Parquet with different compressions
# TODO: Compare sizes (snappy, gzip, zstd, none)
# TODO: Measure read times
# TODO: Determine best compression

print("\nExercise 5: Compression Test")
# Your code here


# Exercise 6: Real-World Pipeline
# TODO: Simulate data pipeline
# TODO: CSV → Parquet → Arrow → Processing
# TODO: Measure end-to-end performance

print("\nExercise 6: Real-World Pipeline")
# Your code here


# Bonus Challenge
# TODO: Create format recommendation function
# TODO: Input: data characteristics (size, use case, schema changes)
# TODO: Output: recommended format with justification

print("\nBonus Challenge: Format Recommender")

def recommend_format(size_mb, use_case, schema_stable, need_human_readable):
    """
    Recommend best data format
    
    Args:
        size_mb: Dataset size in MB
        use_case: 'analytics', 'streaming', 'api', 'export'
        schema_stable: True if schema doesn't change
        need_human_readable: True if humans need to read it
    
    Returns:
        dict with recommendation and reason
    """
    # Your code here
    pass

# Test your recommender
# print(recommend_format(1000, 'analytics', True, False))
