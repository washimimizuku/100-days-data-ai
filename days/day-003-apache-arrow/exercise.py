"""
Day 3: Apache Arrow - Exercises
Complete each exercise below
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import time
import sys

# Exercise 1: Create Arrow Table
# TODO: Create Arrow table from Python dict with 3 columns
# TODO: Print schema and data
# TODO: Convert to Pandas DataFrame

print("Exercise 1: Create Arrow Table")
# Your code here


# Exercise 2: Arrow vs Pandas Memory
# TODO: Create dataset with 1M rows
# TODO: Create Pandas DataFrame
# TODO: Create Arrow Table
# TODO: Compare memory usage

print("\nExercise 2: Arrow vs Pandas Memory")
# Your code here


# Exercise 3: Parquet with Arrow
# TODO: Read sample_data.parquet using Arrow
# TODO: Select specific columns (name, salary)
# TODO: Filter rows where salary > 80000
# TODO: Write filtered data back to Parquet

print("\nExercise 3: Parquet with Arrow")
# Your code here


# Exercise 4: Arrow Compute Functions
# TODO: Create Arrow table with numeric data
# TODO: Calculate mean, max, min using Arrow compute
# TODO: Sort by a column
# TODO: Compare speed with Pandas

print("\nExercise 4: Arrow Compute Functions")
# Your code here


# Exercise 5: Zero-Copy Demo
# TODO: Create large Pandas DataFrame (100K rows)
# TODO: Convert to Arrow
# TODO: Access column (zero-copy)
# TODO: Measure time vs Pandas column access

print("\nExercise 5: Zero-Copy Demo")
# Your code here


# Bonus Challenge
# TODO: Create ETL pipeline using Arrow
# TODO: Read Parquet → Filter → Transform → Write
# TODO: Measure performance vs Pandas approach

print("\nBonus Challenge: Arrow ETL Pipeline")

def arrow_etl_pipeline(input_file, output_file):
    """
    ETL pipeline using Arrow
    
    Args:
        input_file: Input Parquet file
        output_file: Output Parquet file
    
    Returns:
        Processing time
    """
    # Your code here
    pass

# Test your pipeline
# arrow_etl_pipeline('input.parquet', 'output.parquet')
