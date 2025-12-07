"""
Day 2: Parquet Format - Exercises
Complete each exercise below
"""

import pandas as pd
import pyarrow.parquet as pq
import time
import os

# Exercise 1: Basic Parquet Operations
# TODO: Create a DataFrame with 1000 rows (id, name, value, category)
# TODO: Write to Parquet with snappy compression
# TODO: Read it back and verify data

print("Exercise 1: Basic Parquet Operations")
# Your code here


# Exercise 2: Compression Comparison
# TODO: Create DataFrame with 10000 rows
# TODO: Write with snappy, gzip, and no compression
# TODO: Compare file sizes
# TODO: Measure read times for each

print("\nExercise 2: Compression Comparison")
# Your code here


# Exercise 3: Column Selection
# TODO: Create DataFrame with 20 columns
# TODO: Write to Parquet
# TODO: Read only 3 specific columns
# TODO: Compare time vs reading all columns

print("\nExercise 3: Column Selection")
# Your code here


# Exercise 4: CSV to Parquet Conversion
# TODO: Read the employees.csv from data/day-001/
# TODO: Convert to Parquet
# TODO: Compare file sizes
# TODO: Compare read performance

print("\nExercise 4: CSV to Parquet Conversion")
# Your code here


# Exercise 5: Metadata Inspection
# TODO: Read Parquet file metadata (don't load data)
# TODO: Print schema
# TODO: Print row count
# TODO: Print column statistics

print("\nExercise 5: Metadata Inspection")
# Your code here


# Exercise 6: Partitioned Data
# TODO: Create sales data with date column
# TODO: Extract year and month
# TODO: Write as partitioned Parquet (by year/month)
# TODO: Read specific partition

print("\nExercise 6: Partitioned Data")
# Your code here


# Bonus Challenge
# TODO: Create a function that converts any CSV to Parquet
# TODO: Add compression parameter
# TODO: Return file size comparison

print("\nBonus Challenge: CSV to Parquet Converter")

def csv_to_parquet(csv_file, parquet_file, compression='snappy'):
    """
    Convert CSV file to Parquet format
    
    Args:
        csv_file: Path to input CSV
        parquet_file: Path to output Parquet
        compression: Compression algorithm
    
    Returns:
        dict with size comparison
    """
    # Your code here
    pass

# Test your function
# result = csv_to_parquet('test.csv', 'test.parquet')
# print(result)
