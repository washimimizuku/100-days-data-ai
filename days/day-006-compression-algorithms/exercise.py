"""
Day 6: Compression Algorithms - Exercises
Complete each exercise below
"""

import pandas as pd
import time
import os

# Exercise 1: Compression Benchmark
# TODO: Create dataset with 50K rows (id, name, text)
# TODO: Write with snappy, gzip, zstd, none
# TODO: Measure write times
# TODO: Compare file sizes

print("Exercise 1: Compression Benchmark")
# Your code here


# Exercise 2: Read Performance
# TODO: Read files from Exercise 1
# TODO: Measure read times for each
# TODO: Calculate compression ratios
# TODO: Determine best overall (speed + size)

print("\nExercise 2: Read Performance")
# Your code here


# Exercise 3: ZSTD Levels
# TODO: Test ZSTD with levels 1, 3, 10, 22
# TODO: Compare sizes
# TODO: Compare write/read times
# TODO: Find optimal level

print("\nExercise 3: ZSTD Levels")
# Your code here


# Exercise 4: Data Type Impact
# TODO: Create numeric dataset (all numbers)
# TODO: Create text dataset (all strings)
# TODO: Test compression on both
# TODO: Compare effectiveness

print("\nExercise 4: Data Type Impact")
# Your code here


# Exercise 5: Cost Calculation
# TODO: Assume 1TB of data
# TODO: Calculate storage costs for each compression
# TODO: Use $0.023/GB/month (S3 pricing)
# TODO: Calculate annual savings

print("\nExercise 5: Cost Calculation")
# Your code here


# Exercise 6: Real-World Scenario
# TODO: Create hot data (recent, frequently accessed)
# TODO: Create warm data (monthly, occasional access)
# TODO: Create cold data (archive, rare access)
# TODO: Use appropriate compression for each
# TODO: Calculate total storage savings

print("\nExercise 6: Real-World Scenario")
# Your code here


# Bonus Challenge
# TODO: Create compression recommender function
# TODO: Input: access_pattern, data_size, cpu_available
# TODO: Output: recommended compression with justification

print("\nBonus Challenge: Compression Recommender")

def recommend_compression(access_pattern, data_size_gb, cpu_available):
    """
    Recommend compression algorithm
    
    Args:
        access_pattern: 'hot', 'warm', 'cold'
        data_size_gb: Size in GB
        cpu_available: 'low', 'medium', 'high'
    
    Returns:
        dict with recommendation and reason
    """
    # Your code here
    pass

# Test your recommender
# print(recommend_compression('hot', 100, 'low'))
