"""
Day 1: CSV vs JSON - Exercises
Complete each exercise below
"""

import csv
import json
import pandas as pd
import time

# Exercise 1: Read CSV
# TODO: Read 'employees.csv' and print first 3 rows
# TODO: Count total employees
# TODO: Calculate average salary

print("Exercise 1: Read CSV")
# Your code here


# Exercise 2: Write CSV
# TODO: Create a list of 5 products with: product_id, name, price, category
# TODO: Write to 'products.csv' using csv.DictWriter

print("\nExercise 2: Write CSV")
products = [
    # Add your products here
]

# Your code here


# Exercise 3: Read JSON
# TODO: Read 'config.json'
# TODO: Print the database connection string
# TODO: List all API endpoints
# TODO: Extract the timeout value

print("\nExercise 3: Read JSON")
# Your code here


# Exercise 4: Write JSON
# TODO: Create a user profile dictionary with:
#   - name, email, age
#   - address (nested dict: street, city, country)
#   - skills (list)
#   - is_active (boolean)
# TODO: Write to 'user_profile.json' with indentation

print("\nExercise 4: Write JSON")
user_profile = {
    # Add your user profile here
}

# Your code here


# Exercise 5: CSV to JSON Conversion
# TODO: Read 'sales.csv' using pandas
# TODO: Group by category and sum sales
# TODO: Convert to JSON format
# TODO: Save as 'sales_summary.json'

print("\nExercise 5: CSV to JSON Conversion")
# Your code here


# Exercise 6: Performance Test
# TODO: Create a DataFrame with 10,000 rows (id, name, value, category)
# TODO: Save as CSV and JSON
# TODO: Measure read time for CSV
# TODO: Measure read time for JSON
# TODO: Print comparison

print("\nExercise 6: Performance Test")

# Create test data
# Your code here

# Measure CSV read time
# Your code here

# Measure JSON read time
# Your code here

# Print results
# Your code here


# Bonus Challenge
# TODO: Create a function that converts any CSV file to JSON
# TODO: Handle different delimiters (comma, tab, semicolon)
# TODO: Add error handling for missing files

print("\nBonus Challenge: CSV to JSON Converter")

def csv_to_json(csv_file, json_file, delimiter=','):
    """
    Convert CSV file to JSON format
    
    Args:
        csv_file: Path to input CSV file
        json_file: Path to output JSON file
        delimiter: CSV delimiter (default: comma)
    """
    # Your code here
    pass

# Test your function
# csv_to_json('test.csv', 'test.json')
