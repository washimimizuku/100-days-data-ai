"""
Day 39: Data Validation - Exercises
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta


def exercise_1_type_validation(df, expected_types):
    """
    Exercise 1: Validate column data types
    
    Check that each column has the expected data type.
    
    Args:
        df: DataFrame to validate
        expected_types: Dict mapping column names to expected dtypes
    
    Return dict with validation results
    """
    # TODO: Check each column exists
    # TODO: Check each column has correct dtype
    # TODO: Return dict with passed/failed and errors list
    pass


def exercise_2_range_validation(df, column, min_val, max_val):
    """
    Exercise 2: Validate values are within range
    
    Check that all values in column are between min_val and max_val.
    
    Return tuple: (passed: bool, invalid_indices: list)
    """
    # TODO: Check values are >= min_val
    # TODO: Check values are <= max_val
    # TODO: Get indices of invalid values
    # TODO: Return (passed, invalid_indices)
    pass


def exercise_3_format_validation(df):
    """
    Exercise 3: Validate email and phone formats
    
    Validate:
    - email matches pattern: ^[\w\.-]+@[\w\.-]+\.\w+$
    - phone matches pattern: ^\d{10}$
    
    Return dict with validation results for each field
    """
    # TODO: Define email regex pattern
    # TODO: Define phone regex pattern
    # TODO: Validate email column
    # TODO: Validate phone column
    # TODO: Return results dict
    pass


def exercise_4_uniqueness_validation(df, columns):
    """
    Exercise 4: Check for duplicates
    
    Check that specified columns have no duplicate values.
    
    Args:
        df: DataFrame
        columns: List of column names to check
    
    Return dict with duplicate info
    """
    # TODO: Check for duplicates in specified columns
    # TODO: Get duplicate rows
    # TODO: Count duplicates
    # TODO: Return results dict
    pass


def exercise_5_business_rules(df):
    """
    Exercise 5: Validate business rules
    
    Rules:
    1. Active customers must have email
    2. Order amount must be positive
    3. Age must be >= 18 for account holders
    
    Return list of rule violations
    """
    # TODO: Check active customers have email
    # TODO: Check amounts are positive
    # TODO: Check age >= 18
    # TODO: Return list of violations
    pass


def exercise_6_build_validator(df):
    """
    Exercise 6: Build validation framework
    
    Create a validator that checks:
    1. customer_id not null
    2. customer_id unique
    3. age between 0 and 120
    4. email format valid
    5. status in allowed values
    
    Return validation results
    """
    # TODO: Create validation rules
    # TODO: Run each rule
    # TODO: Collect results
    # TODO: Return overall validation result
    pass


def exercise_7_validation_pipeline(df, quarantine_path):
    """
    Exercise 7: ETL with validation and quarantine
    
    1. Validate data
    2. Separate valid and invalid records
    3. Save invalid records to quarantine
    4. Return valid records
    
    Return tuple: (valid_df, invalid_df, validation_results)
    """
    # TODO: Run validation
    # TODO: Collect invalid row indices
    # TODO: Split into valid and invalid DataFrames
    # TODO: Save invalid to quarantine_path
    # TODO: Return (valid_df, invalid_df, results)
    pass


def exercise_8_schema_validation(df, schema):
    """
    Exercise 8: Comprehensive schema validation
    
    Validate DataFrame against schema definition.
    
    Schema format:
    {
        'column_name': {
            'type': 'int64',
            'nullable': False,
            'unique': True,
            'min': 0,
            'max': 100,
            'allowed_values': ['a', 'b', 'c']
        }
    }
    
    Return tuple: (passed: bool, errors: list)
    """
    # TODO: Check all required columns exist
    # TODO: Validate types
    # TODO: Check nullable constraints
    # TODO: Check uniqueness constraints
    # TODO: Check range constraints
    # TODO: Check allowed values
    # TODO: Return (passed, errors)
    pass


if __name__ == "__main__":
    print("Day 39: Data Validation - Exercises\n")
    
    # Create sample dataset with validation issues
    np.random.seed(42)
    n = 1000
    
    sample_data = {
        'customer_id': list(range(1, n + 1)) + [100],  # Duplicate
        'name': [f'Customer {i}' if i % 10 != 0 else None for i in range(n + 1)],
        'email': [f'user{i}@example.com' if i % 15 != 0 else 'invalid-email' for i in range(n + 1)],
        'phone': [f'{1000000000 + i}' if i % 20 != 0 else '123' for i in range(n + 1)],
        'age': list(np.random.randint(18, 80, n)) + [150],  # Invalid age
        'amount': list(np.random.uniform(10, 1000, n)) + [-50],  # Negative amount
        'status': np.random.choice(['active', 'inactive', 'suspended', 'unknown'], n + 1),
        'created_at': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n + 1)]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Sample dataset created with validation issues")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}\n")
    
    # Define expected types
    expected_types = {
        'customer_id': 'int64',
        'name': 'object',
        'email': 'object',
        'phone': 'object',
        'age': 'int64',
        'amount': 'float64',
        'status': 'object'
    }
    
    # Define schema
    schema = {
        'customer_id': {
            'type': 'int64',
            'nullable': False,
            'unique': True
        },
        'age': {
            'type': 'int64',
            'nullable': False,
            'min': 0,
            'max': 120
        },
        'status': {
            'type': 'object',
            'nullable': False,
            'allowed_values': ['active', 'inactive', 'suspended']
        }
    }
    
    # Uncomment to run exercises
    # print("Exercise 1: Type Validation")
    # print(exercise_1_type_validation(df, expected_types))
    
    # print("\nExercise 2: Range Validation (age)")
    # print(exercise_2_range_validation(df, 'age', 0, 120))
    
    # print("\nExercise 3: Format Validation")
    # print(exercise_3_format_validation(df))
    
    # print("\nExercise 4: Uniqueness Validation")
    # print(exercise_4_uniqueness_validation(df, ['customer_id']))
    
    # print("\nExercise 5: Business Rules")
    # print(exercise_5_business_rules(df))
    
    # print("\nExercise 6: Build Validator")
    # print(exercise_6_build_validator(df))
    
    # print("\nExercise 7: Validation Pipeline")
    # valid_df, invalid_df, results = exercise_7_validation_pipeline(df, 'quarantine.csv')
    # print(f"Valid: {len(valid_df)}, Invalid: {len(invalid_df)}")
    
    # print("\nExercise 8: Schema Validation")
    # print(exercise_8_schema_validation(df, schema))
