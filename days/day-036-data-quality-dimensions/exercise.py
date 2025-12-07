"""
Day 36: Data Quality Dimensions - Exercises
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def exercise_1_check_accuracy(df):
    """
    Exercise 1: Check data accuracy
    
    Check the following:
    1. Email addresses match valid format
    2. Ages are between 0 and 120
    3. Phone numbers have 10 digits
    
    Return accuracy scores for each check
    """
    # TODO: Check email format using regex
    # TODO: Check age range
    # TODO: Check phone number length
    # TODO: Calculate accuracy rate for each
    # TODO: Return dictionary with scores
    pass


def exercise_2_check_completeness(df):
    """
    Exercise 2: Check data completeness
    
    Check completeness for critical fields:
    - customer_id, name, email, phone, created_at
    
    Return completeness score for each field and overall
    """
    # TODO: Define critical fields
    # TODO: Calculate null percentage for each field
    # TODO: Calculate completeness score (1 - null_rate)
    # TODO: Calculate overall completeness
    # TODO: Return dictionary with scores
    pass


def exercise_3_check_consistency(df):
    """
    Exercise 3: Check data consistency
    
    Check:
    1. Date format consistency (YYYY-MM-DD)
    2. Country code consistency (2-letter ISO)
    3. Status values consistency
    
    Return consistency scores
    """
    # TODO: Check date format consistency
    # TODO: Check country code format (2 uppercase letters)
    # TODO: Check status values against allowed list
    # TODO: Calculate consistency rate for each
    # TODO: Return dictionary with scores
    pass


def exercise_4_check_validity(df):
    """
    Exercise 4: Check data validity
    
    Validate business rules:
    1. Status must be: active, inactive, suspended
    2. Age must be positive
    3. Created_at must be in the past
    
    Return validity scores
    """
    # TODO: Check status against allowed values
    # TODO: Check age is positive
    # TODO: Check created_at is in the past
    # TODO: Calculate validity rate for each rule
    # TODO: Return dictionary with scores
    pass


def exercise_5_check_uniqueness(df):
    """
    Exercise 5: Check data uniqueness
    
    Check for duplicates:
    1. customer_id should be unique
    2. email should be unique
    3. Overall record duplicates
    
    Return uniqueness scores
    """
    # TODO: Check customer_id duplicates
    # TODO: Check email duplicates
    # TODO: Check full record duplicates
    # TODO: Calculate uniqueness rate (1 - duplicate_rate)
    # TODO: Return dictionary with scores
    pass


def exercise_6_check_timeliness(df):
    """
    Exercise 6: Check data timeliness
    
    Check:
    1. Records updated in last 7 days
    2. Records updated in last 30 days
    3. Average data age
    
    Return timeliness metrics
    """
    # TODO: Calculate records updated in last 7 days
    # TODO: Calculate records updated in last 30 days
    # TODO: Calculate average age in days
    # TODO: Calculate timeliness score
    # TODO: Return dictionary with metrics
    pass


def exercise_7_calculate_quality_score(df):
    """
    Exercise 7: Calculate overall quality score
    
    Combine all dimension scores with weights:
    - Accuracy: 20%
    - Completeness: 20%
    - Consistency: 15%
    - Validity: 20%
    - Uniqueness: 15%
    - Timeliness: 10%
    
    Return overall score and grade
    """
    # TODO: Call all check functions
    # TODO: Apply weights to each dimension
    # TODO: Calculate weighted average
    # TODO: Assign letter grade (A: 95+, B: 85+, C: 75+, D: 65+, F: <65)
    # TODO: Return score and grade
    pass


def exercise_8_generate_quality_report(df):
    """
    Exercise 8: Generate quality report
    
    Create a comprehensive report with:
    1. Dataset summary (rows, columns)
    2. Quality scores for each dimension
    3. Overall score and grade
    4. Top 5 quality issues
    
    Return formatted report string
    """
    # TODO: Get dataset summary
    # TODO: Calculate all quality scores
    # TODO: Identify top issues
    # TODO: Format report as string
    # TODO: Return report
    pass


if __name__ == "__main__":
    print("Day 36: Data Quality Dimensions - Exercises\n")
    
    # Create sample dataset
    np.random.seed(42)
    n = 1000
    
    sample_data = {
        'customer_id': range(1, n + 1),
        'name': [f'Customer {i}' if i % 10 != 0 else None for i in range(n)],
        'email': [f'user{i}@example.com' if i % 15 != 0 else 'invalid-email' for i in range(n)],
        'phone': [f'555{i:07d}' if i % 20 != 0 else None for i in range(n)],
        'age': np.random.randint(-5, 130, n),
        'country': np.random.choice(['US', 'UK', 'CA', 'USA', 'GB'], n),
        'status': np.random.choice(['active', 'inactive', 'suspended', 'unknown'], n),
        'created_at': [datetime.now() - timedelta(days=np.random.randint(0, 100)) for _ in range(n)],
        'updated_at': [datetime.now() - timedelta(days=np.random.randint(0, 60)) for _ in range(n)]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Sample dataset created with quality issues\n")
    
    # Uncomment to run exercises
    # print("Exercise 1: Accuracy")
    # print(exercise_1_check_accuracy(df))
    
    # print("\nExercise 2: Completeness")
    # print(exercise_2_check_completeness(df))
    
    # print("\nExercise 3: Consistency")
    # print(exercise_3_check_consistency(df))
    
    # print("\nExercise 4: Validity")
    # print(exercise_4_check_validity(df))
    
    # print("\nExercise 5: Uniqueness")
    # print(exercise_5_check_uniqueness(df))
    
    # print("\nExercise 6: Timeliness")
    # print(exercise_6_check_timeliness(df))
    
    # print("\nExercise 7: Overall Quality Score")
    # print(exercise_7_calculate_quality_score(df))
    
    # print("\nExercise 8: Quality Report")
    # print(exercise_8_generate_quality_report(df))
