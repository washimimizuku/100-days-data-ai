"""
Day 37: Great Expectations - Exercises
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def exercise_1_setup_context():
    """
    Exercise 1: Set up Great Expectations context
    
    1. Import great_expectations
    2. Create a data context
    3. Print context configuration
    
    Return the context object
    """
    # TODO: Import great_expectations as gx
    # TODO: Create context with gx.get_context()
    # TODO: Print context root directory
    # TODO: Return context
    pass


def exercise_2_create_expectation_suite(context, df):
    """
    Exercise 2: Create expectation suite
    
    Create a suite named "customer_suite" with expectations:
    1. customer_id should not be null
    2. customer_id should be unique
    3. age should be between 0 and 120
    4. email should match email regex pattern
    5. status should be in ['active', 'inactive', 'suspended']
    
    Return the validator
    """
    # TODO: Create expectation suite
    # TODO: Create batch request for the dataframe
    # TODO: Get validator with batch request and suite
    # TODO: Add 5 expectations listed above
    # TODO: Save expectation suite
    # TODO: Return validator
    pass


def exercise_3_validate_data(validator):
    """
    Exercise 3: Validate data and analyze results
    
    1. Run validation
    2. Check if validation passed
    3. Print statistics
    4. List failed expectations
    
    Return validation results
    """
    # TODO: Run validator.validate()
    # TODO: Check results["success"]
    # TODO: Print statistics from results["statistics"]
    # TODO: Loop through results["results"] and print failed expectations
    # TODO: Return results
    pass


def exercise_4_add_more_expectations(validator):
    """
    Exercise 4: Add comprehensive expectations
    
    Add expectations for:
    1. Table row count between 1 and 1,000,000
    2. name should not be null
    3. email should not be null
    4. phone should match 10-digit pattern
    5. created_at should be datetime type
    6. country should be in ['US', 'UK', 'CA']
    
    Return updated validator
    """
    # TODO: Add table row count expectation
    # TODO: Add name not null expectation
    # TODO: Add email not null expectation
    # TODO: Add phone regex expectation (10 digits)
    # TODO: Add created_at type expectation
    # TODO: Add country set membership expectation
    # TODO: Save expectation suite
    # TODO: Return validator
    pass


def exercise_5_create_checkpoint(context, validator):
    """
    Exercise 5: Create a checkpoint
    
    Create a checkpoint named "customer_checkpoint" that:
    1. Uses the customer_suite
    2. Can be run repeatedly
    3. Returns validation results
    
    Return the checkpoint
    """
    # TODO: Create checkpoint configuration
    # TODO: Add checkpoint to context
    # TODO: Return checkpoint
    pass


def exercise_6_run_checkpoint(checkpoint):
    """
    Exercise 6: Run checkpoint and handle results
    
    1. Run the checkpoint
    2. Check if it passed
    3. Print summary
    4. Return results
    """
    # TODO: Run checkpoint
    # TODO: Check checkpoint_result["success"]
    # TODO: Print validation summary
    # TODO: Return checkpoint_result
    pass


def exercise_7_custom_expectation(validator, df):
    """
    Exercise 7: Create custom business rule
    
    Add custom expectation:
    - If status is 'active', email must not be null
    
    Use expect_column_pair_values_to_be_equal or custom logic
    """
    # TODO: Filter active customers
    # TODO: Check that active customers have emails
    # TODO: Add appropriate expectation
    # TODO: Return validation result
    pass


def exercise_8_generate_report(context):
    """
    Exercise 8: Generate Data Docs
    
    1. Build data docs
    2. Get data docs sites
    3. Print location of generated docs
    
    Return data docs configuration
    """
    # TODO: Build data docs with context.build_data_docs()
    # TODO: Get data docs sites
    # TODO: Print docs location
    # TODO: Return docs configuration
    pass


if __name__ == "__main__":
    print("Day 37: Great Expectations - Exercises\n")
    
    # Create sample dataset
    np.random.seed(42)
    n = 1000
    
    sample_data = {
        'customer_id': list(range(1, n + 1)) + [100],  # Duplicate
        'name': [f'Customer {i}' if i % 10 != 0 else None for i in range(n + 1)],
        'email': [f'user{i}@example.com' if i % 15 != 0 else None for i in range(n + 1)],
        'phone': [f'{1000000000 + i}' if i % 20 != 0 else '123' for i in range(n + 1)],
        'age': list(np.random.randint(18, 80, n)) + [150],  # Invalid age
        'country': np.random.choice(['US', 'UK', 'CA', 'FR'], n + 1),
        'status': np.random.choice(['active', 'inactive', 'suspended'], n + 1),
        'created_at': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n + 1)]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Sample dataset created with intentional quality issues\n")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}\n")
    
    # Uncomment to run exercises
    # print("Exercise 1: Setup Context")
    # context = exercise_1_setup_context()
    
    # print("\nExercise 2: Create Expectation Suite")
    # validator = exercise_2_create_expectation_suite(context, df)
    
    # print("\nExercise 3: Validate Data")
    # results = exercise_3_validate_data(validator)
    
    # print("\nExercise 4: Add More Expectations")
    # validator = exercise_4_add_more_expectations(validator)
    
    # print("\nExercise 5: Create Checkpoint")
    # checkpoint = exercise_5_create_checkpoint(context, validator)
    
    # print("\nExercise 6: Run Checkpoint")
    # checkpoint_result = exercise_6_run_checkpoint(checkpoint)
    
    # print("\nExercise 7: Custom Expectation")
    # exercise_7_custom_expectation(validator, df)
    
    # print("\nExercise 8: Generate Report")
    # exercise_8_generate_report(context)
