"""
Day 37: Great Expectations - Solutions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Note: Great Expectations requires installation: pip install great-expectations
# This solution shows the structure; actual execution requires GX installed


def exercise_1_setup_context():
    """Set up Great Expectations context"""
    try:
        import great_expectations as gx
        
        # Create context
        context = gx.get_context()
        
        print("✅ Great Expectations context created")
        print(f"Context root: {context.root_directory}")
        
        return context
    except ImportError:
        print("⚠️  Great Expectations not installed")
        print("Install with: pip install great-expectations")
        return None


def exercise_2_create_expectation_suite(context, df):
    """Create expectation suite"""
    if context is None:
        print("⚠️  Context not available")
        return None
    
    import great_expectations as gx
    
    # Create or get expectation suite
    suite_name = "customer_suite"
    
    try:
        suite = context.get_expectation_suite(suite_name)
        print(f"Using existing suite: {suite_name}")
    except:
        suite = context.add_expectation_suite(suite_name)
        print(f"Created new suite: {suite_name}")
    
    # Create batch request
    batch_request = gx.core.batch.RuntimeBatchRequest(
        datasource_name="pandas_datasource",
        data_connector_name="runtime_data_connector",
        data_asset_name="customers",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default"}
    )
    
    # Get validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )
    
    # Add expectations
    validator.expect_column_values_to_not_be_null("customer_id")
    validator.expect_column_values_to_be_unique("customer_id")
    validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
    validator.expect_column_values_to_match_regex(
        "email",
        regex=r'^[\w\.-]+@[\w\.-]+\.\w+$'
    )
    validator.expect_column_values_to_be_in_set(
        "status",
        value_set=["active", "inactive", "suspended"]
    )
    
    # Save suite
    validator.save_expectation_suite(discard_failed_expectations=False)
    
    print(f"✅ Created suite with 5 expectations")
    
    return validator


def exercise_3_validate_data(validator):
    """Validate data and analyze results"""
    if validator is None:
        print("⚠️  Validator not available")
        return None
    
    # Run validation
    results = validator.validate()
    
    # Check success
    success = results["success"]
    print(f"\nValidation {'✅ PASSED' if success else '❌ FAILED'}")
    
    # Print statistics
    stats = results["statistics"]
    print(f"\nStatistics:")
    print(f"  Evaluated: {stats['evaluated_expectations']}")
    print(f"  Successful: {stats['successful_expectations']}")
    print(f"  Failed: {stats['unsuccessful_expectations']}")
    print(f"  Success rate: {stats['success_percent']:.1f}%")
    
    # List failed expectations
    print(f"\nFailed Expectations:")
    for result in results["results"]:
        if not result["success"]:
            exp_type = result['expectation_config']['expectation_type']
            column = result['expectation_config']['kwargs'].get('column', 'N/A')
            print(f"  ❌ {exp_type} on column '{column}'")
    
    return results


def exercise_4_add_more_expectations(validator):
    """Add comprehensive expectations"""
    if validator is None:
        print("⚠️  Validator not available")
        return None
    
    # Table-level expectations
    validator.expect_table_row_count_to_be_between(min_value=1, max_value=1000000)
    
    # Column existence and nulls
    validator.expect_column_values_to_not_be_null("name")
    validator.expect_column_values_to_not_be_null("email")
    
    # Phone format (10 digits)
    validator.expect_column_values_to_match_regex(
        "phone",
        regex=r'^\d{10}$'
    )
    
    # Type checks
    validator.expect_column_values_to_be_of_type("created_at", "datetime64")
    
    # Set membership
    validator.expect_column_values_to_be_in_set(
        "country",
        value_set=["US", "UK", "CA"]
    )
    
    # Save suite
    validator.save_expectation_suite(discard_failed_expectations=False)
    
    print("✅ Added 6 more expectations")
    
    return validator


def exercise_5_create_checkpoint(context, validator):
    """Create a checkpoint"""
    if context is None or validator is None:
        print("⚠️  Context or validator not available")
        return None
    
    checkpoint_name = "customer_checkpoint"
    
    # Create checkpoint configuration
    checkpoint_config = {
        "name": checkpoint_name,
        "config_version": 1,
        "class_name": "SimpleCheckpoint",
        "validations": [
            {
                "batch_request": validator.active_batch_request,
                "expectation_suite_name": validator.expectation_suite_name
            }
        ]
    }
    
    # Add checkpoint
    try:
        checkpoint = context.add_checkpoint(**checkpoint_config)
        print(f"✅ Created checkpoint: {checkpoint_name}")
        return checkpoint
    except Exception as e:
        print(f"⚠️  Error creating checkpoint: {e}")
        return None


def exercise_6_run_checkpoint(checkpoint):
    """Run checkpoint and handle results"""
    if checkpoint is None:
        print("⚠️  Checkpoint not available")
        return None
    
    # Run checkpoint
    checkpoint_result = checkpoint.run()
    
    # Check success
    success = checkpoint_result["success"]
    print(f"\nCheckpoint {'✅ PASSED' if success else '❌ FAILED'}")
    
    # Print summary
    print(f"\nValidation Summary:")
    for validation in checkpoint_result["run_results"].values():
        stats = validation["validation_result"]["statistics"]
        print(f"  Evaluated: {stats['evaluated_expectations']}")
        print(f"  Successful: {stats['successful_expectations']}")
        print(f"  Failed: {stats['unsuccessful_expectations']}")
    
    return checkpoint_result


def exercise_7_custom_expectation(validator, df):
    """Create custom business rule"""
    if validator is None:
        print("⚠️  Validator not available")
        return None
    
    # Custom rule: Active customers must have email
    active_customers = df[df['status'] == 'active']
    active_with_email = active_customers['email'].notna().sum()
    total_active = len(active_customers)
    
    print(f"\nCustom Business Rule Check:")
    print(f"  Active customers: {total_active}")
    print(f"  Active with email: {active_with_email}")
    print(f"  Compliance: {active_with_email / total_active * 100:.1f}%")
    
    # Add expectation for active customers
    # Note: This is a simplified check; GX supports custom expectations
    if active_with_email < total_active:
        print(f"  ❌ {total_active - active_with_email} active customers missing email")
    else:
        print(f"  ✅ All active customers have email")
    
    return active_with_email == total_active


def exercise_8_generate_report(context):
    """Generate Data Docs"""
    if context is None:
        print("⚠️  Context not available")
        return None
    
    try:
        # Build data docs
        context.build_data_docs()
        
        # Get data docs sites
        sites = context.get_docs_sites_urls()
        
        print("\n✅ Data Docs generated")
        print(f"Sites: {sites}")
        
        return sites
    except Exception as e:
        print(f"⚠️  Error generating docs: {e}")
        return None



if __name__ == "__main__":
    print("Day 37: Great Expectations - Solutions\n")
    print("=" * 60)
    
    # Create sample dataset with quality issues
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
    
    print(f"Sample dataset created")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("=" * 60)
    
    # Exercise 1: Setup Context
    print("\n📋 Exercise 1: Setup Context")
    print("-" * 60)
    context = exercise_1_setup_context()
    
    if context is not None:
        # Exercise 2: Create Expectation Suite
        print("\n📋 Exercise 2: Create Expectation Suite")
        print("-" * 60)
        validator = exercise_2_create_expectation_suite(context, df)
        
        # Exercise 3: Validate Data
        print("\n📋 Exercise 3: Validate Data")
        print("-" * 60)
        results = exercise_3_validate_data(validator)
        
        # Exercise 4: Add More Expectations
        print("\n📋 Exercise 4: Add More Expectations")
        print("-" * 60)
        validator = exercise_4_add_more_expectations(validator)
        
        # Exercise 5: Create Checkpoint
        print("\n📋 Exercise 5: Create Checkpoint")
        print("-" * 60)
        checkpoint = exercise_5_create_checkpoint(context, validator)
        
        # Exercise 6: Run Checkpoint
        if checkpoint is not None:
            print("\n📋 Exercise 6: Run Checkpoint")
            print("-" * 60)
            checkpoint_result = exercise_6_run_checkpoint(checkpoint)
        
        # Exercise 7: Custom Expectation
        print("\n📋 Exercise 7: Custom Business Rule")
        print("-" * 60)
        exercise_7_custom_expectation(validator, df)
        
        # Exercise 8: Generate Report
        print("\n📋 Exercise 8: Generate Data Docs")
        print("-" * 60)
        exercise_8_generate_report(context)
    else:
        print("\n⚠️  Great Expectations not installed")
        print("To run this solution, install with:")
        print("  pip install great-expectations")
        print("\nThe code structure demonstrates how to:")
        print("  1. Set up GX context")
        print("  2. Create expectation suites")
        print("  3. Validate data")
        print("  4. Create checkpoints")
        print("  5. Generate documentation")
    
    print("\n" + "=" * 60)
    print("✅ All exercises completed!")
    print("=" * 60)
