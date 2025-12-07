"""
Day 39: Data Validation - Solutions
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta


def exercise_1_type_validation(df, expected_types):
    """Validate column data types"""
    errors = []
    
    for col, expected_type in expected_types.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
        elif str(df[col].dtype) != expected_type:
            errors.append(f"{col}: expected {expected_type}, got {df[col].dtype}")
    
    passed = len(errors) == 0
    
    return {
        'passed': passed,
        'errors': errors,
        'checked_columns': len(expected_types)
    }


def exercise_2_range_validation(df, column, min_val, max_val):
    """Validate values are within range"""
    invalid = df[(df[column] < min_val) | (df[column] > max_val)]
    passed = len(invalid) == 0
    
    return passed, invalid.index.tolist()


def exercise_3_format_validation(df):
    """Validate email and phone formats"""
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    phone_pattern = r'^\d{10}$'
    
    # Validate emails
    valid_emails = df['email'].dropna().apply(lambda x: bool(re.match(email_pattern, str(x))))
    email_passed = valid_emails.all()
    invalid_email_count = (~valid_emails).sum()
    
    # Validate phones
    valid_phones = df['phone'].dropna().apply(lambda x: bool(re.match(phone_pattern, str(x))))
    phone_passed = valid_phones.all()
    invalid_phone_count = (~valid_phones).sum()
    
    return {
        'email': {
            'passed': email_passed,
            'invalid_count': invalid_email_count,
            'valid_percentage': valid_emails.sum() / len(valid_emails) * 100
        },
        'phone': {
            'passed': phone_passed,
            'invalid_count': invalid_phone_count,
            'valid_percentage': valid_phones.sum() / len(valid_phones) * 100
        }
    }


def exercise_4_uniqueness_validation(df, columns):
    """Check for duplicates"""
    duplicates = df.duplicated(subset=columns, keep=False)
    duplicate_df = df[duplicates]
    
    return {
        'passed': not duplicates.any(),
        'duplicate_count': duplicates.sum(),
        'duplicate_rows': duplicate_df.index.tolist(),
        'unique_percentage': (1 - duplicates.sum() / len(df)) * 100
    }


def exercise_5_business_rules(df):
    """Validate business rules"""
    violations = []
    
    # Rule 1: Active customers must have email
    active_no_email = df[(df['status'] == 'active') & (df['email'].isna())]
    if len(active_no_email) > 0:
        violations.append({
            'rule': 'Active customers must have email',
            'violation_count': len(active_no_email),
            'indices': active_no_email.index.tolist()
        })
    
    # Rule 2: Amount must be positive
    negative_amounts = df[df['amount'] <= 0]
    if len(negative_amounts) > 0:
        violations.append({
            'rule': 'Amount must be positive',
            'violation_count': len(negative_amounts),
            'indices': negative_amounts.index.tolist()
        })
    
    # Rule 3: Age must be >= 18
    underage = df[df['age'] < 18]
    if len(underage) > 0:
        violations.append({
            'rule': 'Age must be >= 18',
            'violation_count': len(underage),
            'indices': underage.index.tolist()
        })
    
    return {
        'passed': len(violations) == 0,
        'violations': violations,
        'rules_checked': 3
    }


def exercise_6_build_validator(df):
    """Build validation framework"""
    results = []
    
    # Rule 1: customer_id not null
    null_ids = df['customer_id'].isna().sum()
    results.append({
        'rule': 'customer_id not null',
        'passed': null_ids == 0,
        'message': f"Found {null_ids} null values" if null_ids > 0 else "No nulls"
    })
    
    # Rule 2: customer_id unique
    dup_ids = df['customer_id'].duplicated().sum()
    results.append({
        'rule': 'customer_id unique',
        'passed': dup_ids == 0,
        'message': f"Found {dup_ids} duplicates" if dup_ids > 0 else "All unique"
    })
    
    # Rule 3: age between 0 and 120
    invalid_age = df[(df['age'] < 0) | (df['age'] > 120)]
    results.append({
        'rule': 'age between 0 and 120',
        'passed': len(invalid_age) == 0,
        'message': f"Found {len(invalid_age)} invalid ages" if len(invalid_age) > 0 else "All valid"
    })
    
    # Rule 4: email format valid
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    invalid_emails = df['email'].dropna().apply(lambda x: not bool(re.match(email_pattern, str(x))))
    results.append({
        'rule': 'email format valid',
        'passed': not invalid_emails.any(),
        'message': f"Found {invalid_emails.sum()} invalid emails" if invalid_emails.any() else "All valid"
    })
    
    # Rule 5: status in allowed values
    allowed_statuses = ['active', 'inactive', 'suspended']
    invalid_status = ~df['status'].isin(allowed_statuses)
    results.append({
        'rule': 'status in allowed values',
        'passed': not invalid_status.any(),
        'message': f"Found {invalid_status.sum()} invalid statuses" if invalid_status.any() else "All valid"
    })
    
    passed = all(r['passed'] for r in results)
    
    return {
        'passed': passed,
        'total_rules': len(results),
        'passed_rules': sum(1 for r in results if r['passed']),
        'failed_rules': sum(1 for r in results if not r['passed']),
        'results': results
    }


def exercise_7_validation_pipeline(df, quarantine_path):
    """ETL with validation and quarantine"""
    # Run validation
    validation_results = exercise_6_build_validator(df)
    
    # Collect invalid row indices
    invalid_indices = set()
    
    # Check each validation rule
    if df['customer_id'].isna().any():
        invalid_indices.update(df[df['customer_id'].isna()].index)
    
    if df['customer_id'].duplicated().any():
        invalid_indices.update(df[df['customer_id'].duplicated(keep=False)].index)
    
    invalid_age = df[(df['age'] < 0) | (df['age'] > 120)]
    invalid_indices.update(invalid_age.index)
    
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    invalid_emails = df[df['email'].dropna().apply(lambda x: not bool(re.match(email_pattern, str(x))))]
    invalid_indices.update(invalid_emails.index)
    
    allowed_statuses = ['active', 'inactive', 'suspended']
    invalid_status = df[~df['status'].isin(allowed_statuses)]
    invalid_indices.update(invalid_status.index)
    
    # Split valid and invalid
    valid_df = df[~df.index.isin(invalid_indices)]
    invalid_df = df[df.index.isin(invalid_indices)]
    
    # Save invalid records
    if len(invalid_df) > 0:
        invalid_df.to_csv(quarantine_path, index=False)
    
    return valid_df, invalid_df, validation_results


def exercise_8_schema_validation(df, schema):
    """Comprehensive schema validation"""
    errors = []
    
    for column, rules in schema.items():
        # Check column exists
        if column not in df.columns:
            errors.append(f"Missing required column: {column}")
            continue
        
        # Check type
        if 'type' in rules and str(df[column].dtype) != rules['type']:
            errors.append(f"{column}: wrong type {df[column].dtype}, expected {rules['type']}")
        
        # Check nullable
        if not rules.get('nullable', True):
            null_count = df[column].isna().sum()
            if null_count > 0:
                errors.append(f"{column}: {null_count} null values (not allowed)")
        
        # Check unique
        if rules.get('unique', False):
            dup_count = df[column].duplicated().sum()
            if dup_count > 0:
                errors.append(f"{column}: {dup_count} duplicates (must be unique)")
        
        # Check range
        if 'min' in rules:
            below_min = (df[column] < rules['min']).sum()
            if below_min > 0:
                errors.append(f"{column}: {below_min} values below min {rules['min']}")
        
        if 'max' in rules:
            above_max = (df[column] > rules['max']).sum()
            if above_max > 0:
                errors.append(f"{column}: {above_max} values above max {rules['max']}")
        
        # Check allowed values
        if 'allowed_values' in rules:
            invalid = ~df[column].isin(rules['allowed_values'])
            invalid_count = invalid.sum()
            if invalid_count > 0:
                errors.append(f"{column}: {invalid_count} invalid values")
    
    passed = len(errors) == 0
    
    return passed, errors



if __name__ == "__main__":
    print("Day 39: Data Validation - Solutions\n")
    print("=" * 70)
    
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
    
    print(f"Sample dataset: {len(df)} rows, {len(df.columns)} columns")
    print("=" * 70)
    
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
    
    print("\n📋 Exercise 1: Type Validation")
    print("-" * 70)
    result = exercise_1_type_validation(df, expected_types)
    print(f"Passed: {result['passed']}")
    print(f"Checked: {result['checked_columns']} columns")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    
    print("\n📋 Exercise 2: Range Validation (age)")
    print("-" * 70)
    passed, invalid_indices = exercise_2_range_validation(df, 'age', 0, 120)
    print(f"Passed: {passed}")
    print(f"Invalid values: {len(invalid_indices)}")
    
    print("\n📋 Exercise 3: Format Validation")
    print("-" * 70)
    result = exercise_3_format_validation(df)
    print(f"Email - Passed: {result['email']['passed']}, Invalid: {result['email']['invalid_count']}")
    print(f"Phone - Passed: {result['phone']['passed']}, Invalid: {result['phone']['invalid_count']}")
    
    print("\n📋 Exercise 4: Uniqueness Validation")
    print("-" * 70)
    result = exercise_4_uniqueness_validation(df, ['customer_id'])
    print(f"Passed: {result['passed']}")
    print(f"Duplicates: {result['duplicate_count']}")
    
    print("\n📋 Exercise 5: Business Rules")
    print("-" * 70)
    result = exercise_5_business_rules(df)
    print(f"Passed: {result['passed']}")
    print(f"Violations: {len(result['violations'])}")
    for v in result['violations']:
        print(f"  - {v['rule']}: {v['violation_count']} violations")
    
    print("\n📋 Exercise 6: Build Validator")
    print("-" * 70)
    result = exercise_6_build_validator(df)
    print(f"Overall: {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
    print(f"Rules: {result['passed_rules']}/{result['total_rules']} passed")
    for r in result['results']:
        status = '✅' if r['passed'] else '❌'
        print(f"  {status} {r['rule']}: {r['message']}")
    
    print("\n📋 Exercise 7: Validation Pipeline")
    print("-" * 70)
    valid_df, invalid_df, validation_results = exercise_7_validation_pipeline(df, 'quarantine.csv')
    print(f"Valid records: {len(valid_df)}")
    print(f"Invalid records: {len(invalid_df)}")
    print(f"Quarantined to: quarantine.csv")
    
    print("\n📋 Exercise 8: Schema Validation")
    print("-" * 70)
    passed, errors = exercise_8_schema_validation(df, schema)
    print(f"Passed: {passed}")
    print(f"Errors: {len(errors)}")
    for error in errors[:5]:
        print(f"  - {error}")
    
    print("\n" + "=" * 70)
    print("✅ All exercises completed!")
    print("=" * 70)
