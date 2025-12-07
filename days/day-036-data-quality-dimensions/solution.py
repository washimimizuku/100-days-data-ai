"""
Day 36: Data Quality Dimensions - Solutions
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta


def exercise_1_check_accuracy(df):
    """Check data accuracy"""
    results = {}
    
    # Email format check
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    valid_emails = df['email'].dropna().apply(lambda x: bool(re.match(email_pattern, str(x))))
    results['email_accuracy'] = valid_emails.sum() / len(df['email'].dropna())
    
    # Age range check
    valid_ages = df['age'].between(0, 120)
    results['age_accuracy'] = valid_ages.sum() / len(df)
    
    # Phone number check (10 digits)
    valid_phones = df['phone'].dropna().apply(lambda x: len(str(x)) == 10)
    results['phone_accuracy'] = valid_phones.sum() / len(df['phone'].dropna())
    
    results['overall_accuracy'] = np.mean(list(results.values()))
    
    return results


def exercise_2_check_completeness(df):
    """Check data completeness"""
    critical_fields = ['customer_id', 'name', 'email', 'phone', 'created_at']
    results = {}
    
    for field in critical_fields:
        if field in df.columns:
            completeness = 1 - (df[field].isna().sum() / len(df))
            results[f'{field}_completeness'] = completeness
    
    results['overall_completeness'] = np.mean(list(results.values()))
    
    return results


def exercise_3_check_consistency(df):
    """Check data consistency"""
    results = {}
    
    # Date format consistency
    try:
        consistent_dates = pd.to_datetime(df['created_at'], errors='coerce').notna()
        results['date_consistency'] = consistent_dates.sum() / len(df)
    except:
        results['date_consistency'] = 0.0
    
    # Country code consistency (2 uppercase letters)
    country_pattern = r'^[A-Z]{2}$'
    valid_countries = df['country'].apply(lambda x: bool(re.match(country_pattern, str(x))))
    results['country_consistency'] = valid_countries.sum() / len(df)
    
    # Status values consistency
    allowed_statuses = ['active', 'inactive', 'suspended']
    valid_statuses = df['status'].isin(allowed_statuses)
    results['status_consistency'] = valid_statuses.sum() / len(df)
    
    results['overall_consistency'] = np.mean(list(results.values()))
    
    return results


def exercise_4_check_validity(df):
    """Check data validity"""
    results = {}
    
    # Status validity
    allowed_statuses = ['active', 'inactive', 'suspended']
    valid_status = df['status'].isin(allowed_statuses)
    results['status_validity'] = valid_status.sum() / len(df)
    
    # Age validity (positive)
    valid_age = df['age'] > 0
    results['age_validity'] = valid_age.sum() / len(df)
    
    # Created_at in the past
    now = datetime.now()
    past_dates = pd.to_datetime(df['created_at']) < now
    results['date_validity'] = past_dates.sum() / len(df)
    
    results['overall_validity'] = np.mean(list(results.values()))
    
    return results


def exercise_5_check_uniqueness(df):
    """Check data uniqueness"""
    results = {}
    
    # Customer ID uniqueness
    duplicate_ids = df['customer_id'].duplicated().sum()
    results['customer_id_uniqueness'] = 1 - (duplicate_ids / len(df))
    
    # Email uniqueness
    duplicate_emails = df['email'].duplicated().sum()
    results['email_uniqueness'] = 1 - (duplicate_emails / len(df))
    
    # Overall record uniqueness
    duplicate_records = df.duplicated().sum()
    results['record_uniqueness'] = 1 - (duplicate_records / len(df))
    
    results['overall_uniqueness'] = np.mean(list(results.values()))
    
    return results


def exercise_6_check_timeliness(df):
    """Check data timeliness"""
    results = {}
    now = datetime.now()
    
    # Records updated in last 7 days
    updated_at = pd.to_datetime(df['updated_at'])
    last_7_days = (now - updated_at) <= timedelta(days=7)
    results['updated_7_days'] = last_7_days.sum() / len(df)
    
    # Records updated in last 30 days
    last_30_days = (now - updated_at) <= timedelta(days=30)
    results['updated_30_days'] = last_30_days.sum() / len(df)
    
    # Average data age
    avg_age = (now - updated_at).dt.days.mean()
    results['avg_age_days'] = avg_age
    
    # Timeliness score (based on 30-day threshold)
    results['timeliness_score'] = results['updated_30_days']
    
    return results


def exercise_7_calculate_quality_score(df):
    """Calculate overall quality score"""
    # Get all dimension scores
    accuracy = exercise_1_check_accuracy(df)
    completeness = exercise_2_check_completeness(df)
    consistency = exercise_3_check_consistency(df)
    validity = exercise_4_check_validity(df)
    uniqueness = exercise_5_check_uniqueness(df)
    timeliness = exercise_6_check_timeliness(df)
    
    # Define weights
    weights = {
        'accuracy': 0.20,
        'completeness': 0.20,
        'consistency': 0.15,
        'validity': 0.20,
        'uniqueness': 0.15,
        'timeliness': 0.10
    }
    
    # Calculate weighted score
    weighted_score = (
        accuracy['overall_accuracy'] * weights['accuracy'] +
        completeness['overall_completeness'] * weights['completeness'] +
        consistency['overall_consistency'] * weights['consistency'] +
        validity['overall_validity'] * weights['validity'] +
        uniqueness['overall_uniqueness'] * weights['uniqueness'] +
        timeliness['timeliness_score'] * weights['timeliness']
    )
    
    # Assign grade
    if weighted_score >= 0.95:
        grade = 'A'
    elif weighted_score >= 0.85:
        grade = 'B'
    elif weighted_score >= 0.75:
        grade = 'C'
    elif weighted_score >= 0.65:
        grade = 'D'
    else:
        grade = 'F'
    
    return {
        'dimensions': {
            'accuracy': accuracy['overall_accuracy'],
            'completeness': completeness['overall_completeness'],
            'consistency': consistency['overall_consistency'],
            'validity': validity['overall_validity'],
            'uniqueness': uniqueness['overall_uniqueness'],
            'timeliness': timeliness['timeliness_score']
        },
        'overall_score': weighted_score,
        'grade': grade
    }



def exercise_8_generate_quality_report(df):
    """Generate quality report"""
    quality_score = exercise_7_calculate_quality_score(df)
    
    report = f"""
{'='*60}
DATA QUALITY REPORT
{'='*60}

Dataset Summary:
  Rows:        {len(df):,}
  Columns:     {len(df.columns)}
  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
Quality Scores by Dimension:
{'='*60}
  Accuracy:      {quality_score['dimensions']['accuracy']:.1%}
  Completeness:  {quality_score['dimensions']['completeness']:.1%}
  Consistency:   {quality_score['dimensions']['consistency']:.1%}
  Validity:      {quality_score['dimensions']['validity']:.1%}
  Uniqueness:    {quality_score['dimensions']['uniqueness']:.1%}
  Timeliness:    {quality_score['dimensions']['timeliness']:.1%}

{'='*60}
Overall Quality Score: {quality_score['overall_score']:.1%} (Grade: {quality_score['grade']})
{'='*60}

Top Quality Issues:
  1. Invalid email formats: {(1 - exercise_1_check_accuracy(df)['email_accuracy']) * 100:.1f}%
  2. Missing names: {df['name'].isna().sum()} records
  3. Invalid ages: {(df['age'] < 0).sum() + (df['age'] > 120).sum()} records
  4. Inconsistent country codes: {(1 - exercise_3_check_consistency(df)['country_consistency']) * 100:.1f}%
  5. Invalid status values: {(1 - exercise_3_check_consistency(df)['status_consistency']) * 100:.1f}%

Recommendations:
  • Implement email validation at data entry point
  • Make 'name' field mandatory in source systems
  • Add age range validation (0-120)
  • Standardize country codes to ISO 3166-1 alpha-2
  • Restrict status values to allowed list
  • Increase data refresh frequency for timeliness

{'='*60}
"""
    return report


if __name__ == "__main__":
    print("Day 36: Data Quality Dimensions - Solutions\n")
    
    # Create sample dataset with quality issues
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
    
    print("Exercise 1: Accuracy Check")
    accuracy = exercise_1_check_accuracy(df)
    for key, value in accuracy.items():
        print(f"  {key}: {value:.2%}")
    
    print("\nExercise 2: Completeness Check")
    completeness = exercise_2_check_completeness(df)
    for key, value in completeness.items():
        print(f"  {key}: {value:.2%}")
    
    print("\nExercise 3: Consistency Check")
    consistency = exercise_3_check_consistency(df)
    for key, value in consistency.items():
        print(f"  {key}: {value:.2%}")
    
    print("\nExercise 4: Validity Check")
    validity = exercise_4_check_validity(df)
    for key, value in validity.items():
        print(f"  {key}: {value:.2%}")
    
    print("\nExercise 5: Uniqueness Check")
    uniqueness = exercise_5_check_uniqueness(df)
    for key, value in uniqueness.items():
        print(f"  {key}: {value:.2%}")
    
    print("\nExercise 6: Timeliness Check")
    timeliness = exercise_6_check_timeliness(df)
    for key, value in timeliness.items():
        if 'days' in key:
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value:.2%}")
    
    print("\nExercise 7: Overall Quality Score")
    quality = exercise_7_calculate_quality_score(df)
    print(f"  Overall Score: {quality['overall_score']:.2%}")
    print(f"  Grade: {quality['grade']}")
    
    print("\nExercise 8: Quality Report")
    report = exercise_8_generate_quality_report(df)
    print(report)
