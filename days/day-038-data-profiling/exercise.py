"""
Day 38: Data Profiling - Exercises
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def exercise_1_basic_profiling(df):
    """
    Exercise 1: Basic data profiling
    
    Generate basic statistics:
    1. Dataset shape (rows, columns)
    2. Column data types
    3. Memory usage
    4. Missing value counts
    
    Return dictionary with profile info
    """
    # TODO: Get shape
    # TODO: Get dtypes
    # TODO: Get memory usage
    # TODO: Get missing value counts
    # TODO: Return profile dictionary
    pass


def exercise_2_numerical_profiling(df):
    """
    Exercise 2: Profile numerical columns
    
    For each numerical column, calculate:
    1. Mean, median, std, min, max
    2. Quartiles (25%, 50%, 75%)
    3. Skewness and kurtosis
    4. Number of zeros
    5. Number of outliers (IQR method)
    
    Return dictionary with stats per column
    """
    # TODO: Select numerical columns
    # TODO: Calculate descriptive statistics
    # TODO: Calculate skewness and kurtosis
    # TODO: Count zeros
    # TODO: Detect outliers using IQR method
    # TODO: Return stats dictionary
    pass


def exercise_3_categorical_profiling(df):
    """
    Exercise 3: Profile categorical columns
    
    For each categorical column, calculate:
    1. Number of unique values
    2. Most common values (top 5)
    3. Value counts
    4. Cardinality ratio (unique/total)
    
    Return dictionary with stats per column
    """
    # TODO: Select categorical columns
    # TODO: Count unique values
    # TODO: Get top 5 most common values
    # TODO: Calculate cardinality ratio
    # TODO: Return stats dictionary
    pass


def exercise_4_missing_data_analysis(df):
    """
    Exercise 4: Analyze missing data patterns
    
    1. Count missing values per column
    2. Calculate missing percentage
    3. Identify columns with >50% missing
    4. Check for missing data patterns
    
    Return missing data report
    """
    # TODO: Count missing per column
    # TODO: Calculate missing percentage
    # TODO: Identify high-missing columns (>50%)
    # TODO: Check if missing data is random or patterned
    # TODO: Return missing data report
    pass


def exercise_5_correlation_analysis(df):
    """
    Exercise 5: Correlation analysis
    
    1. Calculate correlation matrix for numerical columns
    2. Find highly correlated pairs (|corr| > 0.7)
    3. Identify potential multicollinearity
    
    Return correlation report
    """
    # TODO: Select numerical columns
    # TODO: Calculate correlation matrix
    # TODO: Find pairs with |correlation| > 0.7
    # TODO: Return correlation report
    pass


def exercise_6_distribution_analysis(df, column):
    """
    Exercise 6: Analyze distribution of a column
    
    For a given column:
    1. Check if normally distributed (Shapiro-Wilk test)
    2. Calculate distribution statistics
    3. Identify distribution type (normal, skewed, uniform)
    
    Return distribution analysis
    """
    # TODO: Perform Shapiro-Wilk test for normality
    # TODO: Calculate skewness and kurtosis
    # TODO: Classify distribution type
    # TODO: Return distribution analysis
    pass


def exercise_7_outlier_detection(df, column):
    """
    Exercise 7: Detect outliers
    
    Use multiple methods:
    1. IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    2. Z-score method (|z| > 3)
    3. Percentile method (< 1st or > 99th percentile)
    
    Return outlier indices and counts
    """
    # TODO: Implement IQR method
    # TODO: Implement Z-score method
    # TODO: Implement percentile method
    # TODO: Return outlier report with indices
    pass


def exercise_8_generate_profile_report(df):
    """
    Exercise 8: Generate comprehensive profile report
    
    Combine all profiling functions to create:
    1. Dataset overview
    2. Numerical column profiles
    3. Categorical column profiles
    4. Missing data analysis
    5. Correlation analysis
    6. Data quality score
    
    Return formatted report string
    """
    # TODO: Call all profiling functions
    # TODO: Combine results
    # TODO: Calculate overall data quality score
    # TODO: Format as readable report
    # TODO: Return report string
    pass


if __name__ == "__main__":
    print("Day 38: Data Profiling - Exercises\n")
    
    # Create sample dataset
    np.random.seed(42)
    n = 1000
    
    sample_data = {
        'customer_id': range(1, n + 1),
        'age': np.random.normal(45, 15, n).astype(int),
        'income': np.random.lognormal(10.5, 0.5, n),
        'credit_score': np.random.normal(700, 50, n).astype(int),
        'account_balance': np.random.exponential(5000, n),
        'num_transactions': np.random.poisson(20, n),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'status': np.random.choice(['active', 'inactive', 'suspended'], n, p=[0.7, 0.2, 0.1]),
        'signup_date': [datetime.now() - timedelta(days=np.random.randint(0, 1000)) for _ in range(n)]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some missing values
    df.loc[df.sample(frac=0.1).index, 'income'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'credit_score'] = np.nan
    
    # Add some outliers
    df.loc[df.sample(n=10).index, 'income'] = df['income'].max() * 10
    
    print("Sample dataset created")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Uncomment to run exercises
    # print("Exercise 1: Basic Profiling")
    # print(exercise_1_basic_profiling(df))
    
    # print("\nExercise 2: Numerical Profiling")
    # print(exercise_2_numerical_profiling(df))
    
    # print("\nExercise 3: Categorical Profiling")
    # print(exercise_3_categorical_profiling(df))
    
    # print("\nExercise 4: Missing Data Analysis")
    # print(exercise_4_missing_data_analysis(df))
    
    # print("\nExercise 5: Correlation Analysis")
    # print(exercise_5_correlation_analysis(df))
    
    # print("\nExercise 6: Distribution Analysis (age)")
    # print(exercise_6_distribution_analysis(df, 'age'))
    
    # print("\nExercise 7: Outlier Detection (income)")
    # print(exercise_7_outlier_detection(df, 'income'))
    
    # print("\nExercise 8: Profile Report")
    # print(exercise_8_generate_profile_report(df))
