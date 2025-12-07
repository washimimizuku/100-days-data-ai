"""
Day 58: Feature Engineering - Exercises
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def exercise_1_missing_values():
    """
    Exercise 1: Handle Missing Values
    
    Apply different strategies for handling missing data.
    """
    data = {
        'age': [25, 30, np.nan, 45, 50, np.nan, 35],
        'income': [50000, np.nan, 55000, np.nan, 80000, 60000, 70000],
        'score': [85, 90, np.nan, 88, 92, np.nan, 87]
    }
    df = pd.DataFrame(data)
    
    # TODO: Print missing value counts
    # TODO: Calculate percentage of missing values per column
    # TODO: Impute 'age' with median
    # TODO: Impute 'income' with mean
    # TODO: Impute 'score' with forward fill
    # TODO: Print cleaned dataframe
    pass


def exercise_2_categorical_encoding():
    """
    Exercise 2: Encode Categorical Variables
    
    Apply appropriate encoding for different categorical types.
    """
    data = {
        'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
        'color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
        'size': ['Small', 'Medium', 'Large', 'Medium', 'Small', 'Large']
    }
    df = pd.DataFrame(data)
    
    # TODO: Label encode 'education' (ordinal: High School < Bachelor < Master < PhD)
    # TODO: One-hot encode 'color' (nominal: no order)
    # TODO: Map 'size' to numbers (Small=1, Medium=2, Large=3)
    # TODO: Print encoded dataframe
    pass


def exercise_3_feature_scaling():
    """
    Exercise 3: Apply Feature Scaling
    
    Compare different scaling techniques.
    """
    data = {
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 55000, 80000, 70000],
        'score': [85, 90, 88, 92, 87]
    }
    df = pd.DataFrame(data)
    
    # TODO: Apply StandardScaler to all features
    # TODO: Apply MinMaxScaler to all features
    # TODO: Print original and scaled dataframes
    # TODO: Compare the results
    pass


def exercise_4_feature_creation():
    """
    Exercise 4: Create New Features
    
    Generate new features from existing ones.
    """
    data = {
        'price': [200000, 300000, 250000, 400000, 350000],
        'sqft': [1000, 1500, 1200, 2000, 1800],
        'bedrooms': [2, 3, 2, 4, 3],
        'bathrooms': [1, 2, 1, 3, 2],
        'year_built': [1990, 2000, 1995, 2010, 2005]
    }
    df = pd.DataFrame(data)
    
    # TODO: Create 'price_per_sqft' feature
    # TODO: Create 'total_rooms' feature (bedrooms + bathrooms)
    # TODO: Create 'age' feature (2024 - year_built)
    # TODO: Create 'is_new' feature (1 if age < 10, else 0)
    # TODO: Print dataframe with new features
    pass


def exercise_5_complete_pipeline():
    """
    Exercise 5: Complete Feature Engineering Pipeline
    
    Build end-to-end pipeline for real dataset.
    """
    data = {
        'age': [25, 30, np.nan, 45, 50, 35, np.nan, 40],
        'income': [50000, np.nan, 55000, 80000, 70000, 60000, 65000, np.nan],
        'education': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Master', 'Bachelor', 'PhD', 'Master'],
        'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'NYC', 'SF', 'LA'],
        'experience': [3, 5, 4, 10, 8, 6, 12, 7]
    }
    df = pd.DataFrame(data)
    
    # TODO: 1. Handle missing values (median for numerical)
    # TODO: 2. Create 'income_per_age' feature
    # TODO: 3. Encode 'education' (ordinal mapping)
    # TODO: 4. One-hot encode 'city'
    # TODO: 5. Scale numerical features (StandardScaler)
    # TODO: 6. Print final transformed dataframe
    # TODO: 7. Print feature names
    pass


if __name__ == "__main__":
    print("Day 58: Feature Engineering - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Missing Values")
    print("=" * 60)
    # exercise_1_missing_values()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Categorical Encoding")
    print("=" * 60)
    # exercise_2_categorical_encoding()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Feature Scaling")
    print("=" * 60)
    # exercise_3_feature_scaling()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Feature Creation")
    print("=" * 60)
    # exercise_4_feature_creation()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Complete Pipeline")
    print("=" * 60)
    # exercise_5_complete_pipeline()
