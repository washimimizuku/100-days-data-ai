"""
Day 58: Feature Engineering - Solutions
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def exercise_1_missing_values():
    """Exercise 1: Handle Missing Values"""
    data = {
        'age': [25, 30, np.nan, 45, 50, np.nan, 35],
        'income': [50000, np.nan, 55000, np.nan, 80000, 60000, 70000],
        'score': [85, 90, np.nan, 88, 92, np.nan, 87]
    }
    df = pd.DataFrame(data)
    
    print("Original Data:")
    print(df)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nMissing Percentage:")
    print((df.isnull().sum() / len(df) * 100).round(2))
    
    imputer_median = SimpleImputer(strategy='median')
    df['age'] = imputer_median.fit_transform(df[['age']])
    
    imputer_mean = SimpleImputer(strategy='mean')
    df['income'] = imputer_mean.fit_transform(df[['income']])
    
    df['score'] = df['score'].fillna(method='ffill')
    
    print("\nCleaned Data:")
    print(df)


def exercise_2_categorical_encoding():
    """Exercise 2: Encode Categorical Variables"""
    data = {
        'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
        'color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
        'size': ['Small', 'Medium', 'Large', 'Medium', 'Small', 'Large']
    }
    df = pd.DataFrame(data)
    
    print("Original Data:")
    print(df)
    
    education_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df['education_encoded'] = df['education'].map(education_map)
    
    df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')
    
    size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
    df_encoded['size_encoded'] = df_encoded['size'].map(size_map)
    
    print("\nEncoded Data:")
    print(df_encoded)
    
    print("\nEducation Mapping:")
    for edu, val in education_map.items():
        print(f"  {edu}: {val}")


def exercise_3_feature_scaling():
    """Exercise 3: Apply Feature Scaling"""
    data = {
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 55000, 80000, 70000],
        'score': [85, 90, 88, 92, 87]
    }
    df = pd.DataFrame(data)
    
    print("Original Data:")
    print(df)
    print("\nOriginal Statistics:")
    print(df.describe())
    
    standard_scaler = StandardScaler()
    df_standard = pd.DataFrame(
        standard_scaler.fit_transform(df),
        columns=[f'{col}_standard' for col in df.columns]
    )
    
    print("\nStandardScaler (mean=0, std=1):")
    print(df_standard)
    print("\nStandard Scaled Statistics:")
    print(df_standard.describe())
    
    minmax_scaler = MinMaxScaler()
    df_minmax = pd.DataFrame(
        minmax_scaler.fit_transform(df),
        columns=[f'{col}_minmax' for col in df.columns]
    )
    
    print("\nMinMaxScaler (range [0,1]):")
    print(df_minmax)
    print("\nMinMax Scaled Statistics:")
    print(df_minmax.describe())


def exercise_4_feature_creation():
    """Exercise 4: Create New Features"""
    data = {
        'price': [200000, 300000, 250000, 400000, 350000],
        'sqft': [1000, 1500, 1200, 2000, 1800],
        'bedrooms': [2, 3, 2, 4, 3],
        'bathrooms': [1, 2, 1, 3, 2],
        'year_built': [1990, 2000, 1995, 2010, 2005]
    }
    df = pd.DataFrame(data)
    
    print("Original Data:")
    print(df)
    
    df['price_per_sqft'] = df['price'] / df['sqft']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['age'] = 2024 - df['year_built']
    df['is_new'] = (df['age'] < 10).astype(int)
    
    print("\nData with New Features:")
    print(df)
    
    print("\nNew Features Summary:")
    print(f"Price per sqft range: ${df['price_per_sqft'].min():.2f} - ${df['price_per_sqft'].max():.2f}")
    print(f"Total rooms range: {df['total_rooms'].min()} - {df['total_rooms'].max()}")
    print(f"Age range: {df['age'].min()} - {df['age'].max()} years")
    print(f"New houses: {df['is_new'].sum()} out of {len(df)}")


def exercise_5_complete_pipeline():
    """Exercise 5: Complete Feature Engineering Pipeline"""
    data = {
        'age': [25, 30, np.nan, 45, 50, 35, np.nan, 40],
        'income': [50000, np.nan, 55000, 80000, 70000, 60000, 65000, np.nan],
        'education': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Master', 'Bachelor', 'PhD', 'Master'],
        'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'NYC', 'SF', 'LA'],
        'experience': [3, 5, 4, 10, 8, 6, 12, 7]
    }
    df = pd.DataFrame(data)
    
    print("Step 1: Original Data")
    print(df)
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    print("\nStep 2: Handle Missing Values")
    imputer = SimpleImputer(strategy='median')
    df[['age', 'income']] = imputer.fit_transform(df[['age', 'income']])
    print(f"Missing values after imputation:\n{df.isnull().sum()}")
    
    print("\nStep 3: Create New Feature")
    df['income_per_age'] = df['income'] / df['age']
    print(f"Created 'income_per_age' feature")
    
    print("\nStep 4: Encode Education (Ordinal)")
    education_map = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
    df['education_encoded'] = df['education'].map(education_map)
    print(f"Education mapping: {education_map}")
    
    print("\nStep 5: One-Hot Encode City")
    df = pd.get_dummies(df, columns=['city'], prefix='city')
    print(f"City columns created: {[col for col in df.columns if col.startswith('city_')]}")
    
    print("\nStep 6: Scale Numerical Features")
    numerical_cols = ['age', 'income', 'experience', 'income_per_age']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print(f"Scaled features: {numerical_cols}")
    
    print("\nStep 7: Final Transformed Data")
    print(df)
    
    print("\nStep 8: Feature Summary")
    print(f"Total features: {len(df.columns)}")
    print(f"Feature names: {df.columns.tolist()}")
    print(f"\nNumerical features (scaled): {numerical_cols}")
    print(f"Categorical features (encoded): education_encoded, city_*")


if __name__ == "__main__":
    print("Day 58: Feature Engineering - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Missing Values")
    print("=" * 60)
    exercise_1_missing_values()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Categorical Encoding")
    print("=" * 60)
    exercise_2_categorical_encoding()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Feature Scaling")
    print("=" * 60)
    exercise_3_feature_scaling()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Feature Creation")
    print("=" * 60)
    exercise_4_feature_creation()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Complete Pipeline")
    print("=" * 60)
    exercise_5_complete_pipeline()
