"""
Day 38: Data Profiling - Solutions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats


def exercise_1_basic_profiling(df):
    """Basic data profiling"""
    profile = {
        'shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'dtypes': df.dtypes.value_counts().to_dict(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isna().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    return profile


def exercise_2_numerical_profiling(df):
    """Profile numerical columns"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    profiles = {}
    
    for col in numerical_cols:
        series = df[col].dropna()
        
        # IQR outlier detection
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
        
        profiles[col] = {
            'count': df[col].count(),
            'missing': df[col].isna().sum(),
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'q25': Q1,
            'q75': Q3,
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'zeros': (df[col] == 0).sum(),
            'outliers': len(outliers)
        }
    
    return profiles


def exercise_3_categorical_profiling(df):
    """Profile categorical columns"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    profiles = {}
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        
        profiles[col] = {
            'count': df[col].count(),
            'missing': df[col].isna().sum(),
            'unique': df[col].nunique(),
            'cardinality_ratio': df[col].nunique() / len(df),
            'top_5_values': value_counts.head(5).to_dict(),
            'mode': df[col].mode()[0] if len(df[col].mode()) > 0 else None
        }
    
    return profiles


def exercise_4_missing_data_analysis(df):
    """Analyze missing data patterns"""
    missing_stats = []
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        
        missing_stats.append({
            'column': col,
            'missing_count': missing_count,
            'missing_percentage': missing_pct,
            'dtype': str(df[col].dtype)
        })
    
    missing_df = pd.DataFrame(missing_stats)
    missing_df = missing_df.sort_values('missing_percentage', ascending=False)
    
    high_missing = missing_df[missing_df['missing_percentage'] > 50]
    
    return {
        'missing_by_column': missing_df.to_dict('records'),
        'high_missing_columns': high_missing['column'].tolist(),
        'total_missing_cells': df.isna().sum().sum(),
        'missing_percentage': df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
    }


def exercise_5_correlation_analysis(df):
    """Correlation analysis"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) < 2:
        return {'message': 'Not enough numerical columns for correlation'}
    
    corr_matrix = df[numerical_cols].corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'column1': corr_matrix.columns[i],
                    'column2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'high_correlations': high_corr_pairs,
        'multicollinearity_risk': len(high_corr_pairs) > 0
    }


def exercise_6_distribution_analysis(df, column):
    """Analyze distribution of a column"""
    series = df[column].dropna()
    
    # Shapiro-Wilk test for normality
    if len(series) > 3:
        statistic, p_value = stats.shapiro(series[:5000])  # Limit for performance
        is_normal = p_value > 0.05
    else:
        statistic, p_value, is_normal = None, None, None
    
    skewness = series.skew()
    kurtosis = series.kurtosis()
    
    # Classify distribution
    if is_normal:
        dist_type = 'Normal'
    elif abs(skewness) < 0.5:
        dist_type = 'Approximately symmetric'
    elif skewness > 0.5:
        dist_type = 'Right-skewed (positive skew)'
    else:
        dist_type = 'Left-skewed (negative skew)'
    
    return {
        'shapiro_statistic': statistic,
        'shapiro_p_value': p_value,
        'is_normal': is_normal,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'distribution_type': dist_type
    }


def exercise_7_outlier_detection(df, column):
    """Detect outliers using multiple methods"""
    series = df[column].dropna()
    
    # IQR method
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
    
    # Z-score method
    z_scores = np.abs(stats.zscore(series))
    zscore_outliers = series[z_scores > 3]
    
    # Percentile method
    p1 = series.quantile(0.01)
    p99 = series.quantile(0.99)
    percentile_outliers = series[(series < p1) | (series > p99)]
    
    return {
        'iqr_method': {
            'count': len(iqr_outliers),
            'percentage': len(iqr_outliers) / len(series) * 100,
            'lower_bound': Q1 - 1.5*IQR,
            'upper_bound': Q3 + 1.5*IQR
        },
        'zscore_method': {
            'count': len(zscore_outliers),
            'percentage': len(zscore_outliers) / len(series) * 100
        },
        'percentile_method': {
            'count': len(percentile_outliers),
            'percentage': len(percentile_outliers) / len(series) * 100,
            'lower_bound': p1,
            'upper_bound': p99
        }
    }



def exercise_8_generate_profile_report(df):
    """Generate comprehensive profile report"""
    # Get all profiles
    basic = exercise_1_basic_profiling(df)
    numerical = exercise_2_numerical_profiling(df)
    categorical = exercise_3_categorical_profiling(df)
    missing = exercise_4_missing_data_analysis(df)
    correlation = exercise_5_correlation_analysis(df)
    
    # Calculate quality score
    completeness = 1 - (missing['missing_percentage'] / 100)
    quality_score = completeness * 100
    
    report = f"""
{'='*70}
DATA PROFILING REPORT
{'='*70}

DATASET OVERVIEW
{'-'*70}
Rows:              {basic['shape']['rows']:,}
Columns:           {basic['shape']['columns']}
Memory Usage:      {basic['memory_mb']:.2f} MB
Duplicates:        {basic['duplicates']}

DATA TYPES
{'-'*70}
"""
    
    for dtype, count in basic['dtypes'].items():
        report += f"{str(dtype):20} {count} columns\n"
    
    report += f"""
MISSING DATA
{'-'*70}
Total Missing:     {missing['total_missing_cells']:,} cells
Missing %:         {missing['missing_percentage']:.2f}%
High Missing Cols: {len(missing['high_missing_columns'])}
"""
    
    if missing['high_missing_columns']:
        report += f"  Columns: {', '.join(missing['high_missing_columns'])}\n"
    
    report += f"""
NUMERICAL COLUMNS ({len(numerical)})
{'-'*70}
"""
    
    for col, stats in numerical.items():
        report += f"\n{col}:\n"
        report += f"  Count: {stats['count']}, Missing: {stats['missing']}\n"
        report += f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}\n"
        report += f"  Std: {stats['std']:.2f}, Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n"
        report += f"  Skewness: {stats['skewness']:.2f}, Outliers: {stats['outliers']}\n"
    
    report += f"""
CATEGORICAL COLUMNS ({len(categorical)})
{'-'*70}
"""
    
    for col, stats in categorical.items():
        report += f"\n{col}:\n"
        report += f"  Unique: {stats['unique']}, Cardinality: {stats['cardinality_ratio']:.2%}\n"
        report += f"  Mode: {stats['mode']}\n"
        report += f"  Top values: {list(stats['top_5_values'].keys())[:3]}\n"
    
    report += f"""
CORRELATIONS
{'-'*70}
High Correlations: {len(correlation.get('high_correlations', []))}
"""
    
    for corr in correlation.get('high_correlations', [])[:5]:
        report += f"  {corr['column1']} <-> {corr['column2']}: {corr['correlation']:.3f}\n"
    
    report += f"""
DATA QUALITY SCORE
{'-'*70}
Completeness:      {completeness:.2%}
Overall Score:     {quality_score:.1f}/100

{'='*70}
"""
    
    return report


if __name__ == "__main__":
    print("Day 38: Data Profiling - Solutions\n")
    
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
    
    print("Exercise 1: Basic Profiling")
    basic = exercise_1_basic_profiling(df)
    print(f"  Rows: {basic['shape']['rows']}, Columns: {basic['shape']['columns']}")
    print(f"  Memory: {basic['memory_mb']:.2f} MB")
    print(f"  Duplicates: {basic['duplicates']}")
    
    print("\nExercise 2: Numerical Profiling")
    numerical = exercise_2_numerical_profiling(df)
    print(f"  Profiled {len(numerical)} numerical columns")
    for col in list(numerical.keys())[:2]:
        print(f"  {col}: mean={numerical[col]['mean']:.2f}, outliers={numerical[col]['outliers']}")
    
    print("\nExercise 3: Categorical Profiling")
    categorical = exercise_3_categorical_profiling(df)
    print(f"  Profiled {len(categorical)} categorical columns")
    for col in categorical:
        print(f"  {col}: {categorical[col]['unique']} unique values")
    
    print("\nExercise 4: Missing Data Analysis")
    missing = exercise_4_missing_data_analysis(df)
    print(f"  Total missing: {missing['total_missing_cells']} cells ({missing['missing_percentage']:.2f}%)")
    print(f"  High missing columns: {len(missing['high_missing_columns'])}")
    
    print("\nExercise 5: Correlation Analysis")
    correlation = exercise_5_correlation_analysis(df)
    print(f"  High correlations found: {len(correlation.get('high_correlations', []))}")
    
    print("\nExercise 6: Distribution Analysis (age)")
    dist = exercise_6_distribution_analysis(df, 'age')
    print(f"  Distribution type: {dist['distribution_type']}")
    print(f"  Skewness: {dist['skewness']:.3f}")
    
    print("\nExercise 7: Outlier Detection (income)")
    outliers = exercise_7_outlier_detection(df, 'income')
    print(f"  IQR method: {outliers['iqr_method']['count']} outliers ({outliers['iqr_method']['percentage']:.1f}%)")
    print(f"  Z-score method: {outliers['zscore_method']['count']} outliers")
    
    print("\nExercise 8: Comprehensive Profile Report")
    report = exercise_8_generate_profile_report(df)
    print(report)
