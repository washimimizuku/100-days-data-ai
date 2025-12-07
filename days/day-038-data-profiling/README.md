# Day 38: Data Profiling

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand what data profiling is and why it's important
- Learn to generate statistical summaries of datasets
- Detect data quality issues automatically
- Analyze distributions and correlations
- Use profiling tools like pandas-profiling and ydata-profiling
- Create comprehensive data profile reports

---

## What is Data Profiling?

Data profiling is the process of examining data to understand its structure, content, quality, and relationships. It helps you:

- **Understand data**: Get quick insights into dataset characteristics
- **Detect issues**: Find missing values, outliers, and anomalies
- **Assess quality**: Measure completeness, accuracy, and consistency
- **Plan transformations**: Identify what cleaning is needed
- **Document data**: Create metadata and documentation

**When to profile data:**
- Before starting analysis or modeling
- After data ingestion
- During data quality monitoring
- When onboarding new data sources

---

## Types of Profiling

### 1. Structure Profiling
Understanding the dataset structure:
```python
# Basic structure
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Data types
print(df.dtypes)

# Column names
print(df.columns.tolist())
```

### 2. Content Profiling
Analyzing the actual data values:
```python
# Descriptive statistics
print(df.describe())

# Value counts
print(df['category'].value_counts())

# Unique values
print(df['status'].nunique())
```

### 3. Relationship Profiling
Understanding relationships between columns:
```python
# Correlation matrix
corr_matrix = df.corr()

# Highly correlated pairs
high_corr = corr_matrix[abs(corr_matrix) > 0.7]
```

---

## Numerical Column Profiling

### Descriptive Statistics
```python
def profile_numerical(df, column):
    """Profile a numerical column"""
    stats = {
        'count': df[column].count(),
        'missing': df[column].isna().sum(),
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q25': df[column].quantile(0.25),
        'q75': df[column].quantile(0.75),
        'skewness': df[column].skew(),
        'kurtosis': df[column].kurtosis()
    }
    return stats
```

### Distribution Analysis
```python
from scipy import stats

def check_normality(series):
    """Check if data is normally distributed"""
    # Shapiro-Wilk test
    statistic, p_value = stats.shapiro(series.dropna())
    
    is_normal = p_value > 0.05
    
    return {
        'is_normal': is_normal,
        'p_value': p_value,
        'interpretation': 'Normal' if is_normal else 'Not normal'
    }
```

### Outlier Detection
```python
def detect_outliers_iqr(series):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    return {
        'count': len(outliers),
        'percentage': len(outliers) / len(series) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_indices': outliers.index.tolist()
    }

def detect_outliers_zscore(series):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(series.dropna()))
    outliers = series[z_scores > 3]
    
    return {
        'count': len(outliers),
        'percentage': len(outliers) / len(series) * 100
    }
```

---

## Categorical Column Profiling

### Value Analysis
```python
def profile_categorical(df, column):
    """Profile a categorical column"""
    value_counts = df[column].value_counts()
    
    stats = {
        'count': df[column].count(),
        'missing': df[column].isna().sum(),
        'unique': df[column].nunique(),
        'cardinality_ratio': df[column].nunique() / len(df),
        'most_common': value_counts.head(5).to_dict(),
        'least_common': value_counts.tail(5).to_dict(),
        'mode': df[column].mode()[0] if len(df[column].mode()) > 0 else None
    }
    
    return stats
```

### Cardinality Analysis
```python
def analyze_cardinality(df, column):
    """Analyze cardinality of categorical column"""
    unique_count = df[column].nunique()
    total_count = len(df)
    ratio = unique_count / total_count
    
    if ratio < 0.01:
        category = "Low cardinality (< 1%)"
    elif ratio < 0.5:
        category = "Medium cardinality (1-50%)"
    else:
        category = "High cardinality (> 50%)"
    
    return {
        'unique_values': unique_count,
        'total_values': total_count,
        'cardinality_ratio': ratio,
        'category': category
    }
```

---

## Missing Data Analysis

### Missing Patterns
```python
def analyze_missing_data(df):
    """Analyze missing data patterns"""
    missing_stats = []
    
    for column in df.columns:
        missing_count = df[column].isna().sum()
        missing_pct = missing_count / len(df) * 100
        
        missing_stats.append({
            'column': column,
            'missing_count': missing_count,
            'missing_percentage': missing_pct,
            'data_type': df[column].dtype
        })
    
    missing_df = pd.DataFrame(missing_stats)
    missing_df = missing_df.sort_values('missing_percentage', ascending=False)
    
    return missing_df

def check_missing_patterns(df):
    """Check if missing data follows patterns"""
    # Check if missing values correlate
    missing_matrix = df.isna().astype(int)
    missing_corr = missing_matrix.corr()
    
    # Find columns with correlated missingness
    high_corr_missing = []
    for i in range(len(missing_corr.columns)):
        for j in range(i+1, len(missing_corr.columns)):
            if abs(missing_corr.iloc[i, j]) > 0.5:
                high_corr_missing.append({
                    'col1': missing_corr.columns[i],
                    'col2': missing_corr.columns[j],
                    'correlation': missing_corr.iloc[i, j]
                })
    
    return high_corr_missing
```

---

## Correlation Analysis

### Numerical Correlations
```python
def analyze_correlations(df, threshold=0.7):
    """Find highly correlated numerical columns"""
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr_pairs.append({
                    'column1': corr_matrix.columns[i],
                    'column2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    return high_corr_pairs
```

---

## Automated Profiling Tools

### ydata-profiling (formerly pandas-profiling)
```python
from ydata_profiling import ProfileReport

# Generate comprehensive report
profile = ProfileReport(df, title="Data Profile Report")

# Save to HTML
profile.to_file("data_profile.html")

# Get specific sections
overview = profile.get_description()
```

### sweetviz
```python
import sweetviz as sv

# Generate report
report = sv.analyze(df)
report.show_html("sweetviz_report.html")

# Compare two datasets
comparison = sv.compare([df_train, "Training"], [df_test, "Testing"])
comparison.show_html("comparison_report.html")
```

---

## Comprehensive Profile Report

```python
def generate_profile_report(df):
    """Generate comprehensive data profile"""
    report = {
        'overview': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': df.duplicated().sum()
        },
        'numerical_columns': {},
        'categorical_columns': {},
        'missing_data': analyze_missing_data(df),
        'correlations': analyze_correlations(df)
    }
    
    # Profile numerical columns
    for col in df.select_dtypes(include=[np.number]).columns:
        report['numerical_columns'][col] = profile_numerical(df, col)
    
    # Profile categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        report['categorical_columns'][col] = profile_categorical(df, col)
    
    return report
```

---

## 💻 Exercises (40 min)

### Exercise 1: Basic Profiling
Generate basic statistics for a dataset including shape, types, and memory usage.

### Exercise 2: Numerical Profiling
Profile all numerical columns with descriptive statistics and outlier detection.

### Exercise 3: Categorical Profiling
Analyze categorical columns including cardinality and value distributions.

### Exercise 4: Missing Data Analysis
Identify missing data patterns and high-missing columns.

### Exercise 5: Correlation Analysis
Find highly correlated numerical columns.

### Exercise 6: Distribution Analysis
Analyze the distribution of a specific column.

### Exercise 7: Outlier Detection
Detect outliers using multiple methods (IQR, Z-score, percentile).

### Exercise 8: Comprehensive Report
Generate a complete data profile report combining all analyses.

---

## ✅ Quiz (5 min)

Test your understanding of data profiling concepts and techniques.

---

## 🎯 Key Takeaways

- **Profiling first**: Always profile data before analysis or modeling
- **Multiple dimensions**: Profile structure, content, and relationships
- **Automated tools**: Use ydata-profiling or sweetviz for quick insights
- **Custom profiling**: Build custom profiles for specific needs
- **Documentation**: Profiling creates valuable data documentation
- **Quality assessment**: Profiling reveals data quality issues early

---

## 📚 Resources

- [ydata-profiling Documentation](https://docs.profiling.ydata.ai/)
- [Sweetviz Documentation](https://github.com/fbdesignpro/sweetviz)
- [Pandas Profiling Guide](https://pandas.pydata.org/docs/user_guide/style.html)
- [Data Profiling Best Practices](https://www.dataversity.net/data-profiling-best-practices/)

---

## Tomorrow: Day 39 - Data Validation

Learn how to implement comprehensive data validation rules, create validation pipelines, and ensure data quality at ingestion time using various validation frameworks.
