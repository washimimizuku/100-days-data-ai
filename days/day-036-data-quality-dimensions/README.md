# Day 36: Data Quality Dimensions

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand the six core dimensions of data quality
- Learn how to measure and assess data quality
- Implement data quality checks in Python
- Create data quality reports and scorecards
- Apply quality frameworks to real-world datasets

---

## What is Data Quality?

Data quality refers to the fitness of data for its intended use. High-quality data is accurate, complete, consistent, and reliable. Poor data quality costs organizations an average of $15M per year and leads to:

- **Bad decisions**: Analytics based on flawed data
- **Lost revenue**: Incorrect pricing, inventory, or customer data
- **Compliance issues**: Regulatory violations and fines
- **Wasted resources**: Time spent cleaning and fixing data

---

## The Six Dimensions of Data Quality

### 1. Accuracy
**Definition**: Data correctly represents the real-world entity or event.

**Examples**:
- Customer email addresses are valid and deliverable
- Product prices match the actual selling price
- Dates are in the correct format and valid

**How to measure**:
```python
# Check email format
valid_emails = df['email'].str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$')
accuracy_rate = valid_emails.sum() / len(df)

# Check value ranges
valid_ages = df['age'].between(0, 120)
accuracy_rate = valid_ages.sum() / len(df)
```

### 2. Completeness
**Definition**: All required data is present; no missing values in critical fields.

**Examples**:
- Customer records have name, email, and phone
- Orders have customer_id, product_id, and amount
- Transactions have timestamps

**How to measure**:
```python
# Check for nulls in critical columns
completeness = 1 - (df['email'].isna().sum() / len(df))

# Check multiple columns
critical_cols = ['customer_id', 'email', 'phone']
completeness_scores = {}
for col in critical_cols:
    completeness_scores[col] = 1 - (df[col].isna().sum() / len(df))
```

### 3. Consistency
**Definition**: Data is uniform across systems and doesn't contradict itself.

**Examples**:
- Date formats are consistent (all YYYY-MM-DD)
- Country codes follow ISO standards
- Customer data matches across CRM and billing systems

**How to measure**:
```python
# Check date format consistency
date_format = '%Y-%m-%d'
consistent_dates = pd.to_datetime(df['date'], format=date_format, errors='coerce').notna()
consistency_rate = consistent_dates.sum() / len(df)

# Check referential integrity
orders_with_valid_customers = df['customer_id'].isin(customers_df['customer_id'])
consistency_rate = orders_with_valid_customers.sum() / len(df)
```

### 4. Validity
**Definition**: Data conforms to defined business rules and constraints.

**Examples**:
- Order status is one of: pending, shipped, delivered, cancelled
- Quantity is a positive integer
- Email domain is in allowed list

**How to measure**:
```python
# Check against allowed values
valid_statuses = ['pending', 'shipped', 'delivered', 'cancelled']
validity_rate = df['status'].isin(valid_statuses).sum() / len(df)

# Check business rules
valid_orders = (df['quantity'] > 0) & (df['amount'] > 0)
validity_rate = valid_orders.sum() / len(df)
```

### 5. Uniqueness
**Definition**: No duplicate records exist; each entity is represented once.

**Examples**:
- Customer IDs are unique
- Order numbers don't repeat
- Email addresses appear only once

**How to measure**:
```python
# Check for duplicates
duplicate_count = df.duplicated(subset=['customer_id']).sum()
uniqueness_rate = 1 - (duplicate_count / len(df))

# Check for duplicate emails
duplicate_emails = df['email'].duplicated().sum()
uniqueness_rate = 1 - (duplicate_emails / len(df))
```

### 6. Timeliness
**Definition**: Data is up-to-date and available when needed.

**Examples**:
- Stock prices are updated in real-time
- Customer addresses reflect recent moves
- Inventory counts are current

**How to measure**:
```python
from datetime import datetime, timedelta

# Check data freshness
now = datetime.now()
fresh_data = (now - pd.to_datetime(df['updated_at'])) < timedelta(days=7)
timeliness_rate = fresh_data.sum() / len(df)

# Check SLA compliance
sla_hours = 24
data_age_hours = (now - pd.to_datetime(df['created_at'])).dt.total_seconds() / 3600
timeliness_rate = (data_age_hours <= sla_hours).sum() / len(df)
```

---

## Data Quality Framework

### Quality Score Calculation

```python
def calculate_quality_score(df, rules):
    """Calculate overall data quality score"""
    scores = {}
    
    # Accuracy
    scores['accuracy'] = check_accuracy(df, rules['accuracy'])
    
    # Completeness
    scores['completeness'] = check_completeness(df, rules['required_fields'])
    
    # Consistency
    scores['consistency'] = check_consistency(df, rules['formats'])
    
    # Validity
    scores['validity'] = check_validity(df, rules['constraints'])
    
    # Uniqueness
    scores['uniqueness'] = check_uniqueness(df, rules['unique_fields'])
    
    # Timeliness
    scores['timeliness'] = check_timeliness(df, rules['freshness'])
    
    # Overall score (weighted average)
    weights = rules.get('weights', {
        'accuracy': 0.20,
        'completeness': 0.20,
        'consistency': 0.15,
        'validity': 0.20,
        'uniqueness': 0.15,
        'timeliness': 0.10
    })
    
    overall_score = sum(scores[dim] * weights[dim] for dim in scores)
    
    return {
        'scores': scores,
        'overall': overall_score,
        'grade': get_grade(overall_score)
    }

def get_grade(score):
    """Convert score to letter grade"""
    if score >= 0.95: return 'A'
    elif score >= 0.85: return 'B'
    elif score >= 0.75: return 'C'
    elif score >= 0.65: return 'D'
    else: return 'F'
```

---

## Data Quality Report

### Report Structure

```python
def generate_quality_report(df, rules):
    """Generate comprehensive quality report"""
    report = {
        'dataset': {
            'name': rules.get('dataset_name', 'Unknown'),
            'rows': len(df),
            'columns': len(df.columns),
            'timestamp': datetime.now().isoformat()
        },
        'quality_scores': calculate_quality_score(df, rules),
        'issues': find_quality_issues(df, rules),
        'recommendations': generate_recommendations(df, rules)
    }
    return report
```

### Example Report Output

```
Data Quality Report
===================
Dataset: customer_data
Rows: 10,000
Timestamp: 2024-12-05 10:30:00

Quality Scores:
- Accuracy:      92% (A)
- Completeness:  88% (B)
- Consistency:   95% (A)
- Validity:      85% (B)
- Uniqueness:    98% (A)
- Timeliness:    75% (C)

Overall Score: 89% (B)

Critical Issues:
1. 1,200 records missing email addresses (12%)
2. 500 records with invalid phone formats (5%)
3. 2,500 records older than 30 days (25%)

Recommendations:
1. Implement email validation at data entry
2. Standardize phone number format
3. Increase data refresh frequency
```

---

## 💻 Exercises (40 min)

### Exercise 1: Implement Quality Checks
Create functions to check each quality dimension for a customer dataset.

### Exercise 2: Calculate Quality Scores
Build a scoring system that calculates dimension scores and overall quality.

### Exercise 3: Generate Quality Report
Create a comprehensive quality report with issues and recommendations.

### Exercise 4: Quality Dashboard
Build a simple dashboard showing quality metrics over time.

---

## ✅ Quiz (5 min)

Test your understanding of data quality dimensions and measurement techniques.

---

## 🎯 Key Takeaways

- **Six dimensions**: Accuracy, Completeness, Consistency, Validity, Uniqueness, Timeliness
- **Measurement**: Each dimension has specific metrics and thresholds
- **Scoring**: Weighted scores provide overall quality assessment
- **Reporting**: Clear reports help stakeholders understand quality issues
- **Continuous**: Quality monitoring should be ongoing, not one-time
- **Business impact**: Poor quality costs money; good quality enables decisions

---

## 📚 Resources

- [DAMA Data Quality Framework](https://www.dama.org/)
- [ISO 8000 Data Quality Standards](https://www.iso.org/standard/50798.html)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Data Quality: The Accuracy Dimension](https://mitpress.mit.edu/books/data-quality)

---

## Tomorrow: Day 37 - Great Expectations

Learn how to use Great Expectations, a powerful Python library for data validation and quality testing with declarative expectations and automated documentation.
