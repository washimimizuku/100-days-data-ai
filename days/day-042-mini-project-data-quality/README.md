# Day 42: Mini Project - Data Quality Framework

## 🎯 Project Overview

Build a comprehensive data quality framework that integrates profiling, validation, testing, and monitoring into a unified system. This project combines concepts from Days 36-41.

**Time**: 2 hours

---

## Learning Objectives

- Integrate multiple quality tools into one framework
- Build automated quality monitoring
- Generate quality reports and dashboards
- Implement quality gates for data pipelines
- Create reusable quality components

---

## Project Requirements

### 1. Quality Profiler
Automatically profile datasets and generate statistics:
- Numerical column profiling (mean, median, std, outliers)
- Categorical column profiling (cardinality, value counts)
- Missing data analysis
- Distribution analysis
- Correlation analysis

### 2. Validation Engine
Validate data against rules:
- Type validation
- Range validation
- Format validation (email, phone, etc.)
- Uniqueness validation
- Business rule validation
- Schema validation

### 3. Quality Tests
Implement automated tests:
- Not null tests
- Uniqueness tests
- Referential integrity tests
- Custom business rule tests
- Threshold tests (e.g., missing rate < 5%)

### 4. Quality Scoring
Calculate overall quality scores:
- Dimension scores (accuracy, completeness, etc.)
- Weighted overall score
- Quality grade (A-F)
- Trend tracking

### 5. Quality Reports
Generate comprehensive reports:
- Executive summary
- Detailed findings
- Quality trends
- Recommendations
- Alerts for critical issues

### 6. Quality Gates
Implement pipeline gates:
- Pass/fail decisions
- Quarantine invalid records
- Alert on failures
- Block downstream processing

---

## Project Structure

```
data_quality_framework/
├── quality_framework.py      # Main framework
├── profiler.py               # Data profiling
├── validator.py              # Data validation
├── tests.py                  # Quality tests
├── scorer.py                 # Quality scoring
├── reporter.py               # Report generation
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
└── test_framework.sh         # Test script
```

---

## Implementation Guide

### Phase 1: Core Framework (30 min)
1. Create `QualityFramework` class
2. Implement profiling module
3. Implement validation module
4. Add basic tests

### Phase 2: Scoring & Reporting (30 min)
1. Implement quality scoring
2. Create report generator
3. Add visualization helpers
4. Generate sample reports

### Phase 3: Integration (30 min)
1. Integrate all components
2. Add quality gates
3. Implement quarantine logic
4. Create pipeline integration

### Phase 4: Testing & Documentation (30 min)
1. Test with sample data
2. Document usage
3. Create examples
4. Write test script

---

## Key Features

### 1. Automated Profiling
```python
profiler = DataProfiler()
profile = profiler.profile(df)
# Returns: stats, distributions, correlations, issues
```

### 2. Rule-Based Validation
```python
validator = DataValidator()
validator.add_rule(NotNullRule('customer_id'))
validator.add_rule(RangeRule('age', 0, 120))
results = validator.validate(df)
```

### 3. Quality Scoring
```python
scorer = QualityScorer()
score = scorer.calculate_score(df, rules)
# Returns: {overall: 0.85, grade: 'B', dimensions: {...}}
```

### 4. Comprehensive Reports
```python
reporter = QualityReporter()
report = reporter.generate_report(df, profile, validation, score)
# Returns: formatted report with all findings
```

### 5. Pipeline Integration
```python
framework = QualityFramework(config)
valid_df, invalid_df = framework.process(df)
# Validates, scores, reports, and quarantines
```

---

## Sample Configuration

```python
quality_config = {
    'profiling': {
        'enabled': True,
        'detect_outliers': True,
        'analyze_correlations': True
    },
    'validation': {
        'rules': [
            {'type': 'not_null', 'column': 'customer_id'},
            {'type': 'unique', 'column': 'customer_id'},
            {'type': 'range', 'column': 'age', 'min': 0, 'max': 120},
            {'type': 'format', 'column': 'email', 'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'}
        ]
    },
    'scoring': {
        'weights': {
            'accuracy': 0.20,
            'completeness': 0.20,
            'consistency': 0.15,
            'validity': 0.20,
            'uniqueness': 0.15,
            'timeliness': 0.10
        },
        'thresholds': {
            'A': 0.95,
            'B': 0.85,
            'C': 0.75,
            'D': 0.65
        }
    },
    'reporting': {
        'format': 'text',  # or 'html', 'json'
        'include_recommendations': True,
        'alert_on_failure': True
    },
    'gates': {
        'enabled': True,
        'min_score': 0.70,
        'quarantine_invalid': True
    }
}
```

---

## Expected Outputs

### 1. Quality Profile
```
Dataset Profile
===============
Rows: 10,000
Columns: 15
Memory: 2.5 MB

Numerical Columns (8):
  age: mean=45.2, std=15.3, outliers=12
  income: mean=75000, std=25000, outliers=45

Categorical Columns (5):
  status: 3 unique values
  region: 4 unique values

Missing Data:
  email: 5% missing
  phone: 10% missing

Quality Issues:
  - 12 outliers in age column
  - 500 missing emails
  - 45 duplicate customer_ids
```

### 2. Validation Results
```
Validation Results
==================
Total Rules: 10
Passed: 7
Failed: 3

Failed Rules:
  ❌ customer_id uniqueness: 45 duplicates
  ❌ email not_null: 500 nulls
  ❌ age range: 12 values outside [0, 120]
```

### 3. Quality Score
```
Quality Score: 82% (B)
======================
Accuracy:      85%
Completeness:  90%
Consistency:   95%
Validity:      75%
Uniqueness:    70%
Timeliness:    85%
```

### 4. Quality Report
```
DATA QUALITY REPORT
===================
Dataset: customer_data
Date: 2024-12-05
Overall Score: 82% (Grade B)

Critical Issues (3):
  1. 45 duplicate customer IDs
  2. 500 missing email addresses
  3. 12 invalid age values

Recommendations:
  1. Implement deduplication logic
  2. Make email mandatory at source
  3. Add age validation at entry point

Quality Trend:
  Last 7 days: 78% → 82% (↑ 4%)
```

---

## Success Criteria

- [ ] All components implemented and working
- [ ] Profiling generates comprehensive statistics
- [ ] Validation catches all rule violations
- [ ] Scoring produces accurate quality scores
- [ ] Reports are clear and actionable
- [ ] Quality gates block bad data
- [ ] Framework is reusable and configurable
- [ ] Code is under 400 lines per file
- [ ] Test script validates functionality

---

## Bonus Features

1. **HTML Reports**: Generate interactive HTML reports
2. **Trend Tracking**: Store and visualize quality over time
3. **Alerting**: Send alerts on quality failures
4. **Custom Rules**: Easy-to-add custom validation rules
5. **Performance**: Optimize for large datasets

---

## Testing

Run the test script to validate your implementation:
```bash
./test_framework.sh
```

Expected output:
- All tests pass
- Sample reports generated
- Quality scores calculated
- Invalid records quarantined

---

## Resources

- Days 36-41 materials
- Great Expectations documentation
- Pandas profiling examples
- Data quality best practices

---

## Next Steps

After completing this project:
1. Review your implementation
2. Test with real datasets
3. Identify improvements
4. Consider production deployment
5. Move to Day 43: Git Workflows

---

## Reflection Questions

1. What was the most challenging part?
2. How would you scale this for production?
3. What additional features would be valuable?
4. How would you integrate with existing pipelines?
5. What monitoring would you add?
