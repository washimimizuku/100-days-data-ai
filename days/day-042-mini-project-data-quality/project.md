# Data Quality Framework - Project Specification

## Overview

Build an integrated data quality framework that combines profiling, validation, testing, scoring, and reporting into a unified system.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Quality Framework                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │ Profiler │  │Validator │  │  Tests   │  │ Scorer  ││
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘│
│       │             │              │             │      │
│       └─────────────┴──────────────┴─────────────┘      │
│                         │                               │
│                    ┌────▼────┐                          │
│                    │Reporter │                          │
│                    └─────────┘                          │
└─────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. DataProfiler
**Purpose**: Analyze dataset characteristics

**Methods**:
- `profile(df)`: Generate comprehensive profile
- `profile_numerical(df, column)`: Profile numerical column
- `profile_categorical(df, column)`: Profile categorical column
- `detect_outliers(series)`: Find outliers
- `analyze_correlations(df)`: Find correlations

**Output**: Profile dictionary with stats, distributions, issues

### 2. DataValidator
**Purpose**: Validate data against rules

**Methods**:
- `add_rule(rule)`: Add validation rule
- `validate(df)`: Run all rules
- `get_failed_rules()`: Get failures
- `get_invalid_rows()`: Get invalid row indices

**Output**: Validation results with pass/fail per rule

### 3. QualityTests
**Purpose**: Run automated quality tests

**Methods**:
- `test_not_null(df, column)`: Check for nulls
- `test_unique(df, column)`: Check uniqueness
- `test_range(df, column, min, max)`: Check range
- `test_format(df, column, pattern)`: Check format
- `run_all_tests(df)`: Execute all tests

**Output**: Test results with pass/fail

### 4. QualityScorer
**Purpose**: Calculate quality scores

**Methods**:
- `calculate_dimension_scores(df, rules)`: Score each dimension
- `calculate_overall_score(dimension_scores, weights)`: Weighted score
- `assign_grade(score)`: Convert score to grade
- `track_trend(scores)`: Track over time

**Output**: Score dictionary with overall, dimensions, grade

### 5. QualityReporter
**Purpose**: Generate quality reports

**Methods**:
- `generate_report(df, profile, validation, score)`: Full report
- `generate_summary()`: Executive summary
- `generate_recommendations()`: Actionable recommendations
- `format_report(format)`: Format as text/HTML/JSON

**Output**: Formatted report string

### 6. QualityFramework
**Purpose**: Orchestrate all components

**Methods**:
- `process(df)`: Run full quality pipeline
- `apply_gates(df, score)`: Apply quality gates
- `quarantine_invalid(df, invalid_indices)`: Separate bad data
- `generate_alerts(results)`: Send alerts

**Output**: Valid DataFrame, invalid DataFrame, results

---

## Data Flow

```
Input DataFrame
      │
      ▼
  Profile Data ────────┐
      │                │
      ▼                │
 Validate Data ────────┤
      │                │
      ▼                │
  Run Tests ───────────┤
      │                │
      ▼                │
Calculate Score ───────┤
      │                │
      ▼                │
 Apply Gates ──────────┤
      │                │
      ▼                ▼
Split Valid/Invalid  Generate Report
      │
      ▼
Return Results
```

---

## Configuration Schema

```python
{
    'profiling': {
        'enabled': bool,
        'detect_outliers': bool,
        'analyze_correlations': bool,
        'outlier_method': 'iqr' | 'zscore'
    },
    'validation': {
        'rules': [
            {
                'type': 'not_null' | 'unique' | 'range' | 'format',
                'column': str,
                'min': float,  # for range
                'max': float,  # for range
                'pattern': str  # for format
            }
        ]
    },
    'scoring': {
        'weights': {
            'accuracy': float,
            'completeness': float,
            'consistency': float,
            'validity': float,
            'uniqueness': float,
            'timeliness': float
        },
        'thresholds': {
            'A': float,
            'B': float,
            'C': float,
            'D': float
        }
    },
    'gates': {
        'enabled': bool,
        'min_score': float,
        'quarantine_invalid': bool,
        'fail_on_critical': bool
    },
    'reporting': {
        'format': 'text' | 'html' | 'json',
        'include_recommendations': bool,
        'alert_on_failure': bool
    }
}
```

---

## Implementation Steps

### Step 1: Core Classes (30 min)
1. Implement `DataProfiler` with basic profiling
2. Implement `DataValidator` with rule system
3. Implement `QualityTests` with common tests
4. Test each component independently

### Step 2: Scoring & Reporting (30 min)
1. Implement `QualityScorer` with dimension scoring
2. Implement `QualityReporter` with report generation
3. Add formatting options
4. Test scoring and reporting

### Step 3: Framework Integration (30 min)
1. Implement `QualityFramework` orchestrator
2. Add quality gates logic
3. Add quarantine functionality
4. Integrate all components

### Step 4: Testing & Polish (30 min)
1. Create test script
2. Test with sample data
3. Generate example reports
4. Document usage

---

## Sample Usage

```python
from quality_framework import QualityFramework
from config import quality_config

# Initialize framework
framework = QualityFramework(quality_config)

# Process data
valid_df, invalid_df, results = framework.process(df)

# Access results
print(f"Quality Score: {results['score']['overall']:.1%}")
print(f"Grade: {results['score']['grade']}")
print(f"Valid records: {len(valid_df)}")
print(f"Invalid records: {len(invalid_df)}")

# Generate report
report = results['report']
print(report)

# Save invalid records
if len(invalid_df) > 0:
    invalid_df.to_csv('quarantine.csv', index=False)
```

---

## Testing Checklist

- [ ] Profiler generates accurate statistics
- [ ] Validator catches all rule violations
- [ ] Tests identify quality issues
- [ ] Scorer calculates correct scores
- [ ] Reporter generates readable reports
- [ ] Framework integrates all components
- [ ] Quality gates work correctly
- [ ] Quarantine separates invalid data
- [ ] Configuration is flexible
- [ ] Code is modular and reusable

---

## Deliverables

1. `quality_framework.py` - Main framework class
2. `profiler.py` - Data profiling module
3. `validator.py` - Data validation module
4. `tests.py` - Quality tests module
5. `scorer.py` - Quality scoring module
6. `reporter.py` - Report generation module
7. `config.py` - Configuration management
8. `requirements.txt` - Dependencies
9. `test_framework.sh` - Test script
10. Sample reports and outputs

---

## Success Metrics

- All components work independently
- Integration is seamless
- Reports are clear and actionable
- Quality gates prevent bad data
- Framework is reusable
- Code is clean and documented
- Tests pass successfully
