# Day 37: Great Expectations

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand what Great Expectations is and why it's used
- Learn to create and validate expectations
- Build data quality test suites
- Generate data documentation automatically
- Integrate Great Expectations into data pipelines

---

## What is Great Expectations?

Great Expectations (GX) is an open-source Python library for data validation, documentation, and profiling. It helps you:

- **Define expectations**: Declare what your data should look like
- **Validate data**: Check if data meets expectations
- **Document data**: Auto-generate data documentation
- **Prevent bad data**: Catch issues before they propagate

**Why use Great Expectations?**
- Declarative syntax (say what, not how)
- 300+ built-in expectations
- Automatic documentation generation
- Integration with data pipelines
- Version control for data quality rules

---

## Core Concepts

### 1. Expectations
Assertions about your data. Examples:
- `expect_column_values_to_not_be_null`
- `expect_column_values_to_be_between`
- `expect_column_values_to_match_regex`

### 2. Expectation Suites
Collections of expectations for a dataset:
```python
suite = context.add_expectation_suite("customer_suite")
```

### 3. Validators
Objects that validate data against expectations:
```python
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="customer_suite"
)
```

### 4. Checkpoints
Reusable validation workflows:
```python
checkpoint = context.add_checkpoint(
    name="customer_checkpoint",
    validator=validator
)
```

### 5. Data Docs
Auto-generated HTML documentation showing validation results.

---

## Getting Started

### Installation
```bash
pip install great-expectations
```

### Initialize Project
```python
import great_expectations as gx

# Create data context
context = gx.get_context()
```

---

## Common Expectations

### Null Checks
```python
# Column should not have nulls
validator.expect_column_values_to_not_be_null("customer_id")

# Column can have nulls
validator.expect_column_to_exist("middle_name")
```

### Value Ranges
```python
# Age between 0 and 120
validator.expect_column_values_to_be_between(
    "age", min_value=0, max_value=120
)

# Positive amounts
validator.expect_column_values_to_be_greater_than("amount", 0)
```

### String Patterns
```python
# Email format
validator.expect_column_values_to_match_regex(
    "email",
    regex=r'^[\w\.-]+@[\w\.-]+\.\w+$'
)

# Phone format (10 digits)
validator.expect_column_values_to_match_regex(
    "phone",
    regex=r'^\d{10}$'
)
```

### Set Membership
```python
# Status must be in allowed values
validator.expect_column_values_to_be_in_set(
    "status",
    value_set=["active", "inactive", "suspended"]
)

# Country codes
validator.expect_column_values_to_be_in_set(
    "country",
    value_set=["US", "UK", "CA", "AU"]
)
```

### Uniqueness
```python
# Customer ID must be unique
validator.expect_column_values_to_be_unique("customer_id")

# Email must be unique
validator.expect_column_values_to_be_unique("email")
```

### Type Checks
```python
# Column must be integer
validator.expect_column_values_to_be_of_type("age", "int64")

# Column must be datetime
validator.expect_column_values_to_be_of_type("created_at", "datetime64")
```

---

## Building an Expectation Suite

```python
import great_expectations as gx
import pandas as pd

# Create context
context = gx.get_context()

# Load data
df = pd.read_csv("customers.csv")

# Create expectation suite
suite = context.add_expectation_suite("customer_suite")

# Get validator
validator = context.get_validator(
    batch_request=gx.core.batch.RuntimeBatchRequest(
        datasource_name="pandas_datasource",
        data_connector_name="runtime_data_connector",
        data_asset_name="customers",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default"}
    ),
    expectation_suite_name="customer_suite"
)

# Add expectations
validator.expect_table_row_count_to_be_between(min_value=1, max_value=1000000)
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_be_unique("customer_id")
validator.expect_column_values_to_be_between("age", 0, 120)
validator.expect_column_values_to_match_regex("email", r'^[\w\.-]+@[\w\.-]+\.\w+$')

# Save suite
validator.save_expectation_suite(discard_failed_expectations=False)
```

---

## Validating Data

```python
# Run validation
results = validator.validate()

# Check if validation passed
if results["success"]:
    print("✅ All expectations passed!")
else:
    print("❌ Some expectations failed")
    
# Get statistics
stats = results["statistics"]
print(f"Evaluated: {stats['evaluated_expectations']}")
print(f"Successful: {stats['successful_expectations']}")
print(f"Failed: {stats['unsuccessful_expectations']}")
print(f"Success rate: {stats['success_percent']:.1f}%")

# Get failed expectations
for result in results["results"]:
    if not result["success"]:
        print(f"Failed: {result['expectation_config']['expectation_type']}")
        print(f"  Column: {result['expectation_config']['kwargs'].get('column')}")
```

---

## Checkpoints

Checkpoints make validation reusable:

```python
# Create checkpoint
checkpoint = context.add_checkpoint(
    name="customer_checkpoint",
    config_version=1,
    class_name="SimpleCheckpoint",
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": "customer_suite"
        }
    ]
)

# Run checkpoint
checkpoint_result = checkpoint.run()

# Check results
if checkpoint_result["success"]:
    print("✅ Checkpoint passed!")
else:
    print("❌ Checkpoint failed!")
```

---

## Data Docs

Generate beautiful HTML documentation:

```python
# Build data docs
context.build_data_docs()

# Open in browser
context.open_data_docs()
```

Data Docs show:
- All expectation suites
- Validation results
- Data profiling statistics
- Historical trends

---

## Integration with Pipelines

### Pandas Pipeline
```python
def validate_customers(df):
    """Validate customer data"""
    context = gx.get_context()
    
    validator = context.get_validator(
        batch_request=create_batch_request(df),
        expectation_suite_name="customer_suite"
    )
    
    results = validator.validate()
    
    if not results["success"]:
        raise ValueError("Data validation failed!")
    
    return df
```

### Spark Pipeline
```python
def validate_spark_df(spark_df):
    """Validate Spark DataFrame"""
    # Convert to Pandas for validation
    sample_df = spark_df.limit(10000).toPandas()
    
    validator = context.get_validator(
        batch_request=create_batch_request(sample_df),
        expectation_suite_name="data_suite"
    )
    
    results = validator.validate()
    return results["success"]
```

---

## 💻 Exercises (40 min)

### Exercise 1: Create Basic Expectations
Set up Great Expectations and create basic expectations for a customer dataset.

### Exercise 2: Build Expectation Suite
Create a comprehensive suite with 10+ expectations covering all data quality dimensions.

### Exercise 3: Validate Data
Run validation and handle failed expectations.

### Exercise 4: Create Checkpoint
Build a reusable checkpoint for automated validation.

### Exercise 5: Custom Expectations
Create custom expectations for business-specific rules.

---

## ✅ Quiz (5 min)

Test your understanding of Great Expectations concepts and usage.

---

## 🎯 Key Takeaways

- **Declarative**: Define what data should look like, not how to check
- **300+ expectations**: Built-in expectations cover most use cases
- **Automatic docs**: Data Docs provide beautiful documentation
- **Pipeline integration**: Easy to add to ETL/ELT workflows
- **Version control**: Expectation suites can be versioned with code
- **Fail fast**: Catch data issues early before they propagate

---

## 📚 Resources

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Expectation Gallery](https://greatexpectations.io/expectations/)
- [Great Expectations GitHub](https://github.com/great-expectations/great_expectations)
- [GX University](https://greatexpectations.io/gx-university/)

---

## Tomorrow: Day 38 - Data Profiling

Learn how to automatically profile datasets to understand data distributions, detect anomalies, and generate insights using pandas-profiling and other tools.
