# Day 39: Data Validation

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand data validation vs profiling vs quality checks
- Implement validation rules for different data types
- Build validation pipelines for data ingestion
- Handle validation failures gracefully
- Create reusable validation frameworks
- Integrate validation into ETL workflows

---

## What is Data Validation?

Data validation is the process of ensuring data meets defined rules and constraints before it's processed or stored. Unlike profiling (which discovers data characteristics), validation enforces requirements.

**Key differences:**
- **Profiling**: "What does the data look like?"
- **Validation**: "Does the data meet our requirements?"
- **Quality checks**: "How good is the data?"

**Why validate data?**
- Prevent bad data from entering systems
- Catch errors at ingestion time
- Ensure business rules are met
- Maintain data integrity
- Enable fail-fast behavior

---

## Validation Types

### 1. Type Validation
Ensure data has correct types:
```python
def validate_types(df, schema):
    """Validate column data types"""
    errors = []
    
    for col, expected_type in schema.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
        elif df[col].dtype != expected_type:
            errors.append(f"{col}: expected {expected_type}, got {df[col].dtype}")
    
    return errors
```

### 2. Range Validation
Check values fall within acceptable ranges:
```python
def validate_range(series, min_val, max_val):
    invalid = series[(series < min_val) | (series > max_val)]
    return len(invalid) == 0, invalid.index.tolist()
```

### 3. Format Validation
Verify data matches expected patterns using regex for emails, phones, etc.

### 4. Uniqueness Validation
Ensure no duplicates in key columns:
```python
def validate_uniqueness(df, columns):
    duplicates = df.duplicated(subset=columns, keep=False)
    return not duplicates.any(), df[duplicates]
```

### 5. Referential Integrity
Validate foreign key relationships exist in reference tables.

### 6. Business Rule Validation
Enforce domain-specific rules:
```python
def validate_business_rules(df):
    """Validate business-specific rules"""
    errors = []
    
    # Rule: Active customers must have email
    active_no_email = df[(df['status'] == 'active') & (df['email'].isna())]
    if len(active_no_email) > 0:
        errors.append(f"{len(active_no_email)} active customers missing email")
    
    # Rule: Order amount must be positive
    negative_amounts = df[df['amount'] <= 0]
    if len(negative_amounts) > 0:
        errors.append(f"{len(negative_amounts)} orders with non-positive amounts")
    
    return errors
```

---

## Building a Validation Framework

### Validation Rule Class
```python
class ValidationRule:
    """Base class for validation rules"""
    
    def __init__(self, name, column=None):
        self.name = name
        self.column = column
    
    def validate(self, df):
        """Override in subclasses"""
        raise NotImplementedError
    
    def get_result(self, passed, message, invalid_rows=None):
        """Format validation result"""
        return {
            'rule': self.name,
            'column': self.column,
            'passed': passed,
            'message': message,
            'invalid_count': len(invalid_rows) if invalid_rows is not None else 0,
            'invalid_rows': invalid_rows
        }

class NotNullRule(ValidationRule):
    """Validate column has no nulls"""
    
    def validate(self, df):
        if self.column not in df.columns:
            return self.get_result(False, f"Column {self.column} not found")
        
        null_count = df[self.column].isna().sum()
        passed = null_count == 0
        message = f"Found {null_count} null values" if not passed else "No nulls"
        
        return self.get_result(passed, message, df[df[self.column].isna()].index)

class RangeRule(ValidationRule):
    """Validate values are within range"""
    
    def __init__(self, name, column, min_val, max_val):
        super().__init__(name, column)
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, df):
        invalid = df[(df[self.column] < self.min_val) | (df[self.column] > self.max_val)]
        passed = len(invalid) == 0
        message = f"Found {len(invalid)} values outside [{self.min_val}, {self.max_val}]"
        
        return self.get_result(passed, message, invalid.index)
```

### Validator Class
```python
class DataValidator:
    """Orchestrate multiple validation rules"""
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, rule):
        """Add validation rule"""
        self.rules.append(rule)
        return self
    
    def validate(self, df):
        """Run all validation rules"""
        results = []
        
        for rule in self.rules:
            result = rule.validate(df)
            results.append(result)
        
        passed = all(r['passed'] for r in results)
        
        return {
            'passed': passed,
            'total_rules': len(results),
            'passed_rules': sum(1 for r in results if r['passed']),
            'failed_rules': sum(1 for r in results if not r['passed']),
            'results': results
        }
```

---

## Validation Pipeline

### ETL with Validation
```python
def etl_with_validation(source_path, target_path, validator):
    """ETL pipeline with validation"""
    
    # Extract
    df = pd.read_csv(source_path)
    print(f"Extracted {len(df)} rows")
    
    # Validate
    validation_results = validator.validate(df)
    
    if not validation_results['passed']:
        print("❌ Validation failed!")
        for result in validation_results['results']:
            if not result['passed']:
                print(f"  - {result['rule']}: {result['message']}")
        
        # Decide: fail or quarantine
        raise ValueError("Data validation failed")
    
    print("✅ Validation passed")
    
    # Transform
    df_transformed = transform_data(df)
    
    # Load
    df_transformed.to_parquet(target_path)
    print(f"Loaded {len(df_transformed)} rows")
```

### Quarantine Invalid Records
```python
def validate_with_quarantine(df, validator, quarantine_path):
    """Validate and quarantine invalid records"""
    
    validation_results = validator.validate(df)
    
    # Collect all invalid row indices
    invalid_indices = set()
    for result in validation_results['results']:
        if not result['passed'] and result['invalid_rows'] is not None:
            invalid_indices.update(result['invalid_rows'])
    
    # Split valid and invalid
    valid_df = df[~df.index.isin(invalid_indices)]
    invalid_df = df[df.index.isin(invalid_indices)]
    
    # Save invalid records
    if len(invalid_df) > 0:
        invalid_df.to_csv(quarantine_path, index=False)
        print(f"⚠️  Quarantined {len(invalid_df)} invalid records")
    
    return valid_df, invalid_df
```

---

## Schema Validation

### Define Schema
```python
schema = {
    'customer_id': {
        'type': 'int64',
        'nullable': False,
        'unique': True
    },
    'name': {
        'type': 'object',
        'nullable': False,
        'max_length': 100
    },
    'email': {
        'type': 'object',
        'nullable': False,
        'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'
    },
    'age': {
        'type': 'int64',
        'nullable': False,
        'min': 0,
        'max': 120
    },
    'status': {
        'type': 'object',
        'nullable': False,
        'allowed_values': ['active', 'inactive', 'suspended']
    }
}
```

### Schema Validator
```python
def validate_schema(df, schema):
    """Validate dataframe against schema"""
    errors = []
    
    for column, rules in schema.items():
        # Check column exists
        if column not in df.columns:
            errors.append(f"Missing required column: {column}")
            continue
        
        # Check type
        if 'type' in rules and df[column].dtype != rules['type']:
            errors.append(f"{column}: wrong type {df[column].dtype}, expected {rules['type']}")
        
        # Check nullable
        if not rules.get('nullable', True):
            null_count = df[column].isna().sum()
            if null_count > 0:
                errors.append(f"{column}: {null_count} null values (not allowed)")
        
        # Check unique
        if rules.get('unique', False):
            dup_count = df[column].duplicated().sum()
            if dup_count > 0:
                errors.append(f"{column}: {dup_count} duplicates (must be unique)")
        
        # Check range
        if 'min' in rules:
            below_min = (df[column] < rules['min']).sum()
            if below_min > 0:
                errors.append(f"{column}: {below_min} values below min {rules['min']}")
        
        if 'max' in rules:
            above_max = (df[column] > rules['max']).sum()
            if above_max > 0:
                errors.append(f"{column}: {above_max} values above max {rules['max']}")
        
        # Check allowed values
        if 'allowed_values' in rules:
            invalid = ~df[column].isin(rules['allowed_values'])
            invalid_count = invalid.sum()
            if invalid_count > 0:
                errors.append(f"{column}: {invalid_count} invalid values")
    
    return len(errors) == 0, errors
```

---

## 💻 Exercises (40 min)

### Exercise 1: Type Validation
Implement type validation for all columns in a dataset.

### Exercise 2: Range Validation
Validate numerical columns are within acceptable ranges.

### Exercise 3: Format Validation
Validate email and phone number formats using regex.

### Exercise 4: Uniqueness Validation
Check for duplicates in key columns.

### Exercise 5: Business Rules
Implement custom business rule validation.

### Exercise 6: Build Validator
Create a reusable validation framework with multiple rules.

### Exercise 7: Validation Pipeline
Build an ETL pipeline with validation and quarantine.

### Exercise 8: Schema Validation
Implement comprehensive schema validation.

---

## ✅ Quiz (5 min)

Test your understanding of data validation concepts and implementation.

---

## 🎯 Key Takeaways

- **Fail fast**: Validate at ingestion to prevent bad data propagation
- **Multiple layers**: Use type, range, format, and business rule validation
- **Quarantine**: Don't lose invalid data; save it for investigation
- **Reusable**: Build validation frameworks for consistency
- **Schema-driven**: Define schemas to automate validation
- **Clear errors**: Provide actionable error messages

---

## 📚 Resources

- [Pandera Documentation](https://pandera.readthedocs.io/)
- [Cerberus Validation](https://docs.python-cerberus.org/)
- [JSON Schema](https://json-schema.org/)
- [Data Validation Best Practices](https://www.dataversity.net/data-validation-best-practices/)

---

## Tomorrow: Day 40 - Data Lineage

Learn how to track data lineage, understand data flow through systems, implement lineage tracking, and use tools like Apache Atlas and OpenLineage.
