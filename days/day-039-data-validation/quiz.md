# Day 39 Quiz: Data Validation

## Questions

1. **What is the main difference between data profiling and data validation?**
   - A) They are the same thing
   - B) Profiling discovers characteristics; validation enforces requirements
   - C) Profiling is faster
   - D) Validation is only for databases

2. **What is type validation?**
   - A) Checking file types
   - B) Ensuring columns have the expected data types (int, string, etc.)
   - C) Validating user types
   - D) Checking database types

3. **What does the IQR method do in validation?**
   - A) Validates data types
   - B) Detects outliers that fall outside Q1 - 1.5*IQR or Q3 + 1.5*IQR
   - C) Checks for missing values
   - D) Validates uniqueness

4. **What is referential integrity validation?**
   - A) Checking data types
   - B) Validating foreign key relationships exist in reference tables
   - C) Checking for duplicates
   - D) Validating file references

5. **What should you do with invalid records in a validation pipeline?**
   - A) Delete them permanently
   - B) Quarantine them for investigation while processing valid records
   - C) Ignore them
   - D) Always fail the entire pipeline


6. **What is schema validation?**
   - A) Validating database schemas
   - B) Validating data against a defined schema with types, constraints, and rules
   - C) Validating JSON schemas only
   - D) Validating file schemas

7. **What is a validation rule in a validation framework?**
   - A) A database constraint
   - B) A reusable check that validates a specific aspect of data
   - C) A SQL query
   - D) A file format

8. **When should validation occur in a data pipeline?**
   - A) Only at the end
   - B) At ingestion time to fail fast and prevent bad data propagation
   - C) Never, it slows things down
   - D) Only in production

9. **What is format validation?**
   - A) Checking file formats
   - B) Verifying data matches expected patterns (email, phone, date formats)
   - C) Validating code formatting
   - D) Checking database formats

10. **What is the benefit of quarantining invalid records?**
    - A) It deletes bad data
    - B) Allows processing valid data while preserving invalid data for investigation
    - C) It fixes the data automatically
    - D) It prevents all data processing

---

## Answers

1. **B** - Profiling discovers what the data looks like (characteristics, distributions, patterns), while validation enforces requirements (rules, constraints, business logic).

2. **B** - Type validation ensures columns have the expected data types (int64, float64, object, datetime64, etc.) to prevent type errors in downstream processing.

3. **B** - The IQR (Interquartile Range) method detects outliers as values that fall outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR], where IQR = Q3 - Q1.

4. **B** - Referential integrity validation checks that foreign key values in one table exist as primary key values in the referenced table, ensuring valid relationships.

5. **B** - Quarantine invalid records by saving them separately for investigation while allowing valid records to continue processing. This prevents data loss and enables debugging.

6. **B** - Schema validation validates data against a defined schema that specifies types, nullable constraints, uniqueness requirements, ranges, and allowed values for each column.

7. **B** - A validation rule is a reusable, encapsulated check that validates a specific aspect of data (e.g., not null, within range, matches pattern).

8. **B** - Validation should occur at ingestion time to fail fast and prevent bad data from propagating through the pipeline, causing issues downstream.

9. **B** - Format validation verifies data matches expected patterns using regex or other methods (e.g., email format, phone format, date format, postal codes).

10. **B** - Quarantining allows processing valid data to continue while preserving invalid data for investigation, debugging, and potential correction without data loss.

---

## Scoring

- **9-10 correct**: Excellent! You understand data validation deeply.
- **7-8 correct**: Good! Review validation types and pipeline integration.
- **5-6 correct**: Fair. Revisit validation rules and quarantine strategies.
- **Below 5**: Review the README and examples again.
