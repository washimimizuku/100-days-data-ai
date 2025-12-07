# Day 37 Quiz: Great Expectations

## Questions

1. **What is Great Expectations?**
   - A) A database management system
   - B) An open-source Python library for data validation and documentation
   - C) A machine learning framework
   - D) A data visualization tool

2. **What is an expectation in Great Expectations?**
   - A) A database query
   - B) An assertion about what your data should look like
   - C) A machine learning prediction
   - D) A data transformation

3. **What is an expectation suite?**
   - A) A single expectation
   - B) A collection of expectations for a dataset
   - C) A database schema
   - D) A test framework

4. **What does expect_column_values_to_not_be_null do?**
   - A) Deletes null values
   - B) Validates that a column has no null values
   - C) Fills null values
   - D) Counts null values

5. **What is a validator in Great Expectations?**
   - A) A database connection
   - B) An object that validates data against expectations
   - C) A user authentication system
   - D) A data transformation tool


6. **What is a checkpoint in Great Expectations?**
   - A) A database backup
   - B) A reusable validation workflow that can be run repeatedly
   - C) A data transformation step
   - D) A version control commit

7. **What are Data Docs?**
   - A) Database documentation
   - B) Auto-generated HTML documentation showing validation results
   - C) API documentation
   - D) User manuals

8. **How many built-in expectations does Great Expectations provide?**
   - A) About 50
   - B) About 100
   - C) About 300+
   - D) About 1000

9. **What does expect_column_values_to_be_between check?**
   - A) Column data type
   - B) Column values fall within a specified min and max range
   - C) Column length
   - D) Column uniqueness

10. **How do you integrate Great Expectations into a data pipeline?**
    - A) It cannot be integrated
    - B) Create validator, run validation, check results, raise error if failed
    - C) Only through command line
    - D) Only through web interface

---

## Answers

1. **B** - Great Expectations is an open-source Python library for data validation, documentation, and profiling. It helps define expectations about data and validates whether data meets those expectations.

2. **B** - An expectation is an assertion about what your data should look like. For example, "customer_id should not be null" or "age should be between 0 and 120".

3. **B** - An expectation suite is a collection of expectations for a dataset. It groups related expectations together (e.g., all expectations for customer data).

4. **B** - expect_column_values_to_not_be_null validates that a specified column has no null values. It doesn't modify data, only checks it.

5. **B** - A validator is an object that validates data against expectations. It combines a batch of data with an expectation suite to perform validation.

6. **B** - A checkpoint is a reusable validation workflow that can be run repeatedly. It packages together data, expectations, and actions to take on validation results.

7. **B** - Data Docs are auto-generated HTML documentation showing validation results, expectation suites, data profiling statistics, and historical trends in a beautiful web interface.

8. **C** - Great Expectations provides 300+ built-in expectations covering common data quality checks like nulls, ranges, patterns, uniqueness, types, and more.

9. **B** - expect_column_values_to_be_between checks that column values fall within a specified minimum and maximum range. For example, age between 0 and 120.

10. **B** - Integrate GX by creating a validator, running validation, checking results["success"], and raising an error if validation fails. This prevents bad data from propagating through the pipeline.

---

## Scoring

- **9-10 correct**: Excellent! You understand Great Expectations.
- **7-8 correct**: Good! Review expectations and checkpoints.
- **5-6 correct**: Fair. Revisit core concepts and validation workflow.
- **Below 5**: Review the README and examples again.
