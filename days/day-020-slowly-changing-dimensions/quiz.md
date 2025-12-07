# Day 20 Quiz: Slowly Changing Dimensions

## Questions

1. **What are slowly changing dimensions?**
   - A) Dimensions that never change
   - B) Dimensions that change slowly and unpredictably over time
   - C) Dimensions that change every day
   - D) Temporary dimensions

2. **What is SCD Type 1?**
   - A) Add new row
   - B) Overwrite old value with new value (no history)
   - C) Add new column
   - D) Create history table

3. **What is SCD Type 2?**
   - A) Overwrite
   - B) Add new row for each change, keeping full history
   - C) Add new column
   - D) Delete old records

4. **When should you use SCD Type 1?**
   - A) When history is critical
   - B) When corrections or unimportant changes don't need history
   - C) Always
   - D) Never

5. **What are effective_date and expiration_date in SCD Type 2?**
   - A) Creation dates
   - B) Dates defining when a dimension version is valid
   - C) Deletion dates
   - D) Backup dates

6. **What is the is_current flag for in SCD Type 2?**
   - A) To mark deleted records
   - B) To quickly identify the current/active version of a dimension
   - C) To mark errors
   - D) To track updates

7. **How do you query dimension state as of a specific date?**
   - A) Use is_current flag
   - B) Filter WHERE date BETWEEN effective_date AND expiration_date
   - C) Use MAX(date)
   - D) Not possible

8. **Why use surrogate keys with SCD Type 2?**
   - A) They're required by law
   - B) Natural keys repeat across versions; surrogates uniquely identify each version
   - C) They're faster
   - D) They save storage

9. **What is SCD Type 3?**
   - A) Add new row
   - B) Add column for previous value (limited history)
   - C) Overwrite
   - D) Delete records

10. **Which SCD type is most commonly used?**
    - A) Type 1
    - B) Type 2 (full history with new rows)
    - C) Type 3
    - D) Type 0

---

## Answers

1. **B** - Slowly changing dimensions are dimension attributes that change slowly and unpredictably over time (e.g., customer address, product price).

2. **B** - SCD Type 1 overwrites the old value with the new value. No history is kept. Used for corrections or when history doesn't matter.

3. **B** - SCD Type 2 adds a new row for each change, keeping full history. Each version has effective/expiration dates and is_current flag.

4. **B** - Use Type 1 for corrections, typo fixes, or changes where historical values don't matter (e.g., fixing misspelled name).

5. **B** - effective_date and expiration_date define the validity period for each dimension version. Used for point-in-time queries.

6. **B** - is_current flag (TRUE/FALSE) allows fast filtering to get only current/active dimension records without date comparisons.

7. **B** - Point-in-time query: `WHERE '2024-01-15' BETWEEN effective_date AND expiration_date` returns dimension state as of that date.

8. **B** - Natural keys (customer_id) repeat across versions. Surrogate keys (customer_key) uniquely identify each version/row in Type 2.

9. **B** - SCD Type 3 adds columns for previous values (e.g., current_address, previous_address). Tracks limited history (usually 1 previous value).

10. **B** - SCD Type 2 is most common because it provides full history while remaining relatively simple to implement and query.

---

## Scoring

- **9-10 correct**: Excellent! You understand SCD patterns.
- **7-8 correct**: Good! Review Type 2 implementation and point-in-time queries.
- **5-6 correct**: Fair. Revisit the different SCD types and use cases.
- **Below 5**: Review the README and examples again.
