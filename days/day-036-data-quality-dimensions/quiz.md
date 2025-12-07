# Day 36 Quiz: Data Quality Dimensions

## Questions

1. **What are the six core dimensions of data quality?**
   - A) Speed, Size, Security, Scalability, Stability, Structure
   - B) Accuracy, Completeness, Consistency, Validity, Uniqueness, Timeliness
   - C) Format, Schema, Type, Size, Location, Owner
   - D) Read, Write, Update, Delete, Query, Index

2. **What does data accuracy measure?**
   - A) How fast data can be processed
   - B) Whether data correctly represents real-world entities or events
   - C) How much data is stored
   - D) How many users access the data

3. **What is data completeness?**
   - A) All data is compressed
   - B) All required data is present with no missing values in critical fields
   - C) All data is encrypted
   - D) All data is backed up

4. **What does consistency mean in data quality?**
   - A) Data is always available
   - B) Data is uniform across systems and doesn't contradict itself
   - C) Data is stored in one location
   - D) Data is never updated

5. **What is data validity?**
   - A) Data is old
   - B) Data conforms to defined business rules and constraints
   - C) Data is large
   - D) Data is public


6. **What does uniqueness measure?**
   - A) How many copies of data exist
   - B) No duplicate records exist; each entity is represented once
   - C) How rare the data is
   - D) How many users have access

7. **What is data timeliness?**
   - A) How long data takes to process
   - B) Data is up-to-date and available when needed
   - C) The time data was created
   - D) How often data is accessed

8. **How do you calculate an overall data quality score?**
   - A) Add all dimension scores
   - B) Use weighted average of dimension scores based on importance
   - C) Take the minimum score
   - D) Count the number of errors

9. **What is a typical weight for accuracy in quality scoring?**
   - A) 5%
   - B) 20% (one of the most important dimensions)
   - C) 50%
   - D) 100%

10. **What does a data quality grade of 'B' typically represent?**
    - A) 95%+ quality score
    - B) 85-94% quality score
    - C) 75-84% quality score
    - D) Below 65% quality score

---

## Answers

1. **B** - The six core dimensions of data quality are: Accuracy (correct representation), Completeness (no missing data), Consistency (uniform across systems), Validity (conforms to rules), Uniqueness (no duplicates), and Timeliness (up-to-date).

2. **B** - Data accuracy measures whether data correctly represents the real-world entities or events it's supposed to describe. For example, valid email addresses, correct prices, and accurate dates.

3. **B** - Data completeness means all required data is present with no missing values in critical fields. For example, customer records should have name, email, and phone number.

4. **B** - Consistency means data is uniform across systems and doesn't contradict itself. For example, date formats are consistent (all YYYY-MM-DD), country codes follow standards, and customer data matches across systems.

5. **B** - Data validity means data conforms to defined business rules and constraints. For example, status must be from allowed values, quantities must be positive, and dates must be in valid ranges.

6. **B** - Uniqueness measures that no duplicate records exist and each entity is represented once. For example, customer IDs are unique, order numbers don't repeat, and email addresses appear only once.

7. **B** - Data timeliness means data is up-to-date and available when needed. For example, stock prices are current, customer addresses reflect recent changes, and inventory counts are fresh.

8. **B** - Overall quality score is calculated using a weighted average of dimension scores based on their importance to the business. Different dimensions can have different weights (e.g., accuracy 20%, completeness 20%, etc.).

9. **B** - Accuracy typically receives a 20% weight in quality scoring as it's one of the most important dimensions. Common weights: Accuracy 20%, Completeness 20%, Validity 20%, Consistency 15%, Uniqueness 15%, Timeliness 10%.

10. **B** - A grade of 'B' typically represents 85-94% quality score. Grading scale: A (95%+), B (85-94%), C (75-84%), D (65-74%), F (<65%).

---

## Scoring

- **9-10 correct**: Excellent! You understand data quality dimensions.
- **7-8 correct**: Good! Review dimension definitions and measurement.
- **5-6 correct**: Fair. Revisit the six dimensions and scoring methods.
- **Below 5**: Review the README and examples again.
