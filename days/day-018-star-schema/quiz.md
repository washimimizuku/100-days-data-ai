# Day 18 Quiz: Star Schema

## Questions

1. **What is a star schema?**
   - A) A database backup strategy
   - B) A dimensional model with central fact table surrounded by dimension tables
   - C) A type of index
   - D) A query optimization technique

2. **What is a fact table?**
   - A) A table storing descriptive attributes
   - B) A table storing measurable business events with metrics and foreign keys
   - C) A lookup table
   - D) A temporary table

3. **What is a dimension table?**
   - A) A table storing metrics
   - B) A table providing descriptive context for facts
   - C) A fact table
   - D) A staging table

4. **What is a surrogate key?**
   - A) A business identifier
   - B) A system-generated integer key used instead of natural keys
   - C) A foreign key
   - D) A composite key

5. **What is grain in a fact table?**
   - A) Data quality
   - B) The level of detail (e.g., one row per order line)
   - C) Storage size
   - D) Query performance

6. **Why denormalize dimensions in star schema?**
   - A) To save storage
   - B) To reduce joins and improve query performance
   - C) To increase complexity
   - D) To follow normalization rules

7. **What are additive measures?**
   - A) Measures that cannot be summed
   - B) Measures that can be summed across all dimensions (e.g., quantity, amount)
   - C) Only percentages
   - D) Only averages

8. **When should you use star schema?**
   - A) For OLTP systems
   - B) For data warehouses and analytics/BI workloads
   - C) For real-time transactions
   - D) Never

9. **How many joins are typically needed in star schema queries?**
   - A) Many (5-10)
   - B) Few (1-4, one per dimension)
   - C) None
   - D) Always exactly 3

10. **What is the main advantage of star schema over normalized schemas?**
    - A) Less storage
    - B) Faster queries due to fewer joins and denormalization
    - C) Better for updates
    - D) More complex

---

## Answers

1. **B** - Star schema is a dimensional modeling technique where a central fact table is surrounded by dimension tables, forming a star shape.

2. **B** - Fact tables store measurable business events (facts) with metrics/measures and foreign keys to dimension tables.

3. **B** - Dimension tables provide descriptive context (who, what, where, when, why) for the facts, containing attributes like names, dates, locations.

4. **B** - Surrogate keys are system-generated integer keys (e.g., customer_key) used instead of natural business keys for performance and stability.

5. **B** - Grain defines the level of detail in a fact table, such as one row per order line (atomic) or one row per daily aggregate.

6. **B** - Denormalization reduces the number of joins needed in queries, significantly improving read performance for analytics workloads.

7. **B** - Additive measures can be summed across all dimensions (e.g., quantity, revenue). Semi-additive can sum across some dimensions (e.g., inventory balance).

8. **B** - Star schema is designed for data warehouses and analytics/BI workloads where read performance is critical. Not suitable for OLTP.

9. **B** - Star schema queries typically need few joins (1-4), one per dimension table, making them fast and simple.

10. **B** - The main advantage is faster query performance due to fewer joins and denormalized structure optimized for read-heavy analytics.

---

## Scoring

- **9-10 correct**: Excellent! You understand star schema design.
- **7-8 correct**: Good! Review fact vs dimension tables and grain.
- **5-6 correct**: Fair. Revisit surrogate keys and denormalization.
- **Below 5**: Review the README and examples again.
