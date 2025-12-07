# Day 19 Quiz: Snowflake Schema

## Questions

1. **What is a snowflake schema?**
   - A) A cloud data warehouse
   - B) A normalized dimensional model with dimension hierarchies
   - C) A type of star schema
   - D) A query optimization technique

2. **How is snowflake schema different from star schema?**
   - A) They are the same
   - B) Snowflake normalizes dimensions into multiple related tables; star keeps them denormalized
   - C) Snowflake is faster
   - D) Snowflake uses different fact tables

3. **Why normalize dimensions in snowflake schema?**
   - A) To make queries faster
   - B) To reduce storage by eliminating redundancy
   - C) To increase complexity
   - D) It's required by law

4. **What are the trade-offs of snowflake schema?**
   - A) No trade-offs
   - B) Less storage but slower queries due to more joins
   - C) More storage but faster queries
   - D) Same as star schema

5. **When should you use snowflake schema?**
   - A) Always
   - B) When storage optimization is more important than query speed
   - C) Never
   - D) Only for small databases

6. **What are materialized views in snowflake schema?**
   - A) Regular tables
   - B) Pre-joined dimension hierarchies stored for faster queries
   - C) Temporary tables
   - D) Backup tables

7. **What is selective normalization?**
   - A) Normalizing all dimensions
   - B) Normalizing only large dimensions while keeping small ones denormalized (hybrid)
   - C) Never normalizing
   - D) Random normalization

8. **Which is faster: star or snowflake schema?**
   - A) Snowflake (more joins)
   - B) Star (fewer joins, denormalized)
   - C) Same speed
   - D) Depends on database size only

9. **How many joins are needed in snowflake schema for a 3-level hierarchy?**
   - A) 1 join
   - B) 3 joins (one per level)
   - C) No joins
   - D) Always 10 joins

10. **What is the main advantage of snowflake schema?**
    - A) Faster queries
    - B) Reduced storage through normalization and better data integrity
    - C) Simpler design
    - D) No advantages

---

## Answers

1. **B** - Snowflake schema is a normalized dimensional model where dimension tables are split into multiple related tables forming hierarchies.

2. **B** - Snowflake normalizes dimensions (e.g., Customer → City → State → Country), while star keeps all attributes in one denormalized dimension table.

3. **B** - Normalization reduces storage by eliminating redundant data (e.g., country name repeated for every customer in same country).

4. **B** - Trade-off: Less storage and better data integrity, but slower queries due to more joins needed to traverse hierarchies.

5. **B** - Use snowflake when storage optimization is critical, dimensions are large with redundancy, and query performance is acceptable.

6. **B** - Materialized views pre-join dimension hierarchies and store results, providing star-like query performance with snowflake storage benefits.

7. **B** - Selective normalization (hybrid approach) normalizes only large dimensions while keeping small dimensions denormalized for optimal balance.

8. **B** - Star schema is faster because it requires fewer joins (denormalized). Snowflake requires multiple joins to traverse hierarchies.

9. **B** - A 3-level hierarchy requires 3 joins (e.g., Customer → City → State → Country = 3 joins).

10. **B** - Main advantage is reduced storage through normalization and improved data integrity, at the cost of query performance.

---

## Scoring

- **9-10 correct**: Excellent! You understand snowflake schema trade-offs.
- **7-8 correct**: Good! Review normalization benefits and materialized views.
- **5-6 correct**: Fair. Revisit star vs snowflake comparison.
- **Below 5**: Review the README and examples again.
