# Day 40 Quiz: Data Lineage

## Questions

1. **What is data lineage?**
   - A) A database backup system
   - B) Documentation of data's journey through systems - where it comes from, how it's transformed, where it goes
   - C) A data visualization tool
   - D) A data storage format

2. **What are the two main types of lineage?**
   - A) Fast and slow
   - B) Table-level and column-level lineage
   - C) Input and output
   - D) Source and target

3. **What is table-level lineage?**
   - A) Tracking individual rows
   - B) Tracking relationships between tables/datasets
   - C) Tracking database schemas
   - D) Tracking table sizes

4. **What is column-level lineage?**
   - A) Tracking column data types
   - B) Tracking individual column transformations and dependencies
   - C) Tracking column names
   - D) Tracking column sizes

5. **Why is data lineage important for debugging?**
   - A) It makes code faster
   - B) It allows tracing data quality issues back to their source
   - C) It reduces storage costs
   - D) It improves query performance


6. **What is impact analysis in lineage?**
   - A) Analyzing data size
   - B) Understanding downstream effects of changes to a dataset
   - C) Analyzing query performance
   - D) Analyzing storage costs

7. **What is OpenLineage?**
   - A) A database
   - B) An open standard for collecting and sharing lineage metadata
   - C) A visualization tool
   - D) A query language

8. **What does upstream lineage show?**
   - A) Future data
   - B) Source datasets and transformations that produce the current dataset
   - C) Downstream consumers
   - D) Data quality metrics

9. **What does downstream lineage show?**
   - A) Source data
   - B) Datasets and processes that consume the current dataset
   - C) Historical data
   - D) Backup locations

10. **Why is lineage important for compliance (GDPR, CCPA)?**
    - A) It's not important
    - B) It proves data provenance and helps with data deletion requests
    - C) It encrypts data
    - D) It reduces storage costs

---

## Answers

1. **B** - Data lineage is the documentation of data's journey through systems, showing where it comes from (sources), how it's transformed (processing), and where it goes (consumers).

2. **B** - The two main types are table-level lineage (tracking relationships between tables/datasets) and column-level lineage (tracking individual column transformations).

3. **B** - Table-level lineage tracks relationships between tables/datasets, showing which tables feed into which other tables through transformations.

4. **B** - Column-level lineage tracks individual column transformations and dependencies, showing how specific columns are derived from source columns.

5. **B** - Lineage allows tracing data quality issues back to their source by following the data flow upstream to identify where problems originated.

6. **B** - Impact analysis uses lineage to understand the downstream effects of changes to a dataset, showing which other datasets and processes will be affected.

7. **B** - OpenLineage is an open standard for collecting and sharing lineage metadata across different tools and platforms in a consistent format.

8. **B** - Upstream lineage shows the source datasets and transformations that produce the current dataset, tracing data back to its origins.

9. **B** - Downstream lineage shows the datasets and processes that consume the current dataset, revealing who depends on this data.

10. **B** - Lineage is critical for compliance because it proves data provenance (where data came from) and helps fulfill data deletion requests by identifying all locations where personal data exists.

---

## Scoring

- **9-10 correct**: Excellent! You understand data lineage deeply.
- **7-8 correct**: Good! Review lineage types and use cases.
- **5-6 correct**: Fair. Revisit upstream/downstream concepts.
- **Below 5**: Review the README and examples again.
