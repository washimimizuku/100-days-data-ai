# Day 16 Quiz: Medallion Architecture

## Questions

1. **What are the three layers in medallion architecture?**
   - A) Input, Process, Output
   - B) Bronze (Raw), Silver (Cleaned), Gold (Curated)
   - C) Source, Transform, Target
   - D) Dev, Test, Prod

2. **What is stored in the bronze layer?**
   - A) Only clean data
   - B) Raw data exactly as received, including duplicates and bad data
   - C) Aggregated metrics
   - D) Only validated records

3. **What transformations happen in the silver layer?**
   - A) None, data is unchanged
   - B) Deduplication, validation, cleaning, type conversions
   - C) Only aggregations
   - D) Only schema changes

4. **What is the purpose of the gold layer?**
   - A) Store raw data
   - B) Business-level aggregations and metrics optimized for consumption
   - C) Backup data
   - D) Archive old data

5. **Should the bronze layer contain duplicates?**
   - A) No, always deduplicate
   - B) Yes, bronze preserves raw data as-is for audit trail
   - C) Only for testing
   - D) Duplicates are not possible

6. **How should you handle bad data in the silver layer?**
   - A) Delete it permanently
   - B) Filter out or quarantine for investigation
   - C) Fix it automatically
   - D) Ignore it

7. **What is incremental processing in medallion architecture?**
   - A) Processing all data every time
   - B) Processing only new or changed data since last run
   - C) Processing data in parallel
   - D) Processing data slowly

8. **Why use Delta Lake or Iceberg for medallion architecture?**
   - A) They're faster
   - B) ACID transactions, time travel, schema evolution
   - C) They're cheaper
   - D) They're required by law

9. **What metadata should be added in the bronze layer?**
   - A) None
   - B) Ingestion timestamp, source system, file name
   - C) Only timestamps
   - D) Only business dates

10. **Which layer should BI tools typically query?**
    - A) Bronze (too raw)
    - B) Silver (still technical)
    - C) Gold (optimized for business consumption)
    - D) All layers equally

---

## Answers

1. **B** - Medallion architecture uses Bronze (raw data), Silver (cleaned data), and Gold (curated/aggregated data) layers.

2. **B** - Bronze stores raw data exactly as received, including duplicates, nulls, and bad data to maintain a complete audit trail.

3. **B** - Silver layer applies data quality: deduplication, null filtering, validation, type conversions, and business logic.

4. **B** - Gold layer contains business-level aggregations, KPIs, and feature tables optimized for BI tools and ML models.

5. **B** - Yes, bronze should preserve all data including duplicates to maintain an exact copy of source data for audit and reprocessing.

6. **B** - Bad data should be filtered out or moved to a quarantine table for investigation, not deleted permanently.

7. **B** - Incremental processing means processing only new or changed data since the last run, improving efficiency and reducing costs.

8. **B** - Delta Lake and Iceberg provide ACID transactions, time travel, and schema evolution needed for reliable medallion pipelines.

9. **B** - Bronze should add metadata like ingestion timestamp, source system identifier, and source file name for traceability.

10. **C** - BI tools should query the Gold layer, which contains clean, aggregated, business-ready data optimized for analytics.

---

## Scoring

- **9-10 correct**: Excellent! You understand medallion architecture.
- **7-8 correct**: Good! Review layer purposes and transformations.
- **5-6 correct**: Fair. Revisit the layer comparison table.
- **Below 5**: Review the README and examples again.
