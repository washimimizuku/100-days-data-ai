# Day 15 Quiz: Data Warehouse vs Data Lake vs Lakehouse

## Questions

1. **What is the main difference between a data warehouse and data lake?**
   - A) Warehouse is faster
   - B) Warehouse stores structured data with schema-on-write; lake stores all data types with schema-on-read
   - C) Lake is more expensive
   - D) Warehouse doesn't support SQL

2. **What problem does a data lakehouse solve?**
   - A) Storage capacity
   - B) Combines warehouse performance/ACID with lake flexibility/cost
   - C) Network speed
   - D) User authentication

3. **Which architecture supports ACID transactions?**
   - A) Only data warehouse
   - B) Only data lake
   - C) Warehouse and lakehouse
   - D) None of them

4. **Which is cheapest for storing 10TB of data?**
   - A) Data warehouse ($230/month)
   - B) Data lake ($23/month)
   - C) Data lakehouse ($25/month)
   - D) All cost the same

5. **Which architecture is best for ML workloads?**
   - A) Data warehouse (limited ML support)
   - B) Data lake or lakehouse (flexible, all data types)
   - C) None support ML
   - D) Only cloud warehouses

6. **What is schema-on-write vs schema-on-read?**
   - A) Same thing
   - B) Schema-on-write enforces schema before storing; schema-on-read applies schema when reading
   - C) Schema-on-write is faster
   - D) Schema-on-read is more secure

7. **What technologies enable data lakehouses?**
   - A) MySQL and PostgreSQL
   - B) Delta Lake, Apache Iceberg, Apache Hudi
   - C) MongoDB and Cassandra
   - D) Redis and Memcached

8. **When should you choose a data warehouse?**
   - A) When you need to store images and videos
   - B) When you have only structured data, BI workloads, and sufficient budget
   - C) When you want the cheapest option
   - D) When you need ML capabilities

9. **What is a data swamp?**
   - A) A type of database
   - B) A data lake with poor governance and quality issues
   - C) A backup system
   - D) A caching layer

10. **What is the Lambda architecture?**
    - A) AWS-specific architecture
    - B) Combines batch layer (lake) and speed layer (streaming) feeding a warehouse
    - C) A type of data warehouse
    - D) A query optimization technique

---

## Answers

1. **B** - Warehouses store structured data with predefined schemas (schema-on-write), while lakes store all data types in raw format with schemas applied at read time (schema-on-read).

2. **B** - Lakehouses combine the best of both: warehouse-like performance and ACID transactions with lake-like flexibility and low storage costs.

3. **C** - Data warehouses and lakehouses support ACID transactions. Traditional data lakes do not (though lakehouse formats like Delta/Iceberg add ACID to lakes).

4. **B** - Data lakes use cheap object storage (S3, ADLS, GCS) at ~$0.023/GB/month = $23/TB. Warehouses cost 10-20x more for storage.

5. **B** - Data lakes and lakehouses are best for ML because they support all data types (structured, semi-structured, unstructured) and flexible schemas needed for ML workflows.

6. **B** - Schema-on-write validates and enforces schema before storing data (warehouse). Schema-on-read applies schema when querying data (lake), allowing flexibility.

7. **B** - Delta Lake, Apache Iceberg, and Apache Hudi are open table formats that add ACID transactions and warehouse features to data lakes, enabling lakehouses.

8. **B** - Choose warehouses when you have primarily structured data, traditional BI/reporting workloads, strong governance needs, and budget for higher costs.

9. **B** - A data swamp is a data lake that has become disorganized with poor data quality, lack of governance, and difficult-to-use data due to inadequate management.

10. **B** - Lambda architecture uses a batch layer (data lake) for historical data and a speed layer (streaming) for real-time data, both feeding a serving layer (often a warehouse).

---

## Scoring

- **9-10 correct**: Excellent! You understand data architecture evolution.
- **7-8 correct**: Good! Review lakehouse benefits and use cases.
- **5-6 correct**: Fair. Revisit the comparison table and examples.
- **Below 5**: Review the README and architecture patterns again.
