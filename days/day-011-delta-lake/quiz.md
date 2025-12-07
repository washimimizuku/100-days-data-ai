# Day 11 Quiz: Delta Lake

## Questions

1. **What is the Delta Lake transaction log?**
   - A) A backup of all data files
   - B) JSON files in _delta_log/ tracking all table changes
   - C) A database storing metadata
   - D) Parquet files with transaction data

2. **What file format does Delta Lake use for data storage?**
   - A) Avro
   - B) ORC
   - C) Parquet
   - D) JSON

3. **How do you perform an UPDATE in Delta Lake?**
   - A) Rewrite all data files
   - B) Use DeltaTable.update() or UPDATE SQL statement
   - C) Delete and re-insert records
   - D) Updates are not supported

4. **What is the difference between OPTIMIZE and VACUUM?**
   - A) They do the same thing
   - B) OPTIMIZE compacts small files; VACUUM removes old files
   - C) OPTIMIZE deletes data; VACUUM compresses data
   - D) VACUUM is faster than OPTIMIZE

5. **How do you query a previous version in Delta Lake?**
   - A) `spark.read.format("delta").version(2).load(path)`
   - B) `spark.read.format("delta").option("versionAsOf", 2).load(path)`
   - C) `spark.read.delta(path, version=2)`
   - D) Previous versions cannot be queried

6. **What is schema enforcement in Delta Lake?**
   - A) Automatic schema creation
   - B) Validation that prevents writes with incompatible schemas
   - C) Schema compression
   - D) Schema encryption

7. **What does the MERGE operation do?**
   - A) Combines two tables permanently
   - B) Performs upsert (update if exists, insert if not)
   - C) Merges small files
   - D) Merges schemas

8. **How long does VACUUM retain old files by default?**
   - A) 24 hours
   - B) 7 days (168 hours)
   - C) 30 days
   - D) Forever

9. **What happens when you write data with a new column and mergeSchema=true?**
   - A) Write fails
   - B) New column is added to schema; existing rows have null
   - C) New column is ignored
   - D) Table is recreated

10. **Which operation provides ACID guarantees in Delta Lake?**
    - A) Only reads
    - B) Only writes
    - C) All operations (reads, writes, updates, deletes)
    - D) None, Delta Lake doesn't support ACID

---

## Answers

1. **B** - The transaction log is a series of JSON files in the `_delta_log/` directory that records every change to the table.

2. **C** - Delta Lake stores data in Parquet format, adding a transaction log layer on top for ACID guarantees.

3. **B** - Use `DeltaTable.update()` API or SQL `UPDATE` statement to modify records in place.

4. **B** - OPTIMIZE compacts small files into larger ones for better performance; VACUUM removes old data files no longer referenced.

5. **B** - Use `.option("versionAsOf", version_number)` or `.option("timestampAsOf", timestamp)` for time travel.

6. **B** - Schema enforcement validates incoming data against the table schema, rejecting writes with incompatible schemas.

7. **B** - MERGE performs upsert operations: updates matching records and inserts non-matching ones in a single atomic operation.

8. **B** - VACUUM retains files for 7 days (168 hours) by default to allow time travel queries.

9. **B** - With `mergeSchema=true`, the new column is added to the schema. Existing rows will have null for the new column.

10. **C** - Delta Lake provides ACID guarantees for all operations through its transaction log mechanism.

---

## Scoring

- **9-10 correct**: Excellent! You understand Delta Lake fundamentals.
- **7-8 correct**: Good! Review OPTIMIZE/VACUUM and schema management.
- **5-6 correct**: Fair. Revisit transaction log and CRUD operations.
- **Below 5**: Review the README and practice the exercises again.
