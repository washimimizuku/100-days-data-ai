# Day 10 Quiz: Iceberg Time Travel & Snapshots

## Questions

1. **What creates a new snapshot in Apache Iceberg?**
   - A) Only INSERT operations
   - B) Every write operation (INSERT, UPDATE, DELETE, MERGE)
   - C) Only when explicitly requested
   - D) Daily at midnight

2. **How do you query data as of a specific timestamp in Spark?**
   - A) `spark.read.timestamp("2024-01-01").table("table")`
   - B) `spark.read.option("as-of-timestamp", "2024-01-01 00:00:00").table("table")`
   - C) `spark.table("table").where("timestamp = '2024-01-01'")`
   - D) `spark.sql("SELECT * FROM table@2024-01-01")`

3. **What's the difference between rollback_to_snapshot and set_current_snapshot?**
   - A) They are identical operations
   - B) rollback deletes newer snapshots; set_current_snapshot keeps them
   - C) rollback is faster
   - D) set_current_snapshot requires admin privileges

4. **Why should you expire old snapshots?**
   - A) To improve query performance
   - B) To reduce storage costs and metadata overhead
   - C) To prevent data corruption
   - D) Snapshots expire automatically

5. **How do you read only changes between two snapshots?**
   - A) Query both snapshots and compare
   - B) Use `start-snapshot-id` and `end-snapshot-id` options
   - C) Use DIFF command
   - D) Not possible in Iceberg

6. **What are orphan files in Iceberg?**
   - A) Files with no data
   - B) Files not referenced by any snapshot
   - C) Corrupted files
   - D) Temporary files

7. **What metadata does a snapshot contain?**
   - A) Only the snapshot ID
   - B) snapshot_id, parent_id, timestamp, operation, manifest_list
   - C) Just the data file paths
   - D) Only the schema

8. **How long should you typically retain snapshots?**
   - A) Forever for audit purposes
   - B) Based on compliance requirements (7-90 days typical)
   - C) Only 24 hours
   - D) Until storage is full

9. **What happens to data files when you rollback?**
   - A) Old files are restored from backup
   - B) Metadata pointer changes; data files remain unchanged
   - C) All files are rewritten
   - D) Files are deleted and recreated

10. **What's the benefit of incremental reads?**
    - A) Faster queries
    - B) Process only new/changed data instead of full table
    - C) Better compression
    - D) Automatic deduplication

---

## Answers

1. **B** - Every write operation (INSERT, UPDATE, DELETE, MERGE) creates a new immutable snapshot.

2. **B** - Use `.option("as-of-timestamp", "2024-01-01 00:00:00")` to query historical data.

3. **B** - `rollback_to_snapshot` deletes newer snapshots permanently; `set_current_snapshot` keeps them for potential recovery.

4. **B** - Old snapshots consume storage and increase metadata size. Regular expiration reduces costs.

5. **B** - Use `start-snapshot-id` and `end-snapshot-id` options to read incremental changes efficiently.

6. **B** - Orphan files are data files no longer referenced by any snapshot, typically from failed writes.

7. **B** - Snapshots contain snapshot_id, parent_id, timestamp, operation type, summary stats, and manifest_list path.

8. **B** - Retention depends on compliance needs. Common: 7 days (dev), 30 days (staging), 90 days (production).

9. **B** - Rollback only changes metadata pointers. Data files are immutable and remain unchanged.

10. **B** - Incremental reads process only changes between snapshots, avoiding full table scans for efficiency.

---

## Scoring

- **9-10 correct**: Excellent! You master Iceberg time travel.
- **7-8 correct**: Good! Review snapshot management and incremental reads.
- **5-6 correct**: Fair. Revisit rollback operations and expiration.
- **Below 5**: Review the README and practice the exercises again.
