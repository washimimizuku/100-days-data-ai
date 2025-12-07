# Day 12 Quiz: Delta Lake ACID Transactions

## Questions

1. **What does ACID stand for?**
   - A) Automatic, Consistent, Isolated, Durable
   - B) Atomicity, Consistency, Isolation, Durability
   - C) Atomic, Complete, Independent, Distributed
   - D) Available, Consistent, Isolated, Distributed

2. **How does Delta Lake achieve atomicity?**
   - A) Through database locks
   - B) By writing to transaction log only after all data files are written
   - C) Using two-phase commit
   - D) Atomicity is not guaranteed

3. **What is optimistic concurrency control?**
   - A) Assuming conflicts won't happen; validate before commit
   - B) Locking resources pessimistically
   - C) Always allowing concurrent writes
   - D) Disabling concurrency

4. **What happens when two writers conflict in Delta Lake?**
   - A) Both transactions succeed
   - B) First transaction succeeds; second retries or fails
   - C) Both transactions fail
   - D) Data is merged automatically

5. **What isolation level does Delta Lake provide?**
   - A) Read Uncommitted
   - B) Read Committed
   - C) Repeatable Read
   - D) Serializable

6. **Can readers block writers in Delta Lake?**
   - A) Yes, always
   - B) No, readers never block writers
   - C) Only during schema changes
   - D) Only for large tables

7. **What is a transaction log checkpoint?**
   - A) A backup of all data
   - B) Aggregated state of transaction log for faster reads
   - C) A savepoint for rollback
   - D) A lock file

8. **How does Delta Lake handle write failures?**
   - A) Partial data is written
   - B) Table is corrupted
   - C) No data is written; table remains in previous consistent state
   - D) Manual cleanup required

9. **What happens during a concurrent read while a write is in progress?**
   - A) Read is blocked until write completes
   - B) Read sees partial data
   - C) Read sees consistent snapshot from before write started
   - D) Read fails with error

10. **Which operations can conflict in Delta Lake?**
    - A) Only writes to same partition
    - B) Concurrent schema changes, updates to same rows
    - C) All concurrent operations
    - D) Conflicts never occur

---

## Answers

1. **B** - ACID stands for Atomicity, Consistency, Isolation, Durability - the four key properties of reliable transactions.

2. **B** - Delta Lake writes all data files first, then atomically commits by writing to the transaction log. If log write fails, data files are ignored.

3. **A** - Optimistic concurrency assumes conflicts are rare. Transactions proceed without locks, then validate before commit. If conflict detected, retry or fail.

4. **B** - The first transaction to commit succeeds. The second transaction detects the conflict and either retries automatically or fails with ConcurrentModificationException.

5. **D** - Delta Lake provides Serializable isolation, the strongest level. Readers see complete consistent snapshots; no dirty reads or phantom reads.

6. **B** - Readers never block writers in Delta Lake. Readers see consistent snapshots while writers modify data concurrently.

7. **B** - Checkpoints aggregate the transaction log state into a Parquet file, allowing faster table state reconstruction without reading all JSON files.

8. **C** - If a write fails, no partial data is committed. The table remains in its previous consistent state. No manual cleanup needed.

9. **C** - Readers see a consistent snapshot from a specific version. They are isolated from concurrent writes and never see partial or uncommitted data.

10. **B** - Conflicts occur with concurrent schema changes, updates/deletes to the same rows, or operations that modify the same data. Non-overlapping operations succeed.

---

## Scoring

- **9-10 correct**: Excellent! You master Delta Lake ACID transactions.
- **7-8 correct**: Good! Review optimistic concurrency and isolation levels.
- **5-6 correct**: Fair. Revisit transaction log and conflict resolution.
- **Below 5**: Review the README and practice the exercises again.
