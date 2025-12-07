# Day 2 Quiz: Parquet Format

## Questions

1. **What is the main advantage of columnar storage?**
   - A) Faster writes
   - B) Read only needed columns; better compression with similar data together
   - C) Smaller file headers
   - D) Better for row-based queries

2. **Which compression is fastest: snappy, gzip, or zstd?**
   - A) gzip
   - B) zstd
   - C) Snappy is fastest with moderate compression
   - D) All have same speed

3. **When should you NOT use Parquet?**
   - A) For large analytics workloads
   - B) When you need human-readable format, small datasets (<10MB), or frequent schema changes
   - C) For data warehouses
   - D) For columnar analytics

4. **How do you read only specific columns from Parquet?**
   - A) Read all then filter
   - B) pd.read_parquet('file.parquet', columns=['col1', 'col2'])
   - C) Not possible
   - D) Use SQL SELECT

5. **What is partitioning and why is it useful?**
   - A) Splitting files randomly
   - B) Dividing data into folders by column values; read only needed partitions
   - C) Compressing data
   - D) Encrypting data

6. **Is Parquet human-readable like CSV?**
   - A) Yes, it's text-based
   - B) Yes, with a text editor
   - C) No, it's binary and requires tools like pandas or pyarrow
   - D) Only the headers are readable

7. **Which is typically smaller: CSV or Parquet?**
   - A) CSV is always smaller
   - B) Parquet is 50-80% smaller due to columnar storage and compression
   - C) They are the same size
   - D) Depends only on data content

8. **What metadata is stored in Parquet files?**
   - A) Only file size
   - B) Schema, row counts, column statistics (min/max), compression info
   - C) Only column names
   - D) No metadata

9. **What is the difference between row groups and columns in Parquet?**
   - A) They are the same thing
   - B) Row groups are horizontal chunks; columns are vertical partitions
   - C) Row groups are for small files only
   - D) Columns contain row groups

10. **Why is Parquet faster for analytics than CSV?**
    - A) Better file extension
    - B) Columnar storage, better compression, predicate pushdown, efficient encoding
    - C) Newer technology
    - D) Requires less memory

---

## Answers

1. **B** - Columnar storage allows reading only needed columns, making analytics queries faster. Similar data together enables better compression.
2. **C** - Snappy is the fastest compression algorithm with moderate compression ratios. It's the default for Parquet.
3. **B** - Don't use Parquet when you need human-readable format, working with small datasets (<10MB), schema changes frequently, or need to edit files manually.
4. **B** - Use the columns parameter: pd.read_parquet('file.parquet', columns=['col1', 'col2']).
5. **B** - Partitioning divides data into separate folders based on column values (e.g., by year/month). Read only the partitions you need for faster queries.
6. **C** - No, Parquet is a binary format and not human-readable. You need tools like pandas or pyarrow to read it.
7. **B** - Parquet is typically 50-80% smaller than CSV due to columnar storage and built-in compression.
8. **B** - Parquet stores schema (column names and types), row counts, column statistics (min/max values), and compression information.
9. **B** - Row groups are horizontal partitions of data (chunks of rows), while columns are vertical partitions. Parquet stores data in row groups, with each row group containing all columns.
10. **B** - Parquet is faster because: 1) Columnar storage - read only needed columns, 2) Better compression - less I/O, 3) Predicate pushdown - skip irrelevant data, 4) Efficient encoding - optimized for data types.

---

## Scoring

- **9-10 correct**: Excellent! You understand Parquet well.
- **7-8 correct**: Good! Review the areas you missed.
- **5-6 correct**: Fair. Re-read the theory section.
- **Below 5**: Review the material and try again.
