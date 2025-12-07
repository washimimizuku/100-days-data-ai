# Day 5 Quiz: Data Serialization Comparison

## Questions

1. **Which format is smallest: CSV, JSON, or Parquet?**
   - A) CSV is always smallest
   - B) JSON is smallest
   - C) Parquet is smallest (50-80% smaller than CSV)
   - D) All are the same size

2. **Which format is best for Kafka streaming?**
   - A) CSV
   - B) Avro with schema evolution and Schema Registry
   - C) Parquet
   - D) JSON

3. **Which format is human-readable?**
   - A) Parquet and Avro
   - B) CSV and JSON are text-based; Parquet, Avro, Arrow are binary
   - C) Only Arrow
   - D) All formats

4. **Which format is best for analytics?**
   - A) CSV
   - B) JSON
   - C) Parquet with columnar storage and compression
   - D) Avro

5. **Which format supports schema evolution?**
   - A) CSV has the best support
   - B) Avro has best support; Parquet has limited; CSV/JSON have none
   - C) All formats support it equally
   - D) None support it

6. **Which format is columnar?**
   - A) CSV and JSON
   - B) Parquet and Arrow are columnar; CSV, JSON, Avro are row-based
   - C) Only Avro
   - D) All formats

7. **When would you use CSV over Parquet?**
   - A) For large analytics workloads
   - B) For human-readable format, Excel/spreadsheets, small datasets (<10MB)
   - C) For best compression
   - D) For nested data

8. **What is Arrow best used for?**
   - A) Disk storage
   - B) In-memory processing, zero-copy operations, language interoperability
   - C) Streaming data
   - D) Human-readable exports

9. **Why is Parquet faster for analytics than CSV?**
   - A) Newer technology
   - B) Columnar storage, better compression, predicate pushdown, efficient encoding
   - C) Smaller file extension
   - D) Better documentation

10. **What's the relationship between Arrow and Parquet?**
    - A) They are competitors
    - B) Complementary: Parquet for disk storage, Arrow for memory processing
    - C) Arrow replaced Parquet
    - D) They are the same

---

## Answers

1. **C** - Parquet is the smallest due to columnar storage and built-in compression. It's typically 50-80% smaller than CSV and even smaller than JSON.
2. **B** - Avro is best for Kafka streaming because it provides schema evolution, compact binary format, and integrates with Schema Registry.
3. **B** - CSV and JSON are human-readable (text-based). Parquet, Avro, and Arrow are binary formats and not human-readable.
4. **C** - Parquet is best for analytics because of columnar storage (read only needed columns), excellent compression, and optimized for read-heavy workloads.
5. **B** - Avro has the best schema evolution support with backward and forward compatibility. Parquet has limited support, while CSV and JSON have none.
6. **B** - Parquet and Arrow are columnar formats. CSV, JSON, and Avro are row-based formats.
7. **B** - Use CSV when you need human-readable format, working with Excel/spreadsheets, small datasets (<10MB), or maximum compatibility needed.
8. **B** - Arrow is best for in-memory data processing, zero-copy operations between systems, and language interoperability (Python, R, Java, C++).
9. **B** - Parquet is faster because: 1) Columnar storage (read only needed columns), 2) Better compression (less I/O), 3) Predicate pushdown (skip irrelevant data), 4) Efficient encoding for data types.
10. **B** - They're complementary: Parquet stores data on disk (storage format), Arrow loads it into memory (memory format). Together they provide efficient end-to-end data processing.

---

## Scoring

- **9-10 correct**: Excellent! You understand format differences well.
- **7-8 correct**: Good! Review the areas you missed.
- **5-6 correct**: Fair. Re-read the theory section.
- **Below 5**: Review the material and try again.
