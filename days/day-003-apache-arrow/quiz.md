# Day 3 Quiz: Apache Arrow

## Questions

1. **What is the main difference between Arrow and Parquet?**
   - A) Arrow is faster
   - B) Arrow is in-memory (RAM); Parquet is on-disk (files)
   - C) Arrow is newer
   - D) Parquet supports more data types

2. **What does "zero-copy" mean?**
   - A) No data is stored
   - B) Data accessed/shared between systems without copying in memory
   - C) Free software
   - D) Automatic backups

3. **Why is Arrow faster than traditional formats?**
   - A) Better hardware
   - B) Columnar memory layout, zero-copy, optimized for CPUs, vectorized operations
   - C) Smaller file size
   - D) Newer algorithms

4. **Is Arrow a storage format or memory format?**
   - A) Storage format for disk
   - B) Memory format for in-memory processing in RAM
   - C) Both equally
   - D) Neither

5. **Can Arrow handle nested data?**
   - A) No, only flat data
   - B) Yes, supports lists, structs, and maps
   - C) Only with special plugins
   - D) Only in Python

6. **How does Arrow improve interoperability?**
   - A) By being faster
   - B) Provides standard memory format all systems can use, eliminating conversion overhead
   - C) By using JSON
   - D) By compressing data

7. **What are Arrow compute functions?**
   - A) Mathematical formulas
   - B) Built-in operations (filter, sort, mean) that work directly on Arrow data
   - C) Cloud computing services
   - D) Database queries

8. **When would you use Arrow vs Pandas?**
   - A) Always use Arrow
   - B) Arrow for data exchange, zero-copy, large datasets; Pandas for rich analysis features
   - C) Always use Pandas
   - D) They are the same

9. **What is the relationship between Arrow and Parquet?**
   - A) They are competitors
   - B) Complementary: Parquet for disk storage, Arrow for memory processing
   - C) Arrow replaced Parquet
   - D) No relationship

10. **Why is Arrow more memory efficient than Pandas?**
    - A) It uses compression
    - B) Compact memory layout, better data types, no Python object overhead
    - C) It stores less data
    - D) It uses disk instead of memory

---

## Answers

1. **B** - Arrow is an in-memory columnar format (data in RAM), while Parquet is an on-disk storage format (data in files). Arrow is for processing, Parquet is for storage.
2. **B** - Zero-copy means data can be accessed or shared between systems without copying it in memory. This makes operations extremely fast because no data duplication occurs.
3. **B** - Arrow is faster because: columnar memory layout (cache-friendly), zero-copy operations, optimized for modern CPUs, no serialization/deserialization overhead, and vectorized operations.
4. **B** - Arrow is a memory format (in-memory). It's designed for how data is processed in RAM, not how it's stored on disk.
5. **B** - Yes, Arrow supports nested data structures including lists, structs, and maps. It can represent complex hierarchical data.
6. **B** - Arrow provides a standard memory format that all systems (Python, R, Java, C++) can use. This eliminates conversion overhead when exchanging data between different systems.
7. **B** - Arrow compute functions are built-in operations (like filter, sort, mean, max) that work directly on Arrow data without converting to other formats. They're optimized for performance.
8. **B** - Use Arrow when exchanging data between systems, need zero-copy operations, working with very large datasets, or need language interoperability. Use Pandas when you need rich data analysis features, working with smaller datasets, or need extensive data manipulation.
9. **B** - Arrow and Parquet work together: Parquet stores data on disk in columnar format, Arrow loads it into memory in columnar format. They're complementary - Parquet for storage, Arrow for processing.
10. **B** - Arrow uses a more compact memory layout with better data type representations and no Python object overhead. It stores data in contiguous memory blocks optimized for modern CPUs.

---

## Scoring

- **9-10 correct**: Excellent! You understand Arrow well.
- **7-8 correct**: Good! Review the areas you missed.
- **5-6 correct**: Fair. Re-read the theory section.
- **Below 5**: Review the material and try again.
