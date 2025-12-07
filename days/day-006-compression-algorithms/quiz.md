# Day 6 Quiz: Compression Algorithms

## Questions

1. **Which compression is fastest?**
   - A) Gzip
   - B) ZSTD
   - C) LZ4 is fastest with extremely low CPU usage
   - D) Snappy

2. **Which compression has best ratio?**
   - A) LZ4
   - B) Snappy
   - C) ZSTD
   - D) Gzip has best ratio (4-5x) though slowest

3. **What is the default compression for Parquet?**
   - A) Gzip
   - B) LZ4
   - C) Snappy balances speed and compression ratio
   - D) No compression

4. **When should you use Gzip?**
   - A) For real-time processing
   - B) When storage cost is critical, data rarely accessed (cold storage), maximum compression needed
   - C) For fastest writes
   - D) Never

5. **What is ZSTD's advantage?**
   - A) Fastest compression
   - B) Better compression than Snappy, faster than Gzip, adjustable levels (1-22)
   - C) Smallest file size
   - D) Best for streaming

6. **Can you use LZ4 with pandas directly?**
   - A) Yes, pandas.to_parquet(compression='lz4')
   - B) No, need PyArrow: pq.write_table(table, 'file.parquet', compression='lz4')
   - C) Yes, but only for CSV
   - D) LZ4 is not supported

7. **What compression level is ZSTD default?**
   - A) 1
   - B) 5
   - C) 3 provides good balance of speed and ratio
   - D) 10

8. **When should you use no compression?**
   - A) Always
   - B) When data already compressed (images/videos), extremely fast writes critical, temporary files
   - C) For analytics workloads
   - D) For cold storage

9. **What is the trade-off in compression?**
   - A) Size vs speed
   - B) Speed vs compression ratio: faster algorithms have lower ratios
   - C) Cost vs performance
   - D) Security vs speed

10. **How much can compression save on storage costs?**
    - A) 10-20%
    - B) 50-80% savings; e.g., 1TB with ZSTD costs $6.58/month vs $23/month uncompressed
    - C) 90-95%
    - D) No significant savings

---

## Answers

1. **C** - LZ4 is the fastest compression algorithm, followed by Snappy. LZ4 has extremely low CPU usage and is ideal for real-time systems.
2. **D** - Gzip has the best compression ratio (4-5x), though it's the slowest. It's ideal for cold storage and archival data.
3. **C** - Snappy is the default compression for Parquet because it provides a good balance of speed and compression ratio.
4. **B** - Use Gzip when storage cost is critical, data is rarely accessed (cold storage), maximum compression is needed, and speed is not important.
5. **B** - ZSTD provides better compression than Snappy while being faster than Gzip. It also has adjustable compression levels (1-22) for fine-tuning the speed/ratio trade-off.
6. **B** - No, pandas doesn't support LZ4 directly. You need to use PyArrow: pq.write_table(table, 'file.parquet', compression='lz4').
7. **C** - ZSTD default compression level is 3, which provides a good balance of speed and compression ratio.
8. **B** - Use no compression when data is already compressed (images, videos), extremely fast writes are critical, for temporary processing files, or when CPU is severely limited.
9. **B** - The trade-off is between compression speed and compression ratio. Faster algorithms (LZ4, Snappy) have lower ratios, while slower algorithms (Gzip) have higher ratios.
10. **B** - Compression can save 50-80% on storage costs. For example, 1TB with ZSTD (3.5x) costs $6.58/month vs $23/month uncompressed - saving $197/year.

---

## Scoring

- **9-10 correct**: Excellent! You understand compression well.
- **7-8 correct**: Good! Review the areas you missed.
- **5-6 correct**: Fair. Re-read the theory section.
- **Below 5**: Review the material and try again.
