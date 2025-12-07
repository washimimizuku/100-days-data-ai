# Day 4 Quiz: Avro

## Questions

1. **Is Avro row-based or columnar?**
   - A) Columnar like Parquet
   - B) Row-based, storing complete records together
   - C) Hybrid format
   - D) Neither

2. **Why is Avro used with Kafka?**
   - A) It's required by Kafka
   - B) Schema Registry, compact binary format, schema evolution, type safety
   - C) It's the fastest format
   - D) It's human-readable

3. **What is Schema Registry?**
   - A) A database
   - B) Service that stores/manages Avro schemas centrally with IDs
   - C) A file format
   - D) A compression algorithm

4. **What is schema evolution?**
   - A) Deleting old schemas
   - B) Ability to change schemas over time while maintaining compatibility
   - C) Automatic schema generation
   - D) Schema compression

5. **Is Avro smaller than JSON?**
   - A) No, JSON is always smaller
   - B) Yes, Avro is ~70% smaller as binary format without repeated field names
   - C) They are the same size
   - D) Only with compression

6. **When should you use Avro over Parquet?**
   - A) For analytics workloads
   - B) For streaming (Kafka), schema changes, write-heavy workloads, full row access
   - C) For read-heavy workloads
   - D) Never

7. **Can Avro schemas have optional fields?**
   - A) No, all fields are required
   - B) Yes, using union types with null
   - C) Only in version 2.0
   - D) Only for strings

8. **Is Avro human-readable?**
   - A) Yes, it's text-based
   - B) Yes, with any text editor
   - C) No, it's binary and requires tools like avro-tools or Python libraries
   - D) Only the schema is readable

9. **What is the difference between backward and forward compatibility?**
   - A) They are the same
   - B) Backward: new code reads old data; Forward: old code reads new data
   - C) Backward is faster
   - D) Forward is deprecated

10. **How does Avro reduce message size in Kafka?**
    - A) By compressing data
    - B) Stores schema ID (4 bytes) instead of full schema; retrieves from Registry
    - C) By removing field names
    - D) By using smaller data types

---

## Answers

1. **B** - Avro is row-based. It stores complete records together, making it ideal for streaming and write-heavy workloads.
2. **B** - Avro is used with Kafka because: Schema Registry provides centralized schema management, compact binary format (70% smaller than JSON), schema evolution without breaking consumers, type safety and validation, and fast serialization/deserialization.
3. **B** - Schema Registry is a service that stores and manages Avro schemas centrally. It assigns schema IDs, validates compatibility, and enables schema evolution.
4. **B** - Schema evolution is the ability to change schemas over time while maintaining compatibility. You can add optional fields without breaking existing producers or consumers.
5. **B** - Yes, Avro is typically 70% smaller than JSON because it's a binary format and doesn't repeat field names in every record.
6. **B** - Use Avro when: streaming data (Kafka), schema changes frequently, write-heavy workloads, need full row access, or RPC/messaging systems. Use Parquet for analytics and read-heavy workloads.
7. **B** - Yes, Avro supports optional fields using union types with null: {"name": "field", "type": ["null", "string"], "default": null}.
8. **C** - No, Avro is a binary format and not human-readable. You need tools like avro-tools or Python libraries to read it.
9. **B** - Backward compatible: New code can read old data (add optional fields). Forward compatible: Old code can read new data (ignore new fields).
10. **B** - Avro stores the schema ID (4 bytes) instead of the full schema in each message. The actual schema is retrieved from Schema Registry, making messages much smaller.

---

## Scoring

- **9-10 correct**: Excellent! You understand Avro well.
- **7-8 correct**: Good! Review the areas you missed.
- **5-6 correct**: Fair. Re-read the theory section.
- **Below 5**: Review the material and try again.
