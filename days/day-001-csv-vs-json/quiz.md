# Day 1 Quiz: CSV vs JSON

## Questions

1. **What is the main structural difference between CSV and JSON?**
   - A) CSV is faster to parse
   - B) CSV is flat/tabular; JSON is hierarchical with nested structures
   - C) JSON is always smaller in size
   - D) CSV supports more data types

2. **Which format preserves data types?**
   - A) CSV preserves all data types
   - B) Both preserve data types equally
   - C) JSON preserves data types (strings, numbers, booleans, null)
   - D) Neither preserves data types

3. **When would you choose CSV over JSON?**
   - A) When you need nested data structures
   - B) When working with flat/tabular data, spreadsheets, or need smaller file size
   - C) When you need to preserve data types
   - D) When building REST APIs

4. **Can CSV represent nested data?**
   - A) Yes, using special delimiters
   - B) Yes, using multiple files
   - C) No, CSV is flat with only rows and columns
   - D) Yes, using JSON inside CSV fields

5. **Which format is typically smaller in file size?**
   - A) JSON is always smaller
   - B) CSV is typically smaller due to less structural overhead
   - C) They are always the same size
   - D) It depends only on compression

6. **What Python module is best for reading CSV files for data analysis?**
   - A) csv module for all use cases
   - B) json module
   - C) pandas for data analysis; csv module for simple reading
   - D) numpy

7. **How do you pretty-print JSON in Python?**
   - A) json.print(data)
   - B) json.dumps(data, indent=2)
   - C) json.format(data)
   - D) print(json)

8. **What does orient='records' do in pandas to_json()?**
   - A) Sorts records alphabetically
   - B) Converts each DataFrame row to a JSON object in an array
   - C) Compresses the JSON output
   - D) Validates the JSON schema

9. **What is a disadvantage of CSV for data exchange?**
   - A) Too slow to parse
   - B) No standard for handling special characters, newlines, or data types
   - C) Not supported by Excel
   - D) Requires special software

10. **Why is JSON popular for APIs?**
    - A) It's the fastest format
    - B) Supports nested structures, preserves types, human-readable, native JavaScript support
    - C) It's the smallest format
    - D) It's required by HTTP protocol

---

## Answers

1. **B** - CSV is flat/tabular with rows and columns, while JSON is hierarchical and supports nested structures (objects and arrays).
2. **C** - JSON preserves data types (strings, numbers, booleans, null), while CSV treats everything as text/strings.
3. **B** - Choose CSV for flat/tabular data, spreadsheets, when file size is critical, or maximum compatibility is needed.
4. **C** - No, CSV cannot represent nested data because it's a flat format with only rows and columns.
5. **B** - CSV is typically smaller because it has less structural overhead (no brackets, braces, or repeated key names).
6. **C** - pandas is best for data analysis with powerful DataFrame operations; csv module is better for simple reading/writing.
7. **B** - Use json.dumps(data, indent=2) or json.dump(data, file, indent=2) to add indentation for readability.
8. **B** - orient='records' converts each DataFrame row to a JSON object in an array: [{col1: val1, col2: val2}, ...].
9. **B** - CSV has no standard for handling special characters, newlines in fields, or data types, causing parsing issues.
10. **B** - JSON is popular for APIs because it supports nested data, preserves types, is human-readable, and has native support in JavaScript and most languages.

---

## Scoring

- **9-10 correct**: Excellent! You understand the differences well.
- **7-8 correct**: Good! Review the areas you missed.
- **5-6 correct**: Fair. Re-read the theory section.
- **Below 5**: Review the material and try again.
