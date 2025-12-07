# Day 1: CSV vs JSON

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand the differences between CSV and JSON formats
- Know when to use CSV vs JSON
- Read and write both formats in Python
- Compare performance and use cases

---

## Theory

### What are CSV and JSON?

#### CSV (Comma-Separated Values)
A simple, flat file format where data is stored in rows and columns, separated by commas.

```csv
name,age,city,salary
Alice,25,New York,75000
Bob,30,San Francisco,95000
Charlie,28,Austin,80000
```

**Characteristics:**
- Flat structure (rows and columns)
- Human-readable
- Lightweight
- No data types (everything is text)
- Limited to tabular data

#### JSON (JavaScript Object Notation)
A hierarchical format that stores data as key-value pairs and supports nested structures.

```json
{
  "employees": [
    {
      "name": "Alice",
      "age": 25,
      "city": "New York",
      "salary": 75000,
      "skills": ["Python", "SQL"]
    },
    {
      "name": "Bob",
      "age": 30,
      "city": "San Francisco",
      "salary": 95000,
      "skills": ["Java", "Kubernetes"]
    }
  ]
}
```

**Characteristics:**
- Hierarchical structure (nested objects/arrays)
- Human-readable
- Supports data types (strings, numbers, booleans, null)
- Can represent complex relationships
- Widely used in APIs

### When to Use CSV

✅ **Use CSV when:**
- Data is flat/tabular (rows and columns)
- Working with spreadsheet applications (Excel, Google Sheets)
- Need maximum compatibility
- File size is a concern (CSV is smaller)
- Data has no nested structures
- Exporting database tables

**Examples:**
- Sales transactions
- Customer lists
- Time series data
- Database exports

### When to Use JSON

✅ **Use JSON when:**
- Data has nested/hierarchical structure
- Working with APIs (REST, GraphQL)
- Need to preserve data types
- Data has complex relationships
- Configuration files
- Exchanging data between systems

**Examples:**
- API responses
- Configuration files
- User profiles with nested data
- Document databases (MongoDB)

### Reading CSV in Python

```python
import csv

# Method 1: Using csv module
with open('data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)  # Each row is a dictionary

# Method 2: Using pandas (recommended for data analysis)
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head())
```

### Writing CSV in Python

```python
import csv

data = [
    {'name': 'Alice', 'age': 25, 'city': 'New York'},
    {'name': 'Bob', 'age': 30, 'city': 'San Francisco'}
]

with open('output.csv', 'w', newline='') as file:
    fieldnames = ['name', 'age', 'city']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(data)

# Using pandas
df = pd.DataFrame(data)
df.to_csv('output.csv', index=False)
```

### Reading JSON in Python

```python
import json

# Method 1: Using json module
with open('data.json', 'r') as file:
    data = json.load(file)
    print(data)

# Method 2: Using pandas
df = pd.read_json('data.json')
print(df.head())
```

### Writing JSON in Python

```python
import json

data = {
    'employees': [
        {'name': 'Alice', 'age': 25, 'skills': ['Python', 'SQL']},
        {'name': 'Bob', 'age': 30, 'skills': ['Java', 'Kubernetes']}
    ]
}

# Pretty print with indentation
with open('output.json', 'w') as file:
    json.dump(data, file, indent=2)

# Using pandas
df.to_json('output.json', orient='records', indent=2)
```

### Performance Comparison

| Aspect | CSV | JSON |
|--------|-----|------|
| **File Size** | Smaller | Larger (due to structure) |
| **Read Speed** | Faster for flat data | Slower |
| **Write Speed** | Faster | Slower |
| **Complexity** | Simple | Can be complex |
| **Data Types** | No (all strings) | Yes |
| **Nested Data** | No | Yes |

### Real-World Example: Data Pipeline

```python
import pandas as pd
import json

# Read CSV from database export
df = pd.read_csv('sales_data.csv')

# Transform data
df['total'] = df['quantity'] * df['price']

# Convert to JSON for API
result = {
    'summary': {
        'total_sales': df['total'].sum(),
        'record_count': len(df)
    },
    'data': df.to_dict(orient='records')
}

# Send to API
with open('api_payload.json', 'w') as f:
    json.dump(result, f, indent=2)
```

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Read CSV
Read the provided `employees.csv` file and:
- Print the first 3 rows
- Count total employees
- Calculate average salary

### Exercise 2: Write CSV
Create a CSV file with 5 products:
- product_id, name, price, category
- Use the csv module

### Exercise 3: Read JSON
Read the provided `config.json` file and:
- Print the database connection string
- List all API endpoints
- Extract the timeout value

### Exercise 4: Write JSON
Create a JSON file representing a user profile:
- name, email, age
- address (nested: street, city, country)
- skills (array)
- is_active (boolean)

### Exercise 5: CSV to JSON Conversion
Read `sales.csv` and convert it to JSON format:
- Group by category
- Calculate total sales per category
- Save as `sales_summary.json`

### Exercise 6: Performance Test
Compare read times for CSV vs JSON:
- Create a dataset with 10,000 rows
- Measure time to read CSV
- Measure time to read JSON
- Print the results

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is the main structural difference between CSV and JSON?
2. Which format preserves data types?
3. When would you choose CSV over JSON?
4. Can CSV represent nested data? Why or why not?
5. Which format is typically smaller in file size?
6. What Python module is best for reading CSV files for data analysis?
7. How do you pretty-print JSON in Python?
8. What does `orient='records'` do in pandas `to_json()`?

---

## 🎯 Key Takeaways

- **CSV** is best for flat, tabular data (spreadsheets, database exports)
- **JSON** is best for hierarchical data (APIs, configs, nested structures)
- CSV is faster and smaller, but JSON is more flexible
- Use `pandas` for data analysis with both formats
- JSON preserves data types, CSV does not
- Choose the format based on your use case, not preference

---

## 📚 Additional Resources

- [Python CSV Documentation](https://docs.python.org/3/library/csv.html)
- [Python JSON Documentation](https://docs.python.org/3/library/json.html)
- [Pandas I/O Tools](https://pandas.pydata.org/docs/user_guide/io.html)
- [Real Python - Working with JSON](https://realpython.com/python-json/)
- [CSV vs JSON Comparison](https://www.geeksforgeeks.org/difference-between-json-and-csv/)

---

## 🔗 Data Files

Sample data files are provided in `/data/day-001/`:
- `employees.csv` - Employee data
- `config.json` - Configuration example
- `sales.csv` - Sales transactions

---

## Tomorrow: Day 2 - Parquet Format

We'll learn about Parquet, a columnar storage format optimized for analytics workloads.
