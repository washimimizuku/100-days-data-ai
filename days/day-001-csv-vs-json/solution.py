"""
Day 1: CSV vs JSON - Solutions
"""

import csv
import json
import pandas as pd
import time

# Exercise 1: Read CSV
print("Exercise 1: Read CSV")

# Using pandas
df = pd.read_csv('../../data/day-001/employees.csv')
print("First 3 rows:")
print(df.head(3))
print(f"\nTotal employees: {len(df)}")
print(f"Average salary: ${df['salary'].mean():,.2f}")
print()

# Exercise 2: Write CSV
print("Exercise 2: Write CSV")

products = [
    {'product_id': 1, 'name': 'Laptop', 'price': 999.99, 'category': 'Electronics'},
    {'product_id': 2, 'name': 'Mouse', 'price': 29.99, 'category': 'Electronics'},
    {'product_id': 3, 'name': 'Desk', 'price': 299.99, 'category': 'Furniture'},
    {'product_id': 4, 'name': 'Chair', 'price': 199.99, 'category': 'Furniture'},
    {'product_id': 5, 'name': 'Monitor', 'price': 399.99, 'category': 'Electronics'}
]

with open('products.csv', 'w', newline='') as file:
    fieldnames = ['product_id', 'name', 'price', 'category']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(products)

print("Created products.csv with 5 products")
print()

# Exercise 3: Read JSON
print("Exercise 3: Read JSON")

with open('../../data/day-001/config.json', 'r') as file:
    config = json.load(file)

print(f"Database connection: {config['database']['connection_string']}")
print(f"API endpoints: {', '.join(config['api']['endpoints'])}")
print(f"Timeout: {config['api']['timeout']} seconds")
print()

# Exercise 4: Write JSON
print("Exercise 4: Write JSON")

user_profile = {
    'name': 'Alice Johnson',
    'email': 'alice@example.com',
    'age': 28,
    'address': {
        'street': '123 Main St',
        'city': 'San Francisco',
        'country': 'USA'
    },
    'skills': ['Python', 'SQL', 'Machine Learning', 'Data Analysis'],
    'is_active': True
}

with open('user_profile.json', 'w') as file:
    json.dump(user_profile, file, indent=2)

print("Created user_profile.json")
print(json.dumps(user_profile, indent=2))
print()

# Exercise 5: CSV to JSON Conversion
print("Exercise 5: CSV to JSON Conversion")

# Read sales data
df_sales = pd.read_csv('../../data/day-001/sales.csv')

# Group by category and calculate totals
sales_summary = df_sales.groupby('category').agg({
    'quantity': 'sum',
    'price': 'mean',
    'total': 'sum'
}).reset_index()

# Convert to JSON
result = {
    'summary': {
        'total_revenue': df_sales['total'].sum(),
        'total_transactions': len(df_sales),
        'categories': len(sales_summary)
    },
    'by_category': sales_summary.to_dict(orient='records')
}

with open('sales_summary.json', 'w') as file:
    json.dump(result, file, indent=2)

print("Created sales_summary.json")
print(f"Total revenue: ${result['summary']['total_revenue']:,.2f}")
print(f"Categories: {result['summary']['categories']}")
print()

# Exercise 6: Performance Test
print("Exercise 6: Performance Test")

# Create test data
test_data = pd.DataFrame({
    'id': range(10000),
    'name': [f'Item_{i}' for i in range(10000)],
    'value': [i * 1.5 for i in range(10000)],
    'category': [f'Cat_{i % 10}' for i in range(10000)]
})

# Save as CSV and JSON
test_data.to_csv('test_data.csv', index=False)
test_data.to_json('test_data.json', orient='records')

# Measure CSV read time
start = time.time()
df_csv = pd.read_csv('test_data.csv')
csv_time = time.time() - start

# Measure JSON read time
start = time.time()
df_json = pd.read_json('test_data.json')
json_time = time.time() - start

# Get file sizes
import os
csv_size = os.path.getsize('test_data.csv') / 1024  # KB
json_size = os.path.getsize('test_data.json') / 1024  # KB

print(f"CSV read time: {csv_time:.4f} seconds")
print(f"JSON read time: {json_time:.4f} seconds")
print(f"CSV is {json_time/csv_time:.2f}x faster")
print(f"\nCSV file size: {csv_size:.2f} KB")
print(f"JSON file size: {json_size:.2f} KB")
print(f"CSV is {json_size/csv_size:.2f}x smaller")
print()

# Bonus Challenge
print("Bonus Challenge: CSV to JSON Converter")

def csv_to_json(csv_file, json_file, delimiter=','):
    """
    Convert CSV file to JSON format
    
    Args:
        csv_file: Path to input CSV file
        json_file: Path to output JSON file
        delimiter: CSV delimiter (default: comma)
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_file, delimiter=delimiter)
        
        # Convert to JSON
        df.to_json(json_file, orient='records', indent=2)
        
        print(f"Successfully converted {csv_file} to {json_file}")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")

# Test the function
csv_to_json('products.csv', 'products.json')

print("\n✅ All exercises completed!")
