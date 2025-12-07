"""
Day 5: Data Serialization Comparison - Solutions
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import os
import json

# Exercise 1: Format Benchmark
print("Exercise 1: Format Benchmark")

# Create dataset
df = pd.DataFrame({
    'id': range(100000),
    'name': [f'User_{i}' for i in range(100000)],
    'value': [i * 1.5 for i in range(100000)],
    'category': [f'Cat_{i % 10}' for i in range(100000)]
})

formats = {}

# CSV
start = time.time()
df.to_csv('benchmark.csv', index=False)
formats['CSV'] = {
    'write_time': time.time() - start,
    'size': os.path.getsize('benchmark.csv')
}

# JSON
start = time.time()
df.to_json('benchmark.json', orient='records')
formats['JSON'] = {
    'write_time': time.time() - start,
    'size': os.path.getsize('benchmark.json')
}

# Parquet
start = time.time()
df.to_parquet('benchmark.parquet', compression='snappy')
formats['Parquet'] = {
    'write_time': time.time() - start,
    'size': os.path.getsize('benchmark.parquet')
}

print("Write Performance:")
for fmt, stats in formats.items():
    print(f"{fmt}:")
    print(f"  Write time: {stats['write_time']:.3f}s")
    print(f"  File size: {stats['size'] / 1024 / 1024:.2f} MB")
print()

# Exercise 2: Read Performance
print("Exercise 2: Read Performance")

read_times = {}

# CSV
start = time.time()
df_csv = pd.read_csv('benchmark.csv')
read_times['CSV'] = time.time() - start

# JSON
start = time.time()
df_json = pd.read_json('benchmark.json')
read_times['JSON'] = time.time() - start

# Parquet (all columns)
start = time.time()
df_parquet = pd.read_parquet('benchmark.parquet')
read_times['Parquet (all)'] = time.time() - start

# Parquet (specific columns)
start = time.time()
df_parquet_cols = pd.read_parquet('benchmark.parquet', columns=['id', 'name'])
read_times['Parquet (2 cols)'] = time.time() - start

print("Read Performance:")
for fmt, read_time in read_times.items():
    print(f"{fmt}: {read_time:.3f}s")
print()

# Exercise 3: Format Conversion
print("Exercise 3: Format Conversion")

# Read CSV
df_employees = pd.read_csv('../../data/day-001/employees.csv')
print(f"Read {len(df_employees)} employees from CSV")

# Convert to Parquet
df_employees.to_parquet('employees_converted.parquet')
parquet_size = os.path.getsize('employees_converted.parquet')

# Convert to JSON
df_employees.to_json('employees_converted.json', orient='records', indent=2)
json_size = os.path.getsize('employees_converted.json')

# Original CSV size
csv_size = os.path.getsize('../../data/day-001/employees.csv')

# Verify integrity
df_from_parquet = pd.read_parquet('employees_converted.parquet')
df_from_json = pd.read_json('employees_converted.json')

print(f"Data integrity: {df_employees.equals(df_from_parquet) and df_employees.equals(df_from_json)}")
print(f"\nSize comparison:")
print(f"  CSV: {csv_size / 1024:.2f} KB")
print(f"  Parquet: {parquet_size / 1024:.2f} KB ({csv_size/parquet_size:.2f}x smaller)")
print(f"  JSON: {json_size / 1024:.2f} KB ({csv_size/json_size:.2f}x smaller)")
print()

# Exercise 4: Decision Matrix
print("Exercise 4: Decision Matrix")

scenarios = {
    "E-commerce analytics (10M rows, read-heavy)": "Parquet",
    "Real-time events (streaming, schema changes)": "Avro",
    "REST API (nested data, web clients)": "JSON",
    "Database export (1000 rows, for Excel)": "CSV",
    "ML pipeline (in-memory, Python to Spark)": "Arrow"
}

print("Format recommendations:")
for scenario, format_choice in scenarios.items():
    print(f"  {scenario}")
    print(f"    → {format_choice}")

print("\nJustifications:")
print("  Parquet: Columnar storage, best compression, fast for analytics")
print("  Avro: Schema evolution, Kafka standard, compact binary")
print("  JSON: Human-readable, native web support, nested structures")
print("  CSV: Excel compatible, simple, universal")
print("  Arrow: Zero-copy, in-memory speed, language agnostic")
print()

# Exercise 5: Compression Test
print("Exercise 5: Compression Test")

df_compress = pd.DataFrame({
    'id': range(50000),
    'text': [f'Text_{i}' * 20 for i in range(50000)]
})

compressions = ['snappy', 'gzip', 'zstd', None]
compression_results = {}

for comp in compressions:
    comp_name = comp if comp else 'none'
    filename = f'compressed_{comp_name}.parquet'
    
    # Write
    start = time.time()
    df_compress.to_parquet(filename, compression=comp)
    write_time = time.time() - start
    
    # Size
    size = os.path.getsize(filename)
    
    # Read
    start = time.time()
    pd.read_parquet(filename)
    read_time = time.time() - start
    
    compression_results[comp_name] = {
        'size_kb': size / 1024,
        'write_time': write_time,
        'read_time': read_time
    }

print("Compression comparison:")
for comp, stats in compression_results.items():
    print(f"{comp}:")
    print(f"  Size: {stats['size_kb']:.2f} KB")
    print(f"  Write: {stats['write_time']:.3f}s")
    print(f"  Read: {stats['read_time']:.3f}s")

# Find best
best_size = min(compression_results.items(), key=lambda x: x[1]['size_kb'])
best_speed = min(compression_results.items(), key=lambda x: x[1]['read_time'])
print(f"\nBest compression: {best_size[0]}")
print(f"Fastest read: {best_speed[0]}")
print()

# Exercise 6: Real-World Pipeline
print("Exercise 6: Real-World Pipeline")

# Create source CSV
df_source = pd.DataFrame({
    'id': range(10000),
    'value': range(10000),
    'category': ['A', 'B', 'C'] * 3333 + ['A']
})
df_source.to_csv('pipeline_source.csv', index=False)

# Pipeline: CSV → Parquet → Arrow → Processing
start_total = time.time()

# Step 1: CSV to Parquet
start = time.time()
df = pd.read_csv('pipeline_source.csv')
df.to_parquet('pipeline_intermediate.parquet')
step1_time = time.time() - start

# Step 2: Parquet to Arrow
start = time.time()
table = pq.read_table('pipeline_intermediate.parquet')
step2_time = time.time() - start

# Step 3: Arrow processing
start = time.time()
import pyarrow.compute as pc
filtered = table.filter(pc.greater(table['value'], 5000))
result = filtered.to_pandas()
step3_time = time.time() - start

total_time = time.time() - start_total

print("Pipeline performance:")
print(f"  CSV → Parquet: {step1_time:.3f}s")
print(f"  Parquet → Arrow: {step2_time:.3f}s")
print(f"  Arrow processing: {step3_time:.3f}s")
print(f"  Total: {total_time:.3f}s")
print(f"  Processed {len(result)} rows")
print()

# Bonus Challenge
print("Bonus Challenge: Format Recommender")

def recommend_format(size_mb, use_case, schema_stable, need_human_readable):
    """
    Recommend best data format
    """
    # Human-readable requirement
    if need_human_readable:
        if size_mb < 10:
            return {
                'format': 'CSV',
                'reason': 'Small size, human-readable, universal compatibility'
            }
        else:
            return {
                'format': 'JSON',
                'reason': 'Human-readable with support for nested structures'
            }
    
    # Use case based
    if use_case == 'analytics':
        return {
            'format': 'Parquet',
            'reason': 'Columnar storage ideal for analytics, excellent compression'
        }
    elif use_case == 'streaming':
        if schema_stable:
            return {
                'format': 'Avro',
                'reason': 'Kafka standard, compact, schema evolution support'
            }
        else:
            return {
                'format': 'JSON',
                'reason': 'Flexible schema, widely supported in streaming'
            }
    elif use_case == 'api':
        return {
            'format': 'JSON',
            'reason': 'Native web support, human-readable, nested structures'
        }
    elif use_case == 'export':
        return {
            'format': 'CSV',
            'reason': 'Excel compatible, simple, universal'
        }
    else:  # in-memory processing
        return {
            'format': 'Arrow',
            'reason': 'Zero-copy, fastest in-memory processing, language agnostic'
        }

# Test cases
test_cases = [
    (1000, 'analytics', True, False),
    (50, 'streaming', False, False),
    (10, 'api', True, True),
    (1, 'export', True, True),
    (500, 'processing', True, False)
]

print("Format recommendations:")
for size, use, stable, readable in test_cases:
    result = recommend_format(size, use, stable, readable)
    print(f"\n{size}MB, {use}, schema_stable={stable}, readable={readable}")
    print(f"  → {result['format']}: {result['reason']}")

print("\n✅ All exercises completed!")
