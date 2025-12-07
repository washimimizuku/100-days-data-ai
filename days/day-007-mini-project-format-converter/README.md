# Day 7: Mini Project - Format Converter Tool

## 📖 Project Overview (2 hours)

**Time**: 2 hours


Build a command-line tool that converts data between formats (CSV, JSON, Parquet) with compression options and performance benchmarking.

**What You'll Build:**
- CLI tool with argument parsing
- Format conversion (CSV ↔ JSON ↔ Parquet)
- Compression options (snappy, gzip, zstd)
- Performance benchmarking
- Error handling
- Progress reporting

---

## Learning Objectives

By the end of this project, you will:
- Build a production-ready CLI tool
- Handle multiple data formats
- Implement error handling
- Add performance monitoring
- Create user-friendly interfaces
- Apply Week 1 knowledge

---

## Project Requirements

### Core Features (Must Have)

1. **Format Conversion**
   - CSV → Parquet
   - CSV → JSON
   - JSON → Parquet
   - Parquet → CSV
   - Parquet → JSON
   - JSON → CSV

2. **Compression Support**
   - Snappy (default)
   - Gzip
   - ZSTD
   - None

3. **CLI Interface**
   - Input file path
   - Output file path
   - Compression option
   - Verbose mode

4. **Performance Metrics**
   - Conversion time
   - File size comparison
   - Compression ratio
   - Memory usage

### Bonus Features (Nice to Have)

- Batch conversion (multiple files)
- Schema validation
- Data preview
- Format recommendation
- Configuration file support

---

## Architecture

```
format_converter.py
├── parse_arguments()      # CLI argument parsing
├── detect_format()        # Auto-detect input format
├── read_data()           # Read any format
├── write_data()          # Write any format
├── benchmark()           # Performance metrics
├── validate_data()       # Data validation
└── main()                # Main execution
```

---

## Implementation Guide

### Step 1: Setup (15 min)

Create project structure:
```
day-007-mini-project-format-converter/
├── format_converter.py    # Main CLI tool
├── test_data/            # Test files
├── README.md             # This file
└── requirements.txt      # Dependencies
```

### Step 2: CLI Interface (20 min)

```python
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Convert data between formats'
    )
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output file path')
    parser.add_argument(
        '-c', '--compression',
        choices=['snappy', 'gzip', 'zstd', 'none'],
        default='snappy',
        help='Compression algorithm'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()
```

### Step 3: Format Detection (15 min)

```python
def detect_format(filepath):
    """Detect file format from extension"""
    if filepath.endswith('.csv'):
        return 'csv'
    elif filepath.endswith('.json'):
        return 'json'
    elif filepath.endswith('.parquet'):
        return 'parquet'
    else:
        raise ValueError(f"Unsupported format: {filepath}")
```

### Step 4: Read Functions (20 min)

```python
import pandas as pd

def read_data(filepath, format_type):
    """Read data from any format"""
    if format_type == 'csv':
        return pd.read_csv(filepath)
    elif format_type == 'json':
        return pd.read_json(filepath)
    elif format_type == 'parquet':
        return pd.read_parquet(filepath)
```

### Step 5: Write Functions (20 min)

```python
def write_data(df, filepath, format_type, compression):
    """Write data to any format"""
    if format_type == 'csv':
        df.to_csv(filepath, index=False)
    elif format_type == 'json':
        df.to_json(filepath, orient='records', indent=2)
    elif format_type == 'parquet':
        comp = None if compression == 'none' else compression
        df.to_parquet(filepath, compression=comp)
```

### Step 6: Benchmarking (20 min)

```python
import time
import os

def benchmark_conversion(input_file, output_file, df):
    """Measure performance metrics"""
    input_size = os.path.getsize(input_file)
    output_size = os.path.getsize(output_file)
    
    return {
        'input_size_mb': input_size / 1024 / 1024,
        'output_size_mb': output_size / 1024 / 1024,
        'compression_ratio': input_size / output_size,
        'rows': len(df),
        'columns': len(df.columns)
    }
```

### Step 7: Main Function (20 min)

```python
def main():
    args = parse_arguments()
    
    # Detect formats
    input_format = detect_format(args.input)
    output_format = detect_format(args.output)
    
    if args.verbose:
        print(f"Converting {input_format} → {output_format}")
    
    # Read
    start = time.time()
    df = read_data(args.input, input_format)
    read_time = time.time() - start
    
    # Write
    start = time.time()
    write_data(df, args.output, output_format, args.compression)
    write_time = time.time() - start
    
    # Benchmark
    metrics = benchmark_conversion(args.input, args.output, df)
    
    # Report
    print(f"✅ Conversion complete!")
    print(f"   Rows: {metrics['rows']:,}")
    print(f"   Input: {metrics['input_size_mb']:.2f} MB")
    print(f"   Output: {metrics['output_size_mb']:.2f} MB")
    print(f"   Ratio: {metrics['compression_ratio']:.2f}x")
    print(f"   Time: {read_time + write_time:.2f}s")
```

### Step 8: Error Handling (10 min)

```python
try:
    main()
except FileNotFoundError as e:
    print(f"❌ Error: File not found - {e}")
except ValueError as e:
    print(f"❌ Error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

---

## Usage Examples

### Basic Conversion

```bash
# CSV to Parquet
python format_converter.py data.csv data.parquet

# Parquet to JSON
python format_converter.py data.parquet data.json

# JSON to CSV
python format_converter.py data.json data.csv
```

### With Compression

```bash
# Use gzip compression
python format_converter.py data.csv data.parquet -c gzip

# Use zstd compression
python format_converter.py data.csv data.parquet -c zstd

# No compression
python format_converter.py data.csv data.parquet -c none
```

### Verbose Mode

```bash
python format_converter.py data.csv data.parquet -v
```

---

## Testing Your Tool

### Test 1: CSV to Parquet
```bash
python format_converter.py ../../data/day-001/employees.csv employees.parquet
```

### Test 2: With Different Compressions
```bash
python format_converter.py employees.csv employees_snappy.parquet -c snappy
python format_converter.py employees.csv employees_gzip.parquet -c gzip
python format_converter.py employees.csv employees_zstd.parquet -c zstd
```

### Test 3: Round-trip Conversion
```bash
# CSV → Parquet → JSON → CSV
python format_converter.py data.csv data.parquet
python format_converter.py data.parquet data.json
python format_converter.py data.json data_final.csv
# Verify data_final.csv matches data.csv
```

---

## Expected Output

```
✅ Conversion complete!
   Rows: 10,000
   Input: 2.5 MB
   Output: 0.8 MB
   Ratio: 3.12x
   Time: 0.45s
```

---

## Bonus Challenges

### Challenge 1: Batch Conversion
Convert all CSV files in a directory:
```python
def batch_convert(input_dir, output_dir, output_format):
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            # Convert each file
            pass
```

### Challenge 2: Data Preview
Show first 5 rows before conversion:
```python
if args.preview:
    print(df.head())
```

### Challenge 3: Schema Validation
Validate data types and nulls:
```python
def validate_schema(df):
    print(f"Columns: {len(df.columns)}")
    print(f"Nulls: {df.isnull().sum().sum()}")
    print(f"Dtypes:\n{df.dtypes}")
```

### Challenge 4: Format Recommendation
Suggest best format based on data:
```python
def recommend_format(df):
    if df.select_dtypes(include=['object']).shape[1] > 5:
        return 'parquet'  # Many text columns
    elif len(df) < 1000:
        return 'csv'  # Small dataset
    else:
        return 'parquet'  # Default
```

---

## Success Criteria

- [ ] Tool converts between all 6 format combinations
- [ ] Supports 4 compression options
- [ ] Shows performance metrics
- [ ] Handles errors gracefully
- [ ] Has clear CLI interface
- [ ] Works with test data
- [ ] Code is well-documented
- [ ] Completes in under 2 hours

---

## Deliverables

1. **format_converter.py** - Working CLI tool
2. **test_results.txt** - Output from test runs
3. **README.md** - Usage documentation
4. **requirements.txt** - Dependencies

---

## Requirements

```txt
pandas>=2.0.0
pyarrow>=12.0.0
```

---

## Tips

- Start with basic conversion, add features incrementally
- Test each function independently
- Use verbose mode for debugging
- Handle edge cases (empty files, large files)
- Add helpful error messages
- Keep code modular and reusable

---

## What You've Learned

This project combines everything from Week 1:
- ✅ CSV, JSON, Parquet formats (Days 1-2)
- ✅ Apache Arrow (Day 3)
- ✅ Avro concepts (Day 4)
- ✅ Format comparison (Day 5)
- ✅ Compression algorithms (Day 6)

---

## Next Steps

After completing this project:
1. Test with different datasets
2. Add more features (batch, validation)
3. Share your tool on GitHub
4. Use it in future projects
5. Move to Week 2: Modern Table Formats

---

**Estimated Time:** 2 hours
**Difficulty:** Intermediate
**Skills:** Python, CLI, Data Formats, Error Handling
