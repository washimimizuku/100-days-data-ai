#!/bin/bash
# Test script for format converter

echo "=== Format Converter Test Suite ==="
echo ""

# Test 1: CSV to Parquet
echo "Test 1: CSV to Parquet (snappy)"
python format_converter.py ../../data/day-001/employees.csv test_employees.parquet -v
echo ""

# Test 2: Parquet to JSON
echo "Test 2: Parquet to JSON"
python format_converter.py test_employees.parquet test_employees.json -v
echo ""

# Test 3: JSON to CSV
echo "Test 3: JSON to CSV"
python format_converter.py test_employees.json test_employees_final.csv -v
echo ""

# Test 4: Different compressions
echo "Test 4: Testing different compressions"
python format_converter.py ../../data/day-001/employees.csv test_gzip.parquet -c gzip
python format_converter.py ../../data/day-001/employees.csv test_zstd.parquet -c zstd
python format_converter.py ../../data/day-001/employees.csv test_none.parquet -c none
echo ""

# Compare sizes
echo "=== Size Comparison ==="
ls -lh test_*.parquet
echo ""

# Test 5: Preview mode
echo "Test 5: Preview mode"
python format_converter.py ../../data/day-001/employees.csv test_preview.parquet --preview
echo ""

echo "=== All tests complete! ==="
