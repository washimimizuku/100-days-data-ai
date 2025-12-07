#!/usr/bin/env python3
"""
Format Converter Tool
Converts data between CSV, JSON, and Parquet formats with compression options.

Usage:
    python format_converter.py input.csv output.parquet
    python format_converter.py input.csv output.parquet -c gzip -v
"""

import argparse
import pandas as pd
import time
import os
import sys


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert data between CSV, JSON, and Parquet formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv data.parquet
  %(prog)s data.csv data.parquet -c gzip
  %(prog)s data.parquet data.json -v
        """
    )
    
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output file path')
    parser.add_argument(
        '-c', '--compression',
        choices=['snappy', 'gzip', 'zstd', 'none'],
        default='snappy',
        help='Compression algorithm (default: snappy)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show data preview before conversion'
    )
    
    return parser.parse_args()


def detect_format(filepath):
    """Detect file format from extension"""
    filepath_lower = filepath.lower()
    
    if filepath_lower.endswith('.csv'):
        return 'csv'
    elif filepath_lower.endswith('.json'):
        return 'json'
    elif filepath_lower.endswith('.parquet'):
        return 'parquet'
    else:
        raise ValueError(
            f"Unsupported format: {filepath}\n"
            f"Supported formats: .csv, .json, .parquet"
        )


def read_data(filepath, format_type, verbose=False):
    """Read data from any supported format"""
    if verbose:
        print(f"📖 Reading {format_type.upper()} file...")
    
    try:
        if format_type == 'csv':
            df = pd.read_csv(filepath)
        elif format_type == 'json':
            df = pd.read_json(filepath)
        elif format_type == 'parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        if verbose:
            print(f"   ✓ Read {len(df):,} rows, {len(df.columns)} columns")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error reading {format_type} file: {e}")


def write_data(df, filepath, format_type, compression, verbose=False):
    """Write data to any supported format"""
    if verbose:
        print(f"💾 Writing {format_type.upper()} file...")
    
    try:
        if format_type == 'csv':
            df.to_csv(filepath, index=False)
        elif format_type == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format_type == 'parquet':
            comp = None if compression == 'none' else compression
            df.to_parquet(filepath, compression=comp)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        if verbose:
            print(f"   ✓ Written to {filepath}")
    
    except Exception as e:
        raise Exception(f"Error writing {format_type} file: {e}")


def benchmark_conversion(input_file, output_file, df, read_time, write_time):
    """Calculate performance metrics"""
    input_size = os.path.getsize(input_file)
    output_size = os.path.getsize(output_file)
    
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'input_size_mb': input_size / 1024 / 1024,
        'output_size_mb': output_size / 1024 / 1024,
        'compression_ratio': input_size / output_size if output_size > 0 else 1,
        'read_time': read_time,
        'write_time': write_time,
        'total_time': read_time + write_time
    }


def print_metrics(metrics, input_format, output_format, compression):
    """Print performance metrics"""
    print("\n" + "="*50)
    print("✅ Conversion Complete!")
    print("="*50)
    print(f"Format:      {input_format.upper()} → {output_format.upper()}")
    print(f"Compression: {compression}")
    print(f"Rows:        {metrics['rows']:,}")
    print(f"Columns:     {metrics['columns']}")
    print(f"Input size:  {metrics['input_size_mb']:.2f} MB")
    print(f"Output size: {metrics['output_size_mb']:.2f} MB")
    print(f"Ratio:       {metrics['compression_ratio']:.2f}x")
    print(f"Read time:   {metrics['read_time']:.3f}s")
    print(f"Write time:  {metrics['write_time']:.3f}s")
    print(f"Total time:  {metrics['total_time']:.3f}s")
    print("="*50)


def preview_data(df):
    """Show data preview"""
    print("\n" + "="*50)
    print("📊 Data Preview")
    print("="*50)
    print(df.head())
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNull values: {df.isnull().sum().sum()}")
    print("="*50 + "\n")


def main():
    """Main execution function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Check input file exists
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file not found: {args.input}")
            sys.exit(1)
        
        # Detect formats
        input_format = detect_format(args.input)
        output_format = detect_format(args.output)
        
        if args.verbose:
            print(f"\n🔄 Converting {input_format.upper()} → {output_format.upper()}")
            print(f"   Compression: {args.compression}")
        
        # Read data
        start = time.time()
        df = read_data(args.input, input_format, args.verbose)
        read_time = time.time() - start
        
        # Preview if requested
        if args.preview:
            preview_data(df)
        
        # Write data
        start = time.time()
        write_data(df, args.output, output_format, args.compression, args.verbose)
        write_time = time.time() - start
        
        # Calculate metrics
        metrics = benchmark_conversion(
            args.input, args.output, df, read_time, write_time
        )
        
        # Print results
        if not args.verbose:
            print(f"✅ Converted {len(df):,} rows: {args.input} → {args.output}")
        else:
            print_metrics(metrics, input_format, output_format, args.compression)
    
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
