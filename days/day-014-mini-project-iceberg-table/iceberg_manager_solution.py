#!/usr/bin/env python3
"""
Iceberg Table Manager - Complete Solution
"""

import argparse
import json
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *


class IcebergManager:
    def __init__(self, warehouse_path="warehouse"):
        self.spark = SparkSession.builder \
            .appName("IcebergManager") \
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
            .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
            .config("spark.sql.catalog.local.type", "hadoop") \
            .config("spark.sql.catalog.local.warehouse", warehouse_path) \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("ERROR")
    
    def create_table(self, table_name, schema_str):
        type_map = {
            "int": "INT", "long": "BIGINT", "string": "STRING",
            "decimal": "DECIMAL(10,2)", "date": "DATE", "timestamp": "TIMESTAMP"
        }
        
        fields = []
        for field in schema_str.split(","):
            name, dtype = field.strip().split(":")
            sql_type = type_map.get(dtype.lower(), "STRING")
            fields.append(f"{name} {sql_type}")
        
        schema_sql = ", ".join(fields)
        
        self.spark.sql(f"DROP TABLE IF EXISTS local.{table_name}")
        self.spark.sql(f"CREATE TABLE local.{table_name} ({schema_sql}) USING iceberg")
        print(f"✓ Created table: {table_name}")
    
    def insert_data(self, table_name, data_json):
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        if not isinstance(data, list):
            data = [data]
        
        df = self.spark.createDataFrame(data)
        df.writeTo(f"local.{table_name}").append()
        print(f"✓ Inserted {len(data)} record(s)")
    
    def query_table(self, table_name, snapshot_id=None, limit=100):
        if snapshot_id:
            df = self.spark.read.option("snapshot-id", snapshot_id).table(f"local.{table_name}")
            print(f"Querying snapshot {snapshot_id}:")
        else:
            df = self.spark.table(f"local.{table_name}")
            print(f"Querying current data:")
        
        df.show(limit, truncate=False)
        print(f"Total records: {df.count()}")
    
    def list_snapshots(self, table_name):
        snapshots = self.spark.sql(f"""
            SELECT snapshot_id, parent_id, committed_at, operation
            FROM local.{table_name}.snapshots
            ORDER BY committed_at DESC
        """)
        
        print(f"\nSnapshots for {table_name}:")
        snapshots.show(truncate=False)
    
    def rollback_snapshot(self, table_name, snapshot_id):
        self.spark.sql(f"""
            CALL local.system.rollback_to_snapshot('{table_name}', {snapshot_id})
        """)
        print(f"✓ Rolled back to snapshot {snapshot_id}")
    
    def optimize_table(self, table_name):
        print(f"Optimizing {table_name}...")
        self.spark.sql(f"""
            CALL local.system.rewrite_data_files('{table_name}')
        """)
        print(f"✓ Optimization complete")
    
    def show_stats(self, table_name):
        df = self.spark.table(f"local.{table_name}")
        record_count = df.count()
        
        files = self.spark.sql(f"SELECT COUNT(*) as cnt FROM local.{table_name}.files")
        file_count = files.first()[0]
        
        snapshots = self.spark.sql(f"SELECT COUNT(*) as cnt FROM local.{table_name}.snapshots")
        snapshot_count = snapshots.first()[0]
        
        print(f"\n{'='*50}")
        print(f"Statistics for {table_name}")
        print(f"{'='*50}")
        print(f"Records:   {record_count:,}")
        print(f"Files:     {file_count:,}")
        print(f"Snapshots: {snapshot_count:,}")
        print(f"{'='*50}\n")
    
    def close(self):
        self.spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Iceberg Table Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    create_parser = subparsers.add_parser("create", help="Create table")
    create_parser.add_argument("--table", required=True, help="Table name")
    create_parser.add_argument("--schema", required=True, help="Schema (e.g., 'id:int,name:string')")
    
    insert_parser = subparsers.add_parser("insert", help="Insert data")
    insert_parser.add_argument("--table", required=True, help="Table name")
    insert_parser.add_argument("--data", required=True, help="JSON data")
    
    query_parser = subparsers.add_parser("query", help="Query table")
    query_parser.add_argument("--table", required=True, help="Table name")
    query_parser.add_argument("--snapshot-id", type=int, help="Snapshot ID for time travel")
    query_parser.add_argument("--limit", type=int, default=100, help="Max rows")
    
    snapshots_parser = subparsers.add_parser("snapshots", help="List snapshots")
    snapshots_parser.add_argument("--table", required=True, help="Table name")
    
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to snapshot")
    rollback_parser.add_argument("--table", required=True, help="Table name")
    rollback_parser.add_argument("--snapshot-id", type=int, required=True, help="Snapshot ID")
    
    optimize_parser = subparsers.add_parser("optimize", help="Optimize table")
    optimize_parser.add_argument("--table", required=True, help="Table name")
    
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--table", required=True, help="Table name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    manager = IcebergManager()
    
    try:
        if args.command == "create":
            manager.create_table(args.table, args.schema)
        elif args.command == "insert":
            manager.insert_data(args.table, args.data)
        elif args.command == "query":
            manager.query_table(args.table, args.snapshot_id, args.limit)
        elif args.command == "snapshots":
            manager.list_snapshots(args.table)
        elif args.command == "rollback":
            manager.rollback_snapshot(args.table, args.snapshot_id)
        elif args.command == "optimize":
            manager.optimize_table(args.table)
        elif args.command == "stats":
            manager.show_stats(args.table)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
