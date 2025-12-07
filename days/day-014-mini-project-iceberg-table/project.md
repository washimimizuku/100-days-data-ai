# Mini Project: Iceberg Table Manager

## Objective

Build a command-line tool to manage Apache Iceberg tables with full CRUD operations, time travel, and maintenance features.

## Requirements

### Functional Requirements

1. **Table Management**
   - Create tables with custom schemas
   - Drop tables
   - List all tables

2. **Data Operations**
   - Insert records (JSON format)
   - Query data (with optional filters)
   - Update records
   - Delete records

3. **Time Travel**
   - List all snapshots
   - Query specific snapshot
   - Query by timestamp
   - Rollback to previous snapshot

4. **Maintenance**
   - Expire old snapshots
   - Remove orphan files
   - Compact small files
   - Display table statistics

5. **CLI Interface**
   - Argparse-based commands
   - Help documentation
   - Verbose mode
   - Error handling

### Non-Functional Requirements

- Clean, readable code
- Proper error handling
- Comprehensive documentation
- Test coverage
- Performance considerations

## Architecture

```
iceberg_manager.py
├── IcebergManager class
│   ├── __init__(warehouse_path)
│   ├── create_table(name, schema)
│   ├── insert_data(table, data)
│   ├── query_table(table, snapshot_id)
│   ├── list_snapshots(table)
│   ├── rollback(table, snapshot_id)
│   ├── optimize(table)
│   └── stats(table)
└── CLI interface (argparse)
```

## Implementation Steps

1. **Setup** (15 min)
   - Initialize Spark with Iceberg
   - Create IcebergManager class
   - Setup argparse structure

2. **Basic Operations** (30 min)
   - Implement create_table
   - Implement insert_data
   - Implement query_table
   - Test basic flow

3. **Time Travel** (20 min)
   - Implement list_snapshots
   - Implement snapshot queries
   - Implement rollback
   - Test time travel

4. **Maintenance** (20 min)
   - Implement optimize
   - Implement expire_snapshots
   - Implement stats
   - Test maintenance

5. **Testing & Polish** (15 min)
   - Create test script
   - Add error handling
   - Write documentation
   - Final testing

## Deliverables

1. `iceberg_manager.py` - Main implementation
2. `test_manager.sh` - Test script
3. `requirements.txt` - Dependencies
4. `README.md` - Usage guide

## Evaluation Criteria

- **Functionality** (40%) - All features work
- **Code Quality** (30%) - Clean, maintainable code
- **Error Handling** (15%) - Graceful failures
- **Documentation** (15%) - Clear usage guide

## Time Budget

- Total: 2 hours
- Core features: 90 minutes
- Testing/docs: 30 minutes
