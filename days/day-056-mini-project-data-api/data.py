"""
Day 56: Production Data API - Data Layer
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime


# Sample datasets
DATASETS = {
    "sales": pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "product": ["laptop", "mouse", "keyboard", "monitor", "laptop"],
        "amount": [999.99, 29.99, 79.99, 299.99, 899.99],
        "status": ["completed", "completed", "pending", "completed", "completed"]
    }),
    "users": pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
        "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "status": ["active", "active", "inactive"]
    }),
    "products": pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Laptop", "Mouse", "Keyboard", "Monitor"],
        "category": ["electronics", "electronics", "electronics", "electronics"],
        "price": [999.99, 29.99, 79.99, 299.99],
        "stock": [50, 200, 150, 75]
    })
}


def get_dataset_metadata(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Get dataset metadata"""
    if dataset_id not in DATASETS:
        return None
    
    df = DATASETS[dataset_id]
    return {
        "id": dataset_id,
        "name": dataset_id.title(),
        "description": f"{dataset_id.title()} dataset",
        "rows": len(df),
        "columns": len(df.columns),
        "created_at": datetime(2024, 1, 1),
        "updated_at": datetime.now()
    }


def get_dataset_schema(dataset_id: str) -> Optional[List[Dict[str, Any]]]:
    """Get dataset schema"""
    if dataset_id not in DATASETS:
        return None
    
    df = DATASETS[dataset_id]
    schema = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "int" in dtype:
            col_type = "integer"
        elif "float" in dtype:
            col_type = "float"
        else:
            col_type = "string"
        
        schema.append({
            "name": col,
            "type": col_type,
            "nullable": df[col].isna().any()
        })
    
    return schema


def query_dataset(
    dataset_id: str,
    columns: Optional[List[str]] = None,
    filters: Optional[Dict[str, Dict[str, Any]]] = None,
    sort_by: Optional[str] = None,
    sort_order: str = "asc",
    limit: int = 100,
    offset: int = 0
) -> Optional[Dict[str, Any]]:
    """Query dataset with filters and pagination"""
    if dataset_id not in DATASETS:
        return None
    
    df = DATASETS[dataset_id].copy()
    
    # Apply filters
    if filters:
        for col, conditions in filters.items():
            if col not in df.columns:
                continue
            
            for op, value in conditions.items():
                if op == "eq":
                    df = df[df[col] == value]
                elif op == "ne":
                    df = df[df[col] != value]
                elif op == "gt":
                    df = df[df[col] > value]
                elif op == "lt":
                    df = df[df[col] < value]
                elif op == "gte":
                    df = df[df[col] >= value]
                elif op == "lte":
                    df = df[df[col] <= value]
                elif op == "in":
                    df = df[df[col].isin(value)]
    
    # Select columns
    if columns:
        valid_cols = [c for c in columns if c in df.columns]
        if valid_cols:
            df = df[valid_cols]
    
    # Sort
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=(sort_order == "asc"))
    
    # Get total before pagination
    total = len(df)
    
    # Paginate
    df = df.iloc[offset:offset + limit]
    
    return {
        "data": df.to_dict(orient="records"),
        "count": len(df),
        "total": total,
        "has_more": (offset + limit) < total
    }


def list_datasets() -> List[Dict[str, Any]]:
    """List all available datasets"""
    return [get_dataset_metadata(ds_id) for ds_id in DATASETS.keys()]
