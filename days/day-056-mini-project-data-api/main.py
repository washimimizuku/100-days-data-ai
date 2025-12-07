"""
Day 56: Production Data API - FastAPI Application
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import time
from models import (
    DatasetMetadata,
    Column,
    QueryRequest,
    QueryResponse,
    ExportRequest
)
from data import (
    list_datasets,
    get_dataset_metadata,
    get_dataset_schema,
    query_dataset
)

app = FastAPI(
    title="Production Data API",
    description="RESTful API for querying datasets",
    version="1.0.0"
)


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Production Data API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/datasets", response_model=list[DatasetMetadata])
async def get_datasets():
    """List all available datasets"""
    datasets = list_datasets()
    return datasets


@app.get("/datasets/{dataset_id}", response_model=DatasetMetadata)
async def get_dataset(dataset_id: str):
    """Get dataset metadata"""
    metadata = get_dataset_metadata(dataset_id)
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found"
        )
    return metadata


@app.get("/datasets/{dataset_id}/schema", response_model=list[Column])
async def get_schema(dataset_id: str):
    """Get dataset schema"""
    schema = get_dataset_schema(dataset_id)
    if schema is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found"
        )
    return schema


@app.post("/datasets/{dataset_id}/query", response_model=QueryResponse)
async def query_data(dataset_id: str, query: QueryRequest):
    """Query dataset with filters and pagination"""
    start_time = time.time()
    
    result = query_dataset(
        dataset_id=dataset_id,
        columns=query.columns,
        filters=query.filters,
        sort_by=query.sort_by,
        sort_order=query.sort_order,
        limit=query.limit,
        offset=query.offset
    )
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found"
        )
    
    execution_time = (time.time() - start_time) * 1000
    
    return QueryResponse(
        data=result["data"],
        count=result["count"],
        total=result["total"],
        execution_time_ms=round(execution_time, 2),
        has_more=result["has_more"]
    )


@app.post("/datasets/{dataset_id}/export")
async def export_data(dataset_id: str, export: ExportRequest):
    """Export dataset in specified format"""
    result = query_dataset(
        dataset_id=dataset_id,
        columns=export.columns,
        filters=export.filters,
        limit=10000
    )
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found"
        )
    
    return {
        "message": f"Export {export.format} prepared",
        "rows": result["count"],
        "format": export.format
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
