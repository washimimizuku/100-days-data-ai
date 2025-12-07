"""
Day 56: Production Data API - Pydantic Models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime


class Column(BaseModel):
    """Column schema definition"""
    name: str
    type: Literal["string", "integer", "float", "boolean", "date"]
    nullable: bool = False
    description: Optional[str] = None


class DatasetMetadata(BaseModel):
    """Dataset metadata"""
    id: str
    name: str
    description: str
    rows: int
    columns: int
    created_at: datetime
    updated_at: datetime


class QueryRequest(BaseModel):
    """Data query request"""
    columns: Optional[List[str]] = None
    filters: Optional[Dict[str, Dict[str, Any]]] = None
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "asc"
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class QueryResponse(BaseModel):
    """Data query response"""
    data: List[Dict[str, Any]]
    count: int
    total: int
    execution_time_ms: float
    has_more: bool


class ExportRequest(BaseModel):
    """Data export request"""
    format: Literal["csv", "json", "parquet"]
    columns: Optional[List[str]] = None
    filters: Optional[Dict[str, Dict[str, Any]]] = None


class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    error_code: str
    status_code: int
