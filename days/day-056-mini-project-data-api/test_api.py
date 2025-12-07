"""
Day 56: Production Data API - Tests
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# Test root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# Test list datasets
def test_list_datasets():
    response = client.get("/datasets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]


# Test get dataset
def test_get_dataset():
    response = client.get("/datasets/sales")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "sales"
    assert "rows" in data
    assert "columns" in data


def test_get_nonexistent_dataset():
    response = client.get("/datasets/nonexistent")
    assert response.status_code == 404


# Test get schema
def test_get_schema():
    response = client.get("/datasets/sales/schema")
    assert response.status_code == 200
    schema = response.json()
    assert isinstance(schema, list)
    assert len(schema) > 0
    assert "name" in schema[0]
    assert "type" in schema[0]


# Test query dataset
def test_query_dataset():
    response = client.post("/datasets/sales/query", json={
        "limit": 10,
        "offset": 0
    })
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "count" in data
    assert "total" in data
    assert isinstance(data["data"], list)


def test_query_with_filters():
    response = client.post("/datasets/sales/query", json={
        "filters": {
            "amount": {"gt": 100}
        },
        "limit": 10
    })
    assert response.status_code == 200
    data = response.json()
    assert data["count"] <= 10
    for row in data["data"]:
        assert row["amount"] > 100


def test_query_with_columns():
    response = client.post("/datasets/sales/query", json={
        "columns": ["id", "amount"],
        "limit": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) <= 5
    if data["data"]:
        assert set(data["data"][0].keys()) == {"id", "amount"}


def test_query_with_sorting():
    response = client.post("/datasets/sales/query", json={
        "sort_by": "amount",
        "sort_order": "desc",
        "limit": 5
    })
    assert response.status_code == 200
    data = response.json()
    amounts = [row["amount"] for row in data["data"]]
    assert amounts == sorted(amounts, reverse=True)


def test_query_with_pagination():
    # First page
    response1 = client.post("/datasets/sales/query", json={
        "limit": 2,
        "offset": 0
    })
    assert response1.status_code == 200
    data1 = response1.json()
    
    # Second page
    response2 = client.post("/datasets/sales/query", json={
        "limit": 2,
        "offset": 2
    })
    assert response2.status_code == 200
    data2 = response2.json()
    
    # Ensure different data
    if data1["data"] and data2["data"]:
        assert data1["data"][0]["id"] != data2["data"][0]["id"]


def test_query_invalid_limit():
    response = client.post("/datasets/sales/query", json={
        "limit": 10000  # Exceeds max
    })
    assert response.status_code == 422


# Test export
def test_export_dataset():
    response = client.post("/datasets/sales/export", json={
        "format": "csv"
    })
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["format"] == "csv"


def test_export_with_filters():
    response = client.post("/datasets/sales/export", json={
        "format": "json",
        "filters": {
            "status": {"eq": "completed"}
        }
    })
    assert response.status_code == 200


# Test health check
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Integration test
def test_full_workflow():
    # List datasets
    datasets = client.get("/datasets").json()
    assert len(datasets) > 0
    
    # Get first dataset
    dataset_id = datasets[0]["id"]
    
    # Get metadata
    metadata = client.get(f"/datasets/{dataset_id}").json()
    assert metadata["id"] == dataset_id
    
    # Get schema
    schema = client.get(f"/datasets/{dataset_id}/schema").json()
    assert len(schema) > 0
    
    # Query data
    query_result = client.post(f"/datasets/{dataset_id}/query", json={
        "limit": 5
    }).json()
    assert query_result["count"] <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=main", "--cov=data", "--cov=models"])
