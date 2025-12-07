# Day 52: FastAPI Basics

## 📖 Learning Objectives

By the end of this session, you will:
- Set up FastAPI for building data APIs
- Create routes with path and query parameters
- Define request and response models with Pydantic
- Handle different HTTP methods
- Implement basic error handling
- Use automatic API documentation

**Time**: 1 hour  
**Level**: Intermediate

---

## What is FastAPI?

**FastAPI** is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.

**Key Features**: High performance, fast development, fewer bugs, intuitive, easy to learn, standards-based (OpenAPI/JSON Schema), automatic interactive documentation

---

## Installation

```bash
pip install fastapi uvicorn[standard]
```

**FastAPI**: The framework  
**Uvicorn**: ASGI server to run the application

---

## Hello World

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Run with: uvicorn main:app --reload
```

Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

---

## Path Parameters

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id, "name": "Alice"}

@app.get("/products/{product_id}")
def get_product(product_id: str):
    return {"product_id": product_id}
```

**Type validation**: FastAPI validates types automatically.

```python
# Valid: GET /users/123
# Invalid: GET /users/abc (returns 422 error)
```

---

## Query Parameters

```python
@app.get("/items")
def list_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# GET /items?skip=20&limit=50
```

### Optional Parameters

```python
from typing import Optional

@app.get("/search")
def search(q: Optional[str] = None, max_results: int = 10):
    if q:
        return {"query": q, "max_results": max_results}
    return {"message": "No query provided"}

# GET /search?q=python
# GET /search
```

---

## Request Body with Pydantic

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    email: str
    age: int

@app.post("/users")
def create_user(user: User):
    return {"message": "User created", "user": user}
```

**Request**:
```json
POST /users
{
  "name": "Alice",
  "email": "alice@example.com",
  "age": 30
}
```

**Response**:
```json
{
  "message": "User created",
  "user": {
    "name": "Alice",
    "email": "alice@example.com",
    "age": 30
  }
}
```

---

## Response Models

```python
from pydantic import BaseModel

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    # Internal data might have password, etc.
    user_data = {
        "id": user_id,
        "name": "Alice",
        "email": "alice@example.com",
        "password": "secret123"  # Won't be in response
    }
    return user_data
```

**Response** (password excluded):
```json
{
  "id": 1,
  "name": "Alice",
  "email": "alice@example.com"
}
```

---

## HTTP Methods

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

# Create
@app.post("/items")
def create_item(item: Item):
    return {"message": "Item created", "item": item}

# Read
@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id, "name": "Product"}

# Update
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"message": "Item updated", "item_id": item_id}

# Partial Update
@app.patch("/items/{item_id}")
def partial_update(item_id: int, name: str = None, price: float = None):
    updates = {}
    if name: updates["name"] = name
    if price: updates["price"] = price
    return {"item_id": item_id, "updates": updates}

# Delete
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    return {"message": "Item deleted", "item_id": item_id}
```

---

## Status Codes

```python
from fastapi import FastAPI, status

@app.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(user: User):
    return user

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int):
    return None
```

**Common codes**: 200 OK, 201 Created, 204 No Content, 400 Bad Request, 404 Not Found

---

## Error Handling

```python
from fastapi import FastAPI, HTTPException

@app.get("/users/{user_id}")
def get_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    return users_db[user_id]
```

**Custom error response**:
```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_ID",
                "message": "Item ID must be positive",
                "item_id": item_id
            }
        )
    return {"item_id": item_id}
```

---

## Data API Example

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Data API", version="1.0.0")

class DataQuery(BaseModel):
    dataset: str
    filters: Optional[dict] = None
    limit: int = 100

class DataResponse(BaseModel):
    data: List[dict]
    count: int
    execution_time_ms: float

@app.post("/query", response_model=DataResponse)
def query_data(query: DataQuery):
    # Simulate data query
    data = [
        {"id": 1, "value": 100},
        {"id": 2, "value": 200}
    ]
    return {
        "data": data,
        "count": len(data),
        "execution_time_ms": 45.2
    }

@app.get("/datasets")
def list_datasets(
    category: Optional[str] = None,
    limit: int = Query(default=10, le=100)
):
    datasets = [
        {"id": 1, "name": "sales", "category": "business"},
        {"id": 2, "name": "users", "category": "analytics"}
    ]
    return {"datasets": datasets, "count": len(datasets)}
```

---

## Validation with Pydantic

```python
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=0, le=120)

@app.post("/users")
def create_user(user: User):
    return user
```

**Constraints**: min_length, max_length, ge (≥), le (≤), gt (>), lt (<), regex

---



## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Basic Routes
Create GET and POST endpoints for a simple API.

### Exercise 2: Path Parameters
Implement routes with path parameters and type validation.

### Exercise 3: Query Parameters
Add filtering and pagination with query parameters.

### Exercise 4: Request Models
Define Pydantic models for request validation.

### Exercise 5: CRUD Operations
Build complete CRUD API for a resource.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- FastAPI uses Python type hints for automatic validation
- Pydantic models define request/response schemas
- Path parameters: `/users/{user_id}`
- Query parameters: `/items?skip=0&limit=10`
- Use `response_model` to control response structure
- HTTPException for error handling
- Automatic interactive documentation at `/docs`
- Status codes communicate operation results

---

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

---

## Tomorrow: Day 53 - FastAPI Async/Await

Learn asynchronous programming patterns for high-performance APIs.
