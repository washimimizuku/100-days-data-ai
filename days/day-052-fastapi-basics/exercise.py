"""
Day 52: FastAPI Basics - Exercises

Practice building FastAPI endpoints with routes, parameters, and models.
"""
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional


# Exercise 1: Basic Routes
def exercise_1():
    """
    Exercise 1: Create Basic Routes
    
    Create a FastAPI app with:
    1. GET / - Returns welcome message
    2. GET /health - Returns health status
    3. POST /echo - Echoes back the request body
    
    TODO: Implement the routes
    """
    app = FastAPI()
    
    # TODO: Add routes here
    
    return app


# Exercise 2: Path Parameters
def exercise_2():
    """
    Exercise 2: Path Parameters
    
    Create routes with path parameters:
    1. GET /users/{user_id} - Get user by ID (int)
    2. GET /products/{product_id} - Get product by ID (str)
    3. GET /categories/{category}/items/{item_id} - Nested paths
    
    TODO: Implement routes with path parameters
    """
    app = FastAPI()
    
    # TODO: Add routes with path parameters
    
    return app


# Exercise 3: Query Parameters
def exercise_3():
    """
    Exercise 3: Query Parameters
    
    Create a product listing endpoint:
    GET /products
    
    Query parameters:
    - category: Optional[str] - Filter by category
    - min_price: Optional[float] - Minimum price
    - max_price: Optional[float] - Maximum price
    - skip: int = 0 - Pagination offset
    - limit: int = 10 - Items per page (max 100)
    
    TODO: Implement with query parameters
    """
    app = FastAPI()
    
    # TODO: Add route with query parameters
    
    return app


# Exercise 4: Request Models
def exercise_4():
    """
    Exercise 4: Pydantic Request Models
    
    Create models and endpoints:
    
    1. User model:
       - name: str (1-50 chars)
       - email: str
       - age: int (0-120)
    
    2. POST /users - Create user
    3. PUT /users/{user_id} - Update user
    
    TODO: Define models and implement routes
    """
    app = FastAPI()
    
    # TODO: Define Pydantic models
    # class User(BaseModel):
    #     ...
    
    # TODO: Add routes
    
    return app


# Exercise 5: CRUD Operations
def exercise_5():
    """
    Exercise 5: Complete CRUD API
    
    Build a complete CRUD API for books:
    
    Book model:
    - title: str
    - author: str
    - year: int
    - isbn: str
    
    Endpoints:
    - POST /books - Create book (201)
    - GET /books - List books with pagination
    - GET /books/{book_id} - Get specific book
    - PUT /books/{book_id} - Update book
    - DELETE /books/{book_id} - Delete book (204)
    
    Use in-memory storage (dict).
    Handle 404 errors appropriately.
    
    TODO: Implement complete CRUD
    """
    app = FastAPI()
    
    # TODO: Define Book model
    
    # TODO: In-memory storage
    # books = {}
    # next_id = 1
    
    # TODO: Implement all CRUD operations
    
    return app


# Exercise 6: Response Models
def exercise_6():
    """
    Exercise 6: Response Models
    
    Create user API with separate request/response models:
    
    UserCreate (request):
    - name: str
    - email: str
    - password: str
    
    UserResponse (response):
    - id: int
    - name: str
    - email: str
    (password excluded)
    
    POST /users - Create user
    GET /users/{user_id} - Get user
    
    TODO: Define models and routes with response_model
    """
    app = FastAPI()
    
    # TODO: Define UserCreate and UserResponse models
    
    # TODO: Implement routes with response_model
    
    return app


# Exercise 7: Error Handling
def exercise_7():
    """
    Exercise 7: Error Handling
    
    Create product API with proper error handling:
    
    GET /products/{product_id}
    - Return 404 if not found
    - Return 400 if ID is negative
    
    POST /products
    - Return 409 if product already exists (by name)
    
    TODO: Implement with HTTPException
    """
    app = FastAPI()
    
    # TODO: Implement with error handling
    
    return app


# Exercise 8: Data Query API
def exercise_8():
    """
    Exercise 8: Data Query API
    
    Build a data query endpoint:
    
    POST /query
    
    Request:
    - dataset: str
    - filters: Optional[dict]
    - columns: Optional[List[str]]
    - limit: int = 100
    
    Response:
    - data: List[dict]
    - count: int
    - execution_time_ms: float
    
    Simulate querying a dataset and return mock data.
    
    TODO: Implement data query API
    """
    app = FastAPI()
    
    # TODO: Define request/response models
    
    # TODO: Implement query endpoint
    
    return app


if __name__ == "__main__":
    print("Day 52: FastAPI Basics - Exercises\n")
    print("Run with: uvicorn exercise:app --reload")
    print("Replace 'app' with exercise function name")
    print("\nExample:")
    print("  app = exercise_1()")
    print("  uvicorn exercise:app --reload")
