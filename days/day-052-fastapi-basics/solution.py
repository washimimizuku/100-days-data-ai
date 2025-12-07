"""
Day 52: FastAPI Basics - Solutions
"""
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
import time


# Exercise 1: Basic Routes
def exercise_1():
    """Create basic routes"""
    app = FastAPI(title="Exercise 1: Basic Routes")
    
    @app.get("/")
    def read_root():
        return {"message": "Welcome to the API"}
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy", "timestamp": time.time()}
    
    @app.post("/echo")
    def echo(data: dict):
        return {"echoed": data}
    
    return app


# Exercise 2: Path Parameters
def exercise_2():
    """Routes with path parameters"""
    app = FastAPI(title="Exercise 2: Path Parameters")
    
    @app.get("/users/{user_id}")
    def get_user(user_id: int):
        return {"user_id": user_id, "name": f"User {user_id}"}
    
    @app.get("/products/{product_id}")
    def get_product(product_id: str):
        return {"product_id": product_id, "name": f"Product {product_id}"}
    
    @app.get("/categories/{category}/items/{item_id}")
    def get_category_item(category: str, item_id: int):
        return {
            "category": category,
            "item_id": item_id,
            "name": f"{category} item {item_id}"
        }
    
    return app


# Exercise 3: Query Parameters
def exercise_3():
    """Query parameters with filtering and pagination"""
    app = FastAPI(title="Exercise 3: Query Parameters")
    
    # Mock products
    products = [
        {"id": 1, "name": "Laptop", "category": "electronics", "price": 999.99},
        {"id": 2, "name": "Mouse", "category": "electronics", "price": 29.99},
        {"id": 3, "name": "Desk", "category": "furniture", "price": 299.99},
        {"id": 4, "name": "Chair", "category": "furniture", "price": 199.99},
        {"id": 5, "name": "Monitor", "category": "electronics", "price": 399.99},
    ]
    
    @app.get("/products")
    def list_products(
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        skip: int = 0,
        limit: int = Query(default=10, le=100)
    ):
        filtered = products
        
        if category:
            filtered = [p for p in filtered if p["category"] == category]
        if min_price is not None:
            filtered = [p for p in filtered if p["price"] >= min_price]
        if max_price is not None:
            filtered = [p for p in filtered if p["price"] <= max_price]
        
        paginated = filtered[skip:skip + limit]
        
        return {
            "products": paginated,
            "total": len(filtered),
            "skip": skip,
            "limit": limit
        }
    
    return app


# Exercise 4: Request Models
def exercise_4():
    """Pydantic request models"""
    app = FastAPI(title="Exercise 4: Request Models")
    
    class User(BaseModel):
        name: str = Field(..., min_length=1, max_length=50)
        email: EmailStr
        age: int = Field(..., ge=0, le=120)
    
    users = {}
    next_id = 1
    
    @app.post("/users", status_code=status.HTTP_201_CREATED)
    def create_user(user: User):
        nonlocal next_id
        user_id = next_id
        next_id += 1
        users[user_id] = user.dict()
        return {"id": user_id, **user.dict()}
    
    @app.put("/users/{user_id}")
    def update_user(user_id: int, user: User):
        if user_id not in users:
            raise HTTPException(status_code=404, detail="User not found")
        users[user_id] = user.dict()
        return {"id": user_id, **user.dict()}
    
    return app


# Exercise 5: CRUD Operations
def exercise_5():
    """Complete CRUD API for books"""
    app = FastAPI(title="Exercise 5: CRUD Operations")
    
    class Book(BaseModel):
        title: str = Field(..., min_length=1)
        author: str = Field(..., min_length=1)
        year: int = Field(..., ge=1000, le=2100)
        isbn: str = Field(..., min_length=10, max_length=13)
    
    class BookResponse(BaseModel):
        id: int
        title: str
        author: str
        year: int
        isbn: str
    
    books = {}
    next_id = 1
    
    @app.post("/books", response_model=BookResponse, status_code=status.HTTP_201_CREATED)
    def create_book(book: Book):
        nonlocal next_id
        book_id = next_id
        next_id += 1
        books[book_id] = book.dict()
        return {"id": book_id, **book.dict()}
    
    @app.get("/books", response_model=List[BookResponse])
    def list_books(skip: int = 0, limit: int = 10):
        result = []
        for book_id, book in list(books.items())[skip:skip + limit]:
            result.append({"id": book_id, **book})
        return result
    
    @app.get("/books/{book_id}", response_model=BookResponse)
    def get_book(book_id: int):
        if book_id not in books:
            raise HTTPException(status_code=404, detail="Book not found")
        return {"id": book_id, **books[book_id]}
    
    @app.put("/books/{book_id}", response_model=BookResponse)
    def update_book(book_id: int, book: Book):
        if book_id not in books:
            raise HTTPException(status_code=404, detail="Book not found")
        books[book_id] = book.dict()
        return {"id": book_id, **book.dict()}
    
    @app.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_book(book_id: int):
        if book_id not in books:
            raise HTTPException(status_code=404, detail="Book not found")
        del books[book_id]
        return None
    
    return app


# Exercise 6: Response Models
def exercise_6():
    """Separate request/response models"""
    app = FastAPI(title="Exercise 6: Response Models")
    
    class UserCreate(BaseModel):
        name: str
        email: EmailStr
        password: str = Field(..., min_length=8)
    
    class UserResponse(BaseModel):
        id: int
        name: str
        email: str
    
    users = {}
    next_id = 1
    
    @app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    def create_user(user: UserCreate):
        nonlocal next_id
        user_id = next_id
        next_id += 1
        users[user_id] = user.dict()
        return {"id": user_id, **user.dict()}
    
    @app.get("/users/{user_id}", response_model=UserResponse)
    def get_user(user_id: int):
        if user_id not in users:
            raise HTTPException(status_code=404, detail="User not found")
        return {"id": user_id, **users[user_id]}
    
    return app


# Exercise 7: Error Handling
def exercise_7():
    """Error handling with HTTPException"""
    app = FastAPI(title="Exercise 7: Error Handling")
    
    class Product(BaseModel):
        name: str
        price: float
    
    products = {}
    next_id = 1
    
    @app.get("/products/{product_id}")
    def get_product(product_id: int):
        if product_id < 0:
            raise HTTPException(
                status_code=400,
                detail="Product ID must be non-negative"
            )
        if product_id not in products:
            raise HTTPException(
                status_code=404,
                detail=f"Product {product_id} not found"
            )
        return {"id": product_id, **products[product_id]}
    
    @app.post("/products", status_code=status.HTTP_201_CREATED)
    def create_product(product: Product):
        nonlocal next_id
        
        # Check for duplicate name
        for existing in products.values():
            if existing["name"] == product.name:
                raise HTTPException(
                    status_code=409,
                    detail=f"Product with name '{product.name}' already exists"
                )
        
        product_id = next_id
        next_id += 1
        products[product_id] = product.dict()
        return {"id": product_id, **product.dict()}
    
    return app


# Exercise 8: Data Query API
def exercise_8():
    """Data query API"""
    app = FastAPI(title="Exercise 8: Data Query API")
    
    class QueryRequest(BaseModel):
        dataset: str
        filters: Optional[dict] = None
        columns: Optional[List[str]] = None
        limit: int = Field(default=100, le=1000)
    
    class QueryResponse(BaseModel):
        data: List[dict]
        count: int
        execution_time_ms: float
    
    @app.post("/query", response_model=QueryResponse)
    def query_data(query: QueryRequest):
        start_time = time.time()
        
        # Mock data based on dataset
        mock_data = {
            "sales": [
                {"id": 1, "amount": 100.50, "date": "2024-01-01"},
                {"id": 2, "amount": 250.75, "date": "2024-01-02"},
                {"id": 3, "amount": 175.25, "date": "2024-01-03"},
            ],
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
            ]
        }
        
        data = mock_data.get(query.dataset, [])
        
        # Apply column selection
        if query.columns:
            data = [
                {k: v for k, v in row.items() if k in query.columns}
                for row in data
            ]
        
        # Apply limit
        data = data[:query.limit]
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "data": data,
            "count": len(data),
            "execution_time_ms": round(execution_time, 2)
        }
    
    return app


# Default app for testing
app = exercise_1()


if __name__ == "__main__":
    print("Day 52: FastAPI Basics - Solutions\n")
    print("To run an exercise:")
    print("1. Uncomment the desired exercise at the bottom")
    print("2. Run: uvicorn solution:app --reload")
    print("3. Visit: http://localhost:8000/docs")
    print("\nAvailable exercises:")
    print("  - exercise_1(): Basic routes")
    print("  - exercise_2(): Path parameters")
    print("  - exercise_3(): Query parameters")
    print("  - exercise_4(): Request models")
    print("  - exercise_5(): CRUD operations")
    print("  - exercise_6(): Response models")
    print("  - exercise_7(): Error handling")
    print("  - exercise_8(): Data query API")
