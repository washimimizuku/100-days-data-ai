# Day 51: REST API Principles

## 📖 Learning Objectives

By the end of this session, you will:
- Understand REST architectural constraints and principles
- Master HTTP methods and their proper usage
- Learn API design best practices for data services
- Design resource-oriented APIs
- Handle API versioning and documentation

**Time**: 1 hour  
**Level**: Intermediate

---

## What is REST?

**REST** (Representational State Transfer) is an architectural style for designing networked applications. It uses HTTP protocol and treats server objects as resources that can be created, read, updated, or deleted.

### Key Principles

1. **Client-Server**: Separation of concerns
2. **Stateless**: Each request contains all needed information
3. **Cacheable**: Responses must define themselves as cacheable or not
4. **Uniform Interface**: Consistent way to interact with resources
5. **Layered System**: Client can't tell if connected directly to server
6. **Code on Demand** (optional): Server can extend client functionality

---

## HTTP Methods

### CRUD Operations

| Method | Operation | Idempotent | Safe |
|--------|-----------|------------|------|
| GET | Read | Yes | Yes |
| POST | Create | No | No |
| PUT | Update/Replace | Yes | No |
| PATCH | Partial Update | No | No |
| DELETE | Delete | Yes | No |

**Idempotent**: Multiple identical requests have same effect as single request  
**Safe**: Does not modify resource state

### Examples

```python
# GET - Retrieve resources
GET /api/users          # List all users
GET /api/users/123      # Get specific user

# POST - Create new resource
POST /api/users
{
  "name": "Alice",
  "email": "alice@example.com"
}

# PUT - Replace entire resource
PUT /api/users/123
{
  "name": "Alice Smith",
  "email": "alice@example.com",
  "role": "admin"
}

# PATCH - Update specific fields
PATCH /api/users/123
{
  "role": "admin"
}

# DELETE - Remove resource
DELETE /api/users/123
```

---

## HTTP Status Codes

**Success (2xx)**: 200 OK, 201 Created, 204 No Content

**Client Errors (4xx)**: 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 422 Unprocessable Entity

**Server Errors (5xx)**: 500 Internal Server Error, 503 Service Unavailable

```python
# Example responses
{"status": 200, "data": {"id": 123, "name": "Alice"}}
{"status": 404, "error": "User not found", "message": "No user exists with id 123"}
```

---

## Resource Design

### URL Structure

**Good**:
```
GET /api/users                    # Collection
GET /api/users/123                # Specific resource
GET /api/users/123/orders         # Nested resource
GET /api/orders?user_id=123       # Query parameter
```

**Bad**:
```
GET /api/getUsers                 # Verb in URL
GET /api/user/123                 # Inconsistent plural/singular
GET /api/users/123/getOrders      # Verb in nested resource
```

### Naming Conventions

1. **Use nouns, not verbs**: `/users` not `/getUsers`
2. **Use plural for collections**: `/users` not `/user`
3. **Use hyphens for readability**: `/order-items` not `/orderItems`
4. **Lowercase only**: `/users` not `/Users`
5. **No trailing slashes**: `/users` not `/users/`

---

## Query Parameters

### Filtering

```python
GET /api/products?category=electronics
GET /api/products?price_min=100&price_max=500
GET /api/users?role=admin&status=active
```

### Sorting

```python
GET /api/products?sort=price          # Ascending
GET /api/products?sort=-price         # Descending
GET /api/products?sort=category,price # Multiple fields
```

### Pagination

```python
# Offset-based
GET /api/products?limit=20&offset=40

# Cursor-based
GET /api/products?limit=20&cursor=abc123

# Page-based
GET /api/products?page=3&per_page=20
```

### Field Selection

```python
GET /api/users?fields=id,name,email
GET /api/products?fields=id,name,price
```

---

## Request/Response Format

### Request Headers

```http
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token>
```

### Response Structure

```python
# Success response
{
  "status": "success",
  "data": {
    "id": 123,
    "name": "Product Name",
    "price": 99.99
  },
  "meta": {
    "timestamp": "2024-01-01T10:00:00Z"
  }
}

# List response with pagination
{
  "status": "success",
  "data": [
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"}
  ],
  "meta": {
    "total": 100,
    "page": 1,
    "per_page": 20,
    "total_pages": 5
  }
}

# Error response
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {"field": "email", "message": "Invalid email format"}
    ]
  }
}
```

---

## API Versioning

```python
# URL versioning (recommended)
GET /api/v1/users
GET /api/v2/users

# Header versioning
Accept: application/vnd.myapi.v1+json

# Query parameter versioning
GET /api/users?version=1
```

**Best Practice**: Use URL versioning for simplicity and clarity.

---

## Data API Design Patterns

### Pattern 1: Data Query API

```python
# Query datasets
GET /api/datasets
GET /api/datasets/{id}
GET /api/datasets/{id}/query?filter=...&limit=100

# Response
{
  "data": [...],
  "schema": {...},
  "row_count": 100,
  "execution_time_ms": 45
}
```

### Pattern 2: Analytics API

```python
# Get aggregated metrics
GET /api/metrics/sales?start_date=2024-01-01&end_date=2024-01-31
GET /api/metrics/users/active?period=daily

# Response
{
  "metric": "sales",
  "period": "2024-01",
  "value": 125000.50,
  "change_percent": 15.3
}
```

### Pattern 3: ML Model API

```python
# Predict endpoint
POST /api/models/predict
{
  "model_id": "churn-v2",
  "features": {...}
}

# Response
{
  "prediction": 0.85,
  "confidence": 0.92,
  "model_version": "2.1.0"
}
```

---

## Error Handling & Rate Limiting

```python
# Consistent error format
{
  "status": "error",
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "User with id 123 not found",
    "timestamp": "2024-01-01T10:00:00Z",
    "path": "/api/users/123"
  }
}

# Application-specific error codes
USER_NOT_FOUND, INVALID_CREDENTIALS, VALIDATION_ERROR, RATE_LIMIT_EXCEEDED

# Rate limiting headers
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200

# 429 Too Many Requests response
{
  "status": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "retry_after": 60
  }
}
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Design User API
Design RESTful endpoints for user management (CRUD operations).

### Exercise 2: Design Data Query API
Create endpoints for querying datasets with filtering and pagination.

### Exercise 3: HTTP Status Codes
Choose appropriate status codes for various scenarios.

### Exercise 4: Error Responses
Design consistent error response structures.

### Exercise 5: API Versioning
Implement URL-based versioning strategy.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- REST uses HTTP methods to perform CRUD operations on resources
- Use nouns for resources, not verbs in URLs
- HTTP status codes communicate operation results
- Design consistent request/response formats
- Implement pagination for large datasets
- Version APIs to manage breaking changes
- Provide clear error messages with error codes
- Use query parameters for filtering, sorting, pagination

---

## 📚 Resources

- [REST API Tutorial](https://restfulapi.net/)
- [HTTP Status Codes](https://httpstatuses.com/)
- [API Design Best Practices](https://swagger.io/resources/articles/best-practices-in-api-design/)
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines)

---

## Tomorrow: Day 52 - FastAPI Basics

Learn FastAPI framework for building high-performance data APIs.
