# Day 53: FastAPI Async/Await

## 📖 Learning Objectives

By the end of this session, you will:
- Understand asynchronous programming concepts
- Use async/await syntax in FastAPI
- Handle concurrent requests efficiently
- Implement async database operations
- Optimize API performance with async patterns
- Know when to use async vs sync

**Time**: 1 hour  
**Level**: Intermediate

---

## Why Async?

```python
# Synchronous: 10 requests = 20 seconds (blocks)
def get_data():
    result = slow_database_query()  # Blocks for 2 seconds
    return result

# Asynchronous: 10 concurrent requests = ~2 seconds (non-blocking)
async def get_data():
    result = await slow_database_query()  # Doesn't block
    return result
```

**Benefits**: Higher throughput, better resource usage, improved responsiveness, scalability

---

## Async/Await Syntax

```python
import asyncio
from fastapi import FastAPI

app = FastAPI()

# Sync (blocks entire server)
@app.get("/sync")
def sync_endpoint():
    time.sleep(2)
    return {"message": "Done"}

# Async (non-blocking - allows other requests during wait)
@app.get("/async")
async def async_endpoint():
    await asyncio.sleep(2)
    return {"message": "Done"}

# Define and call async functions
async def fetch_data():
    await asyncio.sleep(1)
    return {"data": "value"}

async def main():
    result = await fetch_data()  # Use await inside async functions only
    print(result)

asyncio.run(main())  # Run async function
```

---

## Async in FastAPI

```python
# Simple async endpoint
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    await asyncio.sleep(0.1)  # Simulate async database query
    return {"user_id": user_id, "name": "Alice"}

# Multiple concurrent operations with asyncio.gather()
@app.get("/dashboard")
async def get_dashboard():
    user, orders, stats = await asyncio.gather(
        fetch_user(),
        fetch_orders(),
        fetch_stats()
    )
    return {"user": user, "orders": orders, "stats": stats}
```

---

## Async Database Operations

```python
# PostgreSQL with asyncpg
import asyncpg

async def get_db_pool():
    return await asyncpg.create_pool("postgresql://user:pass@localhost/db")

@app.get("/users")
async def list_users():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM users LIMIT 10")
        return [dict(row) for row in rows]

# MongoDB with Motor
from motor.motor_asyncio import AsyncIOMotorClient

client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client.mydb

@app.get("/products")
async def list_products():
    products = []
    async for product in db.products.find().limit(10):
        products.append(product)
    return products
```

---

## Concurrent Operations

```python
# Sequential (slow): 2 seconds total
@app.get("/slow")
async def slow_endpoint():
    user = await fetch_user()      # 1 second
    orders = await fetch_orders()  # 1 second
    return {"user": user, "orders": orders}

# Concurrent (fast): 1 second total
@app.get("/fast")
async def fast_endpoint():
    user, orders = await asyncio.gather(fetch_user(), fetch_orders())
    return {"user": user, "orders": orders}
```

---

## Error Handling

```python
# Basic error handling
@app.get("/data")
async def get_data():
    try:
        result = await fetch_from_api()
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# With timeout (5 seconds max)
@app.get("/data-timeout")
async def get_data_timeout():
    try:
        result = await asyncio.wait_for(fetch_from_api(), timeout=5.0)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout")
```

---

## Background Tasks

```python
from fastapi import BackgroundTasks

async def send_email(email: str, message: str):
    await asyncio.sleep(2)  # Simulate sending
    print(f"Email sent to {email}")

@app.post("/register")
async def register_user(
    email: str,
    background_tasks: BackgroundTasks
):
    # Add task to run after response
    background_tasks.add_task(send_email, email, "Welcome!")
    return {"message": "User registered"}
```

---

## Async HTTP Requests

```python
import httpx

# Single request
@app.get("/external")
async def fetch_external():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

# Multiple concurrent requests
@app.get("/multiple")
async def fetch_multiple():
    async with httpx.AsyncClient() as client:
        responses = await asyncio.gather(
            client.get("https://api1.example.com/data"),
            client.get("https://api2.example.com/data"),
            client.get("https://api3.example.com/data")
        )
        return [r.json() for r in responses]
```

---

## When to Use Async

**Use Async For**: I/O-bound operations (database, API calls, files), high concurrency, long-running operations

**Use Sync For**: CPU-bound operations (heavy computations), simple operations, blocking libraries

```python
# ✅ Good: I/O-bound
@app.get("/users")
async def get_users():
    return await db.fetch_users()

# ✅ Good: CPU-bound (use sync)
@app.get("/compute")
def compute():
    return heavy_calculation()
```

---

## Async Context Managers & Performance

```python
# Async context manager
class AsyncDatabase:
    async def __aenter__(self):
        self.conn = await connect_to_db()
        return self.conn
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.conn.close()

@app.get("/data")
async def get_data():
    async with AsyncDatabase() as db:
        result = await db.query("SELECT * FROM users")
        return result

# Performance comparison
# Sync: 10 seconds for 10 requests
@app.get("/sync-slow")
def sync_slow():
    return [time.sleep(1) or "done" for _ in range(10)]

# Async: ~1 second for 10 concurrent requests
@app.get("/async-fast")
async def async_fast():
    tasks = [asyncio.sleep(1) for _ in range(10)]
    await asyncio.gather(*tasks)
    return ["done"] * 10
```

---

## Data API Example

```python
# Concurrent data fetching for dataset info
async def fetch_dataset(dataset_id: str):
    await asyncio.sleep(0.5)
    return {"id": dataset_id, "rows": 1000}

async def fetch_schema(dataset_id: str):
    await asyncio.sleep(0.3)
    return {"columns": ["id", "name", "value"]}

async def fetch_stats(dataset_id: str):
    await asyncio.sleep(0.4)
    return {"mean": 42.5, "count": 1000}

@app.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    dataset, schema, stats = await asyncio.gather(
        fetch_dataset(dataset_id),
        fetch_schema(dataset_id),
        fetch_stats(dataset_id)
    )
    return {"dataset": dataset, "schema": schema, "stats": stats}
```

---

## Best Practices & Common Pitfalls

**Best Practices**:
1. Use async for I/O (database, API calls, files)
2. Don't mix sync/async in same function
3. Use asyncio.gather() for concurrent operations
4. Set timeouts to prevent hanging
5. Use async libraries (httpx, asyncpg, motor)

**Common Pitfalls**:
```python
# ❌ Forgetting await
result = fetch_data()  # Returns coroutine, not data!

# ✅ Using await
result = await fetch_data()

# ❌ Sequential when could be concurrent
a = await fetch_a()
b = await fetch_b()

# ✅ Concurrent
a, b = await asyncio.gather(fetch_a(), fetch_b())
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Basic Async
Convert sync endpoints to async.

### Exercise 2: Concurrent Operations
Fetch multiple resources concurrently.

### Exercise 3: Async Database
Implement async database operations.

### Exercise 4: Error Handling
Handle timeouts and errors in async code.

### Exercise 5: Background Tasks
Use background tasks for async operations.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Async is for I/O-bound operations, not CPU-bound
- Use `async def` and `await` for async functions
- `asyncio.gather()` runs operations concurrently
- Async improves throughput for concurrent requests
- Use async libraries (httpx, asyncpg, motor)
- Set timeouts to prevent hanging
- Background tasks run after response is sent
- Profile to verify performance improvements

---

## 📚 Resources

- [FastAPI Async](https://fastapi.tiangolo.com/async/)
- [Python asyncio](https://docs.python.org/3/library/asyncio.html)
- [httpx Documentation](https://www.python-httpx.org/)
- [asyncpg Documentation](https://magicstack.github.io/asyncpg/)

---

## Tomorrow: Day 54 - FastAPI Pydantic Validation

Learn advanced validation patterns with Pydantic models.
