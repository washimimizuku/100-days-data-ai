"""
Day 53: FastAPI Async/Await - Solutions
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
import random
from typing import List


# Exercise 1: Basic Async
def exercise_1():
    """Convert sync to async"""
    app = FastAPI(title="Exercise 1: Basic Async")
    
    @app.get("/users")
    async def get_users():
        await asyncio.sleep(1)
        return {"users": ["Alice", "Bob", "Charlie"]}
    
    @app.get("/products")
    async def get_products():
        await asyncio.sleep(0.5)
        return {"products": ["Laptop", "Mouse", "Keyboard"]}
    
    @app.get("/orders")
    async def get_orders():
        await asyncio.sleep(0.8)
        return {"orders": [{"id": 1, "total": 99.99}]}
    
    return app


# Exercise 2: Concurrent Operations
def exercise_2():
    """Concurrent fetching"""
    app = FastAPI(title="Exercise 2: Concurrent Operations")
    
    async def fetch_user():
        await asyncio.sleep(1)
        return {"user": "Alice", "email": "alice@example.com"}
    
    async def fetch_orders():
        await asyncio.sleep(1)
        return {"orders": [{"id": 1}, {"id": 2}]}
    
    async def fetch_analytics():
        await asyncio.sleep(1)
        return {"views": 1000, "sales": 50}
    
    @app.get("/dashboard")
    async def get_dashboard():
        # Fetch all concurrently
        user, orders, analytics = await asyncio.gather(
            fetch_user(),
            fetch_orders(),
            fetch_analytics()
        )
        
        return {
            "user": user,
            "orders": orders,
            "analytics": analytics
        }
    
    return app


# Exercise 3: Async Database Simulation
def exercise_3():
    """Async CRUD operations"""
    app = FastAPI(title="Exercise 3: Async Database")
    
    class Item(BaseModel):
        name: str
        price: float
    
    items = {}
    next_id = 1
    
    @app.post("/items")
    async def create_item(item: Item):
        nonlocal next_id
        await asyncio.sleep(0.1)
        item_id = next_id
        next_id += 1
        items[item_id] = item.dict()
        return {"id": item_id, **item.dict()}
    
    @app.get("/items")
    async def list_items():
        await asyncio.sleep(0.1)
        return {"items": [{"id": k, **v} for k, v in items.items()]}
    
    @app.get("/items/{item_id}")
    async def get_item(item_id: int):
        await asyncio.sleep(0.1)
        if item_id not in items:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"id": item_id, **items[item_id]}
    
    @app.put("/items/{item_id}")
    async def update_item(item_id: int, item: Item):
        await asyncio.sleep(0.1)
        if item_id not in items:
            raise HTTPException(status_code=404, detail="Item not found")
        items[item_id] = item.dict()
        return {"id": item_id, **item.dict()}
    
    @app.delete("/items/{item_id}")
    async def delete_item(item_id: int):
        await asyncio.sleep(0.1)
        if item_id not in items:
            raise HTTPException(status_code=404, detail="Item not found")
        del items[item_id]
        return {"message": "Item deleted"}
    
    return app


# Exercise 4: Error Handling with Timeout
def exercise_4():
    """Timeout and error handling"""
    app = FastAPI(title="Exercise 4: Timeout Handling")
    
    async def fetch_external_api():
        # Random delay between 1-5 seconds
        delay = random.uniform(1, 5)
        await asyncio.sleep(delay)
        return {"data": "success", "delay": delay}
    
    @app.get("/external")
    async def get_external():
        try:
            result = await asyncio.wait_for(
                fetch_external_api(),
                timeout=2.0
            )
            return result
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Request timeout after 2 seconds"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: {str(e)}"
            )
    
    return app


# Exercise 5: Background Tasks
def exercise_5():
    """Background tasks"""
    app = FastAPI(title="Exercise 5: Background Tasks")
    
    async def process_data(data: dict):
        print(f"Starting background processing: {data}")
        await asyncio.sleep(3)
        print(f"Completed processing: {data}")
    
    @app.post("/process")
    async def process_endpoint(
        data: dict,
        background_tasks: BackgroundTasks
    ):
        background_tasks.add_task(process_data, data)
        return {"message": "Processing started", "data": data}
    
    return app


# Exercise 6: Multiple Concurrent Requests
def exercise_6():
    """Fetch multiple APIs concurrently"""
    app = FastAPI(title="Exercise 6: Multiple APIs")
    
    async def fetch_api_1():
        await asyncio.sleep(1)
        return {"source": "api1", "value": 100}
    
    async def fetch_api_2():
        await asyncio.sleep(1)
        return {"source": "api2", "value": 200}
    
    async def fetch_api_3():
        await asyncio.sleep(1)
        return {"source": "api3", "value": 300}
    
    async def fetch_api_4():
        await asyncio.sleep(1)
        return {"source": "api4", "value": 400}
    
    async def fetch_api_5():
        await asyncio.sleep(1)
        return {"source": "api5", "value": 500}
    
    @app.get("/aggregate")
    async def aggregate_data():
        results = await asyncio.gather(
            fetch_api_1(),
            fetch_api_2(),
            fetch_api_3(),
            fetch_api_4(),
            fetch_api_5()
        )
        
        total = sum(r["value"] for r in results)
        
        return {
            "results": results,
            "total": total,
            "count": len(results)
        }
    
    return app


# Exercise 7: Async Generator
def exercise_7():
    """Async data stream"""
    app = FastAPI(title="Exercise 7: Async Generator")
    
    async def generate_data():
        for i in range(10):
            await asyncio.sleep(0.1)
            yield {"item": i, "value": i * 10}
    
    @app.get("/stream")
    async def stream_data():
        items = []
        async for item in generate_data():
            items.append(item)
        return {"items": items}
    
    return app


# Exercise 8: Async Context Manager
def exercise_8():
    """Async resource management"""
    app = FastAPI(title="Exercise 8: Context Manager")
    
    class AsyncDB:
        async def __aenter__(self):
            print("Opening database connection...")
            await asyncio.sleep(0.5)
            self.connection = {"status": "connected"}
            return self.connection
        
        async def __aexit__(self, exc_type, exc, tb):
            print("Closing database connection...")
            await asyncio.sleep(0.3)
            self.connection = {"status": "closed"}
    
    @app.get("/query")
    async def query_database():
        async with AsyncDB() as db:
            await asyncio.sleep(0.2)
            result = {"data": [1, 2, 3], "connection": db["status"]}
            return result
    
    return app


# Default app
app = exercise_1()


if __name__ == "__main__":
    print("Day 53: FastAPI Async/Await - Solutions\n")
    print("To run an exercise:")
    print("1. Uncomment the desired exercise at the bottom")
    print("2. Run: uvicorn solution:app --reload")
    print("3. Visit: http://localhost:8000/docs")
    print("\nAvailable exercises:")
    print("  - exercise_1(): Basic async")
    print("  - exercise_2(): Concurrent operations")
    print("  - exercise_3(): Async database")
    print("  - exercise_4(): Timeout handling")
    print("  - exercise_5(): Background tasks")
    print("  - exercise_6(): Multiple APIs")
    print("  - exercise_7(): Async generator")
    print("  - exercise_8(): Context manager")
