"""
Day 53: FastAPI Async/Await - Exercises

Practice asynchronous programming patterns in FastAPI.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
import asyncio
from typing import List


# Exercise 1: Basic Async
def exercise_1():
    """
    Exercise 1: Convert to Async
    
    Convert these sync endpoints to async:
    
    1. GET /users - Simulates 1 second database query
    2. GET /products - Simulates 0.5 second database query
    3. GET /orders - Simulates 0.8 second database query
    
    Use asyncio.sleep() to simulate delays.
    
    TODO: Implement async endpoints
    """
    app = FastAPI()
    
    # TODO: Convert to async
    # @app.get("/users")
    # async def get_users():
    #     ...
    
    return app


# Exercise 2: Concurrent Operations
def exercise_2():
    """
    Exercise 2: Concurrent Fetching
    
    Create endpoint GET /dashboard that fetches:
    - User data (1 second)
    - Orders data (1 second)
    - Analytics data (1 second)
    
    Fetch all concurrently using asyncio.gather().
    Should complete in ~1 second, not 3 seconds.
    
    TODO: Implement concurrent fetching
    """
    app = FastAPI()
    
    # TODO: Define async helper functions
    # async def fetch_user():
    #     await asyncio.sleep(1)
    #     return {"user": "Alice"}
    
    # TODO: Implement dashboard endpoint
    
    return app


# Exercise 3: Async Database Simulation
def exercise_3():
    """
    Exercise 3: Async Database Operations
    
    Simulate async database with in-memory dict:
    
    - POST /items - Create item
    - GET /items - List all items
    - GET /items/{item_id} - Get specific item
    - PUT /items/{item_id} - Update item
    - DELETE /items/{item_id} - Delete item
    
    Add 0.1 second delay to each operation.
    
    TODO: Implement async CRUD
    """
    app = FastAPI()
    
    # TODO: In-memory storage
    # items = {}
    
    # TODO: Implement async CRUD operations
    
    return app


# Exercise 4: Error Handling with Timeout
def exercise_4():
    """
    Exercise 4: Timeout and Error Handling
    
    Create GET /external endpoint that:
    1. Simulates fetching from external API (random 1-5 seconds)
    2. Has 2 second timeout
    3. Returns 504 if timeout
    4. Returns 500 for other errors
    
    TODO: Implement with timeout and error handling
    """
    app = FastAPI()
    
    # TODO: Implement with asyncio.wait_for()
    
    return app


# Exercise 5: Background Tasks
def exercise_5():
    """
    Exercise 5: Background Tasks
    
    Create POST /process endpoint that:
    1. Accepts data in request body
    2. Returns immediate response
    3. Processes data in background (3 seconds)
    4. Logs completion
    
    Use FastAPI BackgroundTasks.
    
    TODO: Implement with background tasks
    """
    app = FastAPI()
    
    # TODO: Define background task function
    # async def process_data(data: dict):
    #     await asyncio.sleep(3)
    #     print(f"Processed: {data}")
    
    # TODO: Implement endpoint with background_tasks
    
    return app


# Exercise 6: Multiple Concurrent Requests
def exercise_6():
    """
    Exercise 6: Fetch Multiple APIs
    
    Create GET /aggregate endpoint that:
    1. Fetches from 5 different "APIs" concurrently
    2. Each API takes 1 second
    3. Returns aggregated results
    4. Should complete in ~1 second
    
    TODO: Implement concurrent API fetching
    """
    app = FastAPI()
    
    # TODO: Define async API fetch functions
    
    # TODO: Implement aggregate endpoint
    
    return app


# Exercise 7: Async Generator
def exercise_7():
    """
    Exercise 7: Async Data Stream
    
    Create GET /stream endpoint that:
    1. Generates data items asynchronously
    2. Yields items one by one
    3. Simulates streaming data
    
    Use async generator pattern.
    
    TODO: Implement async generator
    """
    app = FastAPI()
    
    # TODO: Define async generator
    # async def generate_data():
    #     for i in range(10):
    #         await asyncio.sleep(0.1)
    #         yield {"item": i}
    
    # TODO: Implement streaming endpoint
    
    return app


# Exercise 8: Async Context Manager
def exercise_8():
    """
    Exercise 8: Async Resource Management
    
    Create async context manager for database connection:
    1. Simulates opening connection (0.5 seconds)
    2. Provides connection object
    3. Simulates closing connection (0.3 seconds)
    
    Use in GET /query endpoint.
    
    TODO: Implement async context manager
    """
    app = FastAPI()
    
    # TODO: Define async context manager class
    # class AsyncDB:
    #     async def __aenter__(self):
    #         ...
    #     async def __aexit__(self, exc_type, exc, tb):
    #         ...
    
    # TODO: Use in endpoint
    
    return app


if __name__ == "__main__":
    print("Day 53: FastAPI Async/Await - Exercises\n")
    print("Run with: uvicorn exercise:app --reload")
    print("\nExample:")
    print("  app = exercise_1()")
    print("  uvicorn exercise:app --reload")
