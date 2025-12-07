# Day 53: FastAPI Async/Await - Quiz

Test your understanding of asynchronous programming in FastAPI.

---

## Questions

### Question 1
What keyword is used to define an asynchronous function in Python?

A) async  
B) await  
C) asyncio  
D) concurrent

### Question 2
When should you use async/await in FastAPI?

A) For CPU-intensive computations  
B) For I/O-bound operations like database queries  
C) For all endpoints regardless of operation type  
D) Only when using external APIs

### Question 3
What does `await` do in an async function?

A) Blocks the entire server  
B) Pauses the function and allows other tasks to run  
C) Makes the function run faster  
D) Converts sync code to async

### Question 4
How do you run multiple async operations concurrently?

A) Call them sequentially with await  
B) Use asyncio.gather()  
C) Use threading  
D) Use multiprocessing

### Question 5
What happens if you forget to use `await` with an async function?

A) It runs synchronously  
B) It returns a coroutine object instead of the result  
C) It raises an error immediately  
D) Nothing, it works the same

### Question 6
Which library is recommended for async HTTP requests in FastAPI?

A) requests  
B) urllib  
C) httpx  
D) aiohttp

### Question 7
What is the purpose of BackgroundTasks in FastAPI?

A) To run CPU-intensive operations  
B) To run tasks after the response is sent  
C) To make requests faster  
D) To handle concurrent requests

### Question 8
How do you set a timeout for an async operation?

A) asyncio.timeout()  
B) asyncio.wait_for(operation, timeout=5)  
C) operation.timeout(5)  
D) await operation(timeout=5)

### Question 9
What is the main benefit of async over sync for I/O operations?

A) Faster CPU processing  
B) Better memory usage  
C) Higher concurrency and throughput  
D) Simpler code

### Question 10
When should you NOT use async?

A) Database queries  
B) API calls  
C) Heavy CPU computations  
D) File operations

---

## Answers

### Answer 1
**A) async**

Use `async def` to define an asynchronous function. The `async` keyword marks the function as a coroutine that can use `await` inside it.

### Answer 2
**B) For I/O-bound operations like database queries**

Async is beneficial for I/O-bound operations (database, API calls, file I/O) where the program waits for external resources. For CPU-bound operations, use regular sync functions or multiprocessing.

### Answer 3
**B) Pauses the function and allows other tasks to run**

`await` suspends the current coroutine, allowing the event loop to run other tasks while waiting for the awaited operation to complete. This enables concurrency without blocking.

### Answer 4
**B) Use asyncio.gather()**

`asyncio.gather()` runs multiple async operations concurrently and waits for all to complete. Example: `results = await asyncio.gather(task1(), task2(), task3())`

### Answer 5
**B) It returns a coroutine object instead of the result**

Forgetting `await` returns a coroutine object, not the actual result. This is a common bug. Always use `await` when calling async functions.

### Answer 6
**C) httpx**

httpx is the recommended async HTTP client for FastAPI. It has an async API similar to requests. The `requests` library is synchronous only.

### Answer 7
**B) To run tasks after the response is sent**

BackgroundTasks allows you to run operations after sending the response to the client. Useful for logging, sending emails, or cleanup tasks that don't need to delay the response.

### Answer 8
**B) asyncio.wait_for(operation, timeout=5)**

Use `asyncio.wait_for()` to set a timeout. It raises `asyncio.TimeoutError` if the operation exceeds the timeout. Example: `result = await asyncio.wait_for(fetch_data(), timeout=5.0)`

### Answer 9
**C) Higher concurrency and throughput**

Async allows handling many concurrent I/O operations efficiently. While one operation waits for I/O, others can proceed. This dramatically increases throughput for I/O-bound workloads.

### Answer 10
**C) Heavy CPU computations**

Don't use async for CPU-bound operations. Async doesn't make CPU work faster and adds overhead. Use regular sync functions for computations, or multiprocessing for parallel CPU work.

---

## Scoring

- **10/10**: Perfect! You understand async programming
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review async patterns
- **4-5/10**: Fair - Review core async concepts
- **0-3/10**: Needs work - Review all sections

---

## Key Concepts to Remember

1. **async def**: Define async functions
2. **await**: Call async functions (don't forget!)
3. **I/O-bound**: Use async for database, API, file operations
4. **CPU-bound**: Use sync for computations
5. **asyncio.gather()**: Run operations concurrently
6. **httpx**: Async HTTP client
7. **BackgroundTasks**: Run after response
8. **asyncio.wait_for()**: Set timeouts
9. **Concurrency**: Handle many operations simultaneously
10. **Event Loop**: Manages async execution
