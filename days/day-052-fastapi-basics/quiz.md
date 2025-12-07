# Day 52: FastAPI Basics - Quiz

Test your understanding of FastAPI fundamentals.

---

## Questions

### Question 1
What command is used to run a FastAPI application?

A) python app.py  
B) fastapi run app.py  
C) uvicorn main:app --reload  
D) flask run

### Question 2
How do you define a path parameter in FastAPI?

A) @app.get("/users?user_id")  
B) @app.get("/users/<user_id>")  
C) @app.get("/users/{user_id}")  
D) @app.get("/users/:user_id")

### Question 3
What library does FastAPI use for data validation?

A) Marshmallow  
B) Pydantic  
C) Cerberus  
D) Voluptuous

### Question 4
Where can you access the automatic API documentation?

A) /api-docs  
B) /swagger  
C) /docs  
D) /documentation

### Question 5
How do you make a query parameter optional in FastAPI?

A) Use Optional[str] from typing  
B) Add ? after parameter name  
C) Set required=False  
D) Use nullable=True

### Question 6
What decorator is used for a POST endpoint?

A) @app.create()  
B) @app.post()  
C) @app.insert()  
D) @app.add()

### Question 7
How do you raise a 404 error in FastAPI?

A) return 404  
B) raise HTTPException(status_code=404)  
C) throw NotFoundError()  
D) return {"error": 404}

### Question 8
What parameter controls the response structure in FastAPI?

A) return_model  
B) output_model  
C) response_model  
D) schema_model

### Question 9
Which status code should be returned when creating a resource?

A) 200 OK  
B) 201 Created  
C) 202 Accepted  
D) 204 No Content

### Question 10
How do you set a default value for a query parameter?

A) def func(param: int = 10)  
B) def func(param: int default 10)  
C) def func(param: int | 10)  
D) def func(param: int := 10)

---

## Answers

### Answer 1
**C) uvicorn main:app --reload**

Uvicorn is the ASGI server used to run FastAPI applications. The format is `uvicorn filename:app_variable`. The `--reload` flag enables auto-reload during development.

### Answer 2
**C) @app.get("/users/{user_id}")**

Path parameters in FastAPI use curly braces `{}`. The parameter name in the path must match the function parameter name. FastAPI automatically validates the type.

### Answer 3
**B) Pydantic**

FastAPI uses Pydantic for data validation and serialization. Pydantic models define the structure and validation rules for request/response data using Python type hints.

### Answer 4
**C) /docs**

FastAPI automatically generates interactive API documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc). No additional configuration needed.

### Answer 5
**A) Use Optional[str] from typing**

Optional parameters use `Optional[Type]` from the typing module, often with a default value: `param: Optional[str] = None`. This makes the parameter optional in the API.

### Answer 6
**B) @app.post()**

HTTP methods map to decorators: `@app.get()`, `@app.post()`, `@app.put()`, `@app.patch()`, `@app.delete()`. The decorator name matches the HTTP method.

### Answer 7
**B) raise HTTPException(status_code=404)**

Use `HTTPException` from FastAPI to raise HTTP errors. You can include a detail message: `raise HTTPException(status_code=404, detail="Not found")`.

### Answer 8
**C) response_model**

The `response_model` parameter in route decorators controls what fields are included in the response. It filters out fields not in the model, useful for excluding sensitive data like passwords.

### Answer 9
**B) 201 Created**

201 Created indicates successful resource creation. Use `status_code=status.HTTP_201_CREATED` in the decorator. The response typically includes the created resource with its ID.

### Answer 10
**A) def func(param: int = 10)**

Default values use standard Python syntax: `param: Type = default_value`. For query parameters, this makes them optional with the specified default.

---

## Scoring

- **10/10**: Perfect! You understand FastAPI basics
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review Pydantic and routing
- **4-5/10**: Fair - Review core concepts
- **0-3/10**: Needs work - Review all sections

---

## Key Concepts to Remember

1. **Uvicorn**: ASGI server to run FastAPI apps
2. **Path Parameters**: Use `{param}` in route path
3. **Query Parameters**: Function parameters not in path
4. **Pydantic**: Data validation with BaseModel
5. **Type Hints**: Enable automatic validation
6. **HTTPException**: Raise HTTP errors
7. **response_model**: Control response structure
8. **Automatic Docs**: Available at `/docs` and `/redoc`
9. **Status Codes**: Use `status` module constants
10. **Decorators**: `@app.get()`, `@app.post()`, etc.
