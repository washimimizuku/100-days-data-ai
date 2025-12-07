"""
Day 51: REST API Principles - Exercises

Practice designing RESTful APIs for data services.
"""


def exercise_1():
    """
    Exercise 1: Design User API
    
    Design RESTful endpoints for user management with:
    - List all users
    - Get specific user
    - Create new user
    - Update user
    - Delete user
    - Get user's orders
    
    Return dict with endpoint: {method, path, description}
    """
    # TODO: Design endpoints
    endpoints = {
        "list_users": {
            "method": "",
            "path": "",
            "description": ""
        },
        # Add more endpoints
    }
    return endpoints


def exercise_2():
    """
    Exercise 2: Design Data Query API
    
    Design endpoints for querying datasets with:
    - List available datasets
    - Get dataset metadata
    - Query dataset with filters
    - Get dataset schema
    - Export dataset
    
    Include query parameters for:
    - Filtering (where conditions)
    - Sorting (order by)
    - Pagination (limit, offset)
    - Field selection (columns)
    """
    # TODO: Design data query endpoints
    endpoints = {}
    return endpoints


def exercise_3():
    """
    Exercise 3: HTTP Status Codes
    
    Choose appropriate HTTP status code for each scenario:
    
    Scenarios:
    1. User successfully created
    2. User not found
    3. Invalid email format in request
    4. User deleted successfully (no content to return)
    5. Authentication token missing
    6. User authenticated but not authorized for resource
    7. Server database connection failed
    8. Rate limit exceeded
    9. User data retrieved successfully
    10. Duplicate email address (conflict)
    """
    # TODO: Return dict with scenario: status_code
    status_codes = {
        "user_created": 0,
        "user_not_found": 0,
        # Add more
    }
    return status_codes


def exercise_4():
    """
    Exercise 4: Error Responses
    
    Design consistent error response structures for:
    1. Validation error (multiple field errors)
    2. Resource not found
    3. Authentication error
    4. Rate limit exceeded
    5. Internal server error
    
    Each response should include:
    - status
    - error code
    - message
    - additional context (if applicable)
    """
    # TODO: Design error responses
    error_responses = {
        "validation_error": {},
        "not_found": {},
        # Add more
    }
    return error_responses


def exercise_5():
    """
    Exercise 5: API Versioning
    
    Design versioning strategy for breaking changes:
    
    v1 User endpoint returns:
    {
      "id": 123,
      "name": "Alice",
      "email": "alice@example.com"
    }
    
    v2 User endpoint should return:
    {
      "id": 123,
      "first_name": "Alice",
      "last_name": "Smith",
      "email": "alice@example.com",
      "created_at": "2024-01-01T10:00:00Z"
    }
    
    Design:
    1. URL structure for both versions
    2. Migration strategy
    3. Deprecation notice format
    """
    # TODO: Design versioning strategy
    versioning = {
        "v1_endpoint": "",
        "v2_endpoint": "",
        "migration_plan": [],
        "deprecation_notice": {}
    }
    return versioning


def exercise_6():
    """
    Exercise 6: Pagination Design
    
    Design pagination for large datasets:
    
    Dataset: 10,000 products
    Requirements:
    - Support offset-based pagination
    - Include metadata (total, page info)
    - Allow configurable page size
    - Maximum 100 items per page
    
    Design:
    1. Query parameters
    2. Response structure
    3. Navigation links (first, last, next, prev)
    """
    # TODO: Design pagination
    pagination = {
        "query_params": {},
        "response_structure": {},
        "navigation": {}
    }
    return pagination


def exercise_7():
    """
    Exercise 7: Filtering and Sorting
    
    Design query parameters for product API:
    
    Filters:
    - Category (exact match)
    - Price range (min, max)
    - In stock (boolean)
    - Brand (multiple values)
    
    Sorting:
    - By price (asc/desc)
    - By name (asc/desc)
    - By created date (asc/desc)
    - Multiple sort fields
    
    Example URL:
    /api/products?category=electronics&price_min=100&price_max=500&sort=-price
    """
    # TODO: Design query parameters
    query_design = {
        "filters": {},
        "sorting": {},
        "example_urls": []
    }
    return query_design


def exercise_8():
    """
    Exercise 8: Bulk Operations
    
    Design endpoints for bulk operations:
    1. Create multiple users
    2. Update multiple products
    3. Delete multiple records
    
    Consider:
    - Partial success handling
    - Error reporting for failed items
    - Transaction semantics
    - Response format
    """
    # TODO: Design bulk operation endpoints
    bulk_operations = {
        "create_multiple": {},
        "update_multiple": {},
        "delete_multiple": {},
        "error_handling": {}
    }
    return bulk_operations


if __name__ == "__main__":
    print("Day 51: REST API Principles - Exercises\n")
    
    # Uncomment to run exercises
    # print("Exercise 1: Design User API")
    # print(exercise_1())
    
    # print("\nExercise 2: Design Data Query API")
    # print(exercise_2())
    
    # print("\nExercise 3: HTTP Status Codes")
    # print(exercise_3())
    
    # print("\nExercise 4: Error Responses")
    # print(exercise_4())
    
    # print("\nExercise 5: API Versioning")
    # print(exercise_5())
    
    # print("\nExercise 6: Pagination Design")
    # print(exercise_6())
    
    # print("\nExercise 7: Filtering and Sorting")
    # print(exercise_7())
    
    # print("\nExercise 8: Bulk Operations")
    # print(exercise_8())
