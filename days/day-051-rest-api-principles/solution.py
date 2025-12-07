"""
Day 51: REST API Principles - Solutions
"""


def exercise_1():
    """Design User API endpoints"""
    endpoints = {
        "list_users": {
            "method": "GET",
            "path": "/api/users",
            "description": "List all users with pagination"
        },
        "get_user": {
            "method": "GET",
            "path": "/api/users/{id}",
            "description": "Get specific user by ID"
        },
        "create_user": {
            "method": "POST",
            "path": "/api/users",
            "description": "Create new user"
        },
        "update_user": {
            "method": "PUT",
            "path": "/api/users/{id}",
            "description": "Update entire user record"
        },
        "partial_update_user": {
            "method": "PATCH",
            "path": "/api/users/{id}",
            "description": "Update specific user fields"
        },
        "delete_user": {
            "method": "DELETE",
            "path": "/api/users/{id}",
            "description": "Delete user"
        },
        "get_user_orders": {
            "method": "GET",
            "path": "/api/users/{id}/orders",
            "description": "Get all orders for specific user"
        }
    }
    return endpoints


def exercise_2():
    """Design Data Query API"""
    endpoints = {
        "list_datasets": {
            "method": "GET",
            "path": "/api/datasets",
            "description": "List all available datasets",
            "query_params": ["search", "category", "limit", "offset"]
        },
        "get_dataset_metadata": {
            "method": "GET",
            "path": "/api/datasets/{id}",
            "description": "Get dataset metadata (name, size, schema, etc.)"
        },
        "query_dataset": {
            "method": "GET",
            "path": "/api/datasets/{id}/query",
            "description": "Query dataset with filters",
            "query_params": [
                "filter",      # WHERE conditions
                "sort",        # ORDER BY
                "limit",       # LIMIT
                "offset",      # OFFSET
                "fields"       # SELECT columns
            ],
            "example": "/api/datasets/sales/query?filter=amount>100&sort=-date&limit=50&fields=id,amount,date"
        },
        "get_dataset_schema": {
            "method": "GET",
            "path": "/api/datasets/{id}/schema",
            "description": "Get dataset schema (columns, types)"
        },
        "export_dataset": {
            "method": "POST",
            "path": "/api/datasets/{id}/export",
            "description": "Export dataset in specified format",
            "body": {
                "format": "csv|json|parquet",
                "filters": {},
                "columns": []
            }
        }
    }
    return endpoints


def exercise_3():
    """HTTP Status Codes"""
    status_codes = {
        "user_created": 201,                    # Created
        "user_not_found": 404,                  # Not Found
        "invalid_email": 422,                   # Unprocessable Entity
        "user_deleted": 204,                    # No Content
        "missing_auth": 401,                    # Unauthorized
        "not_authorized": 403,                  # Forbidden
        "database_error": 500,                  # Internal Server Error
        "rate_limit": 429,                      # Too Many Requests
        "user_retrieved": 200,                  # OK
        "duplicate_email": 409                  # Conflict
    }
    return status_codes


def exercise_4():
    """Error Response Structures"""
    error_responses = {
        "validation_error": {
            "status": "error",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": [
                    {
                        "field": "email",
                        "message": "Invalid email format",
                        "value": "invalid-email"
                    },
                    {
                        "field": "age",
                        "message": "Must be between 0 and 120",
                        "value": -5
                    }
                ]
            },
            "timestamp": "2024-01-01T10:00:00Z",
            "path": "/api/users"
        },
        "not_found": {
            "status": "error",
            "error": {
                "code": "RESOURCE_NOT_FOUND",
                "message": "User with id 123 not found",
                "resource_type": "user",
                "resource_id": "123"
            },
            "timestamp": "2024-01-01T10:00:00Z",
            "path": "/api/users/123"
        },
        "authentication_error": {
            "status": "error",
            "error": {
                "code": "AUTHENTICATION_REQUIRED",
                "message": "Valid authentication token required",
                "hint": "Include 'Authorization: Bearer <token>' header"
            },
            "timestamp": "2024-01-01T10:00:00Z",
            "path": "/api/users"
        },
        "rate_limit": {
            "status": "error",
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Rate limit exceeded. Try again later.",
                "retry_after": 60,
                "limit": 1000,
                "window": "1 hour"
            },
            "timestamp": "2024-01-01T10:00:00Z"
        },
        "server_error": {
            "status": "error",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "request_id": "req_abc123"
            },
            "timestamp": "2024-01-01T10:00:00Z"
        }
    }
    return error_responses


def exercise_5():
    """API Versioning Strategy"""
    versioning = {
        "v1_endpoint": "/api/v1/users/{id}",
        "v2_endpoint": "/api/v2/users/{id}",
        "migration_plan": [
            "1. Release v2 alongside v1",
            "2. Add deprecation headers to v1 responses",
            "3. Update documentation with migration guide",
            "4. Give 6 months notice before v1 sunset",
            "5. Monitor v1 usage and contact heavy users",
            "6. Sunset v1 after migration period"
        ],
        "deprecation_notice": {
            "headers": {
                "Deprecation": "true",
                "Sunset": "2024-12-31T23:59:59Z",
                "Link": "<https://api.example.com/docs/migration>; rel=\"deprecation\""
            },
            "response_body": {
                "warning": "This API version is deprecated",
                "sunset_date": "2024-12-31",
                "migration_guide": "https://api.example.com/docs/migration",
                "new_version": "/api/v2/users/{id}"
            }
        },
        "v1_response": {
            "id": 123,
            "name": "Alice Smith",
            "email": "alice@example.com"
        },
        "v2_response": {
            "id": 123,
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "alice@example.com",
            "created_at": "2024-01-01T10:00:00Z",
            "updated_at": "2024-01-15T14:30:00Z"
        }
    }
    return versioning


def exercise_6():
    """Pagination Design"""
    pagination = {
        "query_params": {
            "limit": "Number of items per page (default: 20, max: 100)",
            "offset": "Number of items to skip (default: 0)",
            "page": "Alternative: page number (1-indexed)",
            "per_page": "Alternative: items per page"
        },
        "response_structure": {
            "data": [
                {"id": 1, "name": "Product 1"},
                {"id": 2, "name": "Product 2"}
            ],
            "meta": {
                "total": 10000,
                "limit": 20,
                "offset": 40,
                "page": 3,
                "total_pages": 500
            },
            "links": {
                "first": "/api/products?limit=20&offset=0",
                "prev": "/api/products?limit=20&offset=20",
                "self": "/api/products?limit=20&offset=40",
                "next": "/api/products?limit=20&offset=60",
                "last": "/api/products?limit=20&offset=9980"
            }
        },
        "navigation": {
            "has_next": True,
            "has_prev": True,
            "next_offset": 60,
            "prev_offset": 20
        }
    }
    return pagination


def exercise_7():
    """Filtering and Sorting Design"""
    query_design = {
        "filters": {
            "category": "Exact match: ?category=electronics",
            "price_min": "Range: ?price_min=100",
            "price_max": "Range: ?price_max=500",
            "in_stock": "Boolean: ?in_stock=true",
            "brand": "Multiple: ?brand=apple,samsung,sony"
        },
        "sorting": {
            "single_asc": "?sort=price",
            "single_desc": "?sort=-price",
            "multiple": "?sort=category,-price,name"
        },
        "example_urls": [
            "/api/products?category=electronics&price_min=100&price_max=500&sort=-price",
            "/api/products?in_stock=true&brand=apple,samsung&sort=name",
            "/api/products?category=books&sort=-created_at&limit=50"
        ],
        "query_string_format": {
            "filter_operators": {
                "eq": "Equal: ?price=100",
                "gt": "Greater than: ?price_gt=100",
                "gte": "Greater or equal: ?price_gte=100",
                "lt": "Less than: ?price_lt=500",
                "lte": "Less or equal: ?price_lte=500",
                "in": "In list: ?category_in=electronics,books",
                "like": "Pattern match: ?name_like=phone"
            }
        }
    }
    return query_design


def exercise_8():
    """Bulk Operations Design"""
    bulk_operations = {
        "create_multiple": {
            "method": "POST",
            "path": "/api/users/bulk",
            "request": {
                "users": [
                    {"name": "Alice", "email": "alice@example.com"},
                    {"name": "Bob", "email": "bob@example.com"}
                ]
            },
            "response": {
                "status": "partial_success",
                "created": 1,
                "failed": 1,
                "results": [
                    {
                        "index": 0,
                        "status": "success",
                        "data": {"id": 123, "name": "Alice"}
                    },
                    {
                        "index": 1,
                        "status": "error",
                        "error": {
                            "code": "DUPLICATE_EMAIL",
                            "message": "Email already exists"
                        }
                    }
                ]
            }
        },
        "update_multiple": {
            "method": "PATCH",
            "path": "/api/products/bulk",
            "request": {
                "updates": [
                    {"id": 1, "price": 99.99},
                    {"id": 2, "price": 149.99}
                ]
            }
        },
        "delete_multiple": {
            "method": "DELETE",
            "path": "/api/users/bulk",
            "request": {
                "ids": [123, 456, 789]
            },
            "response": {
                "deleted": 2,
                "failed": 1,
                "errors": [
                    {
                        "id": 456,
                        "error": "User not found"
                    }
                ]
            }
        },
        "error_handling": {
            "strategy": "Continue on error, report all failures",
            "transaction": "No transaction - partial success allowed",
            "rollback": "Not supported for bulk operations",
            "recommendation": "Use batch processing for large datasets"
        }
    }
    return bulk_operations


if __name__ == "__main__":
    print("Day 51: REST API Principles - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: User API Design")
    print("=" * 60)
    endpoints = exercise_1()
    for name, config in endpoints.items():
        print(f"{config['method']:6s} {config['path']:30s} - {config['description']}")
    
    print("\n" + "=" * 60)
    print("Exercise 2: Data Query API")
    print("=" * 60)
    data_endpoints = exercise_2()
    for name, config in data_endpoints.items():
        print(f"\n{name}:")
        print(f"  {config['method']} {config['path']}")
        print(f"  {config['description']}")
        if 'example' in config:
            print(f"  Example: {config['example']}")
    
    print("\n" + "=" * 60)
    print("Exercise 3: HTTP Status Codes")
    print("=" * 60)
    codes = exercise_3()
    for scenario, code in codes.items():
        print(f"{scenario:20s}: {code}")
    
    print("\n" + "=" * 60)
    print("Exercise 4: Error Response (Validation)")
    print("=" * 60)
    errors = exercise_4()
    import json
    print(json.dumps(errors["validation_error"], indent=2))
    
    print("\n" + "=" * 60)
    print("Exercise 5: API Versioning")
    print("=" * 60)
    versioning = exercise_5()
    print(f"v1: {versioning['v1_endpoint']}")
    print(f"v2: {versioning['v2_endpoint']}")
    print("\nMigration Plan:")
    for step in versioning['migration_plan']:
        print(f"  {step}")
    
    print("\n" + "=" * 60)
    print("Exercise 6: Pagination Response")
    print("=" * 60)
    pagination = exercise_6()
    print(json.dumps(pagination["response_structure"]["meta"], indent=2))
    
    print("\n" + "=" * 60)
    print("Exercise 7: Filtering Examples")
    print("=" * 60)
    query = exercise_7()
    for url in query["example_urls"]:
        print(url)
    
    print("\n" + "=" * 60)
    print("Exercise 8: Bulk Create Response")
    print("=" * 60)
    bulk = exercise_8()
    print(json.dumps(bulk["create_multiple"]["response"], indent=2))
