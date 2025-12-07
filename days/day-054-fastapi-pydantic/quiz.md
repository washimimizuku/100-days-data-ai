# Day 54: FastAPI Pydantic Validation - Quiz

Test your understanding of Pydantic validation patterns.

---

## Questions

### Question 1
What decorator is used to create a custom validator in Pydantic?

A) @validate  
B) @validator  
C) @field_validator  
D) @custom_validator

### Question 2
Which Field constraint ensures a number is greater than zero?

A) min=0  
B) gt=0  
C) greater_than=0  
D) positive=True

### Question 3
What is the purpose of a root_validator?

A) Validate the root user  
B) Validate a single field  
C) Validate multiple fields together  
D) Validate nested models

### Question 4
How do you make a field optional in Pydantic?

A) Use Optional[Type] from typing  
B) Set required=False  
C) Use nullable=True  
D) Set optional=True

### Question 5
What HTTP status code does FastAPI return for validation errors?

A) 400 Bad Request  
B) 422 Unprocessable Entity  
C) 500 Internal Server Error  
D) 404 Not Found

### Question 6
How do you access other field values in a validator?

A) self.field_name  
B) values['field_name']  
C) cls.field_name  
D) get_field('field_name')

### Question 7
What does the `regex` constraint do in Field()?

A) Validates string matches a pattern  
B) Searches for text  
C) Replaces text  
D) Formats the string

### Question 8
How do you transform data during validation?

A) Return the transformed value from validator  
B) Use a separate transform() method  
C) Set transform=True in Field()  
D) Use @transformer decorator

### Question 9
What is the purpose of Config class in Pydantic models?

A) Configure database connection  
B) Control validation behavior  
C) Set API configuration  
D) Define routes

### Question 10
How do you validate that a list has at least one item?

A) Field(..., min_length=1)  
B) Field(..., min_items=1)  
C) Field(..., required=True)  
D) Field(..., not_empty=True)

---

## Answers

### Answer 1
**B) @validator**

The `@validator` decorator is used to create custom validation logic for fields. Example: `@validator('field_name')` followed by a class method.

### Answer 2
**B) gt=0**

`gt` means "greater than". Use `Field(..., gt=0)` to ensure a number is greater than zero. Other options: `ge` (>=), `lt` (<), `le` (<=).

### Answer 3
**C) Validate multiple fields together**

Root validators check relationships between multiple fields. Use `@root_validator` to validate that fields are consistent with each other (e.g., start_date < end_date).

### Answer 4
**A) Use Optional[Type] from typing**

Make fields optional with `Optional[Type]` and provide a default: `field: Optional[str] = None`. This allows the field to be omitted or set to None.

### Answer 5
**B) 422 Unprocessable Entity**

FastAPI returns 422 for validation errors. The response includes details about which fields failed validation and why. 400 is for malformed requests, 422 is for validation failures.

### Answer 6
**B) values['field_name']**

In validators, access other fields via the `values` dict parameter: `values.get('other_field')`. This is useful for field dependencies.

### Answer 7
**A) Validates string matches a pattern**

The `regex` constraint validates that a string matches a regular expression pattern. Example: `Field(..., regex=r'^[A-Z]{3}-\d{4}$')` for pattern like "ABC-1234".

### Answer 8
**A) Return the transformed value from validator**

Validators can transform data by returning the modified value. Example: `return v.lower()` to lowercase a string. The transformed value is stored in the model.

### Answer 9
**B) Control validation behavior**

The Config class controls Pydantic behavior: `extra` (allow/forbid extra fields), `validate_assignment` (validate on updates), `use_enum_values`, etc.

### Answer 10
**B) Field(..., min_items=1)**

Use `min_items` for lists: `Field(..., min_items=1)` ensures at least one item. Also available: `max_items` to limit list size.

---

## Scoring

- **10/10**: Perfect! You master Pydantic validation
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review validators and constraints
- **4-5/10**: Fair - Review core validation concepts
- **0-3/10**: Needs work - Review all sections

---

## Key Concepts to Remember

1. **@validator**: Custom validation logic
2. **@root_validator**: Validate multiple fields
3. **Field()**: Add constraints (gt, lt, regex, etc.)
4. **Optional[Type]**: Make fields optional
5. **422 Status**: Validation error response
6. **values dict**: Access other fields in validators
7. **Transform**: Return modified value from validator
8. **Config**: Control validation behavior
9. **Nested Models**: Validate complex structures
10. **min_items/max_items**: List size constraints
