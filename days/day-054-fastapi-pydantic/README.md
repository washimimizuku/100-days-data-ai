# Day 54: FastAPI Pydantic Validation

## 📖 Learning Objectives

By the end of this session, you will:
- Master advanced Pydantic validation patterns
- Create custom validators
- Use field constraints and dependencies
- Handle nested models and complex types
- Implement data transformation
- Validate data for data engineering APIs

**Time**: 1 hour  
**Level**: Intermediate

---

## Pydantic Basics Review

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    email: str
    age: int = Field(..., ge=0, le=120)

user = User(name="Alice", email="alice@example.com", age=30)
```

---

## Field Constraints

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0, le=1000000)  # gt/ge/lt/le for numeric
    quantity: int = Field(default=0, ge=0)
    sku: str = Field(..., regex=r'^[A-Z]{3}-\d{4}$')  # Pattern matching
    tags: list[str] = Field(default_factory=list, max_items=10)  # List size
```

---

## Custom Validators

```python
from pydantic import BaseModel, validator

class User(BaseModel):
    username: str
    email: str
    password: str
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('email')
    def email_must_be_valid(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()  # Transform during validation
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8 or not any(c.isupper() for c in v):
            raise ValueError('Password must be 8+ chars with uppercase')
        return v
```

---

## Root Validators

```python
from pydantic import BaseModel, root_validator

# Validate multiple fields together
class DateRange(BaseModel):
    start_date: str
    end_date: str
    
    @root_validator
    def check_dates(cls, values):
        start, end = values.get('start_date'), values.get('end_date')
        if start and end and start > end:
            raise ValueError('start_date must be before end_date')
        return values

class Order(BaseModel):
    quantity: int
    price: float
    discount: float = 0
    
    @root_validator
    def check_discount(cls, values):
        if values.get('discount', 0) > values.get('price', 0):
            raise ValueError('Discount cannot exceed price')
        return values
```

---

## Field Dependencies

```python
from pydantic import BaseModel, validator

class ShippingAddress(BaseModel):
    country: str
    state: str = None
    zip_code: str = None
    
    @validator('state')
    def state_required_for_us(cls, v, values):
        if values.get('country') == 'US' and not v:
            raise ValueError('State required for US addresses')
        return v
    
    @validator('zip_code')
    def validate_zip(cls, v, values):
        if values.get('country') == 'US' and v and len(v) != 5:
            raise ValueError('US zip code must be 5 digits')
        return v
```

---

## Nested Models & Complex Types

```python
from pydantic import BaseModel
from typing import List, Optional, Union

class Address(BaseModel):
    street: str
    city: str
    country: str

class Contact(BaseModel):
    email: str
    phone: str

class User(BaseModel):
    name: str
    address: Address
    contacts: List[Contact]

# Usage with dict input
user = User(
    name="Alice",
    address={"street": "123 Main St", "city": "NYC", "country": "US"},
    contacts=[{"email": "alice@example.com", "phone": "555-0100"}]
)

# Optional and Union types
class DataQuery(BaseModel):
    dataset: str
    filters: Optional[dict] = None
    limit: int = 100
    format: Union[str, None] = "json"
```

---

## Data Transformation & Config

```python
from pydantic import BaseModel, validator

class User(BaseModel):
    name: str
    email: str
    
    @validator('name')
    def capitalize_name(cls, v):
        return v.title()
    
    @validator('email')
    def lowercase_email(cls, v):
        return v.lower()
    
    class Config:
        extra = "forbid"  # or "allow", "ignore"
        validate_assignment = True
        use_enum_values = True

# Transforms: name="alice smith" → "Alice Smith", email="ALICE@..." → "alice@..."
```

---

## Data Engineering Validation

### Dataset Schema Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Literal

class Column(BaseModel):
    name: str = Field(..., regex=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    type: Literal["string", "integer", "float", "boolean", "date"]
    nullable: bool = False
    
class DatasetSchema(BaseModel):
    name: str
    columns: List[Column]
    
    @validator('columns')
    def unique_column_names(cls, v):
        names = [col.name for col in v]
        if len(names) != len(set(names)):
            raise ValueError('Column names must be unique')
        return v
```

### Query Validation

```python
from pydantic import BaseModel, validator
from typing import Optional, List

class DataQuery(BaseModel):
    dataset: str
    columns: Optional[List[str]] = None
    filters: Optional[dict] = None
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)
    
    @validator('filters')
    def validate_filters(cls, v):
        if v:
            allowed_ops = ['eq', 'ne', 'gt', 'lt', 'in']
            for key, value in v.items():
                if isinstance(value, dict):
                    if not any(op in value for op in allowed_ops):
                        raise ValueError(f'Invalid filter operator in {key}')
        return v
```

---

## Special Types & JSON Schema

```python
from pydantic import BaseModel, EmailStr, HttpUrl, validator
from datetime import datetime, date

# Email and URL validation (requires: pip install pydantic[email])
class Contact(BaseModel):
    email: EmailStr
    website: HttpUrl

# Datetime validation
class Event(BaseModel):
    name: str
    start_date: date
    end_date: date
    created_at: datetime
    
    @validator('end_date')
    def end_after_start(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

# JSON schema generation
schema = User.schema()  # Returns OpenAPI-compatible JSON schema
```

---

## FastAPI Integration & Reusable Validators

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator

app = FastAPI()

class CreateUser(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    email: str
    age: int = Field(..., ge=18, le=100)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

@app.post("/users")
def create_user(user: CreateUser):
    return {"message": "User created", "user": user}  # Auto-validation, 422 on error

# Reusable validators
def validate_positive(v):
    if v <= 0:
        raise ValueError('Must be positive')
    return v

class Product(BaseModel):
    price: float
    quantity: int
    
    _validate_price = validator('price', allow_reuse=True)(validate_positive)
    _validate_quantity = validator('quantity', allow_reuse=True)(validate_positive)
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Field Constraints
Create models with various field constraints.

### Exercise 2: Custom Validators
Implement custom validation logic.

### Exercise 3: Nested Models
Build complex nested data structures.

### Exercise 4: Data Transformation
Transform data during validation.

### Exercise 5: Dataset Validation
Validate data engineering schemas.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Pydantic provides automatic validation based on type hints
- Use Field() for constraints (min, max, regex, etc.)
- Custom validators with @validator decorator
- Root validators check multiple fields together
- Nested models for complex structures
- Config class controls validation behavior
- Validation errors return 422 in FastAPI
- Transform data during validation
- JSON schema generation for documentation

---

## 📚 Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic Validators](https://docs.pydantic.dev/usage/validators/)
- [FastAPI with Pydantic](https://fastapi.tiangolo.com/tutorial/body/)
- [Pydantic Types](https://docs.pydantic.dev/usage/types/)

---

## Tomorrow: Day 55 - Testing Data Pipelines

Learn to test data pipelines and APIs with pytest.
