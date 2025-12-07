"""
Day 54: FastAPI Pydantic Validation - Solutions
"""
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Literal
from datetime import date, datetime
import re


# Exercise 1: Field Constraints
class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    sku: str = Field(..., regex=r'^[A-Z]{3}-\d{4}$')
    price: float = Field(..., gt=0, le=1000000)
    quantity: int = Field(default=0, ge=0)
    tags: List[str] = Field(default_factory=list, max_items=10)
    description: Optional[str] = Field(None, max_length=500)


# Exercise 2: Custom Validators
class UserRegistration(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    email: str
    password: str
    age: int = Field(..., ge=18, le=100)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('email')
    def email_lowercase(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v


# Exercise 3: Root Validators
class DateRange(BaseModel):
    start_date: date
    end_date: date
    description: str
    
    @root_validator
    def validate_date_range(cls, values):
        start = values.get('start_date')
        end = values.get('end_date')
        
        if start and end:
            if end < start:
                raise ValueError('end_date must be after start_date')
            
            days_diff = (end - start).days
            if days_diff > 365:
                raise ValueError('Date range cannot exceed 1 year')
        
        return values


# Exercise 4: Nested Models
class Address(BaseModel):
    street: str
    city: str
    country: str
    zip_code: str

class OrderItem(BaseModel):
    product_id: str
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)

class Order(BaseModel):
    order_id: str
    customer_name: str
    shipping_address: Address
    items: List[OrderItem] = Field(..., min_items=1)
    total: Optional[float] = None
    
    @root_validator
    def calculate_total(cls, values):
        items = values.get('items', [])
        if items:
            total = sum(item.quantity * item.price for item in items)
            values['total'] = round(total, 2)
        return values


# Exercise 5: Field Dependencies
class ShippingInfo(BaseModel):
    country: str
    state: Optional[str] = None
    zip_code: Optional[str] = None
    
    @validator('state')
    def state_required_for_us(cls, v, values):
        country = values.get('country')
        if country == 'US' and not v:
            raise ValueError('State is required for US addresses')
        return v
    
    @validator('zip_code')
    def validate_zip_code(cls, v, values):
        country = values.get('country')
        
        if country == 'US' and v:
            if not re.match(r'^\d{5}$', v):
                raise ValueError('US zip code must be 5 digits')
        
        if country == 'CA' and v:
            if not re.match(r'^[A-Z]\d[A-Z] \d[A-Z]\d$', v):
                raise ValueError('Canadian postal code must match pattern A1A 1A1')
        
        return v


# Exercise 6: Data Transformation
class Contact(BaseModel):
    name: str
    email: str
    phone: str
    
    @validator('name')
    def capitalize_name(cls, v):
        return v.strip().title()
    
    @validator('email')
    def lowercase_email(cls, v):
        return v.strip().lower()
    
    @validator('phone')
    def format_phone(cls, v):
        # Remove non-digits
        digits = ''.join(c for c in v if c.isdigit())
        # Format as XXX-XXX-XXXX
        if len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        return digits


# Exercise 7: Dataset Schema Validation
class Column(BaseModel):
    name: str = Field(..., regex=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    type: Literal["string", "integer", "float", "boolean", "date"]
    nullable: bool = False
    description: Optional[str] = None

class DatasetSchema(BaseModel):
    name: str
    columns: List[Column] = Field(..., min_items=1)
    primary_key: Optional[str] = None
    
    @validator('columns')
    def unique_column_names(cls, v):
        names = [col.name for col in v]
        if len(names) != len(set(names)):
            raise ValueError('Column names must be unique')
        return v
    
    @root_validator
    def validate_primary_key(cls, values):
        primary_key = values.get('primary_key')
        columns = values.get('columns', [])
        
        if primary_key:
            column_names = [col.name for col in columns]
            if primary_key not in column_names:
                raise ValueError(f'Primary key "{primary_key}" not found in columns')
        
        return values


# Exercise 8: Query Validation
class DataQuery(BaseModel):
    dataset: str
    columns: Optional[List[str]] = None
    filters: Optional[dict] = None
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "asc"
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)
    
    @validator('filters')
    def validate_filters(cls, v):
        if v:
            allowed_ops = ['eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'in']
            for key, value in v.items():
                if isinstance(value, dict):
                    ops = list(value.keys())
                    invalid_ops = [op for op in ops if op not in allowed_ops]
                    if invalid_ops:
                        raise ValueError(f'Invalid filter operators: {invalid_ops}')
        return v


if __name__ == "__main__":
    print("Day 54: FastAPI Pydantic Validation - Solutions\n")
    
    # Test Exercise 1
    print("=" * 60)
    print("Exercise 1: Product Validation")
    print("=" * 60)
    try:
        product = Product(
            name="Laptop",
            sku="LAP-1234",
            price=999.99,
            quantity=10,
            tags=["electronics", "computers"]
        )
        print(f"✓ Valid product: {product.name}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test Exercise 2
    print("\n" + "=" * 60)
    print("Exercise 2: User Registration")
    print("=" * 60)
    try:
        user = UserRegistration(
            username="alice123",
            email="ALICE@EXAMPLE.COM",
            password="SecurePass123",
            age=25
        )
        print(f"✓ Valid user: {user.username}, {user.email}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test Exercise 3
    print("\n" + "=" * 60)
    print("Exercise 3: Date Range")
    print("=" * 60)
    try:
        date_range = DateRange(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            description="First half of 2024"
        )
        print(f"✓ Valid range: {date_range.start_date} to {date_range.end_date}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test Exercise 4
    print("\n" + "=" * 60)
    print("Exercise 4: Order with Nested Models")
    print("=" * 60)
    try:
        order = Order(
            order_id="ORD-001",
            customer_name="Alice Smith",
            shipping_address={
                "street": "123 Main St",
                "city": "New York",
                "country": "US",
                "zip_code": "10001"
            },
            items=[
                {"product_id": "PROD-1", "quantity": 2, "price": 29.99},
                {"product_id": "PROD-2", "quantity": 1, "price": 49.99}
            ]
        )
        print(f"✓ Valid order: {order.order_id}, Total: ${order.total}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test Exercise 5
    print("\n" + "=" * 60)
    print("Exercise 5: Shipping Info")
    print("=" * 60)
    try:
        shipping = ShippingInfo(
            country="US",
            state="NY",
            zip_code="10001"
        )
        print(f"✓ Valid shipping: {shipping.country}, {shipping.state}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test Exercise 6
    print("\n" + "=" * 60)
    print("Exercise 6: Contact Transformation")
    print("=" * 60)
    try:
        contact = Contact(
            name="  alice smith  ",
            email="  ALICE@EXAMPLE.COM  ",
            phone="(555) 123-4567"
        )
        print(f"✓ Transformed: {contact.name}, {contact.email}, {contact.phone}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test Exercise 7
    print("\n" + "=" * 60)
    print("Exercise 7: Dataset Schema")
    print("=" * 60)
    try:
        schema = DatasetSchema(
            name="users",
            columns=[
                {"name": "id", "type": "integer", "nullable": False},
                {"name": "name", "type": "string", "nullable": False},
                {"name": "email", "type": "string", "nullable": False}
            ],
            primary_key="id"
        )
        print(f"✓ Valid schema: {schema.name} with {len(schema.columns)} columns")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test Exercise 8
    print("\n" + "=" * 60)
    print("Exercise 8: Data Query")
    print("=" * 60)
    try:
        query = DataQuery(
            dataset="sales",
            columns=["id", "amount", "date"],
            filters={"amount": {"gt": 100}, "status": {"eq": "completed"}},
            sort_by="date",
            sort_order="desc",
            limit=50
        )
        print(f"✓ Valid query: {query.dataset}, limit={query.limit}")
    except Exception as e:
        print(f"✗ Error: {e}")
