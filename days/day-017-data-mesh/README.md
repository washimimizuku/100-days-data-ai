# Day 17: Data Mesh

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand data mesh principles
- Compare centralized vs decentralized data architectures
- Design domain-oriented data products
- Implement federated governance
- Apply data mesh concepts to organizations

---

## Theory

### What is Data Mesh?

A **decentralized socio-technical approach** to data architecture that treats data as a product owned by domain teams.

**Created by**: Zhamak Dehghani (ThoughtWorks, 2019)
**Purpose**: Scale data architecture in large organizations
**Paradigm shift**: From centralized data lake/warehouse to distributed domain ownership

---

### The Problem with Centralized Data

**Traditional Architecture**:
```
Domain Teams → Central Data Team → Data Lake/Warehouse → Analytics
```

**Issues**:
- Bottleneck: Central team overwhelmed
- Context loss: Domain knowledge not captured
- Slow delivery: Long queues for data requests
- Poor quality: Central team lacks domain expertise
- Scaling problems: Can't scale with organization

---

### Four Principles of Data Mesh

#### 1. Domain Ownership

**Concept**: Domain teams own their data end-to-end

**Traditional**:
```
Sales Team → Request → Central Data Team → Build Pipeline → Deliver
(Weeks/months delay)
```

**Data Mesh**:
```
Sales Team → Own Sales Data Product → Self-serve
(Immediate access)
```

**Example Domains**:
- Sales (orders, customers, revenue)
- Marketing (campaigns, leads, conversions)
- Product (features, usage, feedback)
- Finance (transactions, invoices, payments)

---

#### 2. Data as a Product

**Concept**: Treat data like a product with quality standards

**Product Thinking**:
- Discoverable (catalog, documentation)
- Addressable (clear APIs/interfaces)
- Trustworthy (quality, SLAs)
- Self-describing (schema, metadata)
- Secure (access controls)
- Interoperable (standard formats)

**Example Data Product**:
```
Sales Orders Data Product
├── API: orders.sales.company.com
├── Schema: Documented in catalog
├── SLA: 99.9% availability, <1hr latency
├── Quality: Validated, no duplicates
├── Access: Role-based permissions
└── Owner: Sales domain team
```

---

#### 3. Self-Serve Data Platform

**Concept**: Platform team provides infrastructure for domain teams

**Platform Capabilities**:
- Data pipeline templates
- Storage infrastructure
- Compute resources
- Monitoring/observability
- Security/governance tools
- CI/CD for data

**Platform vs Domain Responsibilities**:

| Platform Team | Domain Team |
|---------------|-------------|
| Infrastructure | Data products |
| Tools/templates | Business logic |
| Standards | Domain models |
| Governance framework | Data quality |

---

#### 4. Federated Computational Governance

**Concept**: Distributed governance with global standards

**Governance Model**:
```
Global Policies (Platform)
├── Data privacy (GDPR, CCPA)
├── Security standards
├── Interoperability rules
└── Quality baselines

Domain Implementation (Teams)
├── Domain-specific rules
├── Data product SLAs
└── Access policies
```

**Automated Governance**:
- Policy as code
- Automated compliance checks
- Self-service access requests
- Audit trails

---

### Data Mesh Architecture

```
┌─────────────────────────────────────────────────────┐
│              Self-Serve Data Platform               │
│  (Infrastructure, Tools, Governance Framework)      │
└─────────────────────────────────────────────────────┘
           ↓              ↓              ↓
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Sales   │   │Marketing │   │ Product  │
    │  Domain  │   │  Domain  │   │  Domain  │
    └──────────┘   └──────────┘   └──────────┘
         ↓              ↓              ↓
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Orders   │   │Campaigns │   │  Usage   │
    │   Data   │   │   Data   │   │   Data   │
    │ Product  │   │ Product  │   │ Product  │
    └──────────┘   └──────────┘   └──────────┘
         ↓              ↓              ↓
    ┌─────────────────────────────────────────┐
    │     Consumers (Analytics, ML, BI)       │
    └─────────────────────────────────────────┘
```

---

### Data Product Anatomy

```python
class DataProduct:
    # Identity
    name = "sales_orders"
    domain = "sales"
    owner = "sales-team@company.com"
    
    # Interface
    api_endpoint = "orders.sales.company.com"
    schema = OrdersSchema()
    format = "delta"
    
    # Quality
    sla_availability = 99.9
    sla_latency_hours = 1
    quality_checks = [no_duplicates, no_nulls, valid_amounts]
    
    # Governance
    access_policy = "role-based"
    retention_days = 2555  # 7 years
    pii_fields = ["customer_email", "customer_phone"]
    
    # Observability
    metrics = ["row_count", "freshness", "quality_score"]
    alerts = ["data_quality_failure", "sla_breach"]
```

---

### Implementation Example

**Sales Domain Data Product**:

```python
# sales_orders_product.py
from pyspark.sql import SparkSession
from delta import DeltaTable

class SalesOrdersProduct:
    def __init__(self):
        self.spark = SparkSession.builder.getOrCreate()
        self.path = "s3://mesh/sales/orders"
        self.owner = "sales-team"
    
    def publish(self, orders_df):
        """Publish orders data product"""
        # Apply quality checks
        validated_df = self._validate(orders_df)
        
        # Write to Delta
        validated_df.write \
            .format("delta") \
            .mode("append") \
            .save(self.path)
        
        # Update catalog
        self._register_in_catalog()
    
    def _validate(self, df):
        """Apply data quality rules"""
        return df \
            .dropDuplicates(["order_id"]) \
            .filter(col("order_id").isNotNull()) \
            .filter(col("amount") > 0)
    
    def _register_in_catalog(self):
        """Register in data catalog"""
        catalog.register(
            name="sales_orders",
            path=self.path,
            owner=self.owner,
            schema=self._get_schema()
        )
    
    def consume(self, consumer_team):
        """Provide access to consumers"""
        if self._check_access(consumer_team):
            return self.spark.read.format("delta").load(self.path)
        else:
            raise PermissionError("Access denied")
```

---

### Data Mesh vs Traditional

| Aspect | Traditional | Data Mesh |
|--------|-------------|-----------|
| **Ownership** | Central team | Domain teams |
| **Architecture** | Monolithic | Distributed |
| **Scaling** | Vertical | Horizontal |
| **Bottleneck** | Central team | None |
| **Context** | Lost | Preserved |
| **Delivery** | Slow | Fast |
| **Quality** | Variable | High (domain expertise) |
| **Governance** | Centralized | Federated |

---

### When to Use Data Mesh

**Good fit**:
- Large organizations (>100 people)
- Multiple domains/business units
- Central team bottleneck
- Complex data landscape
- Need for scale

**Not a good fit**:
- Small organizations (<50 people)
- Single domain
- Simple data needs
- Limited resources
- Early stage startups

---

### Implementation Challenges

1. **Cultural shift** - From centralized to distributed
2. **Skills gap** - Domain teams need data skills
3. **Platform complexity** - Self-serve platform is hard
4. **Governance** - Balancing autonomy and standards
5. **Duplication** - Risk of redundant data products
6. **Discovery** - Finding the right data product

---

### Best Practices

1. **Start small** - Pilot with one domain
2. **Invest in platform** - Self-serve is critical
3. **Clear ownership** - Define domain boundaries
4. **Automate governance** - Policy as code
5. **Data catalog** - Essential for discovery
6. **Training** - Upskill domain teams
7. **Standards** - Define interoperability rules
8. **Metrics** - Track data product health

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Domain Identification
- Identify domains in e-commerce company
- Define domain boundaries
- List data products per domain

### Exercise 2: Data Product Design
- Design sales orders data product
- Define schema and SLAs
- Specify quality checks

### Exercise 3: Governance Policies
- Define global policies
- Define domain-specific rules
- Implement policy checks

### Exercise 4: Platform Capabilities
- List required platform features
- Design self-serve APIs
- Plan infrastructure

### Exercise 5: Migration Planning
- Plan centralized to mesh migration
- Identify risks
- Estimate timeline

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What are the four principles of data mesh?
2. What does "data as a product" mean?
3. Who owns data in data mesh?
4. What is federated governance?
5. When should you use data mesh?
6. What is a self-serve data platform?
7. How is data mesh different from data lake?
8. What are common challenges?

---

## 🎯 Key Takeaways

- **Decentralized** - Domain teams own their data
- **Product thinking** - Data treated as product
- **Self-serve platform** - Infrastructure for domains
- **Federated governance** - Distributed with standards
- **Scalability** - Horizontal scaling by domain
- **Context preservation** - Domain expertise retained
- **Cultural shift** - Requires organizational change
- **Not for everyone** - Best for large organizations

---

## 📚 Additional Resources

- [Data Mesh Principles](https://martinfowler.com/articles/data-mesh-principles.html)
- [Zhamak Dehghani's Book](https://www.oreilly.com/library/view/data-mesh/9781492092384/)
- [Data Mesh Architecture](https://www.datamesh-architecture.com/)

---

## Tomorrow: Day 18 - Star Schema

We'll explore dimensional modeling for data warehouses.
