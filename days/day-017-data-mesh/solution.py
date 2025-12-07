"""
Day 17: Data Mesh - Solutions
"""


def exercise_1():
    """Identify domains and data products for e-commerce company"""
    domains = {
        "Sales": {
            "boundary": "Order processing, customer management, revenue",
            "data_products": [
                "orders (order_id, customer_id, amount, status, date)",
                "customers (customer_id, name, email, segment)",
                "revenue_metrics (daily_revenue, monthly_revenue, by_category)"
            ]
        },
        "Marketing": {
            "boundary": "Campaigns, leads, conversions, attribution",
            "data_products": [
                "campaigns (campaign_id, name, budget, channel, dates)",
                "leads (lead_id, source, campaign_id, status)",
                "conversions (conversion_id, lead_id, order_id, value)"
            ]
        },
        "Product": {
            "boundary": "Catalog, inventory, reviews, usage",
            "data_products": [
                "catalog (product_id, name, category, price)",
                "inventory (product_id, quantity, warehouse_id)",
                "reviews (review_id, product_id, rating, comment)"
            ]
        },
        "Finance": {
            "boundary": "Transactions, invoices, payments, accounting",
            "data_products": [
                "transactions (transaction_id, order_id, amount, method)",
                "invoices (invoice_id, customer_id, amount, status)",
                "payments (payment_id, invoice_id, amount, date)"
            ]
        }
    }
    
    print("=== E-commerce Data Mesh Domains ===\n")
    for domain, info in domains.items():
        print(f"{domain} Domain:")
        print(f"  Boundary: {info['boundary']}")
        print(f"  Data Products:")
        for product in info['data_products']:
            print(f"    - {product}")
        print()


def exercise_2():
    """Design a data product with full specifications"""
    data_product = {
        "name": "sales_orders",
        "domain": "sales",
        "owner": "sales-team@company.com",
        "description": "Complete order data including customer, products, and payment info",
        
        "schema": {
            "order_id": "STRING (PK)",
            "customer_id": "STRING (FK)",
            "order_date": "DATE",
            "total_amount": "DECIMAL(10,2)",
            "status": "STRING (pending|completed|cancelled)",
            "items": "ARRAY<STRUCT<product_id, quantity, price>>",
            "payment_method": "STRING"
        },
        
        "sla": {
            "availability": "99.9%",
            "latency": "< 1 hour from source",
            "freshness": "Updated every 15 minutes",
            "completeness": "> 99.5%"
        },
        
        "quality_checks": [
            "No duplicate order_ids",
            "No null order_ids or customer_ids",
            "total_amount > 0",
            "order_date <= current_date",
            "status in allowed values",
            "items array not empty"
        ],
        
        "access_policy": {
            "public": ["order_id", "order_date", "status"],
            "internal": ["customer_id", "total_amount", "items"],
            "restricted": ["payment_method"]
        },
        
        "monitoring": {
            "metrics": ["row_count", "freshness_minutes", "quality_score", "null_rate"],
            "alerts": ["sla_breach", "quality_failure", "schema_change"]
        },
        
        "interface": {
            "format": "Delta Lake",
            "api": "orders.sales.company.com",
            "catalog": "data-catalog.company.com/sales/orders"
        }
    }
    
    print("=== Sales Orders Data Product ===\n")
    for key, value in data_product.items():
        print(f"{key.upper()}:")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        else:
            print(f"  {value}")
        print()


def exercise_3():
    """Define governance policies at global and domain levels"""
    governance = {
        "global_policies": {
            "data_privacy": [
                "GDPR compliance for EU customers",
                "CCPA compliance for CA customers",
                "PII must be encrypted at rest",
                "PII access requires justification"
            ],
            "security": [
                "All data encrypted in transit (TLS)",
                "Role-based access control (RBAC)",
                "Audit logs for all access",
                "MFA for production access"
            ],
            "interoperability": [
                "Use Delta Lake or Iceberg format",
                "Standard schema registry",
                "ISO 8601 for dates",
                "UTF-8 encoding"
            ],
            "quality": [
                "Minimum 95% completeness",
                "Maximum 1% error rate",
                "Freshness SLA defined",
                "Schema versioning required"
            ]
        },
        
        "domain_policies": {
            "sales": {
                "pii_fields": ["customer_email", "customer_phone", "shipping_address"],
                "retention": "7 years (legal requirement)",
                "access": "Sales team + approved analysts"
            },
            "finance": {
                "audit_trail": "All changes logged for 10 years",
                "reconciliation": "Daily reconciliation with source",
                "access": "Finance team only (SOX compliance)"
            },
            "marketing": {
                "consent": "Marketing consent required for contact data",
                "retention": "2 years after last interaction",
                "access": "Marketing team + approved partners"
            }
        }
    }
    
    print("=== Federated Governance Policies ===\n")
    print("GLOBAL POLICIES:")
    for category, policies in governance["global_policies"].items():
        print(f"\n{category.upper()}:")
        for policy in policies:
            print(f"  ✓ {policy}")
    
    print("\n\nDOMAIN-SPECIFIC POLICIES:")
    for domain, policies in governance["domain_policies"].items():
        print(f"\n{domain.upper()}:")
        for key, value in policies.items():
            print(f"  {key}: {value}")


def exercise_4():
    """Design self-serve data platform"""
    platform = {
        "infrastructure": {
            "storage": "S3 + Delta Lake",
            "compute": "Spark on EMR/Databricks",
            "orchestration": "Airflow",
            "catalog": "AWS Glue / Unity Catalog",
            "monitoring": "Datadog + custom dashboards"
        },
        
        "self_serve_apis": {
            "create_product": {
                "endpoint": "POST /api/v1/products",
                "payload": {"name", "domain", "schema", "owner"},
                "response": "product_id, storage_path, catalog_url"
            },
            "publish_data": {
                "endpoint": "POST /api/v1/products/{id}/publish",
                "payload": {"data_path", "version"},
                "response": "publish_status, validation_results"
            },
            "request_access": {
                "endpoint": "POST /api/v1/products/{id}/access",
                "payload": {"requester", "justification"},
                "response": "request_id, approval_status"
            }
        },
        
        "templates": [
            "Batch ingestion pipeline (Spark)",
            "Streaming ingestion (Kafka → Delta)",
            "Data quality checks (Great Expectations)",
            "Monitoring dashboard (Grafana)",
            "CI/CD pipeline (GitHub Actions)"
        ],
        
        "capabilities": [
            "One-click data product creation",
            "Automated quality checks",
            "Self-service access requests",
            "Built-in monitoring",
            "Schema evolution support",
            "Cost tracking per product",
            "Lineage visualization"
        ]
    }
    
    print("=== Self-Serve Data Platform ===\n")
    for section, content in platform.items():
        print(f"{section.upper()}:")
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"  {key}:")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"    {value}")
        elif isinstance(content, list):
            for item in content:
                print(f"  ✓ {item}")
        print()


def exercise_5():
    """Plan migration from centralized to data mesh"""
    migration_plan = {
        "current_state": {
            "architecture": "Centralized data lake",
            "team_size": "15 data engineers",
            "domains": "5 business units",
            "data_products": "~50 datasets",
            "bottleneck": "Central team overwhelmed, 3-month backlog"
        },
        
        "phases": {
            "Phase 1 - Pilot (3 months)": [
                "Select Sales domain as pilot",
                "Build platform MVP",
                "Create 3 data products",
                "Train sales team",
                "Establish governance"
            ],
            "Phase 2 - Expand (6 months)": [
                "Onboard Marketing and Product domains",
                "Enhance platform based on feedback",
                "Create 10 more data products",
                "Implement federated governance",
                "Build data catalog"
            ],
            "Phase 3 - Scale (12 months)": [
                "Onboard remaining domains",
                "Full platform automation",
                "Migrate all datasets",
                "Decommission central pipelines",
                "Continuous improvement"
            ]
        },
        
        "risks": {
            "Skills gap": "Domain teams lack data engineering skills → Provide training",
            "Platform complexity": "Self-serve platform is hard → Start simple, iterate",
            "Governance": "Balancing autonomy and standards → Clear policies, automation",
            "Resistance": "Teams prefer central model → Show quick wins, executive support",
            "Duplication": "Multiple teams create similar products → Catalog, discovery tools"
        },
        
        "costs": {
            "Platform development": "$500K (engineers, tools)",
            "Training": "$100K (workshops, certifications)",
            "Infrastructure": "$200K/year (cloud, tools)",
            "Total Year 1": "$800K",
            "Expected savings": "$400K/year (reduced central team, faster delivery)"
        }
    }
    
    print("=== Data Mesh Migration Plan ===\n")
    for section, content in migration_plan.items():
        print(f"{section.upper()}:")
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    • {item}")
                else:
                    print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    print("Day 17: Data Mesh - Solutions\n")
    
    print("Exercise 1: Domain Identification")
    exercise_1()
    
    print("\nExercise 2: Data Product Design")
    exercise_2()
    
    print("\nExercise 3: Governance Policies")
    exercise_3()
    
    print("\nExercise 4: Platform Capabilities")
    exercise_4()
    
    print("\nExercise 5: Migration Planning")
    exercise_5()
