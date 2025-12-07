"""
Day 17: Data Mesh - Exercises
"""


# Exercise 1: Domain Identification
def exercise_1():
    """Identify domains and data products for e-commerce company"""
    # TODO: List business domains (sales, marketing, product, finance, etc.)
    # TODO: Define domain boundaries
    # TODO: Identify data products per domain
    #       Sales: orders, customers, revenue
    #       Marketing: campaigns, leads, conversions
    #       Product: catalog, inventory, reviews
    # TODO: Map data flows between domains
    pass


# Exercise 2: Data Product Design
def exercise_2():
    """Design a data product with full specifications"""
    # TODO: Define data product name and owner
    # TODO: Specify schema (fields, types)
    # TODO: Define SLAs (availability, latency, freshness)
    # TODO: List quality checks (no duplicates, no nulls, valid ranges)
    # TODO: Specify access policies
    # TODO: Define monitoring metrics
    pass


# Exercise 3: Governance Policies
def exercise_3():
    """Define governance policies at global and domain levels"""
    # TODO: Define global policies
    #       - Data privacy (GDPR compliance)
    #       - Security standards (encryption, access control)
    #       - Interoperability (standard formats)
    
    # TODO: Define domain-specific policies
    #       - Sales: PII handling, retention
    #       - Finance: audit requirements
    
    # TODO: Implement policy validation function
    pass


# Exercise 4: Platform Capabilities
def exercise_4():
    """Design self-serve data platform"""
    # TODO: List required platform capabilities
    #       - Data pipeline templates
    #       - Storage (S3, Delta Lake)
    #       - Compute (Spark, Airflow)
    #       - Catalog (metadata management)
    #       - Monitoring (data quality, SLAs)
    
    # TODO: Design self-serve APIs
    #       - Create data product
    #       - Publish data
    #       - Request access
    
    # TODO: Plan infrastructure
    pass


# Exercise 5: Migration Planning
def exercise_5():
    """Plan migration from centralized to data mesh"""
    # TODO: Assess current state
    #       - Central data team size
    #       - Number of domains
    #       - Data products count
    
    # TODO: Define migration phases
    #       Phase 1: Pilot with one domain
    #       Phase 2: Expand to 3 domains
    #       Phase 3: Full rollout
    
    # TODO: Identify risks
    #       - Skills gap in domain teams
    #       - Platform complexity
    #       - Governance challenges
    
    # TODO: Estimate timeline and costs
    pass


if __name__ == "__main__":
    print("Day 17: Data Mesh\n")
    
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
