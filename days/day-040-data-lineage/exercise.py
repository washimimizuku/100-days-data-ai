"""
Day 40: Data Lineage - Exercises
"""
from datetime import datetime


def exercise_1_build_lineage_graph():
    """
    Exercise 1: Build lineage graph
    
    Create a lineage graph for this pipeline:
    - customers_raw → customers_cleaned → customers_enriched
    - orders_raw → orders_cleaned
    - customers_enriched + orders_cleaned → customer_analytics
    
    Return the lineage graph
    """
    # TODO: Create LineageGraph instance
    # TODO: Add nodes for each dataset
    # TODO: Add edges for relationships
    # TODO: Return graph
    pass


def exercise_2_track_column_lineage():
    """
    Exercise 2: Track column lineage
    
    Track these transformations:
    1. customers_raw.email → customers_cleaned.email_normalized (LOWER(TRIM(email)))
    2. customers_raw.first_name + last_name → customers_cleaned.full_name (CONCAT)
    3. customers_cleaned.email_normalized → customer_analytics.contact_email
    
    Return column lineage tracker
    """
    # TODO: Create ColumnLineage instance
    # TODO: Add transformation for email
    # TODO: Add transformation for full_name
    # TODO: Add transformation for contact_email
    # TODO: Return tracker
    pass


def exercise_3_extract_sql_lineage(sql_query):
    """
    Exercise 3: Extract lineage from SQL
    
    Parse SQL query and extract:
    - Source tables (FROM, JOIN)
    - Target tables (INSERT INTO, CREATE TABLE)
    
    Return dict with sources and targets
    """
    # TODO: Parse SQL query
    # TODO: Extract FROM tables
    # TODO: Extract JOIN tables
    # TODO: Extract target table
    # TODO: Return {sources: [], targets: []}
    pass


def exercise_4_impact_analysis(lineage_graph, changed_table):
    """
    Exercise 4: Analyze downstream impact
    
    Given a table that changed, find:
    - Directly affected tables (depth 1)
    - Indirectly affected tables (depth > 1)
    - Total affected count
    
    Return impact report
    """
    # TODO: Get all downstream tables
    # TODO: Get direct dependencies (depth 1)
    # TODO: Get indirect dependencies
    # TODO: Return impact report dict
    pass


def exercise_5_upstream_tracing(lineage_graph, target_table):
    """
    Exercise 5: Trace upstream sources
    
    Find all upstream sources for a given table.
    
    Return list of source tables
    """
    # TODO: Get upstream dependencies
    # TODO: Return list of source tables
    pass


def exercise_6_lineage_visualization(lineage_graph):
    """
    Exercise 6: Visualize lineage
    
    Generate a text-based visualization of the lineage graph.
    
    Return formatted string representation
    """
    # TODO: Iterate through nodes
    # TODO: Show relationships with arrows
    # TODO: Format as tree or graph
    # TODO: Return visualization string
    pass


def exercise_7_metadata_enrichment(lineage_graph):
    """
    Exercise 7: Add metadata to lineage
    
    Enrich lineage nodes with metadata:
    - Owner
    - Description
    - Row count
    - Last updated
    
    Return enriched graph
    """
    # TODO: Add metadata to each node
    # TODO: Include owner, description, stats
    # TODO: Return enriched graph
    pass


def exercise_8_lineage_report(lineage_graph, table_name):
    """
    Exercise 8: Generate lineage report
    
    Create comprehensive report for a table:
    - Upstream sources
    - Downstream consumers
    - Transformation details
    - Metadata
    
    Return formatted report string
    """
    # TODO: Get upstream sources
    # TODO: Get downstream consumers
    # TODO: Get metadata
    # TODO: Format as readable report
    # TODO: Return report string
    pass


# Helper classes (implement these)
class LineageNode:
    """Represents a dataset in lineage graph"""
    
    def __init__(self, name, node_type='dataset'):
        # TODO: Initialize node properties
        pass
    
    def add_input(self, node):
        # TODO: Add upstream dependency
        pass


class LineageGraph:
    """Manages lineage relationships"""
    
    def __init__(self):
        # TODO: Initialize graph
        pass
    
    def add_node(self, name, node_type='dataset'):
        # TODO: Add node to graph
        pass
    
    def add_edge(self, source, target):
        # TODO: Add lineage relationship
        pass
    
    def get_upstream(self, name, depth=None):
        # TODO: Get upstream dependencies
        pass
    
    def get_downstream(self, name, depth=None):
        # TODO: Get downstream consumers
        pass


class ColumnLineage:
    """Track column-level lineage"""
    
    def __init__(self):
        # TODO: Initialize column lineage tracker
        pass
    
    def add_transformation(self, source_table, source_col, target_table, target_col, transformation):
        # TODO: Record column transformation
        pass
    
    def get_column_lineage(self, table, column):
        # TODO: Get lineage for specific column
        pass
    
    def trace_to_source(self, table, column):
        # TODO: Trace column to original sources
        pass


if __name__ == "__main__":
    print("Day 40: Data Lineage - Exercises\n")
    
    # Uncomment to run exercises
    # print("Exercise 1: Build Lineage Graph")
    # graph = exercise_1_build_lineage_graph()
    
    # print("\nExercise 2: Track Column Lineage")
    # col_lineage = exercise_2_track_column_lineage()
    
    # print("\nExercise 3: Extract SQL Lineage")
    # sql = "INSERT INTO customer_analytics SELECT * FROM customers_cleaned JOIN orders_cleaned ON customers_cleaned.id = orders_cleaned.customer_id"
    # lineage = exercise_3_extract_sql_lineage(sql)
    # print(lineage)
    
    # print("\nExercise 4: Impact Analysis")
    # impact = exercise_4_impact_analysis(graph, 'customers_cleaned')
    # print(impact)
    
    # print("\nExercise 5: Upstream Tracing")
    # sources = exercise_5_upstream_tracing(graph, 'customer_analytics')
    # print(sources)
    
    # print("\nExercise 6: Lineage Visualization")
    # viz = exercise_6_lineage_visualization(graph)
    # print(viz)
    
    # print("\nExercise 7: Metadata Enrichment")
    # enriched = exercise_7_metadata_enrichment(graph)
    
    # print("\nExercise 8: Lineage Report")
    # report = exercise_8_lineage_report(graph, 'customer_analytics')
    # print(report)
