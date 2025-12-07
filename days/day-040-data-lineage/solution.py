"""
Day 40: Data Lineage - Solutions
"""
from datetime import datetime


class LineageNode:
    """Represents a dataset in lineage graph"""
    
    def __init__(self, name, node_type='dataset'):
        self.name = name
        self.node_type = node_type
        self.inputs = []
        self.outputs = []
        self.metadata = {}
    
    def add_input(self, node):
        """Add upstream dependency"""
        self.inputs.append(node)
        node.outputs.append(self)
    
    def add_metadata(self, key, value):
        """Add metadata"""
        self.metadata[key] = value


class LineageGraph:
    """Manages lineage relationships"""
    
    def __init__(self):
        self.nodes = {}
    
    def add_node(self, name, node_type='dataset'):
        """Add node to graph"""
        if name not in self.nodes:
            self.nodes[name] = LineageNode(name, node_type)
        return self.nodes[name]
    
    def add_edge(self, source, target):
        """Add lineage relationship"""
        source_node = self.add_node(source)
        target_node = self.add_node(target)
        target_node.add_input(source_node)
    
    def get_upstream(self, name, depth=None):
        """Get all upstream dependencies"""
        if name not in self.nodes:
            return []
        
        visited = set()
        result = []
        
        def traverse(node, current_depth=0):
            if depth and current_depth >= depth:
                return
            if node.name in visited:
                return
            
            visited.add(node.name)
            for input_node in node.inputs:
                result.append(input_node.name)
                traverse(input_node, current_depth + 1)
        
        traverse(self.nodes[name])
        return result
    
    def get_downstream(self, name, depth=None):
        """Get all downstream consumers"""
        if name not in self.nodes:
            return []
        
        visited = set()
        result = []
        
        def traverse(node, current_depth=0):
            if depth and current_depth >= depth:
                return
            if node.name in visited:
                return
            
            visited.add(node.name)
            for output_node in node.outputs:
                result.append(output_node.name)
                traverse(output_node, current_depth + 1)
        
        traverse(self.nodes[name])
        return result


class ColumnLineage:
    """Track column-level lineage"""
    
    def __init__(self):
        self.lineage = {}
    
    def add_transformation(self, source_table, source_col, target_table, target_col, transformation):
        """Record column transformation"""
        key = f"{target_table}.{target_col}"
        
        if key not in self.lineage:
            self.lineage[key] = []
        
        self.lineage[key].append({
            'source_table': source_table,
            'source_column': source_col,
            'transformation': transformation
        })
    
    def get_column_lineage(self, table, column):
        """Get lineage for specific column"""
        key = f"{table}.{column}"
        return self.lineage.get(key, [])
    
    def trace_to_source(self, table, column):
        """Trace column to original sources"""
        key = f"{table}.{column}"
        
        if key not in self.lineage:
            return [(table, column)]
        
        sources = []
        for lineage_entry in self.lineage[key]:
            upstream = self.trace_to_source(
                lineage_entry['source_table'],
                lineage_entry['source_column']
            )
            sources.extend(upstream)
        
        return sources


def exercise_1_build_lineage_graph():
    """Build lineage graph"""
    graph = LineageGraph()
    
    # Add edges for pipeline
    graph.add_edge('customers_raw', 'customers_cleaned')
    graph.add_edge('customers_cleaned', 'customers_enriched')
    graph.add_edge('orders_raw', 'orders_cleaned')
    graph.add_edge('customers_enriched', 'customer_analytics')
    graph.add_edge('orders_cleaned', 'customer_analytics')
    
    return graph


def exercise_2_track_column_lineage():
    """Track column lineage"""
    col_lineage = ColumnLineage()
    
    # Track transformations
    col_lineage.add_transformation(
        'customers_raw', 'email',
        'customers_cleaned', 'email_normalized',
        'LOWER(TRIM(email))'
    )
    
    col_lineage.add_transformation(
        'customers_raw', 'first_name',
        'customers_cleaned', 'full_name',
        'CONCAT(first_name, " ", last_name)'
    )
    
    col_lineage.add_transformation(
        'customers_cleaned', 'email_normalized',
        'customer_analytics', 'contact_email',
        'email_normalized'
    )
    
    return col_lineage


def exercise_3_extract_sql_lineage(sql_query):
    """Extract lineage from SQL"""
    sql_upper = sql_query.upper()
    sources = []
    targets = []
    
    # Extract FROM tables
    if 'FROM' in sql_upper:
        from_idx = sql_upper.index('FROM')
        from_part = sql_query[from_idx:].split('WHERE')[0].split('JOIN')[0]
        table = from_part.replace('FROM', '').strip().split()[0]
        sources.append(table)
    
    # Extract JOIN tables
    if 'JOIN' in sql_upper:
        parts = sql_query.split('JOIN')
        for part in parts[1:]:
            table = part.split('ON')[0].strip().split()[0]
            sources.append(table)
    
    # Extract target
    if 'INSERT INTO' in sql_upper:
        target = sql_query.split('INSERT INTO')[1].strip().split()[0]
        targets.append(target)
    elif 'CREATE TABLE' in sql_upper:
        target = sql_query.split('CREATE TABLE')[1].strip().split()[0]
        targets.append(target)
    
    return {
        'sources': sources,
        'targets': targets
    }


def exercise_4_impact_analysis(lineage_graph, changed_table):
    """Analyze downstream impact"""
    downstream = lineage_graph.get_downstream(changed_table)
    direct = lineage_graph.get_downstream(changed_table, depth=1)
    indirect = [t for t in downstream if t not in direct]
    
    return {
        'changed_table': changed_table,
        'directly_affected': direct,
        'indirectly_affected': indirect,
        'total_affected': len(downstream)
    }


def exercise_5_upstream_tracing(lineage_graph, target_table):
    """Trace upstream sources"""
    return lineage_graph.get_upstream(target_table)


def exercise_6_lineage_visualization(lineage_graph):
    """Visualize lineage"""
    viz = "Lineage Graph:\n" + "=" * 50 + "\n\n"
    
    for name, node in lineage_graph.nodes.items():
        viz += f"{name}\n"
        
        if node.inputs:
            viz += f"  ← Inputs: {', '.join([n.name for n in node.inputs])}\n"
        
        if node.outputs:
            viz += f"  → Outputs: {', '.join([n.name for n in node.outputs])}\n"
        
        viz += "\n"
    
    return viz


def exercise_7_metadata_enrichment(lineage_graph):
    """Add metadata to lineage"""
    # Add sample metadata
    metadata_map = {
        'customers_raw': {
            'owner': 'data-team',
            'description': 'Raw customer data from CRM',
            'row_count': 100000,
            'updated_at': datetime.now()
        },
        'customers_cleaned': {
            'owner': 'data-team',
            'description': 'Cleaned and validated customer data',
            'row_count': 95000,
            'updated_at': datetime.now()
        }
    }
    
    for name, metadata in metadata_map.items():
        if name in lineage_graph.nodes:
            for key, value in metadata.items():
                lineage_graph.nodes[name].add_metadata(key, value)
    
    return lineage_graph


def exercise_8_lineage_report(lineage_graph, table_name):
    """Generate lineage report"""
    if table_name not in lineage_graph.nodes:
        return f"Table {table_name} not found in lineage graph"
    
    node = lineage_graph.nodes[table_name]
    upstream = lineage_graph.get_upstream(table_name)
    downstream = lineage_graph.get_downstream(table_name)
    
    report = f"""
{'='*60}
LINEAGE REPORT: {table_name}
{'='*60}

UPSTREAM SOURCES ({len(upstream)}):
{'-'*60}
"""
    
    for source in upstream:
        report += f"  • {source}\n"
    
    report += f"""
DOWNSTREAM CONSUMERS ({len(downstream)}):
{'-'*60}
"""
    
    for consumer in downstream:
        report += f"  • {consumer}\n"
    
    report += f"""
METADATA:
{'-'*60}
"""
    
    for key, value in node.metadata.items():
        report += f"  {key}: {value}\n"
    
    report += f"\n{'='*60}\n"
    
    return report



if __name__ == "__main__":
    print("Day 40: Data Lineage - Solutions\n")
    print("=" * 70)
    
    print("\n📋 Exercise 1: Build Lineage Graph")
    print("-" * 70)
    graph = exercise_1_build_lineage_graph()
    print(f"Created graph with {len(graph.nodes)} nodes")
    for name in graph.nodes:
        print(f"  • {name}")
    
    print("\n📋 Exercise 2: Track Column Lineage")
    print("-" * 70)
    col_lineage = exercise_2_track_column_lineage()
    print(f"Tracked {len(col_lineage.lineage)} column transformations")
    
    # Trace email column
    sources = col_lineage.trace_to_source('customer_analytics', 'contact_email')
    print(f"contact_email traces to: {sources}")
    
    print("\n📋 Exercise 3: Extract SQL Lineage")
    print("-" * 70)
    sql = "INSERT INTO customer_analytics SELECT * FROM customers_cleaned JOIN orders_cleaned ON customers_cleaned.id = orders_cleaned.customer_id"
    lineage = exercise_3_extract_sql_lineage(sql)
    print(f"Sources: {lineage['sources']}")
    print(f"Targets: {lineage['targets']}")
    
    print("\n📋 Exercise 4: Impact Analysis")
    print("-" * 70)
    impact = exercise_4_impact_analysis(graph, 'customers_cleaned')
    print(f"Changed table: {impact['changed_table']}")
    print(f"Directly affected: {impact['directly_affected']}")
    print(f"Indirectly affected: {impact['indirectly_affected']}")
    print(f"Total affected: {impact['total_affected']}")
    
    print("\n📋 Exercise 5: Upstream Tracing")
    print("-" * 70)
    sources = exercise_5_upstream_tracing(graph, 'customer_analytics')
    print(f"Upstream sources for customer_analytics:")
    for source in sources:
        print(f"  • {source}")
    
    print("\n📋 Exercise 6: Lineage Visualization")
    print("-" * 70)
    viz = exercise_6_lineage_visualization(graph)
    print(viz)
    
    print("\n📋 Exercise 7: Metadata Enrichment")
    print("-" * 70)
    enriched = exercise_7_metadata_enrichment(graph)
    print("Added metadata to lineage nodes")
    for name, node in enriched.nodes.items():
        if node.metadata:
            print(f"  {name}: {len(node.metadata)} metadata fields")
    
    print("\n📋 Exercise 8: Lineage Report")
    print("-" * 70)
    report = exercise_8_lineage_report(graph, 'customer_analytics')
    print(report)
    
    print("=" * 70)
    print("✅ All exercises completed!")
    print("=" * 70)
