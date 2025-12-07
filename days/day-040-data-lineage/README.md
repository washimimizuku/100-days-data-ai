# Day 40: Data Lineage

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand what data lineage is and why it's critical
- Learn to track data flow through systems
- Implement lineage tracking in Python
- Understand column-level vs table-level lineage
- Use lineage for impact analysis and debugging
- Learn about lineage tools (OpenLineage, Apache Atlas)

---

## What is Data Lineage?

Data lineage is the documentation of data's journey through systems - where it comes from, how it's transformed, and where it goes. It answers:

- **Where did this data come from?** (upstream sources)
- **How was it transformed?** (processing steps)
- **Where is it used?** (downstream consumers)
- **Who owns it?** (data ownership)
- **When was it updated?** (temporal tracking)

**Why lineage matters:**
- **Debugging**: Trace data quality issues to their source
- **Impact analysis**: Understand downstream effects of changes
- **Compliance**: Prove data provenance for regulations (GDPR, CCPA)
- **Trust**: Build confidence in data accuracy
- **Optimization**: Identify redundant pipelines

---

## Types of Lineage

### 1. Table-Level Lineage
Tracks relationships between tables/datasets:
```
customers_raw → customers_cleaned → customers_enriched → customer_analytics
```

### 2. Column-Level Lineage
Tracks individual column transformations:
```
customers_raw.email → customers_cleaned.email_normalized → customer_analytics.contact_email
```

### 3. Job-Level Lineage
Tracks which jobs/processes transform data:
```
extract_job → transform_job → load_job → aggregate_job
```

---

## Basic Lineage Tracking

### Simple Lineage Graph
```python
class LineageNode:
    """Represents a dataset in lineage graph"""
    
    def __init__(self, name, node_type='dataset'):
        self.name = name
        self.node_type = node_type  # dataset, transformation, job
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
```

---

## Column-Level Lineage

### Track Column Transformations
```python
class ColumnLineage:
    """Track column-level lineage"""
    
    def __init__(self):
        self.lineage = {}
    
    def add_transformation(self, source_table, source_col, target_table, target_col, transformation):
        key = f"{target_table}.{target_col}"
        if key not in self.lineage:
            self.lineage[key] = []
        self.lineage[key].append({
            'source_table': source_table,
            'source_column': source_col,
            'transformation': transformation
        })
    
    def trace_to_source(self, table, column):
        """Recursively trace column to original sources"""
        key = f"{table}.{column}"
        if key not in self.lineage:
            return [(table, column)]
        
        sources = []
        for entry in self.lineage[key]:
            upstream = self.trace_to_source(entry['source_table'], entry['source_column'])
            sources.extend(upstream)
        return sources
```

---

## Lineage from SQL Parsing

### Extract Lineage from SQL
```python
def extract_lineage_from_sql(sql):
    """Extract table lineage from SQL query (simplified)"""
    sources = []
    targets = []
    sql_upper = sql.upper()
    
    # Extract FROM and JOIN tables
    if 'FROM' in sql_upper:
        from_clause = sql[sql_upper.index('FROM'):].split('WHERE')[0]
        tables = [t.strip() for t in from_clause.replace('FROM', '').split('JOIN')]
        sources.extend([t.split()[0] for t in tables if t.strip()])
    
    # Extract INSERT/CREATE target
    if 'INSERT INTO' in sql_upper:
        targets.append(sql_upper.split('INSERT INTO')[1].split()[0])
    elif 'CREATE TABLE' in sql_upper:
        targets.append(sql_upper.split('CREATE TABLE')[1].split()[0])
    
    return {
        'sources': sources,
        'targets': targets
    }
```

---

## Impact Analysis

### Analyze Downstream Impact
```python
def analyze_impact(lineage_graph, changed_table):
    """Analyze impact of changes to a table"""
    downstream = lineage_graph.get_downstream(changed_table)
    
    impact_report = {
        'changed_table': changed_table,
        'directly_affected': [],
        'indirectly_affected': [],
        'total_affected': len(downstream)
    }
    
    # Get direct dependencies (depth 1)
    direct = lineage_graph.get_downstream(changed_table, depth=1)
    impact_report['directly_affected'] = direct
    
    # Get indirect dependencies
    indirect = [t for t in downstream if t not in direct]
    impact_report['indirectly_affected'] = indirect
    
    return impact_report
```

---

## Lineage Metadata

### Capture Rich Metadata
```python
class LineageMetadata:
    """Store lineage metadata"""
    
    def __init__(self):
        self.metadata = {}
    
    def add_dataset_metadata(self, dataset, metadata):
        """Add metadata for a dataset"""
        self.metadata[dataset] = {
            'type': 'dataset',
            'owner': metadata.get('owner'),
            'description': metadata.get('description'),
            'schema': metadata.get('schema'),
            'location': metadata.get('location'),
            'created_at': metadata.get('created_at'),
            'updated_at': metadata.get('updated_at'),
            'row_count': metadata.get('row_count'),
            'size_bytes': metadata.get('size_bytes')
        }
    
    def add_transformation_metadata(self, transformation, metadata):
        """Add metadata for a transformation"""
        self.metadata[transformation] = {
            'type': 'transformation',
            'job_name': metadata.get('job_name'),
            'code': metadata.get('code'),
            'runtime': metadata.get('runtime'),
            'status': metadata.get('status'),
            'executed_at': metadata.get('executed_at')
        }
```

---

## OpenLineage Integration

### OpenLineage Event
```python
from datetime import datetime

def create_openlineage_event(job_name, inputs, outputs):
    """Create OpenLineage-compatible event"""
    event = {
        'eventType': 'COMPLETE',
        'eventTime': datetime.utcnow().isoformat(),
        'job': {
            'namespace': 'my_namespace',
            'name': job_name
        },
        'inputs': [
            {
                'namespace': 'my_namespace',
                'name': input_name,
                'facets': {}
            }
            for input_name in inputs
        ],
        'outputs': [
            {
                'namespace': 'my_namespace',
                'name': output_name,
                'facets': {}
            }
            for output_name in outputs
        ],
        'producer': 'my_producer',
        'schemaURL': 'https://openlineage.io/spec/1-0-0/OpenLineage.json'
    }
    return event
```

---

## 💻 Exercises (40 min)

### Exercise 1: Build Lineage Graph
Create a lineage graph for a simple ETL pipeline.

### Exercise 2: Track Column Lineage
Implement column-level lineage tracking.

### Exercise 3: Extract SQL Lineage
Parse SQL queries to extract lineage automatically.

### Exercise 4: Impact Analysis
Analyze downstream impact of table changes.

### Exercise 5: Upstream Tracing
Trace data back to original sources.

### Exercise 6: Lineage Visualization
Generate a visual representation of lineage.

### Exercise 7: Metadata Enrichment
Add rich metadata to lineage nodes.

### Exercise 8: Lineage Report
Generate a comprehensive lineage report.

---

## ✅ Quiz (5 min)

Test your understanding of data lineage concepts and implementation.

---

## 🎯 Key Takeaways

- **Essential for trust**: Lineage builds confidence in data
- **Two levels**: Table-level and column-level lineage
- **Impact analysis**: Understand downstream effects of changes
- **Debugging tool**: Trace issues to their source
- **Compliance**: Required for many regulations
- **Automation**: Extract lineage from SQL and code

---

## 📚 Resources

- [OpenLineage Documentation](https://openlineage.io/)
- [Apache Atlas](https://atlas.apache.org/)
- [Marquez Lineage](https://marquezproject.ai/)
- [Data Lineage Best Practices](https://www.dataversity.net/data-lineage-best-practices/)

---

## Tomorrow: Day 41 - dbt Basics

Learn dbt (data build tool), a transformation framework that brings software engineering practices to analytics, with built-in lineage, testing, and documentation.
