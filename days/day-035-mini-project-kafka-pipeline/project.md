# Real-Time Kafka Pipeline - Project Specification

## Project Structure

```
day-035-mini-project-kafka-pipeline/
├── README.md
├── project.md
├── event_generator.py
├── order_processor.py
├── analytics_engine.py
├── data_sink.py
├── test_pipeline.sh
├── requirements.txt
├── output/
│   └── metrics.json
└── bonus_airflow/
    ├── README.md
    └── pipeline_dag.py
```

---

## Component Specifications

### 1. Event Generator (`event_generator.py`)

**Purpose**: Generate realistic order events

**Implementation**:
```python
from kafka import KafkaProducer
import json
import random
import time
from datetime import datetime

class EventGenerator:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        self.products = ['prod-1', 'prod-2', 'prod-3', 'prod-4', 'prod-5']
    
    def generate_order(self):
        return {
            'order_id': f'order-{int(time.time() * 1000)}',
            'user_id': f'user-{random.randint(1, 100)}',
            'product_id': random.choice(self.products),
            'amount': round(random.uniform(10, 500), 2),
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self, duration=60):
        start = time.time()
        count = 0
        
        while time.time() - start < duration:
            order = self.generate_order()
            self.producer.send('orders', order)
            count += 1
            
            if count % 100 == 0:
                print(f"Generated {count} orders")
            
            time.sleep(random.uniform(0.02, 0.1))  # 10-50 events/sec
        
        self.producer.flush()
        self.producer.close()
        print(f"Total orders generated: {count}")
```

---

### 2. Order Processor (`order_processor.py`)

**Purpose**: Validate and enrich orders

**Implementation**:
```python
from kafka import KafkaConsumer, KafkaProducer
import json

class OrderProcessor:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'orders',
            bootstrap_servers=['localhost:9092'],
            group_id='order-processors',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=False
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.product_names = {
            'prod-1': 'Laptop',
            'prod-2': 'Phone',
            'prod-3': 'Tablet',
            'prod-4': 'Monitor',
            'prod-5': 'Keyboard'
        }
    
    def validate(self, order):
        return (
            order.get('amount', 0) > 0 and
            order.get('product_id') in self.product_names
        )
    
    def enrich(self, order):
        order['product_name'] = self.product_names.get(order['product_id'])
        order['processed_at'] = datetime.now().isoformat()
        return order
    
    def run(self):
        batch = []
        
        for message in self.consumer:
            order = message.value
            
            if self.validate(order):
                enriched = self.enrich(order)
                self.producer.send('processed-orders', enriched)
                batch.append(message)
                
                if len(batch) >= 50:
                    self.consumer.commit()
                    print(f"Processed batch of {len(batch)} orders")
                    batch = []
```

---

### 3. Analytics Engine (`analytics_engine.py`)

**Purpose**: Calculate windowed metrics

**Implementation**:
```python
from kafka import KafkaConsumer, KafkaProducer
from collections import defaultdict
import json
import time

class AnalyticsEngine:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'processed-orders',
            bootstrap_servers=['localhost:9092'],
            group_id='analytics-engine',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.window_size = 60  # 1 minute
        self.windows = defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'total': 0
        }))
    
    def get_window(self, timestamp):
        ts = time.mktime(time.strptime(timestamp[:19], '%Y-%m-%dT%H:%M:%S'))
        return int(ts // self.window_size) * self.window_size
    
    def process(self, order):
        window_start = self.get_window(order['timestamp'])
        product_id = order['product_id']
        amount = order['amount']
        
        stats = self.windows[window_start][product_id]
        stats['count'] += 1
        stats['total'] += amount
        stats['avg'] = stats['total'] / stats['count']
        
        return {
            'window_start': window_start,
            'window_end': window_start + self.window_size,
            'product_id': product_id,
            'product_name': order['product_name'],
            'order_count': stats['count'],
            'total_revenue': round(stats['total'], 2),
            'avg_order_value': round(stats['avg'], 2)
        }
    
    def run(self):
        for message in self.consumer:
            metrics = self.process(message.value)
            self.producer.send('product-metrics', metrics)
```

---

### 4. Data Sink (`data_sink.py`)

**Purpose**: Write metrics to file

**Implementation**:
```python
from kafka import KafkaConsumer
import json
import os

class DataSink:
    def __init__(self, output_file='output/metrics.json'):
        self.consumer = KafkaConsumer(
            'product-metrics',
            bootstrap_servers=['localhost:9092'],
            group_id='data-sink',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        os.makedirs('output', exist_ok=True)
        self.output_file = output_file
    
    def run(self):
        with open(self.output_file, 'a') as f:
            for message in self.consumer:
                metrics = message.value
                f.write(json.dumps(metrics) + '\n')
                f.flush()
                print(f"Wrote metrics: {metrics['product_id']} - "
                      f"${metrics['total_revenue']}")
```

---

## Testing

### Test Script (`test_pipeline.sh`)

```bash
#!/bin/bash
set -e

echo "Starting Kafka Pipeline Test..."

# Start Kafka
echo "Starting Kafka..."
docker run -d --name kafka -p 9092:9092 apache/kafka:latest
sleep 15

# Create topics
echo "Creating topics..."
docker exec kafka kafka-topics --create --topic orders \
  --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

docker exec kafka kafka-topics --create --topic processed-orders \
  --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

docker exec kafka kafka-topics --create --topic product-metrics \
  --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Start pipeline components
echo "Starting pipeline components..."
python event_generator.py &
PID_GEN=$!

sleep 2
python order_processor.py &
PID_PROC=$!

sleep 2
python analytics_engine.py &
PID_ANALYTICS=$!

sleep 2
python data_sink.py &
PID_SINK=$!

# Run for 60 seconds
echo "Running pipeline for 60 seconds..."
sleep 60

# Stop components
echo "Stopping components..."
kill $PID_GEN $PID_PROC $PID_ANALYTICS $PID_SINK 2>/dev/null || true

# Verify output
echo "Verifying output..."
if [ -f "output/metrics.json" ]; then
    echo "✓ Output file created"
    echo "Sample metrics:"
    head -5 output/metrics.json
else
    echo "✗ Output file not found"
    exit 1
fi

# Cleanup
echo "Cleaning up..."
docker stop kafka
docker rm kafka

echo "Test completed successfully!"
```

---

## Requirements (`requirements.txt`)

```
kafka-python==2.0.2
```

---

## Bonus: Airflow Integration

`bonus_airflow/pipeline_dag.py`:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'kafka_pipeline',
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False
)

start_kafka = BashOperator(
    task_id='start_kafka',
    bash_command='docker start kafka || docker run -d --name kafka -p 9092:9092 apache/kafka:latest',
    dag=dag
)

run_generator = BashOperator(
    task_id='run_generator',
    bash_command='python /path/to/event_generator.py',
    dag=dag
)

run_processor = BashOperator(
    task_id='run_processor',
    bash_command='python /path/to/order_processor.py',
    dag=dag
)

run_analytics = BashOperator(
    task_id='run_analytics',
    bash_command='python /path/to/analytics_engine.py',
    dag=dag
)

run_sink = BashOperator(
    task_id='run_sink',
    bash_command='python /path/to/data_sink.py',
    dag=dag
)

start_kafka >> [run_generator, run_processor, run_analytics, run_sink]
```

---

## Success Metrics

- **Throughput**: 10-50 events/second
- **Latency**: < 1 second end-to-end
- **Accuracy**: 100% of valid orders processed
- **Reliability**: No data loss
- **Error Rate**: < 0.1%

---

## Troubleshooting

**Issue**: Kafka not starting
- Solution: Check Docker, increase wait time

**Issue**: No events flowing
- Solution: Verify topics created, check producer/consumer configs

**Issue**: Metrics incorrect
- Solution: Check windowing logic, verify calculations

**Issue**: Output file empty
- Solution: Check consumer group, verify topic has data
