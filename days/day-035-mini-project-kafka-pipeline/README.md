# Day 35: Mini Project - Real-Time Kafka Pipeline

## 🎯 Project Overview

Build an **end-to-end real-time data pipeline** using Kafka ecosystem:
- **Kafka Producers** to generate event data
- **Kafka Consumers** to process events
- **Kafka Streams** for real-time transformations
- **Kafka Connect** for data integration

**Time**: 2 hours  
**Difficulty**: Intermediate  
**Topics**: Kafka, Real-time Processing, Stream Analytics

---

## 📋 Project Requirements

### Scenario
Build a real-time e-commerce analytics pipeline that:
1. Generates order events (producer)
2. Processes orders in real-time (consumer)
3. Calculates metrics with windowing (Kafka Streams)
4. Stores results in files/database (Kafka Connect)

### Components

**1. Event Generator** (Producer)
- Generate order events: `{order_id, user_id, product_id, amount, timestamp}`
- Send to `orders` topic
- Simulate realistic traffic patterns

**2. Order Processor** (Consumer)
- Consume from `orders` topic
- Validate orders
- Enrich with product data
- Send to `processed-orders` topic

**3. Analytics Engine** (Kafka Streams)
- Calculate metrics per product:
  - Order count
  - Total revenue
  - Average order value
- Use 1-minute tumbling windows
- Send to `product-metrics` topic

**4. Data Sink** (Kafka Connect)
- Write metrics to output file
- Optional: Write to database

---

## 🏗️ Architecture

```
Event Generator (Producer)
         ↓
    orders topic
         ↓
Order Processor (Consumer)
         ↓
processed-orders topic
         ↓
Analytics Engine (Kafka Streams)
         ↓
product-metrics topic
         ↓
Data Sink (Kafka Connect)
         ↓
    Output File/DB
```

---

## 💻 Implementation Tasks

### Task 1: Event Generator (30 min)

Create `event_generator.py`:
- Generate realistic order events
- Random user_id, product_id, amounts
- Send to Kafka with proper serialization
- Include error handling

**Requirements**:
- 10-50 events per second
- 5 different products
- Amounts between $10-$500
- Proper logging

### Task 2: Order Processor (30 min)

Create `order_processor.py`:
- Consume from `orders` topic
- Validate order data (amount > 0, valid IDs)
- Enrich with product names
- Handle errors with DLQ
- Manual offset commits

**Requirements**:
- Consumer group for scalability
- Batch processing (50 messages)
- Error handling with retries
- Metrics logging

### Task 3: Analytics Engine (40 min)

Create `analytics_engine.py`:
- Implement stream processing
- Group by product_id
- Calculate windowed metrics
- Emit results per window

**Requirements**:
- 1-minute tumbling windows
- Count, sum, average calculations
- State management
- Window closing logic

### Task 4: Data Sink (20 min)

Create `data_sink.py`:
- Consume from `product-metrics`
- Write to JSON file
- Optional: Kafka Connect configuration

**Requirements**:
- Append to file
- JSON format
- Proper file handling
- Graceful shutdown

---

## 🧪 Testing

### Test Script

Create `test_pipeline.sh`:
```bash
#!/bin/bash

# Start Kafka
docker run -d --name kafka -p 9092:9092 apache/kafka:latest

# Wait for Kafka
sleep 10

# Create topics
kafka-topics --create --topic orders --bootstrap-server localhost:9092
kafka-topics --create --topic processed-orders --bootstrap-server localhost:9092
kafka-topics --create --topic product-metrics --bootstrap-server localhost:9092

# Run pipeline components
python event_generator.py &
python order_processor.py &
python analytics_engine.py &
python data_sink.py &

# Wait and check results
sleep 60

# Verify output
cat output/metrics.json

# Cleanup
pkill -f python
docker stop kafka
docker rm kafka
```

### Validation

Verify:
- [ ] Events generated successfully
- [ ] Orders processed without errors
- [ ] Metrics calculated correctly
- [ ] Output file contains results
- [ ] No data loss
- [ ] Proper error handling

---

## 📊 Expected Output

`output/metrics.json`:
```json
{
  "window_start": 1704470400,
  "window_end": 1704470460,
  "product_id": "prod-1",
  "order_count": 45,
  "total_revenue": 12450.50,
  "avg_order_value": 276.68
}
```

---

## 🎁 Bonus Challenges (Optional)

### Bonus 1: Airflow Orchestration (30 min)
Add Airflow DAG to orchestrate pipeline:
- Start/stop components
- Monitor health
- Handle failures

See `bonus_airflow/README.md`

### Bonus 2: Docker Compose (15 min)
Create `docker-compose.yml`:
- Kafka
- Zookeeper
- Kafka Connect
- All pipeline components

### Bonus 3: Monitoring Dashboard (20 min)
Add monitoring:
- Consumer lag
- Throughput metrics
- Error rates
- Grafana dashboard

---

## 📝 Deliverables

1. `event_generator.py` - Event producer
2. `order_processor.py` - Order consumer
3. `analytics_engine.py` - Stream processor
4. `data_sink.py` - Metrics writer
5. `test_pipeline.sh` - Test script
6. `requirements.txt` - Dependencies
7. `README.md` - Documentation
8. `output/metrics.json` - Results

---

## 🎯 Learning Outcomes

After completing this project, you will:
- Build end-to-end Kafka pipelines
- Implement producers with proper configuration
- Create consumers with error handling
- Use Kafka Streams for analytics
- Integrate with Kafka Connect
- Handle real-time data at scale
- Monitor and test pipelines

---

## 📚 Resources

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [kafka-python](https://kafka-python.readthedocs.io/)
- [Kafka Streams](https://kafka.apache.org/documentation/streams/)
- [Kafka Connect](https://kafka.apache.org/documentation/#connect)

---

## ✅ Success Criteria

- [ ] All components run without errors
- [ ] Events flow through entire pipeline
- [ ] Metrics calculated correctly
- [ ] Output file generated
- [ ] Error handling works
- [ ] Code is well-documented
- [ ] Tests pass

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Start Kafka
docker run -d --name kafka -p 9092:9092 apache/kafka:latest

# Run test
./test_pipeline.sh

# Or run components individually
python event_generator.py
python order_processor.py
python analytics_engine.py
python data_sink.py
```

---

## Tomorrow: Day 36 - Data Quality Dimensions

Learn about data quality frameworks and the six dimensions of data quality.
