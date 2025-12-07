# Day 49: Mini Project - Real-Time Analytics with Spark + Kafka

## 🎯 Project Overview

Build a **production-ready real-time analytics system** that combines everything learned in Week 7:
- Spark Structured Streaming
- Kafka integration
- Windowed aggregations
- Stateful processing
- Stream joins
- Watermarking
- Performance optimization

**Time**: 2 hours  
**Difficulty**: Advanced  
**Topics**: Spark Streaming, Kafka, Real-time Analytics, Production Patterns

---

## 📋 Project Requirements

### Scenario

Build a **real-time e-commerce analytics platform** that:
1. Ingests clickstream and transaction data from Kafka
2. Joins streams to enrich data
3. Calculates real-time metrics with windowing
4. Detects anomalies using stateful processing
5. Outputs results to multiple sinks
6. Handles late data and failures gracefully

### Business Requirements

**Metrics to Calculate**:
- Page views per product (1-minute windows)
- Conversion rate (clicks → purchases)
- Revenue per category (5-minute windows)
- User session analytics
- Anomaly detection (unusual purchase patterns)

**SLAs**:
- Latency: < 10 seconds end-to-end
- Throughput: 1000+ events/second
- Accuracy: No data loss, handle late arrivals
- Availability: Fault-tolerant with checkpointing

---

## 🏗️ Architecture

```
Kafka Topics                Spark Streaming              Outputs
-----------                 ---------------              -------
clickstream    ─┐
                ├──> Join ──> Enrich ──> Aggregate ──> Console
transactions   ─┘                │                       Kafka
                                 │                       Files
                                 └──> Stateful ──────> Monitoring
```

### Data Flow

1. **Input**: Two Kafka topics (clickstream, transactions)
2. **Processing**:
   - Stream-to-stream join (enrich clicks with transactions)
   - Windowed aggregations (metrics per time window)
   - Stateful processing (session tracking, anomaly detection)
3. **Output**: Multiple sinks (console, Kafka, files)

---

## 💻 Implementation

### Part 1: Data Ingestion (20 min)

Create Kafka producers to generate realistic data.

**File**: `data_generator.py`

**Requirements**:
- Generate clickstream events (user_id, product_id, action, timestamp)
- Generate transaction events (user_id, product_id, amount, timestamp)
- Realistic patterns (80% clicks, 20% purchases)
- 100-500 events/second
- Proper error handling

### Part 2: Stream Processing (60 min)

Build the Spark Streaming application.

**File**: `streaming_analytics.py`

**Requirements**:

**A. Stream Ingestion**
- Read from Kafka topics
- Parse JSON data
- Add watermarks (5 minutes)

**B. Stream Join**
- Join clickstream + transactions
- Time constraint: 10 minutes
- Calculate conversion metrics

**C. Windowed Aggregations**
- 1-minute tumbling windows for page views
- 5-minute sliding windows for revenue
- Group by product and category

**D. Stateful Processing**
- Track user sessions (30-minute timeout)
- Detect purchase anomalies (z-score > 3)
- Maintain running statistics

**E. Multiple Outputs**
- Console: Real-time monitoring
- Kafka: Downstream processing
- Parquet: Historical analysis

### Part 3: Monitoring & Testing (20 min)

**File**: `test_pipeline.sh`

**Requirements**:
- Start Kafka
- Create topics
- Run data generator
- Run streaming app
- Verify outputs
- Monitor metrics

### Part 4: Performance Optimization (20 min)

Apply optimizations learned in Day 48:
- Tune trigger intervals
- Optimize partitions
- Minimize state size
- Configure resources

---

## 📁 Project Structure

```
day-049-mini-project-spark-kafka/
├── README.md
├── project.md                    # Detailed specification
├── data_generator.py             # Kafka producers
├── streaming_analytics.py        # Main Spark app
├── config.py                     # Configuration
├── test_pipeline.sh              # Test script
├── requirements.txt              # Dependencies
└── output/                       # Results directory
```

---

## 🚀 Getting Started

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Kafka
docker run -d --name kafka -p 9092:9092 apache/kafka:latest

# Create topics
./test_pipeline.sh setup
```

### 2. Run Pipeline

```bash
# Terminal 1: Data generator
python data_generator.py

# Terminal 2: Streaming analytics
python streaming_analytics.py

# Terminal 3: Monitor
./test_pipeline.sh monitor
```

### 3. Verify Results

```bash
# Check console output
# Check Kafka topic: analytics-results
# Check Parquet files: output/analytics/

# View metrics
cat output/metrics.json
```

---

## 🎯 Success Criteria

### Functional Requirements
- [ ] Ingests data from Kafka successfully
- [ ] Joins streams correctly
- [ ] Calculates all required metrics
- [ ] Detects anomalies accurately
- [ ] Outputs to all sinks
- [ ] Handles late data properly

### Non-Functional Requirements
- [ ] Latency < 10 seconds
- [ ] Throughput > 1000 events/sec
- [ ] No data loss (exactly-once)
- [ ] Fault-tolerant (checkpointing)
- [ ] Monitoring enabled
- [ ] Code well-documented

### Performance Metrics
- [ ] Processing rate > input rate
- [ ] State size bounded
- [ ] Memory usage stable
- [ ] No backpressure

---

## 🎁 Bonus Challenges (Optional)

### Bonus 1: Advanced Analytics (30 min)
- Implement funnel analysis (view → cart → purchase)
- Calculate customer lifetime value
- Predict churn using simple rules

### Bonus 2: Dashboard (30 min)
- Create real-time dashboard with Streamlit
- Visualize metrics
- Show alerts for anomalies

### Bonus 3: Production Deployment (30 min)
- Docker Compose setup
- Kubernetes manifests
- CI/CD pipeline

### Bonus 4: Airflow Integration (30 min)
- DAG for pipeline orchestration
- Monitoring and alerting
- See `bonus_airflow/` folder

---

## 📊 Expected Output

### Console Output
```
-------------------------------------------
Batch: 1
-------------------------------------------
+----------+------------+----------+-------+
|product_id|window_start|page_views|revenue|
+----------+------------+----------+-------+
|prod-001  |10:00:00    |1234      |45678  |
|prod-002  |10:00:00    |987       |23456  |
+----------+------------+----------+-------+

Anomalies Detected: 3
Session Count: 156
```

### Kafka Output (analytics-results topic)
```json
{
  "product_id": "prod-001",
  "window_start": "2024-01-01T10:00:00Z",
  "page_views": 1234,
  "purchases": 45,
  "conversion_rate": 0.036,
  "revenue": 45678.90
}
```

### Parquet Files
```
output/analytics/
├── part-00000.parquet
├── part-00001.parquet
└── _spark_metadata/
```

---

## 🐛 Troubleshooting

### Issue: No data flowing
**Solution**: Check Kafka topics exist, verify producer is running

### Issue: High latency
**Solution**: Tune trigger interval, increase parallelism

### Issue: State growing unbounded
**Solution**: Verify watermarks set, check timeout configuration

### Issue: Join not producing results
**Solution**: Check time constraints, verify watermarks on both streams

---

## 📚 Key Concepts Applied

This project demonstrates:
- ✅ Spark Structured Streaming fundamentals (Day 43)
- ✅ Stream joins and aggregations (Day 44)
- ✅ Watermarking and late data (Day 45)
- ✅ Stream-to-stream joins (Day 46)
- ✅ Stateful processing (Day 47)
- ✅ Performance optimization (Day 48)

---

## 🎓 Learning Outcomes

After completing this project, you will:
- Build production-ready streaming pipelines
- Integrate Spark with Kafka effectively
- Apply advanced streaming patterns
- Optimize for performance and reliability
- Monitor and debug streaming applications
- Handle real-world challenges (late data, failures, scaling)

---

## 📝 Deliverables

1. `data_generator.py` - Working Kafka producers
2. `streaming_analytics.py` - Complete Spark application
3. `config.py` - Configuration management
4. `test_pipeline.sh` - Automated testing
5. `requirements.txt` - Dependencies
6. `output/` - Results and metrics
7. Documentation of design decisions

---

## 🔗 Resources

- [Spark Streaming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- [Kafka Integration](https://spark.apache.org/docs/latest/structured-streaming-kafka-integration.html)
- [Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)

---

## ✅ Completion Checklist

- [ ] Setup complete (Kafka, dependencies)
- [ ] Data generator working
- [ ] Streaming app ingesting data
- [ ] Joins producing results
- [ ] Aggregations calculating correctly
- [ ] Stateful processing working
- [ ] All outputs verified
- [ ] Performance acceptable
- [ ] Tests passing
- [ ] Documentation complete

---

## 🎉 Next Steps

After completing this project:
1. Review your code and identify improvements
2. Compare with reference implementation
3. Try bonus challenges
4. Deploy to production environment
5. Move on to Week 8: Data APIs & Testing

---

## Tomorrow: Day 50 - Checkpoint: Data Engineering Review

Consolidate your knowledge with a comprehensive review of data engineering concepts.
