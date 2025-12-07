#!/bin/bash

# Day 49: Real-Time Analytics - Test Pipeline
# Tests the complete Spark + Kafka streaming pipeline

set -e

KAFKA_CONTAINER="kafka-day49"
KAFKA_PORT=9092

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup: Start Kafka and create topics
setup() {
    print_header "Setting up Kafka"
    
    # Check if Kafka is already running
    if docker ps | grep -q $KAFKA_CONTAINER; then
        print_info "Kafka already running"
    else
        print_info "Starting Kafka..."
        docker run -d \
            --name $KAFKA_CONTAINER \
            -p $KAFKA_PORT:9092 \
            -e KAFKA_NODE_ID=1 \
            -e KAFKA_PROCESS_ROLES=broker,controller \
            -e KAFKA_LISTENERS=PLAINTEXT://localhost:9092,CONTROLLER://localhost:9093 \
            -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
            -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
            -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT \
            -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
            -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
            -e KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1 \
            -e KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1 \
            -e KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=0 \
            -e KAFKA_NUM_PARTITIONS=3 \
            apache/kafka:latest
        
        print_info "Waiting for Kafka to be ready..."
        sleep 10
    fi
    
    # Create topics
    print_info "Creating Kafka topics..."
    docker exec $KAFKA_CONTAINER /opt/kafka/bin/kafka-topics.sh \
        --create --if-not-exists \
        --bootstrap-server localhost:9092 \
        --topic clickstream \
        --partitions 3 \
        --replication-factor 1 || true
    
    docker exec $KAFKA_CONTAINER /opt/kafka/bin/kafka-topics.sh \
        --create --if-not-exists \
        --bootstrap-server localhost:9092 \
        --topic transactions \
        --partitions 3 \
        --replication-factor 1 || true
    
    docker exec $KAFKA_CONTAINER /opt/kafka/bin/kafka-topics.sh \
        --create --if-not-exists \
        --bootstrap-server localhost:9092 \
        --topic analytics-results \
        --partitions 3 \
        --replication-factor 1 || true
    
    print_info "Topics created successfully"
    
    # List topics
    print_info "Available topics:"
    docker exec $KAFKA_CONTAINER /opt/kafka/bin/kafka-topics.sh \
        --list \
        --bootstrap-server localhost:9092
}

# Test: Run the complete pipeline
test() {
    print_header "Testing Pipeline"
    
    # Create output directory
    mkdir -p output/analytics output/checkpoint
    
    print_info "Starting data generator (background)..."
    python data_generator.py --duration 60 --rate 50 > output/generator.log 2>&1 &
    GENERATOR_PID=$!
    
    print_info "Waiting for data to flow..."
    sleep 5
    
    print_info "Starting streaming analytics (30 seconds)..."
    timeout 30 python streaming_analytics.py || true
    
    print_info "Stopping data generator..."
    kill $GENERATOR_PID 2>/dev/null || true
    
    print_info "Pipeline test complete"
}

# Verify: Check outputs
verify() {
    print_header "Verifying Outputs"
    
    # Check generator log
    if [ -f output/generator.log ]; then
        print_info "Data Generator Stats:"
        tail -5 output/generator.log
    fi
    
    # Check Parquet files
    if [ -d output/analytics/revenue ]; then
        print_info "Parquet files created:"
        ls -lh output/analytics/revenue/*.parquet 2>/dev/null | head -5 || echo "No files yet"
    fi
    
    # Check Kafka output topic
    print_info "Checking analytics-results topic..."
    docker exec $KAFKA_CONTAINER /opt/kafka/bin/kafka-console-consumer.sh \
        --bootstrap-server localhost:9092 \
        --topic analytics-results \
        --from-beginning \
        --max-messages 5 \
        --timeout-ms 5000 2>/dev/null || echo "No messages yet"
    
    print_info "Verification complete"
}

# Monitor: Show real-time metrics
monitor() {
    print_header "Monitoring Pipeline"
    
    print_info "Consuming from analytics-results topic..."
    print_info "Press Ctrl+C to stop"
    
    docker exec -it $KAFKA_CONTAINER /opt/kafka/bin/kafka-console-consumer.sh \
        --bootstrap-server localhost:9092 \
        --topic analytics-results \
        --from-beginning
}

# Cleanup: Stop Kafka and remove data
cleanup() {
    print_header "Cleaning Up"
    
    print_info "Stopping Kafka..."
    docker stop $KAFKA_CONTAINER 2>/dev/null || true
    docker rm $KAFKA_CONTAINER 2>/dev/null || true
    
    print_info "Removing output files..."
    rm -rf output/
    
    print_info "Cleanup complete"
}

# Show usage
usage() {
    echo "Usage: $0 {setup|test|verify|monitor|cleanup|all}"
    echo ""
    echo "Commands:"
    echo "  setup    - Start Kafka and create topics"
    echo "  test     - Run the complete pipeline"
    echo "  verify   - Check outputs and results"
    echo "  monitor  - Monitor analytics results in real-time"
    echo "  cleanup  - Stop Kafka and remove data"
    echo "  all      - Run setup, test, and verify"
    exit 1
}

# Main
case "${1:-}" in
    setup)
        setup
        ;;
    test)
        test
        ;;
    verify)
        verify
        ;;
    monitor)
        monitor
        ;;
    cleanup)
        cleanup
        ;;
    all)
        setup
        test
        verify
        ;;
    *)
        usage
        ;;
esac

print_info "Done!"
