#!/bin/bash
set -e

echo "=========================================="
echo "Kafka Pipeline Test Script"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    # Stop Python processes
    pkill -f event_generator.py 2>/dev/null || true
    pkill -f order_processor.py 2>/dev/null || true
    pkill -f analytics_engine.py 2>/dev/null || true
    pkill -f data_sink.py 2>/dev/null || true
    
    # Stop and remove Kafka container
    docker stop kafka 2>/dev/null || true
    docker rm kafka 2>/dev/null || true
    
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"

# Start Kafka
echo -e "\n${YELLOW}Starting Kafka...${NC}"
docker run -d --name kafka -p 9092:9092 apache/kafka:latest

echo "Waiting for Kafka to be ready..."
sleep 15

# Verify Kafka is running
if ! docker ps | grep -q kafka; then
    echo -e "${RED}Error: Kafka failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Kafka started${NC}"

# Create topics
echo -e "\n${YELLOW}Creating Kafka topics...${NC}"
docker exec kafka kafka-topics.sh --create --topic orders \
    --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 \
    --if-not-exists

docker exec kafka kafka-topics.sh --create --topic processed-orders \
    --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 \
    --if-not-exists

docker exec kafka kafka-topics.sh --create --topic product-metrics \
    --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 \
    --if-not-exists

docker exec kafka kafka-topics.sh --create --topic orders-dlq \
    --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 \
    --if-not-exists

echo -e "${GREEN}✓ Topics created${NC}"

# List topics
echo -e "\n${YELLOW}Verifying topics...${NC}"
docker exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092

# Clean output directory
echo -e "\n${YELLOW}Preparing output directory...${NC}"
rm -rf output
mkdir -p output
echo -e "${GREEN}✓ Output directory ready${NC}"

# Start pipeline components
echo -e "\n${YELLOW}Starting pipeline components...${NC}"

echo "Starting data sink..."
python data_sink.py > output/data_sink.log 2>&1 &
SINK_PID=$!
sleep 2

echo "Starting analytics engine..."
python analytics_engine.py > output/analytics_engine.log 2>&1 &
ANALYTICS_PID=$!
sleep 2

echo "Starting order processor..."
python order_processor.py > output/order_processor.log 2>&1 &
PROCESSOR_PID=$!
sleep 2

echo "Starting event generator..."
python event_generator.py > output/event_generator.log 2>&1 &
GENERATOR_PID=$!

echo -e "${GREEN}✓ All components started${NC}"
echo "  - Event Generator PID: $GENERATOR_PID"
echo "  - Order Processor PID: $PROCESSOR_PID"
echo "  - Analytics Engine PID: $ANALYTICS_PID"
echo "  - Data Sink PID: $SINK_PID"

# Run for 60 seconds
echo -e "\n${YELLOW}Running pipeline for 60 seconds...${NC}"
for i in {1..12}; do
    sleep 5
    echo "  $(($i * 5)) seconds elapsed..."
done

# Stop components
echo -e "\n${YELLOW}Stopping components...${NC}"
kill $GENERATOR_PID 2>/dev/null || true
sleep 2
kill $PROCESSOR_PID 2>/dev/null || true
kill $ANALYTICS_PID 2>/dev/null || true
kill $SINK_PID 2>/dev/null || true
sleep 2

echo -e "${GREEN}✓ Components stopped${NC}"

# Verify output
echo -e "\n${YELLOW}Verifying results...${NC}"

if [ -f "output/metrics.json" ]; then
    METRIC_COUNT=$(wc -l < output/metrics.json)
    echo -e "${GREEN}✓ Output file created${NC}"
    echo "  Metrics written: $METRIC_COUNT"
    
    if [ $METRIC_COUNT -gt 0 ]; then
        echo -e "\n${YELLOW}Sample metrics:${NC}"
        head -3 output/metrics.json | python -m json.tool
    else
        echo -e "${RED}Warning: No metrics written${NC}"
    fi
else
    echo -e "${RED}✗ Output file not found${NC}"
    exit 1
fi

# Show component logs
echo -e "\n${YELLOW}Component summaries:${NC}"

if [ -f "output/event_generator.log" ]; then
    echo -e "\n${YELLOW}Event Generator:${NC}"
    tail -10 output/event_generator.log
fi

if [ -f "output/order_processor.log" ]; then
    echo -e "\n${YELLOW}Order Processor:${NC}"
    tail -10 output/order_processor.log
fi

# Final summary
echo -e "\n=========================================="
echo -e "${GREEN}Test completed successfully!${NC}"
echo "=========================================="
echo "Output files:"
echo "  - output/metrics.json (metrics data)"
echo "  - output/*.log (component logs)"
echo ""
echo "To view metrics:"
echo "  cat output/metrics.json | python -m json.tool"
echo ""
echo "To view logs:"
echo "  tail -f output/*.log"
echo "=========================================="
