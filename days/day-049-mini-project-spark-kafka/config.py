"""
Day 49: Real-Time Analytics - Configuration

Centralized configuration for the streaming application.
"""

# Kafka Configuration
KAFKA_BOOTSTRAP = "localhost:9092"
CLICKSTREAM_TOPIC = "clickstream"
TRANSACTIONS_TOPIC = "transactions"
OUTPUT_TOPIC = "analytics-results"

# Spark Configuration
SHUFFLE_PARTITIONS = 50
MAX_OFFSETS_PER_TRIGGER = 10000

# Streaming Configuration
TRIGGER_INTERVAL = "10 seconds"
WATERMARK_DELAY = "5 minutes"

# Checkpoint and Output
CHECKPOINT_DIR = "output/checkpoint"
OUTPUT_DIR = "output/analytics"

# Data Generator Configuration
EVENTS_PER_SECOND = 100
GENERATION_DURATION = 120  # seconds

# Monitoring
STATS_INTERVAL = 1000  # Print stats every N events
