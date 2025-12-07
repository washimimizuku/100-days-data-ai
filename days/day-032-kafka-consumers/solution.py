"""
Day 32: Kafka Consumers & Consumer Groups - Solutions

Prerequisites:
docker run -d --name kafka -p 9092:9092 apache/kafka:latest
pip install kafka-python
"""

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka import ConsumerRebalanceListener
from kafka.errors import KafkaError
import json
import time
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOOTSTRAP_SERVERS = ['localhost:9092']


def exercise_1_basic_consumer():
    """Basic consumer reading messages"""
    print("Creating basic consumer...")
    
    consumer = KafkaConsumer(
        'test-topic',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id='test-group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest'
    )
    
    print("Reading 10 messages...")
    count = 0
    for message in consumer:
        print(f"P{message.partition}@{message.offset}: {message.value}")
        count += 1
        if count >= 10:
            break
    
    consumer.close()
    print("Consumer closed\n")


def exercise_2_consumer_group():
    """Multiple consumers in same group"""
    print("Creating consumer group with 3 consumers...")
    
    def consume_worker(consumer_id):
        consumer = KafkaConsumer(
            'orders', bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id='order-processors', auto_offset_reset='earliest'
        )
        time.sleep(2)
        partitions = consumer.assignment()
        print(f"Consumer {consumer_id}: {[p.partition for p in partitions]}")
        for i, msg in enumerate(consumer):
            if i >= 5: break
        consumer.close()
    
    threads = [threading.Thread(target=consume_worker, args=(i,)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    print("All consumers finished\n")


def exercise_3_manual_offset_management():
    """Manual offset commits"""
    print("Consumer with manual offset management...")
    
    consumer = KafkaConsumer(
        'test-topic',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id='manual-commit-group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        enable_auto_commit=False,
        auto_offset_reset='earliest'
    )
    
    count = 0
    for message in consumer:
        try:
            # Process message
            print(f"Processing: {message.value}")
            
            # Simulate processing
            if message.offset % 5 == 0:
                raise Exception("Simulated error")
            
            # Commit after successful processing
            consumer.commit()
            print(f"✓ Committed offset {message.offset}")
            
        except Exception as e:
            print(f"✗ Error: {e}, not committing")
        
        count += 1
        if count >= 10:
            break
    
    consumer.close()
    print()


def exercise_4_batch_processing():
    """Batch processing for throughput"""
    print("Batch processing consumer...")
    
    consumer = KafkaConsumer(
        'test-topic', bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id='batch-group', enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    batch, batch_size, total = [], 50, 0
    start = time.time()
    
    for msg in consumer:
        batch.append(msg)
        if len(batch) >= batch_size:
            print(f"Processing batch of {len(batch)}...")
            consumer.commit()
            total += len(batch)
            batch = []
        if total >= 200: break
    
    elapsed = time.time() - start
    print(f"Processed {total} in {elapsed:.2f}s ({total/elapsed:.0f} msg/sec)\n")
    consumer.close()


def exercise_5_rebalance_handling():
    """Rebalance listener"""
    print("Consumer with rebalance listener...")
    
    class MyRebalanceListener(ConsumerRebalanceListener):
        def __init__(self, consumer):
            self.consumer = consumer
        
        def on_partitions_revoked(self, revoked):
            print(f"⚠ Partitions revoked: {[p.partition for p in revoked]}")
            self.consumer.commit()
            print("Committed offsets before rebalance")
        
        def on_partitions_assigned(self, assigned):
            print(f"✓ Partitions assigned: {[p.partition for p in assigned]}")
    
    consumer = KafkaConsumer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id='rebalance-group',
        enable_auto_commit=False
    )
    
    listener = MyRebalanceListener(consumer)
    consumer.subscribe(['test-topic'], listener=listener)
    
    count = 0
    for message in consumer:
        print(f"Processing: P{message.partition}@{message.offset}")
        count += 1
        if count >= 10:
            break
    
    consumer.close()
    print()


def exercise_6_consumer_lag_monitoring():
    """Monitor consumer lag"""
    print("Monitoring consumer lag...")
    
    consumer = KafkaConsumer(
        'test-topic', bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id='lag-monitor-group', auto_offset_reset='earliest'
    )
    
    for i, msg in enumerate(consumer):
        if i >= 5: break
    
    print("\nLag Report:")
    total_lag = 0
    for p in consumer.assignment():
        current = consumer.position(p)
        latest = consumer.end_offsets([p])[p]
        lag = latest - current
        total_lag += lag
        print(f"P{p.partition}: current={current}, latest={latest}, lag={lag}")
        if lag > 100: print("  ⚠ High lag!")
    
    print(f"Total lag: {total_lag}\n")
    consumer.close()


def exercise_7_error_handling_dlq():
    """Error handling with DLQ"""
    print("Consumer with DLQ...")
    
    consumer = KafkaConsumer(
        'test-topic',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id='dlq-group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        enable_auto_commit=False,
        auto_offset_reset='earliest'
    )
    
    dlq_producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    def send_to_dlq(message, error):
        dlq_message = {
            'original_topic': message.topic,
            'partition': message.partition,
            'offset': message.offset,
            'value': message.value,
            'error': str(error),
            'timestamp': time.time()
        }
        dlq_producer.send('test-topic.dlq', dlq_message)
        print(f"✗ Sent to DLQ: {error}")
    
    count = 0
    for message in consumer:
        max_retries = 3
        retries = 0
        
        while retries < max_retries:
            try:
                # Simulate occasional failures
                if message.offset % 7 == 0:
                    raise Exception("Processing error")
                
                print(f"✓ Processed: {message.value}")
                consumer.commit()
                break
                
            except Exception as e:
                retries += 1
                print(f"Retry {retries}/{max_retries}: {e}")
                time.sleep(0.1)
                
                if retries == max_retries:
                    send_to_dlq(message, e)
                    consumer.commit()  # Skip bad message
        
        count += 1
        if count >= 15:
            break
    
    consumer.close()
    dlq_producer.close()
    print()


def exercise_8_production_consumer_class():
    """Production-ready consumer class"""
    print("Production consumer class...")
    
    class ProductionConsumer:
        def __init__(self, topics, group_id, bootstrap_servers):
            self.topics = topics
            self.consumer = KafkaConsumer(
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                enable_auto_commit=False,
                max_poll_interval_ms=300000,
                session_timeout_ms=10000
            )
            
            listener = self.RebalanceListener(self.consumer)
            self.consumer.subscribe(topics, listener=listener)
            
            self.running = True
            self.processed_count = 0
        
        class RebalanceListener(ConsumerRebalanceListener):
            def __init__(self, consumer):
                self.consumer = consumer
            
            def on_partitions_revoked(self, revoked):
                logger.info(f"Partitions revoked: {len(revoked)}")
                self.consumer.commit()
            
            def on_partitions_assigned(self, assigned):
                logger.info(f"Partitions assigned: {len(assigned)}")
        
        def consume(self, max_messages=None):
            try:
                for message in self.consumer:
                    if not self.running:
                        break
                    
                    self._process_message(message)
                    self.processed_count += 1
                    
                    if max_messages and self.processed_count >= max_messages:
                        break
            finally:
                self.close()
        
        def _process_message(self, message):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Process message
                    logger.info(f"Processing: {message.value}")
                    self.consumer.commit()
                    return
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        self.consumer.commit()  # Skip
                    else:
                        time.sleep(0.1 * (2 ** attempt))
        
        def get_lag(self):
            partitions = self.consumer.assignment()
            total_lag = 0
            for p in partitions:
                current = self.consumer.position(p)
                end = self.consumer.end_offsets([p])[p]
                total_lag += (end - current)
            return total_lag
        
        def close(self):
            self.running = False
            self.consumer.close()
            logger.info(f"Consumer closed. Processed: {self.processed_count}")
    
    # Test the class
    consumer = ProductionConsumer(
        topics=['test-topic'],
        group_id='production-group',
        bootstrap_servers=BOOTSTRAP_SERVERS
    )
    
    consumer.consume(max_messages=20)
    lag = consumer.get_lag()
    print(f"Final lag: {lag} messages\n")


if __name__ == "__main__":
    print("Day 32: Kafka Consumers & Consumer Groups - Solutions\n")
    print("=" * 60)
    print("\nNote: Requires Kafka running on localhost:9092\n")
    
    try:
        print("\nExercise 1: Basic Consumer")
        print("-" * 60)
        exercise_1_basic_consumer()
        
        print("\nExercise 2: Consumer Group")
        print("-" * 60)
        exercise_2_consumer_group()
        
        print("\nExercise 3: Manual Offset Management")
        print("-" * 60)
        exercise_3_manual_offset_management()
        
        print("\nExercise 4: Batch Processing")
        print("-" * 60)
        exercise_4_batch_processing()
        
        print("\nExercise 5: Rebalance Handling")
        print("-" * 60)
        exercise_5_rebalance_handling()
        
        print("\nExercise 6: Consumer Lag Monitoring")
        print("-" * 60)
        exercise_6_consumer_lag_monitoring()
        
        print("\nExercise 7: Error Handling with DLQ")
        print("-" * 60)
        exercise_7_error_handling_dlq()
        
        print("\nExercise 8: Production Consumer Class")
        print("-" * 60)
        exercise_8_production_consumer_class()
        
        print("=" * 60)
        print("All exercises completed!")
        
    except KafkaError as e:
        print(f"\nKafka Error: {e}")
        print("Make sure Kafka is running on localhost:9092")
