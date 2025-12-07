"""
Day 35: Real-Time Kafka Pipeline - Event Generator
Generates realistic order events and sends to Kafka
"""
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import random
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventGenerator:
    """Generate realistic order events for Kafka pipeline"""
    
    def __init__(self, bootstrap_servers='localhost:9092'):
        """Initialize Kafka producer with reliability settings"""
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=1,  # Ensure ordering
            compression_type='gzip'
        )
        
        self.products = [
            'prod-1',  # Laptop
            'prod-2',  # Phone
            'prod-3',  # Tablet
            'prod-4',  # Monitor
            'prod-5'   # Keyboard
        ]
        
        self.order_count = 0
        self.error_count = 0
    
    def generate_order(self):
        """Generate a single order event"""
        return {
            'order_id': f'order-{int(time.time() * 1000)}-{random.randint(1000, 9999)}',
            'user_id': f'user-{random.randint(1, 100)}',
            'product_id': random.choice(self.products),
            'amount': round(random.uniform(10, 500), 2),
            'timestamp': datetime.now().isoformat()
        }
    
    def send_order(self, order):
        """Send order to Kafka with error handling"""
        try:
            future = self.producer.send('orders', order)
            future.get(timeout=10)  # Block until sent
            self.order_count += 1
            return True
        except KafkaError as e:
            logger.error(f"Failed to send order: {e}")
            self.error_count += 1
            return False
    
    def run(self, duration=60, events_per_second=30):
        """
        Run event generator for specified duration
        
        Args:
            duration: How long to run (seconds)
            events_per_second: Target event rate
        """
        logger.info(f"Starting event generator for {duration} seconds")
        logger.info(f"Target rate: {events_per_second} events/second")
        
        start_time = time.time()
        sleep_time = 1.0 / events_per_second
        
        try:
            while time.time() - start_time < duration:
                order = self.generate_order()
                self.send_order(order)
                
                if self.order_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = self.order_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Generated {self.order_count} orders "
                        f"({rate:.1f} events/sec, {self.error_count} errors)"
                    )
                
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.producer.flush()
            self.producer.close()
            
            elapsed = time.time() - start_time
            rate = self.order_count / elapsed if elapsed > 0 else 0
            
            logger.info("=" * 60)
            logger.info("Event Generator Summary")
            logger.info("=" * 60)
            logger.info(f"Total orders generated: {self.order_count}")
            logger.info(f"Total errors: {self.error_count}")
            logger.info(f"Duration: {elapsed:.1f} seconds")
            logger.info(f"Average rate: {rate:.1f} events/second")
            logger.info("=" * 60)


def main():
    """Main entry point"""
    generator = EventGenerator()
    generator.run(duration=60, events_per_second=30)


if __name__ == "__main__":
    main()
