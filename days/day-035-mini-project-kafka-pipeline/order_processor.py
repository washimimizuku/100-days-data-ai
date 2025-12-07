"""
Day 35: Real-Time Kafka Pipeline - Order Processor
Validates and enriches orders from Kafka
"""
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderProcessor:
    """Process and enrich orders from Kafka"""
    
    def __init__(self, bootstrap_servers='localhost:9092'):
        """Initialize consumer and producer"""
        self.consumer = KafkaConsumer(
            'orders',
            bootstrap_servers=bootstrap_servers,
            group_id='order-processors',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=False,
            max_poll_records=50
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        
        self.product_catalog = {
            'prod-1': {'name': 'Laptop', 'category': 'Electronics'},
            'prod-2': {'name': 'Phone', 'category': 'Electronics'},
            'prod-3': {'name': 'Tablet', 'category': 'Electronics'},
            'prod-4': {'name': 'Monitor', 'category': 'Electronics'},
            'prod-5': {'name': 'Keyboard', 'category': 'Accessories'}
        }
        
        self.processed_count = 0
        self.invalid_count = 0
        self.error_count = 0
    
    def validate_order(self, order):
        """
        Validate order data
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not order.get('order_id'):
            return False, "Missing order_id"
        
        if not order.get('user_id'):
            return False, "Missing user_id"
        
        product_id = order.get('product_id')
        if not product_id or product_id not in self.product_catalog:
            return False, f"Invalid product_id: {product_id}"
        
        amount = order.get('amount', 0)
        if amount <= 0:
            return False, f"Invalid amount: {amount}"
        
        return True, None
    
    def enrich_order(self, order):
        """Add product information to order"""
        product_id = order['product_id']
        product_info = self.product_catalog[product_id]
        
        enriched = order.copy()
        enriched['product_name'] = product_info['name']
        enriched['product_category'] = product_info['category']
        enriched['processed_at'] = datetime.now().isoformat()
        
        return enriched
    
    def send_to_dlq(self, order, error_message):
        """Send invalid order to dead letter queue"""
        dlq_message = {
            'original_order': order,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.producer.send('orders-dlq', dlq_message)
            logger.warning(f"Sent to DLQ: {error_message}")
        except KafkaError as e:
            logger.error(f"Failed to send to DLQ: {e}")
    
    def process_batch(self, messages):
        """Process a batch of messages"""
        for message in messages:
            order = message.value
            
            # Validate
            is_valid, error_msg = self.validate_order(order)
            
            if not is_valid:
                self.invalid_count += 1
                self.send_to_dlq(order, error_msg)
                continue
            
            # Enrich
            try:
                enriched_order = self.enrich_order(order)
                
                # Send to processed topic
                self.producer.send('processed-orders', enriched_order)
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing order: {e}")
                self.error_count += 1
                self.send_to_dlq(order, str(e))
    
    def run(self):
        """Run order processor"""
        logger.info("Starting order processor")
        logger.info("Consuming from 'orders' topic")
        logger.info("Producing to 'processed-orders' topic")
        
        batch = []
        
        try:
            for message in self.consumer:
                batch.append(message)
                
                # Process in batches of 50
                if len(batch) >= 50:
                    self.process_batch(batch)
                    self.consumer.commit()
                    
                    logger.info(
                        f"Processed batch: {self.processed_count} valid, "
                        f"{self.invalid_count} invalid, {self.error_count} errors"
                    )
                    
                    batch = []
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Process remaining messages
            if batch:
                self.process_batch(batch)
                self.consumer.commit()
            
            self.producer.flush()
            self.producer.close()
            self.consumer.close()
            
            logger.info("=" * 60)
            logger.info("Order Processor Summary")
            logger.info("=" * 60)
            logger.info(f"Total processed: {self.processed_count}")
            logger.info(f"Total invalid: {self.invalid_count}")
            logger.info(f"Total errors: {self.error_count}")
            logger.info("=" * 60)


def main():
    """Main entry point"""
    processor = OrderProcessor()
    processor.run()


if __name__ == "__main__":
    main()
