"""
Day 35: Real-Time Kafka Pipeline - Analytics Engine
Calculates windowed metrics using stream processing
"""
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from collections import defaultdict
import json
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Calculate real-time metrics with windowing"""
    
    def __init__(self, bootstrap_servers='localhost:9092', window_size=60):
        """
        Initialize analytics engine
        
        Args:
            bootstrap_servers: Kafka servers
            window_size: Window size in seconds (default 60 = 1 minute)
        """
        self.consumer = KafkaConsumer(
            'processed-orders',
            bootstrap_servers=bootstrap_servers,
            group_id='analytics-engine',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=True
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all'
        )
        
        self.window_size = window_size
        
        # State: window_start -> product_id -> stats
        self.windows = defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'total': 0.0,
            'product_name': None,
            'product_category': None
        }))
        
        self.last_emit_time = time.time()
        self.emit_interval = 10  # Emit metrics every 10 seconds
    
    def get_window_start(self, timestamp_str):
        """
        Calculate window start time for a timestamp
        
        Args:
            timestamp_str: ISO format timestamp
            
        Returns:
            int: Window start timestamp (epoch seconds)
        """
        try:
            dt = datetime.fromisoformat(timestamp_str)
            ts = dt.timestamp()
            return int(ts // self.window_size) * self.window_size
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
            return int(time.time() // self.window_size) * self.window_size
    
    def update_window(self, order):
        """Update window statistics with new order"""
        window_start = self.get_window_start(order['timestamp'])
        product_id = order['product_id']
        amount = order['amount']
        
        stats = self.windows[window_start][product_id]
        stats['count'] += 1
        stats['total'] += amount
        stats['product_name'] = order.get('product_name', product_id)
        stats['product_category'] = order.get('product_category', 'Unknown')
    
    def calculate_metrics(self, window_start, product_id, stats):
        """Calculate final metrics for a window"""
        count = stats['count']
        total = stats['total']
        avg = total / count if count > 0 else 0
        
        return {
            'window_start': window_start,
            'window_end': window_start + self.window_size,
            'product_id': product_id,
            'product_name': stats['product_name'],
            'product_category': stats['product_category'],
            'order_count': count,
            'total_revenue': round(total, 2),
            'avg_order_value': round(avg, 2),
            'timestamp': datetime.now().isoformat()
        }
    
    def emit_metrics(self, force=False):
        """
        Emit metrics for completed windows
        
        Args:
            force: If True, emit all windows regardless of time
        """
        current_time = time.time()
        
        # Only emit periodically unless forced
        if not force and current_time - self.last_emit_time < self.emit_interval:
            return
        
        current_window = int(current_time // self.window_size) * self.window_size
        windows_to_emit = []
        
        # Find completed windows (not current window)
        for window_start in list(self.windows.keys()):
            if force or window_start < current_window:
                windows_to_emit.append(window_start)
        
        # Emit metrics for each completed window
        for window_start in windows_to_emit:
            products = self.windows[window_start]
            
            for product_id, stats in products.items():
                metrics = self.calculate_metrics(window_start, product_id, stats)
                
                try:
                    self.producer.send('product-metrics', metrics)
                    logger.info(
                        f"Emitted metrics: {product_id} - "
                        f"{stats['count']} orders, ${stats['total']:.2f}"
                    )
                except KafkaError as e:
                    logger.error(f"Failed to emit metrics: {e}")
            
            # Remove emitted window
            del self.windows[window_start]
        
        self.last_emit_time = current_time
    
    def run(self):
        """Run analytics engine"""
        logger.info("Starting analytics engine")
        logger.info(f"Window size: {self.window_size} seconds")
        logger.info("Consuming from 'processed-orders' topic")
        logger.info("Producing to 'product-metrics' topic")
        
        message_count = 0
        
        try:
            for message in self.consumer:
                order = message.value
                
                # Update window statistics
                self.update_window(order)
                message_count += 1
                
                # Periodically emit metrics
                self.emit_metrics()
                
                if message_count % 100 == 0:
                    active_windows = len(self.windows)
                    logger.info(
                        f"Processed {message_count} orders, "
                        f"{active_windows} active windows"
                    )
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Emit all remaining metrics
            self.emit_metrics(force=True)
            
            self.producer.flush()
            self.producer.close()
            self.consumer.close()
            
            logger.info("=" * 60)
            logger.info("Analytics Engine Summary")
            logger.info("=" * 60)
            logger.info(f"Total orders processed: {message_count}")
            logger.info("=" * 60)


def main():
    """Main entry point"""
    engine = AnalyticsEngine(window_size=60)
    engine.run()


if __name__ == "__main__":
    main()
