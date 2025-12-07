"""
Day 35: Real-Time Kafka Pipeline - Data Sink
Writes metrics to output file
"""
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSink:
    """Write metrics to file storage"""
    
    def __init__(self, bootstrap_servers='localhost:9092', output_file='output/metrics.json'):
        """
        Initialize data sink
        
        Args:
            bootstrap_servers: Kafka servers
            output_file: Path to output file
        """
        self.consumer = KafkaConsumer(
            'product-metrics',
            bootstrap_servers=bootstrap_servers,
            group_id='data-sink',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=True
        )
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.output_file = output_file
        
        self.metrics_written = 0
        self.error_count = 0
    
    def write_metric(self, metrics):
        """
        Write metrics to file
        
        Args:
            metrics: Metrics dictionary
        """
        try:
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
                f.flush()
            
            self.metrics_written += 1
            
            logger.info(
                f"Wrote metrics: {metrics['product_name']} - "
                f"Window {datetime.fromtimestamp(metrics['window_start']).strftime('%H:%M:%S')} - "
                f"{metrics['order_count']} orders, ${metrics['total_revenue']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error writing metrics: {e}")
            self.error_count += 1
    
    def run(self):
        """Run data sink"""
        logger.info("Starting data sink")
        logger.info("Consuming from 'product-metrics' topic")
        logger.info(f"Writing to: {self.output_file}")
        
        try:
            for message in self.consumer:
                metrics = message.value
                self.write_metric(metrics)
                
                if self.metrics_written % 10 == 0:
                    logger.info(
                        f"Total metrics written: {self.metrics_written} "
                        f"({self.error_count} errors)"
                    )
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.consumer.close()
            
            logger.info("=" * 60)
            logger.info("Data Sink Summary")
            logger.info("=" * 60)
            logger.info(f"Total metrics written: {self.metrics_written}")
            logger.info(f"Total errors: {self.error_count}")
            logger.info(f"Output file: {self.output_file}")
            logger.info("=" * 60)


def main():
    """Main entry point"""
    sink = DataSink()
    sink.run()


if __name__ == "__main__":
    main()
