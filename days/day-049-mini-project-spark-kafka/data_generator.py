"""
Day 49: Real-Time Analytics - Data Generator

Generates realistic clickstream and transaction events to Kafka topics.
"""
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import random
import time
from datetime import datetime
import uuid
import argparse
import signal
import sys


class DataGenerator:
    """Generate realistic e-commerce events"""
    
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1
        )
        
        # Product catalog
        self.products = [
            {"id": "prod_001", "category": "electronics", "price": 299.99, "name": "Wireless Headphones"},
            {"id": "prod_002", "category": "electronics", "price": 499.99, "name": "Smart Watch"},
            {"id": "prod_003", "category": "electronics", "price": 899.99, "name": "Laptop"},
            {"id": "prod_004", "category": "clothing", "price": 49.99, "name": "T-Shirt"},
            {"id": "prod_005", "category": "clothing", "price": 79.99, "name": "Jeans"},
            {"id": "prod_006", "category": "books", "price": 19.99, "name": "Python Guide"},
            {"id": "prod_007", "category": "books", "price": 24.99, "name": "Data Engineering"},
            {"id": "prod_008", "category": "home", "price": 79.99, "name": "Coffee Maker"},
            {"id": "prod_009", "category": "home", "price": 129.99, "name": "Vacuum Cleaner"},
            {"id": "prod_010", "category": "sports", "price": 59.99, "name": "Yoga Mat"}
        ]
        
        self.actions = ["view", "click", "add_to_cart"]
        self.users = [f"user_{i:03d}" for i in range(1, 101)]
        self.referrers = ["google.com", "facebook.com", "twitter.com", "direct", "email"]
        self.payment_methods = ["credit_card", "debit_card", "paypal", "apple_pay"]
        
        self.running = True
        self.stats = {"clicks": 0, "transactions": 0, "errors": 0}
        
    def generate_clickstream(self):
        """Generate clickstream event"""
        user_id = random.choice(self.users)
        product = random.choice(self.products)
        session_id = f"sess_{user_id}_{int(time.time() / 1800)}"
        
        event = {
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "session_id": session_id,
            "product_id": product["id"],
            "category": product["category"],
            "action": random.choice(self.actions),
            "timestamp": datetime.now().isoformat(),
            "page_url": f"/product/{product['id']}",
            "referrer": random.choice(self.referrers)
        }
        
        return event
    
    def generate_transaction(self):
        """Generate transaction event (20% of clicks convert)"""
        user_id = random.choice(self.users)
        product = random.choice(self.products)
        session_id = f"sess_{user_id}_{int(time.time() / 1800)}"
        quantity = random.randint(1, 3)
        
        # Add some anomalies (5% chance of unusually high amount)
        if random.random() < 0.05:
            quantity = random.randint(10, 50)  # Anomaly
        
        event = {
            "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "session_id": session_id,
            "product_id": product["id"],
            "category": product["category"],
            "amount": round(product["price"] * quantity, 2),
            "quantity": quantity,
            "timestamp": datetime.now().isoformat(),
            "payment_method": random.choice(self.payment_methods)
        }
        
        return event
    
    def send_event(self, topic, event):
        """Send event to Kafka with error handling"""
        try:
            future = self.producer.send(topic, event)
            future.get(timeout=10)
            return True
        except KafkaError as e:
            print(f"Error sending to {topic}: {e}")
            self.stats["errors"] += 1
            return False
    
    def run(self, duration=120, events_per_second=100):
        """Run generator for specified duration"""
        print(f"Starting data generator...")
        print(f"Duration: {duration}s, Rate: {events_per_second} events/sec")
        print(f"Topics: clickstream (80%), transactions (20%)")
        print("-" * 60)
        
        start_time = time.time()
        interval = 1.0 / events_per_second
        
        try:
            while self.running and (time.time() - start_time < duration):
                # 80% clickstream, 20% transactions
                if random.random() < 0.8:
                    event = self.generate_clickstream()
                    if self.send_event('clickstream', event):
                        self.stats["clicks"] += 1
                else:
                    event = self.generate_transaction()
                    if self.send_event('transactions', event):
                        self.stats["transactions"] += 1
                
                # Print stats every 1000 events
                total = self.stats["clicks"] + self.stats["transactions"]
                if total > 0 and total % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = total / elapsed
                    print(f"[{elapsed:.1f}s] Generated: {self.stats['clicks']} clicks, "
                          f"{self.stats['transactions']} transactions "
                          f"({rate:.1f} events/s, {self.stats['errors']} errors)")
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\nStopping generator...")
        
        finally:
            self.producer.flush()
            self.producer.close()
            
            elapsed = time.time() - start_time
            total = self.stats["clicks"] + self.stats["transactions"]
            print("\n" + "=" * 60)
            print("Generation Complete")
            print("=" * 60)
            print(f"Duration: {elapsed:.1f}s")
            print(f"Clickstream events: {self.stats['clicks']}")
            print(f"Transaction events: {self.stats['transactions']}")
            print(f"Total events: {total}")
            print(f"Average rate: {total/elapsed:.1f} events/s")
            print(f"Errors: {self.stats['errors']}")
    
    def stop(self):
        """Stop generator gracefully"""
        self.running = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Generate e-commerce events to Kafka')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                       help='Kafka bootstrap servers')
    parser.add_argument('--duration', type=int, default=120,
                       help='Duration in seconds (default: 120)')
    parser.add_argument('--rate', type=int, default=100,
                       help='Events per second (default: 100)')
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    generator = DataGenerator(args.bootstrap_servers)
    generator.run(duration=args.duration, events_per_second=args.rate)


if __name__ == "__main__":
    main()
