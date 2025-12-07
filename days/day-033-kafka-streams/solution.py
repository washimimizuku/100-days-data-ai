"""
Day 33: Kafka Streams - Solutions

Prerequisites:
docker run -d --name kafka -p 9092:9092 apache/kafka:latest
pip install kafka-python
"""

from kafka import KafkaConsumer, KafkaProducer
from collections import defaultdict
import json
import time
import shelve
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOOTSTRAP_SERVERS = ['localhost:9092']


def exercise_1_stateless_transformations():
    """Stateless transformations: map, filter, flatMap"""
    print("Stateless transformations...")
    
    consumer = KafkaConsumer(
        'events',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest'
    )
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    count = 0
    for message in consumer:
        event = message.value
        
        # Map: Add timestamp
        event['processed_at'] = time.time()
        
        # Filter: Only high priority
        if event.get('priority') == 'high':
            producer.send('processed-events', event)
            print(f"✓ Processed high-priority event: {event.get('id')}")
        
        # FlatMap: Emit one message per tag
        if 'tags' in event:
            for tag in event['tags']:
                tag_event = {'tag': tag, 'event_id': event.get('id')}
                producer.send('tag-events', tag_event)
        
        count += 1
        if count >= 10:
            break
    
    producer.flush()
    consumer.close()
    producer.close()
    print()


def exercise_2_word_count():
    """Word count with aggregation"""
    print("Word count aggregation...")
    
    consumer = KafkaConsumer('sentences', bootstrap_servers=BOOTSTRAP_SERVERS,
                            value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS,
                            value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    
    word_counts = defaultdict(int)
    for i, msg in enumerate(consumer):
        for word in msg.value.get('text', '').lower().split():
            word_counts[word] += 1
            producer.send('word-counts', {'word': word, 'count': word_counts[word]})
        if i >= 5: break
    
    print("\nTop 10:", sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    producer.close()
    consumer.close()
    print()


def exercise_3_stream_enrichment():
    """Stream enrichment with join"""
    print("Stream enrichment...")
    
    user_table = {
        'user-1': {'name': 'Alice', 'email': 'alice@example.com'},
        'user-2': {'name': 'Bob', 'email': 'bob@example.com'}
    }
    
    consumer = KafkaConsumer('user-events', bootstrap_servers=BOOTSTRAP_SERVERS,
                            value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS,
                            value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    
    for i, msg in enumerate(consumer):
        user_id = msg.value.get('user_id')
        user_data = user_table.get(user_id, {})
        enriched = {**msg.value, 'user_name': user_data.get('name', 'Unknown')}
        producer.send('enriched-events', enriched)
        print(f"✓ Enriched: {user_data.get('name', 'Unknown')}")
        if i >= 10: break
    
    producer.close()
    consumer.close()
    print()


def exercise_4_tumbling_window():
    """Tumbling window aggregation"""
    print("Tumbling window aggregation...")
    
    consumer = KafkaConsumer(
        'clicks',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest'
    )
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    window_size = 60  # 1 minute
    windows = defaultdict(lambda: defaultdict(int))
    
    count = 0
    for message in consumer:
        timestamp = message.timestamp / 1000
        window_start = int(timestamp // window_size) * window_size
        
        page = message.value.get('page', 'unknown')
        windows[window_start][page] += 1
        
        result = {
            'window_start': window_start,
            'page': page,
            'count': windows[window_start][page]
        }
        producer.send('windowed-counts', result)
        print(f"Window {window_start}: {page} = {windows[window_start][page]}")
        
        count += 1
        if count >= 20:
            break
    
    producer.close()
    consumer.close()
    print()


def exercise_5_session_detection():
    """Session detection with inactivity gap"""
    print("Session detection...")
    
    consumer = KafkaConsumer(
        'user-activity',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest'
    )
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    session_gap = 300  # 5 minutes
    sessions = {}
    
    count = 0
    for message in consumer:
        user_id = message.value.get('user_id')
        timestamp = message.timestamp / 1000
        
        if user_id not in sessions:
            sessions[user_id] = {'start': timestamp, 'end': timestamp, 'count': 0}
        
        session = sessions[user_id]
        
        # Check if new session needed
        if timestamp - session['end'] > session_gap:
            # Close old session
            session_result = {
                'user_id': user_id,
                'start': session['start'],
                'end': session['end'],
                'duration': session['end'] - session['start'],
                'event_count': session['count']
            }
            producer.send('sessions', session_result)
            print(f"✓ Session closed for {user_id}: {session['count']} events")
            
            # Start new session
            sessions[user_id] = {'start': timestamp, 'end': timestamp, 'count': 0}
        
        sessions[user_id]['end'] = timestamp
        sessions[user_id]['count'] += 1
        
        count += 1
        if count >= 15:
            break
    
    producer.close()
    consumer.close()
    print()


def exercise_6_realtime_analytics():
    """Real-time analytics with running stats"""
    print("Real-time analytics...")
    
    consumer = KafkaConsumer(
        'transactions',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest'
    )
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    stats = defaultdict(lambda: {
        'count': 0,
        'sum': 0,
        'min': float('inf'),
        'max': float('-inf')
    })
    
    count = 0
    for message in consumer:
        product_id = message.value.get('product_id')
        amount = message.value.get('amount', 0)
        
        s = stats[product_id]
        s['count'] += 1
        s['sum'] += amount
        s['min'] = min(s['min'], amount)
        s['max'] = max(s['max'], amount)
        s['avg'] = s['sum'] / s['count']
        
        result = {'product_id': product_id, **s}
        producer.send('product-stats', result)
        print(f"Product {product_id}: avg=${s['avg']:.2f}, count={s['count']}")
        
        count += 1
        if count >= 20:
            break
    
    producer.close()
    consumer.close()
    print()


def exercise_7_stream_branching():
    """Stream branching by category"""
    print("Stream branching...")
    
    consumer = KafkaConsumer('all-events', bootstrap_servers=BOOTSTRAP_SERVERS,
                            value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS,
                            value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    
    category_counts = defaultdict(int)
    for i, msg in enumerate(consumer):
        severity = msg.value.get('severity', 'info')
        producer.send(f'{severity}-events', msg.value)
        category_counts[severity] += 1
        print(f"→ {severity}-events")
        if i >= 15: break
    
    print("\nDistribution:", dict(category_counts))
    producer.close()
    consumer.close()
    print()


def exercise_8_stateful_processing():
    """Stateful processing with persistent store"""
    print("Stateful processing with state store...")
    
    consumer = KafkaConsumer(
        'orders',
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest'
    )
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Persistent state store
    state_store = shelve.open('order_state.db')
    
    count = 0
    for message in consumer:
        order_id = message.value.get('order_id')
        status = message.value.get('status')
        
        # Read current state
        current_state = state_store.get(order_id, {})
        old_status = current_state.get('status', 'new')
        
        # Update state
        current_state.update({
            'order_id': order_id,
            'status': status,
            'updated_at': time.time(),
            'transitions': current_state.get('transitions', 0) + 1
        })
        
        state_store[order_id] = current_state
        
        # Emit state change
        state_change = {
            'order_id': order_id,
            'old_status': old_status,
            'new_status': status,
            'transitions': current_state['transitions']
        }
        producer.send('order-status', state_change)
        print(f"Order {order_id}: {old_status} → {status}")
        
        count += 1
        if count >= 10:
            break
    
    state_store.close()
    producer.close()
    consumer.close()
    print()


if __name__ == "__main__":
    print("Day 33: Kafka Streams - Solutions\n")
    print("=" * 60)
    print("\nNote: Requires Kafka running on localhost:9092\n")
    
    try:
        print("\nExercise 1: Stateless Transformations")
        print("-" * 60)
        exercise_1_stateless_transformations()
        
        print("\nExercise 2: Word Count")
        print("-" * 60)
        exercise_2_word_count()
        
        print("\nExercise 3: Stream Enrichment")
        print("-" * 60)
        exercise_3_stream_enrichment()
        
        print("\nExercise 4: Tumbling Window")
        print("-" * 60)
        exercise_4_tumbling_window()
        
        print("\nExercise 5: Session Detection")
        print("-" * 60)
        exercise_5_session_detection()
        
        print("\nExercise 6: Real-Time Analytics")
        print("-" * 60)
        exercise_6_realtime_analytics()
        
        print("\nExercise 7: Stream Branching")
        print("-" * 60)
        exercise_7_stream_branching()
        
        print("\nExercise 8: Stateful Processing")
        print("-" * 60)
        exercise_8_stateful_processing()
        
        print("=" * 60)
        print("All exercises completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Kafka is running on localhost:9092")
