"""
Day 30: Kafka Fundamentals - Solutions
"""

from typing import List, Dict

# Exercise 1: Kafka Concepts Quiz
def exercise_1():
    """Answer conceptual questions about Kafka"""
    print("\nQ1: What is the difference between a topic and a partition?")
    print("A1: A topic is a logical category/feed name. A partition is a physical")
    print("    division of a topic for parallelism. Topics contain 1+ partitions.")
    
    print("\nQ2: Why does Kafka use partitions?")
    print("A2: Partitions enable:")
    print("    - Parallelism (multiple consumers can read simultaneously)")
    print("    - Scalability (distribute load across brokers)")
    print("    - Ordering (messages within a partition are ordered)")
    
    print("\nQ3: What is a broker in Kafka?")
    print("A3: A broker is a Kafka server that stores data and serves clients.")
    print("    Kafka runs as a cluster of brokers for fault tolerance and scalability.")
    
    print("\nQ4: What is the purpose of a consumer group?")
    print("A4: Consumer groups enable parallel consumption and load balancing.")
    print("    Each partition is assigned to one consumer in the group.")
    
    print("\nQ5: Can multiple consumer groups read from the same topic?")
    print("A5: Yes! Multiple consumer groups can independently read the same topic.")
    print("    Each group maintains its own offsets and reads all data.")


# Exercise 2: Partition Assignment
def exercise_2():
    """Calculate which partition a message goes to"""
    num_partitions = 3
    
    messages = [
        {"key": "user123", "value": "login"},
        {"key": "user456", "value": "purchase"},
        {"key": "user123", "value": "logout"},
        {"key": "user789", "value": "signup"},
        {"key": "user456", "value": "view"},
    ]
    
    print("\nPartition Assignment (3 partitions):")
    partition_map = {}
    
    for msg in messages:
        key = msg["key"]
        partition = hash(key) % num_partitions
        
        if partition not in partition_map:
            partition_map[partition] = []
        partition_map[partition].append(msg)
        
        print(f"  Message key='{key}', value='{msg['value']}' → Partition {partition}")
    
    print("\nPartition Contents:")
    for partition in sorted(partition_map.keys()):
        keys = [m["key"] for m in partition_map[partition]]
        print(f"  Partition {partition}: {keys}")
    
    print("\nOrdering Guarantee:")
    print("  ✓ user123 messages (login, logout) in same partition → ordered")
    print("  ✓ user456 messages (purchase, view) in same partition → ordered")
    print("  ✗ Messages across partitions NOT ordered relative to each other")


# Exercise 3: Consumer Group Simulation
def exercise_3():
    """Simulate consumer group behavior"""
    num_partitions = 6
    
    scenarios = [
        {"consumers": 2, "group": "group-A"},
        {"consumers": 3, "group": "group-B"},
        {"consumers": 6, "group": "group-C"},
        {"consumers": 8, "group": "group-D"},
    ]
    
    for scenario in scenarios:
        num_consumers = scenario["consumers"]
        group = scenario["group"]
        
        print(f"\n{group}: {num_consumers} consumers, {num_partitions} partitions")
        
        assignment = assign_partitions(num_partitions, num_consumers)
        
        for consumer_id in sorted(assignment.keys()):
            partitions = assignment[consumer_id]
            if partitions:
                print(f"  Consumer {consumer_id}: Partitions {partitions}")
            else:
                print(f"  Consumer {consumer_id}: IDLE (no partitions)")


# Exercise 4: Offset Management
def exercise_4():
    """Track consumer offsets and simulate crashes"""
    partition_messages = [
        {"offset": 0, "value": "msg0"},
        {"offset": 1, "value": "msg1"},
        {"offset": 2, "value": "msg2"},
        {"offset": 3, "value": "msg3"},
        {"offset": 4, "value": "msg4"},
        {"offset": 5, "value": "msg5"},
    ]
    
    print("\nSimulation: Consumer reading with offset commits")
    
    # Initial state
    committed_offset = -1  # No offset committed yet
    current_offset = 0
    
    print(f"\nInitial state: committed_offset={committed_offset}, current_offset={current_offset}")
    
    # Read 3 messages
    print("\nReading 3 messages:")
    for i in range(3):
        msg = partition_messages[current_offset]
        print(f"  Read offset {msg['offset']}: {msg['value']}")
        current_offset += 1
    
    # Commit offset
    committed_offset = current_offset
    print(f"\nCommit offset: {committed_offset}")
    
    # Consumer crashes
    print("\n💥 Consumer crashes!")
    
    # Consumer restarts
    print("\n🔄 Consumer restarts from last committed offset")
    current_offset = committed_offset
    print(f"  Resuming from offset: {current_offset}")
    
    # Read remaining messages
    print("\nReading remaining messages:")
    while current_offset < len(partition_messages):
        msg = partition_messages[current_offset]
        print(f"  Read offset {msg['offset']}: {msg['value']}")
        current_offset += 1
    
    print("\n✓ No messages lost or duplicated (at-least-once with proper commits)")


# Exercise 5: Message Ordering
def exercise_5():
    """Analyze ordering guarantees"""
    scenarios = [
        {
            "description": "All messages from user123 sent with key='user123'",
            "num_partitions": 3,
            "question": "Are messages from user123 ordered?",
            "answer": "YES",
            "reason": "Same key → same partition → ordering guaranteed"
        },
        {
            "description": "Messages sent without keys to 3 partitions",
            "num_partitions": 3,
            "question": "Are all messages globally ordered?",
            "answer": "NO",
            "reason": "Messages distributed across partitions → no global ordering"
        },
        {
            "description": "Single partition topic",
            "num_partitions": 1,
            "question": "Are all messages ordered?",
            "answer": "YES",
            "reason": "Single partition → all messages in order"
        },
        {
            "description": "2 consumers in same group reading 2 partitions",
            "num_partitions": 2,
            "question": "Does each consumer see messages in order?",
            "answer": "YES (per partition)",
            "reason": "Each consumer reads one partition → sees that partition's order"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}:")
        print(f"  Description: {scenario['description']}")
        print(f"  Question: {scenario['question']}")
        print(f"  Answer: {scenario['answer']}")
        print(f"  Reason: {scenario['reason']}")


# Exercise 6: Architecture Design
def exercise_6():
    """Design Kafka architecture for e-commerce"""
    print("\nUse Case: E-commerce Order Processing")
    print("\nArchitecture Design:")
    print("""
    Topics:
    1. 'orders' (6 partitions, replication=3)
       - Key: order_id
       - Retention: 30 days
       - Producers: Web app, mobile app
       
    2. 'order-validations' (6 partitions, replication=3)
       - Key: order_id
       - Retention: 7 days
       - Producers: Validation service
       
    3. 'payments' (3 partitions, replication=3)
       - Key: payment_id
       - Retention: 90 days (compliance)
       - Producers: Payment service
       
    4. 'shipments' (3 partitions, replication=3)
       - Key: order_id
       - Retention: 60 days
       - Producers: Fulfillment service
    
    Consumer Groups:
    1. 'order-processor' (6 consumers)
       - Reads: 'orders'
       - Purpose: Validate and process orders
       - Writes to: 'order-validations'
       
    2. 'payment-processor' (3 consumers)
       - Reads: 'order-validations'
       - Purpose: Process payments
       - Writes to: 'payments'
       
    3. 'fulfillment' (3 consumers)
       - Reads: 'payments'
       - Purpose: Ship orders
       - Writes to: 'shipments'
       
    4. 'analytics' (1 consumer)
       - Reads: All topics
       - Purpose: Real-time analytics dashboard
       - Writes to: Analytics DB
       
    5. 'notifications' (2 consumers)
       - Reads: 'orders', 'payments', 'shipments'
       - Purpose: Send customer notifications
       - Writes to: Email/SMS service
    
    Data Flow:
    Order → Validation → Payment → Fulfillment → Shipment
    
    Guarantees:
    - Orders from same customer processed in order (key=customer_id)
    - At-least-once delivery (acks=all)
    - 3x replication for fault tolerance
    - Independent scaling of each processing stage
    """)


# Helper functions
def assign_partitions(num_partitions: int, num_consumers: int) -> Dict[int, List[int]]:
    """Assign partitions to consumers evenly"""
    assignment = {i: [] for i in range(num_consumers)}
    
    for partition in range(num_partitions):
        consumer = partition % num_consumers
        assignment[consumer].append(partition)
    
    return assignment


if __name__ == "__main__":
    print("Day 30: Kafka Fundamentals - Solutions\n")
    
    print("=" * 50)
    print("Exercise 1: Kafka Concepts Quiz")
    print("=" * 50)
    exercise_1()
    
    print("\n" + "=" * 50)
    print("Exercise 2: Partition Assignment")
    print("=" * 50)
    exercise_2()
    
    print("\n" + "=" * 50)
    print("Exercise 3: Consumer Group Simulation")
    print("=" * 50)
    exercise_3()
    
    print("\n" + "=" * 50)
    print("Exercise 4: Offset Management")
    print("=" * 50)
    exercise_4()
    
    print("\n" + "=" * 50)
    print("Exercise 5: Message Ordering")
    print("=" * 50)
    exercise_5()
    
    print("\n" + "=" * 50)
    print("Exercise 6: Architecture Design")
    print("=" * 50)
    exercise_6()
