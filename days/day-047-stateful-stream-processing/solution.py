"""
Day 47: Stateful Stream Processing - Solutions
"""
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
from pyspark.sql.streaming import GroupState, GroupStateTimeout
from pyspark.sql.types import *
import time


def exercise_1_running_total():
    """Running total with mapGroupsWithState"""
    spark = SparkSession.builder \
        .appName("RunningTotal") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Create stream
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("user_id", expr("CAST(value % 5 AS STRING)")) \
        .withColumn("amount", expr("10 + (value % 50)"))
    
    # Define state update function
    def update_total(user_id, values, state):
        if state.exists:
            total = state.get
        else:
            total = 0
        
        for value in values:
            total += value.amount
        
        state.update(total)
        return Row(user_id=user_id, total=total)
    
    # Define output schema
    output_schema = StructType([
        StructField("user_id", StringType()),
        StructField("total", LongType())
    ])
    
    # Apply mapGroupsWithState
    result = df.groupByKey(lambda x: x.user_id) \
        .mapGroupsWithState(
            update_total,
            output_schema,
            output_schema,
            GroupStateTimeout.NoTimeout
        )
    
    query = result.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Running total per user")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_2_session_detection():
    """Session detection with timeout"""
    spark = SparkSession.builder \
        .appName("SessionDetection") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Create events stream
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("user_id", expr("CAST(value % 3 AS STRING)")) \
        .withColumn("event_type", lit("click"))
    
    # Session state schema
    session_schema = StructType([
        StructField("start_time", LongType()),
        StructField("end_time", LongType()),
        StructField("event_count", IntegerType())
    ])
    
    # Output schema
    output_schema = StructType([
        StructField("user_id", StringType()),
        StructField("session_start", LongType()),
        StructField("session_end", LongType()),
        StructField("event_count", IntegerType()),
        StructField("status", StringType())
    ])
    
    def detect_sessions(user_id, events, state):
        """Detect sessions with 30-second timeout for demo"""
        
        if state.hasTimedOut:
            # Session timed out, output it
            session = state.get
            state.remove()
            return iter([Row(
                user_id=user_id,
                session_start=session.start_time,
                session_end=session.end_time,
                event_count=session.event_count,
                status="completed"
            )])
        
        # Get current session
        if state.exists:
            session = state.get
        else:
            session = None
        
        outputs = []
        
        for event in events:
            event_time = event.timestamp.timestamp()
            
            if session is None:
                # Start new session
                session = Row(
                    start_time=int(event_time),
                    end_time=int(event_time),
                    event_count=1
                )
            else:
                # Continue session
                session = Row(
                    start_time=session.start_time,
                    end_time=int(event_time),
                    event_count=session.event_count + 1
                )
        
        # Update state with timeout
        if session:
            state.update(session)
            state.setTimeoutDuration("30 seconds")
        
        return iter(outputs)
    
    result = df.groupByKey(lambda x: x.user_id) \
        .flatMapGroupsWithState(
            detect_sessions,
            output_schema,
            session_schema,
            GroupStateTimeout.ProcessingTimeTimeout,
            outputMode="append"
        )
    
    query = result.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Session detection with 30-second timeout")
    query.awaitTermination(timeout=60)
    query.stop()
    spark.stop()


def exercise_3_event_sequencing():
    """Event sequence detection"""
    spark = SparkSession.builder \
        .appName("EventSequencing") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Create events with different types
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("user_id", expr("CAST(value % 3 AS STRING)")) \
        .withColumn("event_type", 
            expr("CASE WHEN value % 3 = 0 THEN 'view' " +
                 "WHEN value % 3 = 1 THEN 'cart' " +
                 "ELSE 'purchase' END"))
    
    # State schema (list of event types)
    state_schema = ArrayType(StringType())
    
    # Output schema
    output_schema = StructType([
        StructField("user_id", StringType()),
        StructField("pattern", StringType()),
        StructField("timestamp", TimestampType())
    ])
    
    def detect_sequence(user_id, events, state):
        """Detect view -> cart -> purchase pattern"""
        
        if state.exists:
            sequence = list(state.get)
        else:
            sequence = []
        
        outputs = []
        last_timestamp = None
        
        for event in events:
            sequence.append(event.event_type)
            last_timestamp = event.timestamp
            
            # Check for conversion pattern
            if len(sequence) >= 3 and \
               sequence[-3:] == ["view", "cart", "purchase"]:
                outputs.append(Row(
                    user_id=user_id,
                    pattern="conversion",
                    timestamp=last_timestamp
                ))
                sequence = []  # Reset after match
        
        # Keep only last 10 events
        if len(sequence) > 10:
            sequence = sequence[-10:]
        
        state.update(sequence)
        return iter(outputs)
    
    result = df.groupByKey(lambda x: x.user_id) \
        .flatMapGroupsWithState(
            detect_sequence,
            output_schema,
            state_schema,
            GroupStateTimeout.NoTimeout,
            outputMode="append"
        )
    
    query = result.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Event sequence detection: view -> cart -> purchase")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


def exercise_4_active_user_tracking():
    """Active user tracking with timeout"""
    spark = SparkSession.builder \
        .appName("ActiveUserTracking") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Create activity stream
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 5) \
        .load() \
        .withColumn("user_id", expr("CAST(value % 5 AS STRING)"))
    
    # State schema
    state_schema = LongType()
    
    # Output schema
    output_schema = StructType([
        StructField("user_id", StringType()),
        StructField("status", StringType()),
        StructField("last_seen", LongType())
    ])
    
    def track_active_users(user_id, events, state):
        """Track with 20-second timeout for demo"""
        
        if state.hasTimedOut:
            # User became inactive
            last_activity = state.get
            state.remove()
            return iter([Row(
                user_id=user_id,
                status="inactive",
                last_seen=last_activity
            )])
        
        # Process events
        latest_time = None
        for event in events:
            latest_time = int(event.timestamp.timestamp())
        
        # Update state
        if latest_time:
            state.update(latest_time)
            state.setTimeoutDuration("20 seconds")
            
            return iter([Row(
                user_id=user_id,
                status="active",
                last_seen=latest_time
            )])
        
        return iter([])
    
    result = df.groupByKey(lambda x: x.user_id) \
        .flatMapGroupsWithState(
            track_active_users,
            output_schema,
            state_schema,
            GroupStateTimeout.ProcessingTimeTimeout,
            outputMode="append"
        )
    
    query = result.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Active user tracking with 20-second timeout")
    query.awaitTermination(timeout=60)
    query.stop()
    spark.stop()


def exercise_5_anomaly_detection():
    """Anomaly detection using statistics"""
    spark = SparkSession.builder \
        .appName("AnomalyDetection") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Create stream with values
    df = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 10) \
        .load() \
        .withColumn("sensor_id", expr("CAST(value % 3 AS STRING)")) \
        .withColumn("reading", expr("50 + (value % 20) + " +
                                   "CASE WHEN value % 50 = 0 THEN 100 ELSE 0 END"))
    
    # State schema (history of values)
    state_schema = StructType([
        StructField("values", ArrayType(DoubleType())),
        StructField("mean", DoubleType()),
        StructField("stddev", DoubleType())
    ])
    
    # Output schema
    output_schema = StructType([
        StructField("sensor_id", StringType()),
        StructField("reading", DoubleType()),
        StructField("z_score", DoubleType()),
        StructField("is_anomaly", BooleanType())
    ])
    
    def detect_anomalies(sensor_id, readings, state):
        """Detect anomalies using z-score"""
        
        if state.exists:
            history = state.get
            values = list(history.values)
            mean = history.mean
            stddev = history.stddev
        else:
            values = []
            mean = 0.0
            stddev = 0.0
        
        outputs = []
        
        for reading in readings:
            value = float(reading.reading)
            
            # Check for anomaly
            if stddev > 0:
                z_score = abs(value - mean) / stddev
                is_anomaly = z_score > 2.5  # 2.5 std devs
                
                if is_anomaly:
                    outputs.append(Row(
                        sensor_id=sensor_id,
                        reading=value,
                        z_score=z_score,
                        is_anomaly=True
                    ))
            
            # Update history
            values.append(value)
            if len(values) > 50:  # Keep last 50
                values = values[-50:]
            
            # Recalculate stats
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                stddev = variance ** 0.5
        
        # Update state
        state.update(Row(values=values, mean=mean, stddev=stddev))
        return iter(outputs)
    
    result = df.groupByKey(lambda x: x.sensor_id) \
        .flatMapGroupsWithState(
            detect_anomalies,
            output_schema,
            state_schema,
            GroupStateTimeout.NoTimeout,
            outputMode="append"
        )
    
    query = result.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    print("Anomaly detection using z-score (threshold: 2.5)")
    query.awaitTermination(timeout=30)
    query.stop()
    spark.stop()


if __name__ == "__main__":
    print("Day 47: Stateful Stream Processing Solutions\n")
    
    # Run exercises
    print("\n" + "="*60)
    print("Exercise 1: Running Total")
    print("="*60)
    exercise_1_running_total()
    
    print("\n" + "="*60)
    print("Exercise 2: Session Detection")
    print("="*60)
    exercise_2_session_detection()
    
    print("\n" + "="*60)
    print("Exercise 3: Event Sequencing")
    print("="*60)
    exercise_3_event_sequencing()
    
    print("\n" + "="*60)
    print("Exercise 4: Active User Tracking")
    print("="*60)
    exercise_4_active_user_tracking()
    
    print("\n" + "="*60)
    print("Exercise 5: Anomaly Detection")
    print("="*60)
    exercise_5_anomaly_detection()
