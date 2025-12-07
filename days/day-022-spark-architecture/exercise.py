"""
Day 22: Spark Architecture - Exercises
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Exercise 1: Initialize Spark
def exercise_1():
    """Create SparkSession with configuration"""
    # TODO: Create SparkSession with app name "Day22"
    # TODO: Set master to local[4]
    # TODO: Configure executor memory to 2g
    # TODO: Print Spark version
    # TODO: Access Spark UI at http://localhost:4040
    pass

# Exercise 2: Understand Execution
def exercise_2():
    """Observe job execution in Spark UI"""
    # TODO: Create DataFrame from range(1000000)
    # TODO: Add transformations (filter, map)
    # TODO: Trigger action (count)
    # TODO: Check Spark UI for job/stages/tasks
    pass

# Exercise 3: Lazy Evaluation
def exercise_3():
    """Demonstrate lazy evaluation"""
    # TODO: Chain 5 transformations
    # TODO: Verify no execution (check Spark UI)
    # TODO: Trigger action
    # TODO: Observe execution in Spark UI
    pass

# Exercise 4: Memory Configuration
def exercise_4():
    """Configure and monitor memory"""
    # TODO: Create DataFrame
    # TODO: Cache it
    # TODO: Trigger action
    # TODO: Check Storage tab in Spark UI
    pass

# Exercise 5: Deployment Modes
def exercise_5():
    """Compare deployment configurations"""
    # TODO: Show local[*] configuration
    # TODO: Show local[4] configuration
    # TODO: Simulate cluster config (10 executors, 4g each)
    pass

if __name__ == "__main__":
    print("Day 22: Spark Architecture\n")
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
