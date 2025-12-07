"""Day 23: Spark DataFrames - Solutions"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pandas as pd

spark = SparkSession.builder.appName("Day23").getOrCreate()

def exercise_1():
    data = [(1, "Alice", 25, "NYC"), (2, "Bob", 30, "LA")]
    df1 = spark.createDataFrame(data, ["id", "name", "age", "city"])
    
    data_dict = [{"id": 3, "name": "Charlie", "age": 35}]
    df2 = spark.createDataFrame(data_dict)
    
    pdf = pd.DataFrame({"id": [4, 5], "name": ["David", "Eve"]})
    df3 = spark.createDataFrame(pdf)
    
    df1.printSchema()
    df1.show()

def exercise_2():
    data = [(1, "Alice", 25), (2, "Bob", 30), (3, "Charlie", 20)]
    df = spark.createDataFrame(data, ["id", "name", "age"])
    
    df.select("name", "age").show()
    df.filter(col("age") > 25).show()
    df.select("name").filter(col("age") > 25).show()

def exercise_3():
    data = [(1, "Alice", 25), (2, "Bob", 30)]
    df = spark.createDataFrame(data, ["id", "name", "age"])
    
    df = df.withColumn("age_plus_10", col("age") + 10)
    df = df.withColumnRenamed("age", "years")
    df = df.withColumn("category", when(col("years") < 30, "young").otherwise("old"))
    df.show()

def exercise_4():
    data = [(1, "Alice", 25, "NYC"), (2, "Bob", 30, "NYC"), (3, "Charlie", 35, "LA")]
    df = spark.createDataFrame(data, ["id", "name", "age", "city"])
    
    df.groupBy("city").count().show()
    df.groupBy("city").agg(avg("age"), max("age")).show()
    
    window = Window.partitionBy("city").orderBy("age")
    df.withColumn("rank", rank().over(window)).show()

def exercise_5():
    data = [(1, "Alice", 25), (2, "Bob", 30)]
    df = spark.createDataFrame(data, ["id", "name", "age"])
    df.createOrReplaceTempView("users")
    
    spark.sql("SELECT * FROM users WHERE age > 25").show()
    spark.sql("SELECT AVG(age) as avg_age FROM users").show()

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    spark.stop()
