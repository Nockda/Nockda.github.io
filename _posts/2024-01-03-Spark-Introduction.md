---
layout: post
title: Introduction to Apache Spark 
subtitle: How to use the Spark
cover-img: /assets/img/spark/spark_cover.png
thumbnail-img: /assets/img/spark/spark_logo.png
share-img: /assets/img/spark/spark_logo.png
tags: [Apache spark, Big data]
comments: true
---

#### Using Language and Tools

**Language:** Python  
**Tool:** Spark (PySpark)

Apache Spark has emerged as one of the most prominent tools in the realm of big data processing. In this guide, I'll provide an introduction on how to utilize Apache Spark, particularly catering to beginners.

**Apache Spark API Reference:** [Spark Python API](https://spark.apache.org/docs/latest/api/python/reference/index.html)

---

## Start Spark

First, we should import pyspark and make SparkContext active upon JVM.

```python
import pyspark 
# When you make a context for the first time.
sc = pyspark.SparkContext('local[*]')
# This line can be used when you run spark again.
# sc = pyspark.SparkContext.getOrCreate()

# class pyspark.SparkContext (
#    master = None,
#    appName = None, 
#    sparkHome = None, 
#    pyFiles = None, 
#    environment = None, 
#    batchSize = 0, 
#    serializer = PickleSerializer(), 
#    conf = None, 
#    gateway = None, 
#    jsc = None, 
#    profiler_cls = <class 'pyspark.profiler.BasicProfiler'>
# )

# Master − It is the URL of the cluster it connects to.
# appName − Name of your job.
# sparkHome − Spark installation directory.
# pyFiles − The .zip or .py files to send to the cluster and add to the PYTHONPATH.
# Environment − Worker nodes environment variables.
# batchSize − The number of Python objects represented as a single Java object. Set 1 to disable batching, 0 to automatically choose the batch size based on object sizes, or -1 to use an unlimited batch size.
# Serializer − RDD serializer.
# Conf − An object of L{SparkConf} to set all the Spark properties.
# Gateway − Use an existing gateway and JVM, otherwise initializing a new JVM.
# JSC − The JavaSparkContext instance.
# profiler_cls − A class of custom Profiler used to do profiling (the default is pyspark.profiler.BasicProfiler).

# SparkContext uses Py4J to launch a JVM and creates a JavaSparkContext. 
# By default, PySpark has SparkContext available as ‘sc’, so creating a new SparkContext won't work.

```

In here, pyspark.SparkContext should be active once.

## Counting Words

For this example, I used 'Lorem ipsum' text file. This is the dummy text file.

```python
# Create or get SparkContext
sc = pyspark.SparkContext.getOrCreate()

# import text file
test_file = "file:/__filelocation__"
text_file = sc.textFile(test_file)

# flatMap() -> map -> reduceByKey
counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)

print(counts.collect())

```

1. __flatMap()__
This operation splits each line into words, resulting in a flat list of words.

2. __map()__
Each word is transformed into a key-value pair, where the word is the key, and the value is initially set to 1.

3. __reduceByKey()__
This operation groups the key-value pairs by the word and sums up the associated values, giving the total count of each word.

4. __Collect()__
The results are collected and printed. Because of lazy evaluation, the previous code delays the execution of operations until this action(collect()) is invoked. 

## Count by key

grade.txt is like this.

      tom 70
      sara 80
      joon 100
      kevin 90
      John 90

if I want to count the number by the grade I should use countByValue()

```python
import collections
import pyspark

# Create or get SparkContext
sc = pyspark.SparkContext.getOrCreate()

test_file = "file:/__grade.txt__"
text_file = sc.textFile(test_file)

# Extract the second value as grade.
grade = text_file.map(lambda line: line.split(" ")[1])

## Use countByValue() to count the occurrences of each unique grade
# Return the count of each unique value in this RDD as a dictionary of (value, count) pairs.
grade_count = grade.countByValue()

for grade, count in sorted(grade_count.items(), key=lambda item: item[1], reverse=True):
    print(f"{grade}: {count}")
```

Print output like this.

      90: 2
      70: 1
      80: 1
      100: 1


## Average the value

If I want to calculate the value, I can use this code.

example.scv is like below
| City    | Price | Count |
|---------|------------|----------------------|
| Seoul   | 10,000     | 3                    |
| Seoul   | 10,000     | 5                    |
| Seoul   | 40,000     | 7                    |
| Busan   | 5,000      | 7                    |
| Incheon | 4,000      | 2                    |
| Busan   | 9,000      | 4                    |
| Incheon | 5,000      | 7                    |
| Seoul   | 4,000      | 2                    |
| Busan   | 8,000      | 9                    |



```python
import pyspark 

sc = pyspark.SparkContext.getOrCreate()
test_file = "file:/__example.scv__"

# Read the CSV file and parse the lines
def parse_line(line: str):
    city, price, count = line.split(',')
    return (int(price), int(count))

# Map values to (count, 1) and then reduce by key to get total count and total number of entries
lines = sc.textFile(test_file)
price_count = lines.map(parse_line)
# [(10000, 3), (10000, 5), (40000, 7), (5000, 7), (4000, 2), (9000, 4), (5000, 7), (4000, 2), (8000, 9)]

# Calculate average count for each price
sum_of_count = price_count.mapValues(lambda count: (count, 1))\
                .reduceByKey(lambda a, b: (int(a[0]) + int(b[0]), int(a[1]) + int(b[1]))) 

# ('10000', (3, 1)), ('10000', (5, 1)) ...
# [('10000', (8, 2)), ('4000', (4, 2)), ('9000', ('4', 1)), ('8000', ('9', 1)), ('40000', ('7', 1)), ('5000', (14, 2))]

# Collect and print the results
avg_by_count = sum_of_count.mapValues(lambda total_count: int(total_count[0]) / total_count[1])
results = avg_by_count.collect()
print(results)
```


1. Parse Data:
The provided code parses the CSV data, extracting the price and count for each line.
2. MapReduce Transformation:
It then performs a MapReduce transformation to sum the counts and occurrences for each price.
3. Calculate Average:
After the transformation, the code calculates the average count for each price.
4. Collect and Print:
Finally, the results are collected and printed.


## Find Min & Max(Filtering)

temperature.csv is the dummy file automatically generated.(Not a real data.) The length of this file is 48,750. this is the top 10 row of the file.

| record_id | month | day | year | MaxTemp | MinTemp |
|-----------|-------|-----|------|------------------------|------------------------------------|
| 474376    | 1     | 1   | 1853 | NA                     | NA                                 |
| 474377    | 2     | 1   | 1853 | NA                     | NA                                 |
| 474378    | 3     | 1   | 1853 | NA                     | NA                                 |
| 474379    | 4     | 1   | 1853 | NA                     | NA                                 |
| 474380    | 5     | 1   | 1853 | NA                     | NA                                 |
| 474381    | 6     | 1   | 1853 | 51.9062                | 36.9572                            |
| 474382    | 7     | 1   | 1853 | 52.3886                | 34.5488                            |
| 474383    | 8     | 1   | 1853 | 52.853                 | 33.5498                            |
| 474384    | 9     | 1   | 1853 | 52.5776                | 33.638                             |



```python
# Filter
# Return a new RDD containing only the elements that satisfy a predicate.

import pyspark

sc = pyspark.SparkContext.getOrCreate()
test_file = "file:/__temperature.csv__"

def get_data(line, header):
    if line != header:
        col = line.split(',')
        city = col[6].strip("\"")
        avg_temp_fahr = col[4]
        yield (city, avg_temp_fahr)
        # "yield" is for lazy return

lines = sc.textFile(test_file)

# Get the header string
header = lines.first()

# Parse the lines and filter out "NA" values
parsed_line = lines.flatMap(lambda line: get_data(line, header))
filtered_line = parsed_line.filter(lambda x: "NA" not in x[1])
```

This is the filtering Max temperature

```python
# finding max temperature
min_temp = filtered_line.reduceByKey(lambda x, y: max(float(x), float(y)))
final_list = min_temp.collect()
print("Max temperature\n")
for city, temperature in final_list:
    print(f"{city}: {temperature}")
```

This is the filtering Min temperature

```python
# finding min temperature
min_temp = filtered_line.reduceByKey(lambda x, y: min(float(x), float(y)))
final_list = min_temp.collect()
print("Min temperature\n")
for city, temperature in final_list:
    print(f"{city}: {temperature}")
```

This code snippet filters out "NA" values from the temperature data and then calculates both the maximum and minimum temperatures for each city. The data displayed represents a dummy dataset for illustration purposes.



## Map vs FlatMap

**Map Transformation:**
The `map` transformation applies a function to each row in a DataFrame/Dataset and returns the new transformed Dataset.

Example:
```
1 => 1
```

**flatMap Transformation:**
The `flatMap` transformation flattens the DataFrame/Dataset after applying the function to every element and returns a new transformed Dataset. The resulting Dataset will contain more rows than the original DataFrame, making it a one-to-many transformation function.

Example:
```
1 => Many
```

One valuable use case of `flatMap()` is to flatten columns that contain arrays, lists, or any nested collections. This transformation is particularly useful when dealing with nested structures in the data.

#### Map

```python
rdd = sc.parallelize([("name", "joe,sarah,tom"), ("car", "hyundai")])
result = rdd.map(lambda x: x[1].split(","))
print(result.collect())
```
output of __Map__ function : [['joe', 'sarah', 'tom'], ['hyundai']]

#### FlatMap

```python
rdd = sc.parallelize([("name", "joe,sarah,tom"), ("car", "hyundai")])
result = rdd.flatMap(lambda x: x[1].split(","))
print(result.collect())

```
output of __FlatMap__ funtion : ['joe', 'sarah', 'tom', 'hyundai']




## Using SQL

The top 3 rows of `income.txt` are as follows:

| Name             | Country        | Email                            | Income  |
|------------------|----------------|----------------------------------|---------|
| Kam Long         | Dominica       | VinThomas@example.taobao          | 137,611 |
| Jamey Warner     | Botswana       | badlittleduck@test.gf             | 134,999 |
| Theola Page      | Malawi         | sharvin@test.mint                 | 171,808 |

```python
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import col

def parse_line(line: str):
    fields = line.split('|')
    return Row(
        name=str(fields[0]),
        country=str(fields[1]),
        email=str(fields[2]),
        compensation=int(fields[3]))

# Create Spark session
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# Read data from income.txt
lines = spark.sparkContext.textFile("file:/__income.txt__")
income_data = lines.map(parse_line)

# Create a DataFrame and cache it
# Creates a DataFrame from an RDD, a list or a pandas.DataFrame.
# SparkSession.createDataFrame(data, schema=None, samplingRatio=None, verifySchema=True)[source]
# .cache() => data is too small. So upload all the data to memory.
schema_income = spark.createDataFrame(data=income_data).cache()

# Creates or replaces a local temporary view with this DataFrame. = naming table name. -> "income"
schema_income.createOrReplaceTempView("income")

# Query the DataFrame using SQL
medium_income_df = spark.sql("SELECT * FROM income WHERE compensation >= 70,000 AND compensation <= 100,000")

# Show the result
#.show() => show the summary chart
medium_income_df.show()

# Group by country and display the count in descending order
schema_income.groupBy("country").count().orderBy(col("count").desc()).show()
```


## Analyze CSV file

In this PySpark script, we use the PySpark SQL API to explore and analyze a CSV file containing information about individuals.

#### Import Libraries

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, round as rnd
```

#### Create Spark Session and Load Data

Top 3 row of age.csv is like below.

| Name            | Age | Country                        |
|-----------------|-----|--------------------------------|
| Neville Hardy   | 56  | Niue                           |
| Dacia Cohen     | 74  | Falkland Islands (Malvinas)   |
| Kathey Daniel   | 10  | Slovenia                       |


```python
spark = SparkSession.builder.appName("sql_import_csv").getOrCreate()
csv_file_path = "file:/__age.csv__"
data = spark.read.option("header", "true")\
                 .option("inferSchema", "true")\
                 .csv(csv_file_path)
```

Here, we create a Spark session named "sql_import_csv" and load CSV data into a DataFrame. We use options like `header` and `inferSchema` to handle CSV specifics.

#### Display Schema

```python
data.printSchema()
```

This line prints the schema of the DataFrame, showing the data types of each column.

#### Explore Data

The following lines are commented out, but they demonstrate various operations you can perform on the DataFrame:

- **Select specific columns:**
  ```python
  data.select("name", "age").show()
  ```

- **Filter data for individuals aged 20 and above:**
  ```python
  data.filter(data.age > 20).show()
  ```

- **Group by age and calculate counts:**
  ```python
  data.groupBy("age").count().show()
  ```

- **Custom arithmetic operation (subtract 10 from age):**
  ```python
  data.select(data.name, data.age, data.age - 10).show()
  ```

- **Create a column alias for age:**
  ```python
  data.select(data.name, col("age").alias("age1")).show()
  ```

- **Calculate the average age per country and display the result:**
  ```python
  data.select(data.name, data.age, data.country)\
          .groupBy("country")\
          .avg("age").show()
  ```

- **Calculate the average age per country, sort by average age:**
  ```python
  data.select(data.name, data.age, data.country)\
          .groupBy("country")\
          .avg("age").sort("avg(age)").show()
  ```

- **Calculate the average age per country, round the result, and display:**
  ```python
  data.select(data.name, data.age, data.country)\
          .groupBy("country")\
          .agg(rnd(avg("age"), 2).alias("avg_age")).show()
  ```

These operations showcase the flexibility and power of PySpark SQL in exploring and analyzing data in csv file.