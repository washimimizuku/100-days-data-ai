# Day 22 Quiz: Spark Architecture

## Questions

1. **What is the role of the driver in Spark?**
   - A) Execute tasks
   - B) Orchestrate application, schedule tasks, collect results
   - C) Store data
   - D) Manage memory

2. **What is the role of executors?**
   - A) Schedule jobs
   - B) Execute tasks and store data in memory/disk
   - C) Manage cluster
   - D) Optimize queries

3. **What is lazy evaluation?**
   - A) Slow execution
   - B) Transformations delayed until action is called
   - C) Caching data
   - D) Parallel processing

4. **What triggers job execution in Spark?**
   - A) Transformations
   - B) Actions (show, count, collect, write)
   - C) Creating SparkSession
   - D) Reading data

5. **What is the difference between job and stage?**
   - A) No difference
   - B) Job contains stages; stages separated by shuffles
   - C) Stage contains jobs
   - D) Jobs are faster

6. **What are Spark's main components?**
   - A) Only Spark Core
   - B) Core, SQL, Streaming, MLlib, GraphX
   - C) Only Spark SQL
   - D) Driver and Executor

7. **What is the Spark UI used for?**
   - A) Writing code
   - B) Monitoring jobs, stages, tasks, and debugging
   - C) Deploying applications
   - D) Managing clusters

8. **What is a cluster manager?**
   - A) A database
   - B) Allocates resources across applications (YARN, K8s, Mesos)
   - C) A monitoring tool
   - D) A storage system

9. **What happens during a shuffle?**
   - A) Data is cached
   - B) Data is redistributed across partitions
   - C) Data is deleted
   - D) Data is compressed

10. **Why is Spark faster than MapReduce?**
    - A) Better algorithms
    - B) In-memory computing and lazy evaluation
    - C) More servers
    - D) Simpler code

---

## Answers

1. **B** - Driver orchestrates the application, converts code to execution plan, schedules tasks, and collects results.
2. **B** - Executors run tasks assigned by driver and store data in memory/disk for caching.
3. **B** - Lazy evaluation means transformations are not executed until an action is called, enabling optimization.
4. **B** - Actions like show(), count(), collect(), write() trigger job execution.
5. **B** - A job contains one or more stages. Stages are separated by shuffle boundaries (e.g., groupBy, join).
6. **B** - Spark has Core (RDD), SQL (DataFrames), Streaming, MLlib (ML), and GraphX (graphs).
7. **B** - Spark UI monitors job execution, shows stages/tasks, memory usage, and helps debug performance issues.
8. **B** - Cluster manager (YARN, Kubernetes, Mesos, Standalone) allocates resources across Spark applications.
9. **B** - Shuffle redistributes data across partitions, required for operations like groupBy, join, repartition.
10. **B** - Spark is 100x faster due to in-memory computing, lazy evaluation, and optimized execution plans.

---

## Scoring

- **9-10 correct**: Excellent! You understand Spark architecture.
- **7-8 correct**: Good! Review driver/executor roles and lazy evaluation.
- **5-6 correct**: Fair. Revisit execution hierarchy and components.
- **Below 5**: Review the README and architecture diagrams again.
