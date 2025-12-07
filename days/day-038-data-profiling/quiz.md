# Day 38 Quiz: Data Profiling

## Questions

1. **What is data profiling?**
   - A) Creating user profiles
   - B) Examining data to understand structure, content, quality, and relationships
   - C) Optimizing database performance
   - D) Encrypting sensitive data

2. **What are the three main types of data profiling?**
   - A) Fast, Medium, Slow
   - B) Structure, Content, Relationship profiling
   - C) Input, Process, Output
   - D) Read, Write, Update

3. **What does the IQR method detect?**
   - A) Missing values
   - B) Outliers using interquartile range (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
   - C) Data types
   - D) Correlations

4. **What is cardinality in categorical columns?**
   - A) The data type
   - B) The ratio of unique values to total values
   - C) The number of missing values
   - D) The column length

5. **What does the Shapiro-Wilk test check?**
   - A) Missing data patterns
   - B) Whether data is normally distributed
   - C) Correlation strength
   - D) Outlier presence


6. **What is skewness in data distribution?**
   - A) Missing data percentage
   - B) Measure of asymmetry in distribution (positive = right-skewed, negative = left-skewed)
   - C) Number of outliers
   - D) Data quality score

7. **What does a high cardinality ratio (>50%) indicate?**
   - A) Many missing values
   - B) Many unique values relative to total rows
   - C) High correlation
   - D) Normal distribution

8. **What is the Z-score method for outlier detection?**
   - A) Values with |z-score| > 3 are outliers
   - B) Values below mean are outliers
   - C) Values above median are outliers
   - D) All negative values are outliers

9. **Why is correlation analysis important in profiling?**
   - A) To find missing values
   - B) To identify relationships between variables and potential multicollinearity
   - C) To detect outliers
   - D) To check data types

10. **What is ydata-profiling (formerly pandas-profiling)?**
    - A) A database tool
    - B) An automated tool that generates comprehensive HTML data profile reports
    - C) A data visualization library
    - D) A machine learning framework

---

## Answers

1. **B** - Data profiling is the process of examining data to understand its structure (shape, types), content (values, distributions), quality (missing, outliers), and relationships (correlations).

2. **B** - The three main types are: Structure profiling (understanding dataset structure), Content profiling (analyzing actual data values), and Relationship profiling (understanding relationships between columns).

3. **B** - The IQR (Interquartile Range) method detects outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR, where IQR = Q3 - Q1.

4. **B** - Cardinality is the ratio of unique values to total values. Low cardinality (<1%) means few unique values; high cardinality (>50%) means many unique values.

5. **B** - The Shapiro-Wilk test checks whether data follows a normal distribution. If p-value > 0.05, data is considered normally distributed.

6. **B** - Skewness measures asymmetry in distribution. Positive skewness (>0) indicates right-skewed (tail on right), negative skewness (<0) indicates left-skewed (tail on left).

7. **B** - High cardinality ratio (>50%) indicates many unique values relative to total rows. This might suggest the column is an identifier or has high variability.

8. **A** - The Z-score method identifies outliers as values with |z-score| > 3 (more than 3 standard deviations from the mean).

9. **B** - Correlation analysis identifies relationships between variables and potential multicollinearity (high correlation between predictors), which is important for feature selection and model building.

10. **B** - ydata-profiling (formerly pandas-profiling) is an automated tool that generates comprehensive HTML reports with statistics, distributions, correlations, missing data analysis, and more.

---

## Scoring

- **9-10 correct**: Excellent! You understand data profiling deeply.
- **7-8 correct**: Good! Review outlier detection and distribution analysis.
- **5-6 correct**: Fair. Revisit profiling types and statistical concepts.
- **Below 5**: Review the README and examples again.
