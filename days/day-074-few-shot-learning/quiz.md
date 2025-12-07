# Day 74: Few-Shot Learning - Quiz

Test your understanding of few-shot learning and in-context learning.

## Questions

### 1. What is the main advantage of few-shot learning over fine-tuning?
a) Better accuracy
b) No training required, works immediately
c) Uses less memory
d) Works with smaller models

**Correct Answer: b**

### 2. How many examples are typically optimal for few-shot learning?
a) 1-2 examples
b) 3-8 examples
c) 20-50 examples
d) 100+ examples

**Correct Answer: b**

### 3. What is the most important criterion when selecting few-shot examples?
a) Length of examples
b) Alphabetical order
c) Diversity and representativeness
d) Recency of examples

**Correct Answer: c**

### 4. What is in-context learning?
a) Training a model on new data
b) Learning patterns from examples in the prompt
c) Using a larger context window
d) Fine-tuning with few examples

**Correct Answer: b**

### 5. Why is format consistency important in few-shot prompts?
a) It looks more professional
b) It helps the model recognize and follow the pattern
c) It reduces token count
d) It's required by the API

**Correct Answer: b**

### 6. What is dynamic example selection?
a) Randomly selecting examples
b) Selecting examples based on similarity to the input
c) Changing examples during inference
d) Using different examples for each model

**Correct Answer: b**

### 7. When should you use fine-tuning instead of few-shot learning?
a) When you need immediate results
b) When you have limited data
c) When you need highest accuracy and have sufficient data
d) When the task changes frequently

**Correct Answer: c**

### 8. What is a common failure mode in few-shot learning?
a) Model runs out of memory
b) Model overfits to the specific examples
c) Model refuses to answer
d) Model becomes slower

**Correct Answer: b**

### 9. How should examples be balanced in few-shot classification?
a) All examples from the most common class
b) Equal representation across classes
c) More examples from rare classes
d) Balance doesn't matter

**Correct Answer: b**

### 10. What should you include when few-shot examples alone aren't enough?
a) More examples (20+)
b) Longer examples
c) Clear instructions explaining the task
d) Random examples

**Correct Answer: c**

## Scoring Guide
- 9-10 correct: Excellent! You understand few-shot learning well.
- 7-8 correct: Good job! Review the topics you missed.
- 5-6 correct: Fair. Revisit the README and practice more.
- Below 5: Review the material and work through the exercises again.

## Answer Key
1. b - Few-shot learning requires no training, works immediately with examples
2. b - 3-8 examples is typically the sweet spot for most tasks
3. c - Diverse, representative examples improve generalization
4. b - In-context learning means learning from examples in the prompt
5. b - Consistent format helps the model recognize and follow patterns
6. b - Dynamic selection chooses examples similar to the input
7. c - Fine-tuning is better when accuracy is critical and data is available
8. b - Models may copy examples too literally instead of generalizing
9. b - Balanced examples across classes prevents bias
10. c - Instructions help when examples alone aren't sufficient
