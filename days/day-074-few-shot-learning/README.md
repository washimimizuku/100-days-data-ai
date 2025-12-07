# Day 74: Few-Shot Learning

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand in-context learning and how it works
- Master example selection strategies for few-shot prompts
- Compare few-shot learning with fine-tuning
- Learn techniques to optimize few-shot performance
- Handle edge cases and failure modes
- Apply few-shot learning to various tasks

---

## What is Few-Shot Learning?

Few-shot learning enables models to perform tasks with just a few examples, without updating weights:

```python
# Zero-shot: "Classify sentiment: 'I love this product!'"
# Few-shot: "Text: 'Amazing!' → positive | 'Terrible' → negative | 'I love this!' → Sentiment:"

# In-context learning: No training, immediate, flexible, limited by context window
# Pattern recognition: Model identifies Input → Output mapping from examples
# Context window: GPT-3 (2048), GPT-4 (8192), Claude (100k) - trade-off between examples and input length
```

---

## Example Selection Strategies

```python
# 1. Diversity: Cover different cases
"Code: print('Hi') → Python | console.log('Hi') → JavaScript | System.out.println('Hi') → Java"

# 2. Relevance: Similar to expected inputs
"Ticket: 'Order not arrived' → Shipping | 'Product defective' → Quality | 'Need setup help' → Support"

# 3. Difficulty progression: Simple to complex
"Q: Show users → SELECT * FROM users; | Q: Find John → SELECT * WHERE name='John'; | Q: Count by country → ..."

# 4. Edge cases: Include boundaries
"Email: 'user@example.com' → true | 'invalid.email' → false | '' → false | 'user@domain.co.uk' → true"
```

---

## Few-Shot vs Fine-Tuning

### Comparison

| Aspect | Few-Shot | Fine-Tuning |
|--------|----------|-------------|
| Training | None | Required |
| Examples | 1-10 | 100s-1000s |
| Cost | Low | High |
| Speed | Instant | Hours/Days |
| Flexibility | High | Low |
| Performance | Good | Better |
| Customization | Limited | Extensive |

### When to Use Few-Shot

```python
# Use few-shot when:
# - Quick prototyping
# - Limited training data
# - Task changes frequently
# - No GPU resources
# - Need immediate results

# Use fine-tuning when:
# - High accuracy required
# - Large training dataset available
# - Task is stable
# - Have compute resources
# - Can afford training time
```

---

## Optimizing Few-Shot Performance

```python
# 1. Example quality: Clear, accurate, representative
"Text: 'John Smith (john@email.com) called 2024-01-15' → {'name': 'John Smith', 'email': 'john@email.com', 'date': '2024-01-15'}"

# 2. Example order: Random, easy-to-hard, similar-first, or balanced (experiment!)

# 3. Number of examples: Sweet spot 3-8 (too few = poor learning, too many = token waste)

# 4. Format consistency: Use same format for all examples
# ✅ "Input: {text}\nOutput: {label}" | ❌ "Text: {text} -> {label}" then "{text}: {label}"
```

---

## Advanced Techniques

```python
# Dynamic example selection: Choose k most similar examples to query
def select_examples(query, example_pool, k=3):
    similarities = [(cosine_similarity(embed(query), embed(ex['input'])), ex) for ex in example_pool]
    similarities.sort(reverse=True)
    return [ex for _, ex in similarities[:k]]

# Example augmentation: Generate variations
"'Great product!' → positive" → "'Excellent product!' → positive", "'Amazing product!' → positive"

# Instruction + examples: Combine for clarity
"Task: Classify sentiment. Instructions: positive/negative/neutral, consider tone. Examples: 'Love it!' → positive | 'Disappointed' → negative | 'It's okay' → neutral. Now: '{text}' →"
```

---

## Common Patterns

```python
# Classification: "Classify into {categories}:\n{examples}\nText: {input}\nCategory:"
# Extraction: "Extract {fields}:\n{examples}\nText: {input}\nExtracted:"
# Transformation: "Transform as shown:\n{examples}\nInput: {input}\nOutput:"
# Generation: "Generate {output_type}:\n{examples}\nInput: {input}\nGenerated:"
```

---

## Handling Failure Modes & Best Practices

```python
# Failure modes:
# 1. Inconsistent outputs → Specify exact format: "Output: JSON with 'label' and 'confidence'"
# 2. Overfitting to examples → Add diverse examples + "Apply logic to new inputs, don't copy"
# 3. Ignoring instructions → Put instructions after examples + "IMPORTANT: Follow pattern above"

# Best practices:
# 1. Start simple: Begin with 2-3 clear examples, add more only if needed
# 2. Balance examples: Equal representation (Positive: 3, Negative: 3, Neutral: 3)
# 3. Clear separators: Use delimiters "Example 1:\n---\nInput: ...\nOutput: ..."
# 4. Test thoroughly: Typical, edge, ambiguous, out-of-distribution cases
```

---

## Practical Applications

```python
# Text classification
"Article: 'Stock market high...' → Business | 'New vaccine...' → Health | 'Team wins...' → Sports | '{new}' → Category:"

# Named entity recognition
"Text: 'Apple CEO Tim Cook...' → [{'text': 'Apple', 'type': 'ORG'}, {'text': 'Tim Cook', 'type': 'PERSON'}] | '{new}' → Entities:"

# Code generation
"Task: Sum list → def sum_list(numbers): return sum(numbers) | Task: Find max → def find_max(numbers): return max(numbers) | Task: {new} → Code:"
```

---

## 💻 Exercises (40 min)

Practice few-shot learning in `exercise.py`:

### Exercise 1: Example Selection
Select optimal examples for a classification task.

### Exercise 2: Dynamic Selection
Implement similarity-based example selection.

### Exercise 3: Format Consistency
Create consistent few-shot prompts.

### Exercise 4: Performance Optimization
Optimize few-shot prompts through iteration.

### Exercise 5: Edge Case Handling
Design prompts that handle edge cases.

---

## ✅ Quiz

Test your understanding of few-shot learning in `quiz.md`.

---

## 🎯 Key Takeaways

- **Few-shot learning** uses examples to teach tasks without training
- **Example selection** is crucial for performance
- **Diversity** in examples improves generalization
- **3-8 examples** is usually optimal
- **Consistent formatting** is essential
- **Order matters** - experiment with different orderings
- **Dynamic selection** can improve results
- **Balance** examples across classes
- **Test thoroughly** with edge cases
- **Few-shot** is faster but fine-tuning may be more accurate

---

## 📚 Resources

- [Language Models are Few-Shot Learners (GPT-3 Paper)](https://arxiv.org/abs/2005.14165)
- [In-Context Learning Survey](https://arxiv.org/abs/2301.00234)
- [What Makes Good In-Context Examples?](https://arxiv.org/abs/2101.06804)
- [OpenAI Few-Shot Learning Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)

---

## Tomorrow: Day 75 - Chain of Thought

Learn how to improve reasoning by asking models to show their step-by-step thinking process.
