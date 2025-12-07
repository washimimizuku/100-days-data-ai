# Day 73: Prompt Engineering

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand prompt engineering principles and best practices
- Master zero-shot and few-shot prompting techniques
- Learn instruction following patterns
- Create effective prompt templates
- Handle common prompting challenges
- Optimize prompts for different tasks

---

## What is Prompt Engineering?

### Definition

Prompt engineering is the practice of designing inputs to get desired outputs from language models:

```python
# Bad prompt
"Write about AI"

# Good prompt
"Write a 200-word explanation of artificial intelligence 
for a 10-year-old, using simple analogies and avoiding 
technical jargon."
```

### Why It Matters

- **No fine-tuning needed**: Get results with just text
- **Cost-effective**: Cheaper than training custom models
- **Flexible**: Adapt to new tasks quickly
- **Accessible**: Anyone can do it

---

## Core Principles

```python
# 1. Be Specific and Clear
"Summarize this article in 3 bullet points, focusing on main findings and implications"

# 2. Provide Context
"I'm building a real-time data pipeline processing 1M events/sec. Best approach for backpressure and consistency?"

# 3. Use Examples (Few-Shot)
"""Classify sentiment: Text: "I love this!" → positive | Text: "It's okay" → neutral | Text: "Terrible!" → negative
Text: "This product is amazing!" → Sentiment:"""

# 4. Specify Format
"List 5 benefits in JSON: {'benefits': [{'title': '...', 'description': '...'}, ...]}"
```

---

## Zero-Shot & Few-Shot Prompting

```python
# Zero-shot: Direct instructions
prompt = "Translate to French: 'Hello, how are you?' → French:"

# Zero-shot: Task description
prompt = "Extract person names, locations, dates from: 'John Smith visited Paris on January 15, 2024.'"

# Few-shot: Pattern learning (SQL)
prompt = """Convert to SQL:
Q: Show all users → SELECT * FROM users;
Q: Find users older than 25 → SELECT * FROM users WHERE age > 25;
Q: List top 10 users by score → SQL:"""

# Few-shot: Classification
prompt = """Classify feedback:
"Fast delivery!" → Shipping | "Poor quality" → Product Quality | "Great value" → Category:"""
```

---

## Instruction Following & Templates

```python
# System message
system_message = "You are a Python expert. Include: type hints, docstrings, error handling, examples."

# Role playing
prompt = "You are a senior data engineer. Review this code for quality, performance, best practices."

# Question answering template
qa_template = "Context: {context}\nQuestion: {question}\nAnswer based only on context. If not found, say 'I don't know.'\nAnswer:"

# Summarization template
summary_template = "Summarize in {num_sentences} sentences, focusing on {focus_area}:\nText: {text}\nSummary:"

# Code generation template
code_template = "Task: {task}\nLanguage: {language}\nRequirements: {requirements}\nGenerate with type hints, error handling, examples.\nCode:"
```

---

## Advanced Techniques

```python
# Chain of Thought (CoT): Step-by-step reasoning
prompt = """Problem: Store has 15 apples, sells 7 morning, 5 afternoon. How many left?
Let's solve step by step: 1) Start: 15 | 2) Morning: 15-7=8 | 3) Afternoon: 8-5=3 | Answer: 3"""

# Self-Consistency: Multiple reasoning paths
prompt = "Solve using 3 different approaches:\nProblem: {problem}\nApproach 1:\nApproach 2:\nApproach 3:\nMost consistent:"

# Prompt Chaining: Break into steps
# Step 1: extract_prompt = "Extract facts: {text}"
# Step 2: analyze_prompt = "Analyze: {facts}"
# Step 3: summary_prompt = "Summarize: {analysis}"
```

---

## Common Patterns

```python
# Classification
classification_prompt = "Classify into: {categories}\nText: {text}\nCategory:"

# Entity extraction
extraction_prompt = "Extract: Names (PERSON), Organizations (ORG), Locations (LOC), Dates (DATE)\nText: {text}\nEntities:"

# Sentiment analysis
sentiment_prompt = "Analyze sentiment. Provide: overall (positive/negative/neutral), confidence (0-1), key phrases.\nText: {text}\nAnalysis:"
```

---

## Best Practices & Optimization

```python
# 1. Iterate and refine: "Summarize" → "Summarize in 3 sentences" → "Summarize in 3 sentences, key findings, bullet points, include numbers"

# 2. Use delimiters: "Summarize text between triple backticks: ```{text}``` Summary:"

# 3. Specify constraints: "Generate product description: Length 50-75 words, tone professional/friendly, include features/benefits/audience, avoid jargon"

# 4. Handle edge cases: "Answer from context. If not found: 'Not in context'. If unclear: ask clarification. Cite sources."

# 5. Temperature control: Creative (0.7-1.0), Factual (0.0-0.3), Balanced (0.5-0.7)

# 6. Token management: "Summarize in 50 words" or "Target 100-150 words, focus on main arguments"

# Common pitfalls:
# ❌ Ambiguous: "Make it better" → ✅ Specific: "Improve readability: shorter sentences, subheadings, examples"
# ❌ Overloading: "Summarize, translate, analyze, extract" → ✅ One task: "Summarize in 3 sentences"
# ❌ Assuming knowledge: "Use standard approach" → ✅ Explicit: "Use MapReduce: split, process parallel, combine"
```

---

## 💻 Exercises (40 min)

Practice prompt engineering in `exercise.py`:

### Exercise 1: Zero-Shot Classification
Create prompts for sentiment classification without examples.

### Exercise 2: Few-Shot Learning
Design few-shot prompts for custom classification tasks.

### Exercise 3: Instruction Following
Write clear instructions for code generation.

### Exercise 4: Prompt Templates
Create reusable templates for common tasks.

### Exercise 5: Prompt Optimization
Refine prompts to improve output quality.

---

## ✅ Quiz

Test your understanding of prompt engineering in `quiz.md`.

---

## 🎯 Key Takeaways

- **Be specific**: Clear instructions yield better results
- **Provide context**: Help the model understand the task
- **Use examples**: Few-shot learning improves accuracy
- **Specify format**: Structure outputs for easy parsing
- **Iterate**: Refine prompts based on results
- **Use delimiters**: Clearly separate instructions from content
- **Set constraints**: Define length, tone, and style
- **Handle edge cases**: Account for unexpected inputs
- **Chain prompts**: Break complex tasks into steps

---

## 📚 Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Learn Prompting](https://learnprompting.org/)
- [Best Practices for Prompt Engineering](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)

---

## Tomorrow: Day 74 - Few-Shot Learning

Deep dive into few-shot learning techniques and how to select effective examples for in-context learning.
