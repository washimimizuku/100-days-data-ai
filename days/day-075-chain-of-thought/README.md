# Day 75: Chain of Thought

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand Chain of Thought (CoT) reasoning
- Master step-by-step prompting techniques
- Learn self-consistency methods
- Apply CoT to complex problem solving
- Optimize reasoning quality
- Handle multi-step reasoning tasks

---

## What is Chain of Thought?

Chain of Thought (CoT) prompting shows reasoning step-by-step:

```python
# Without CoT: "What is 15% of 80?" → Answer: 12
# With CoT: "What is 15% of 80? Let's solve step by step." → Step 1: 0.15 | Step 2: 0.15×80=12 | Answer: 12

# Why it works: Breaks complexity, reduces errors, shows reasoning, improves accuracy (math, logic, reasoning)

# Trigger phrases: "Let's think step by step", "Let's solve this step by step", "Let's break this down"

# Example: "Store has 15 apples, sells 7 morning, 5 afternoon. How many left?"
# "Step 1: Start 15 | Step 2: 15-7=8 | Step 3: 8-5=3 | Answer: 3 apples"
```

---

## Zero-Shot & Few-Shot CoT

```python
# Zero-shot: Just add "Let's think step by step"
"Train: 60 km/h for 2h, then 80 km/h for 1.5h. Total distance? Let's think step by step:"
# → Step 1: 60×2=120km | Step 2: 80×1.5=120km | Step 3: 120+120=240km | Answer: 240km

# Few-shot: Provide examples with step-by-step reasoning
"Problem: Book $12, buy 3. Cost? → Step 1: $12/book | Step 2: $12×3=$36 | Answer: $36
Problem: Rectangle 8m×5m. Area? → Step 1: length×width | Step 2: 8×5=40m² | Answer: 40m²
Problem: Car uses 8L/100km. For 250km? → Solution:"
```

---

## Self-Consistency

```python
# Generate multiple solutions and pick most common answer
"Problem: {problem} → Approach 1: {r1} | Approach 2: {r2} | Approach 3: {r3} | Most consistent:"

# Majority voting implementation
def self_consistency(problem, num_samples=5):
    answers = [extract_final_answer(generate(f"{problem}\n\nLet's think step by step:")) for _ in range(num_samples)]
    return most_common(answers)
```

---

## Complex Reasoning Tasks

```python
# Multi-step math: "120 employees, 40% remote, 60% of remote in different zones. How many?"
# "Step 1: 120×0.40=48 remote | Step 2: 48×0.60=28.8 | Step 3: Round to 29 | Answer: 29"

# Logical reasoning: "All cats are mammals. All mammals are animals. Is Fluffy (cat) an animal?"
# "1) Fluffy is cat | 2) Cats are mammals → Fluffy is mammal | 3) Mammals are animals → Fluffy is animal | Answer: Yes"

# Word problems: "Sarah has 2× Tom's books. Tom has 5 more than Lisa. Lisa has 8. Sarah has?"
# "1) Lisa: 8 | 2) Tom: 8+5=13 | 3) Sarah: 13×2=26 | Answer: 26 books"
```

---

## Advanced Techniques

```python
# Least-to-Most: Break into subproblems
"Problem: {complex} → Subproblem 1: {sub1} Solution: {sol1} | Subproblem 2: {sub2} Solution: {sol2} | Final: {combine}"

# Tree of Thoughts: Explore branches
"Problem: {problem} → Approaches: A) {a} B) {b} C) {c} | Evaluate: A) Pros/Cons | B) Pros/Cons | C) Pros/Cons | Best: {selected} → Solution"

# Verification: Self-checking
"Problem: {problem} → Solution: {steps} → Verify: Check step 1: {v1} | Check step 2: {v2} | Check final: {vf} | Confirmed: {answer}"
```

---

## Domain-Specific CoT

```python
# Code debugging: "Error: ZeroDivisionError in calculate_average([]) → 1) Error: div by 0 | 2) Cause: empty list | 3) Fix: check if not numbers: return 0"

# Data analysis: "Sales: Q1 +20%, Q2 -10%, Q3 +15% → 1) Q1: +20% | 2) Q2: -10% (net +8%) | 3) Q3: +15% (net +24.2%) | Conclusion: Positive trend, investigate Q2"

# Decision making: "Migrate to cloud? → Current: $10k/mo | Cloud: $7k/mo | Migration: $50k | Savings: $3k/mo | Break-even: 17mo | Decision: Yes (positive ROI)"
```

---

## Best Practices & Optimization

```python
# Best practices:
# 1. Clear step labels: "Step 1: Calculate X | Step 2: Calculate Y" (not "First, then, finally...")
# 2. Show intermediate results: "15×0.20=3, 15+3=18" (not "Calculate, Add, Answer: 18")
# 3. Explicit reasoning: "Since X>Y, use formula A" (not just "Use formula A")
# 4. Verify logic: "Check: 18-3=15 ✓" (not just "Answer: 18")

# Common pitfalls:
# 1. Skipping steps: Include all steps, don't jump from Step 1 to Step 3
# 2. Incorrect intermediate steps: Use verification or self-consistency
# 3. Over-complication: Use 3-4 clear steps, not 10 steps for simple problems

# Optimization:
# 1. Prompt engineering: "Solve step by step: 1) Identify knowns 2) Find unknowns 3) Show calculations 4) Verify"
# 2. Temperature: Reasoning (0.1-0.3), Creative (0.7-0.9)
# 3. Example quality: Clear labels, explicit reasoning, correct solutions, similar complexity
```

---

## 💻 Exercises (40 min)

Practice Chain of Thought reasoning in `exercise.py`:

### Exercise 1: Basic CoT
Create step-by-step solutions for math problems.

### Exercise 2: Self-Consistency
Implement multiple reasoning paths with voting.

### Exercise 3: Complex Reasoning
Solve multi-step word problems with CoT.

### Exercise 4: Verification
Add verification steps to solutions.

### Exercise 5: Domain Application
Apply CoT to code debugging and data analysis.

---

## ✅ Quiz

Test your understanding of Chain of Thought in `quiz.md`.

---

## 🎯 Key Takeaways

- **CoT prompting** shows step-by-step reasoning
- **"Let's think step by step"** activates CoT
- **Self-consistency** uses multiple paths and voting
- **Breaks down complexity** into manageable steps
- **Improves accuracy** especially for reasoning tasks
- **Verification** catches errors in reasoning
- **Clear step labels** improve readability
- **Show intermediate results** for transparency
- **Domain-specific** CoT for specialized tasks
- **Lower temperature** for reasoning tasks

---

## 📚 Resources

- [Chain-of-Thought Prompting Paper](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Paper](https://arxiv.org/abs/2203.11171)
- [Tree of Thoughts Paper](https://arxiv.org/abs/2305.10601)
- [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625)
- [OpenAI CoT Guide](https://platform.openai.com/docs/guides/prompt-engineering)

---

## Tomorrow: Day 76 - Local LLMs with Ollama

Learn how to run large language models locally using Ollama for privacy and control.
