# Day 86: ReAct Pattern - Quiz

Test your understanding of the ReAct pattern.

---

### Question 1
What does ReAct stand for?

A) React to Actions  
B) Reasoning and Acting  
C) Recursive Action  
D) Real-time Acting  

**Correct Answer: B**

---

### Question 2
What is the correct order of steps in a ReAct loop?

A) Action → Thought → Observation  
B) Observation → Thought → Action  
C) Thought → Action → Observation  
D) Thought → Observation → Action  

**Correct Answer: C**

---

### Question 3
What is the main advantage of ReAct over direct prompting?

A) It's faster  
B) It uses less tokens  
C) It provides explicit reasoning and can use tools  
D) It doesn't need an LLM  

**Correct Answer: C**

---

### Question 4
Which action type is used to return the final answer in ReAct?

A) return  
B) answer  
C) finish  
D) complete  

**Correct Answer: C**

---

### Question 5
What happens after an action is executed in ReAct?

A) The agent terminates  
B) An observation is generated  
C) The question changes  
D) All tools are reset  

**Correct Answer: B**

---

### Question 6
Which of these is NOT a typical ReAct action type?

A) search  
B) calculate  
C) compile  
D) lookup  

**Correct Answer: C**

---

### Question 7
What is a key challenge when implementing ReAct agents?

A) They are too simple  
B) Managing hallucinations and token costs  
C) They can't use tools  
D) They don't support reasoning  

**Correct Answer: B**

---

### Question 8
In ReAct, what is the purpose of the "Thought" step?

A) To execute actions  
B) To reason about what to do next  
C) To store observations  
D) To terminate the loop  

**Correct Answer: B**

---

### Question 9
How does ReAct differ from Chain of Thought?

A) ReAct doesn't use reasoning  
B) ReAct can take actions and use tools  
C) ReAct is faster  
D) ReAct doesn't use LLMs  

**Correct Answer: B**

---

### Question 10
What should a ReAct agent do when max iterations are reached?

A) Continue indefinitely  
B) Return an error or synthesize available information  
C) Delete all observations  
D) Restart from the beginning  

**Correct Answer: B**

---

## Answer Key
1. B - Reasoning and Acting
2. C - Thought → Action → Observation
3. C - It provides explicit reasoning and can use tools
4. C - finish
5. B - An observation is generated
6. C - compile
7. B - Managing hallucinations and token costs
8. B - To reason about what to do next
9. B - ReAct can take actions and use tools
10. B - Return an error or synthesize available information

## Scoring
- 10/10: Expert - Ready to build production ReAct agents
- 7-9/10: Proficient - Good understanding of ReAct pattern
- 4-6/10: Developing - Review the ReAct loop and examples
- 0-3/10: Beginner - Revisit the theory section
