# Day 87: Tool Use & Function Calling - Quiz

Test your understanding of tool use and function calling.

---

### Question 1
What is the primary purpose of function calling in LLMs?

A) To make LLMs faster  
B) To enable LLMs to interact with external systems  
C) To reduce token usage  
D) To improve text generation  

**Correct Answer: B**

---

### Question 2
What format is typically used to define function schemas?

A) XML  
B) YAML  
C) JSON Schema  
D) Plain text  

**Correct Answer: C**

---

### Question 3
Which field is NOT required in a function schema?

A) name  
B) description  
C) parameters  
D) version  

**Correct Answer: D**

---

### Question 4
What does the LLM output when making a function call?

A) Plain text  
B) Structured JSON with function name and arguments  
C) Python code  
D) SQL query  

**Correct Answer: B**

---

### Question 5
How should function descriptions be written?

A) As long as possible  
B) Clear and specific about when to use the function  
C) In technical jargon  
D) Without any details  

**Correct Answer: B**

---

### Question 6
What is a best practice for the number of tools to provide to an LLM?

A) As many as possible  
B) Exactly 100  
C) Keep under 20 to avoid confusion  
D) Only 1 tool at a time  

**Correct Answer: C**

---

### Question 7
What should you do when a function execution fails?

A) Ignore the error  
B) Restart the entire system  
C) Return an error message to the LLM  
D) Delete the function  

**Correct Answer: C**

---

### Question 8
Which parameter type allows only specific values?

A) string  
B) number  
C) enum  
D) boolean  

**Correct Answer: C**

---

### Question 9
What is an advantage of function calling over plain text responses?

A) Uses fewer tokens  
B) Provides structured, parseable output  
C) Doesn't require an LLM  
D) Works without internet  

**Correct Answer: B**

---

### Question 10
When should parameters be validated?

A) Never  
B) Only after execution  
C) Before executing the function  
D) Only on weekends  

**Correct Answer: C**

---

## Answer Key
1. B - To enable LLMs to interact with external systems
2. C - JSON Schema
3. D - version
4. B - Structured JSON with function name and arguments
5. B - Clear and specific about when to use the function
6. C - Keep under 20 to avoid confusion
7. C - Return an error message to the LLM
8. C - enum
9. B - Provides structured, parseable output
10. C - Before executing the function

## Scoring
- 10/10: Expert - Ready to build production function calling systems
- 7-9/10: Proficient - Good understanding of function calling
- 4-6/10: Developing - Review schemas and execution flow
- 0-3/10: Beginner - Revisit the theory section
