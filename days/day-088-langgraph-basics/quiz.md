# Day 88: LangGraph Basics - Quiz

Test your understanding of LangGraph.

---

### Question 1
What is the primary purpose of LangGraph?

A) To make LLMs faster  
B) To build stateful, multi-actor applications with explicit control flow  
C) To replace LangChain entirely  
D) To reduce API costs  

**Correct Answer: B**

---

### Question 2
What are the main components of a LangGraph?

A) Only nodes  
B) Nodes, edges, and state  
C) Just functions  
D) Only LLMs  

**Correct Answer: B**

---

### Question 3
What is a node in LangGraph?

A) A database  
B) A function that processes state  
C) An LLM model  
D) A file  

**Correct Answer: B**

---

### Question 4
How is state passed between nodes?

A) Through global variables  
B) As a shared dictionary/TypedDict  
C) Through files  
D) It's not passed  

**Correct Answer: B**

---

### Question 5
What do conditional edges enable?

A) Faster execution  
B) Dynamic routing based on state  
C) Parallel processing  
D) Error prevention  

**Correct Answer: B**

---

### Question 6
What is the purpose of set_entry_point()?

A) To end the graph  
B) To define where the graph execution starts  
C) To add nodes  
D) To save state  

**Correct Answer: B**

---

### Question 7
How can you create loops in LangGraph?

A) You can't  
B) By adding conditional edges that route back to previous nodes  
C) By using while loops in nodes  
D) By duplicating nodes  

**Correct Answer: B**

---

### Question 8
What is checkpointing used for?

A) Deleting state  
B) Saving and restoring graph state  
C) Speeding up execution  
D) Adding nodes  

**Correct Answer: B**

---

### Question 9
What's the difference between LangChain and LangGraph?

A) They're the same  
B) LangGraph adds graph-based workflows with explicit state management  
C) LangChain is newer  
D) LangGraph doesn't use LLMs  

**Correct Answer: B**

---

### Question 10
What should each node function return?

A) Nothing  
B) The updated state  
C) A string  
D) An error  

**Correct Answer: B**

---

## Answer Key
1. B - To build stateful, multi-actor applications with explicit control flow
2. B - Nodes, edges, and state
3. B - A function that processes state
4. B - As a shared dictionary/TypedDict
5. B - Dynamic routing based on state
6. B - To define where the graph execution starts
7. B - By adding conditional edges that route back to previous nodes
8. B - Saving and restoring graph state
9. B - LangGraph adds graph-based workflows with explicit state management
10. B - The updated state

## Scoring
- 10/10: Expert - Ready to build complex LangGraph applications
- 7-9/10: Proficient - Good understanding of LangGraph
- 4-6/10: Developing - Review state management and routing
- 0-3/10: Beginner - Revisit the theory section
