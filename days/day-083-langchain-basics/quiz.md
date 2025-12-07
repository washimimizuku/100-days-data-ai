# Day 83: LangChain Basics - Quiz

Test your understanding of LangChain framework for LLM applications.

---

## Question 1
What is the primary purpose of LangChain?

A) To train language models  
B) To provide abstractions and tools for building LLM applications  
C) To host language models in the cloud  
D) To compress language models  

**Correct Answer: B**

---

## Question 2
What do Prompt Templates in LangChain enable?

A) Faster model inference  
B) Reusable, parameterized prompts with variable substitution  
C) Model fine-tuning  
D) Automatic prompt generation  

**Correct Answer: B**

---

## Question 3
What is the purpose of Chains in LangChain?

A) To compress data  
B) To connect multiple LLM calls or operations sequentially  
C) To train models  
D) To store embeddings  

**Correct Answer: B**

---

## Question 4
What does ConversationBufferMemory do?

A) Compresses conversation history  
B) Stores and maintains conversation context across turns  
C) Deletes old messages automatically  
D) Encrypts conversation data  

**Correct Answer: B**

---

## Question 5
What is the difference between ConversationBufferMemory and ConversationBufferWindowMemory?

A) Window memory is faster  
B) Window memory keeps only the last k messages  
C) Buffer memory is more accurate  
D) They are the same  

**Correct Answer: B**

---

## Question 6
What does RetrievalQA chain implement?

A) Question generation  
B) RAG pattern with retrieval and generation  
C) Model training  
D) Data compression  

**Correct Answer: B**

---

## Question 7
What is the purpose of Text Splitters in LangChain?

A) To translate text  
B) To chunk documents into smaller pieces for processing  
C) To compress text  
D) To encrypt text  

**Correct Answer: B**

---

## Question 8
What does the "stuff" chain type in RetrievalQA do?

A) Compresses all documents  
B) Stuffs all retrieved documents into the prompt  
C) Removes irrelevant documents  
D) Summarizes documents first  

**Correct Answer: B**

---

## Question 9
What is the purpose of Output Parsers?

A) To speed up generation  
B) To structure and validate LLM responses into specific formats  
C) To compress outputs  
D) To translate outputs  

**Correct Answer: B**

---

## Question 10
Which LangChain component would you use to load documents from a directory?

A) TextLoader  
B) DirectoryLoader  
C) FileReader  
D) DocumentManager  

**Correct Answer: B**

---

## Scoring Guide
- 9-10 correct: Expert - You've mastered LangChain basics!
- 7-8 correct: Proficient - Strong understanding with minor gaps
- 5-6 correct: Intermediate - Good foundation, review key concepts
- Below 5: Beginner - Review the material and practice more

## Key Takeaways
- LangChain provides abstractions for LLM applications
- Prompt templates enable reusable, parameterized prompts
- Chains connect multiple LLM operations sequentially
- Memory maintains conversation context
- ConversationBufferMemory stores all messages
- ConversationBufferWindowMemory keeps only last k messages
- Document loaders handle various file formats
- Text splitters chunk documents for processing
- Vector stores enable semantic search
- RetrievalQA implements RAG patterns
- Output parsers structure LLM responses
- Components are modular and composable
