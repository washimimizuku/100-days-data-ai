# Day 76: Local LLMs with Ollama - Quiz

Test your understanding of running LLMs locally with Ollama.

---

## Question 1
What is the PRIMARY benefit of running LLMs locally with Ollama?

A) Faster inference than cloud APIs  
B) Data privacy and control over sensitive information  
C) Access to larger models than cloud providers  
D) No need for GPU hardware  

**Correct Answer: B**

---

## Question 2
Which component does Ollama use for optimized LLM inference?

A) TensorFlow  
B) PyTorch  
C) llama.cpp  
D) ONNX Runtime  

**Correct Answer: C**

---

## Question 3
What is the approximate memory requirement for running a 7B parameter model?

A) ~2GB RAM  
B) ~4GB RAM  
C) ~8GB RAM  
D) ~16GB RAM  

**Correct Answer: C**

---

## Question 4
Which Ollama command is used to download a model?

A) ollama download llama2  
B) ollama pull llama2  
C) ollama get llama2  
D) ollama install llama2  

**Correct Answer: B**

---

## Question 5
What does the `stream=True` parameter do in Ollama's generate function?

A) Processes multiple prompts simultaneously  
B) Returns tokens as they are generated instead of waiting for completion  
C) Enables GPU acceleration  
D) Compresses the response data  

**Correct Answer: B**

---

## Question 6
Which temperature setting is most appropriate for factual Q&A tasks?

A) 0.0 - 0.3 (low temperature for deterministic outputs)  
B) 0.5 - 0.7 (medium temperature)  
C) 0.8 - 1.0 (high temperature)  
D) 1.5 - 2.0 (very high temperature)  

**Correct Answer: A**

---

## Question 7
What is a Modelfile in Ollama?

A) A Python script for model training  
B) A configuration file for customizing model behavior and parameters  
C) A binary file containing model weights  
D) A log file for model inference  

**Correct Answer: B**

---

## Question 8
How does Ollama's embeddings API help with semantic search?

A) It compresses documents for faster search  
B) It generates vector representations of text for similarity comparison  
C) It indexes documents in a database  
D) It translates text to multiple languages  

**Correct Answer: B**

---

## Question 9
What is the purpose of context truncation in conversational AI?

A) To reduce model size  
B) To keep conversation history within the model's context window limit  
C) To remove profanity from responses  
D) To speed up inference  

**Correct Answer: B**

---

## Question 10
Which model would be most appropriate for quick, simple factual queries?

A) llama2:70b (large, high-quality model)  
B) mistral (balanced 7B model)  
C) phi (small, efficient 2.7B model)  
D) codellama (code-specialized model)  

**Correct Answer: C**

---

## Scoring Guide
- 9-10 correct: Expert - You've mastered local LLM deployment!
- 7-8 correct: Proficient - Strong understanding with minor gaps
- 5-6 correct: Intermediate - Good foundation, review key concepts
- Below 5: Beginner - Review the material and practice more

## Key Takeaways
- Ollama simplifies running LLMs locally with easy model management
- Local deployment provides privacy, cost savings, and offline capability
- Trade-offs include resource requirements and slower inference
- Model selection depends on task requirements (speed vs quality)
- Temperature controls randomness (low for factual, high for creative)
- Streaming enables real-time response generation
- Embeddings enable semantic search and similarity comparison
- Context management is crucial for long conversations
- Modelfiles allow customization of model behavior
- Python API provides programmatic access to all Ollama features
