# Day 75: Chain of Thought - Quiz

Test your understanding of Chain of Thought (CoT) prompting techniques.

---

## Question 1
What is the primary purpose of Chain of Thought (CoT) prompting?

A) To reduce the number of tokens in LLM responses  
B) To make the model show its reasoning steps explicitly  
C) To speed up model inference time  
D) To eliminate the need for few-shot examples  

**Correct Answer: B**

---

## Question 2
Which of the following is a key characteristic of Zero-Shot CoT?

A) Requires multiple examples with reasoning steps  
B) Uses the phrase "Let's think step by step" without examples  
C) Only works with mathematical problems  
D) Requires fine-tuning the model  

**Correct Answer: B**

---

## Question 3
In Few-Shot CoT, what should each example include?

A) Only the final answer  
B) The question and answer without reasoning  
C) The question, step-by-step reasoning, and final answer  
D) Multiple possible answers  

**Correct Answer: C**

---

## Question 4
What is "self-consistency" in CoT prompting?

A) Ensuring the prompt is grammatically correct  
B) Generating multiple reasoning paths and selecting the most common answer  
C) Using the same prompt for all questions  
D) Verifying the model's confidence score  

**Correct Answer: B**

---

## Question 5
Which type of problem benefits MOST from CoT prompting?

A) Simple factual recall questions  
B) Multi-step reasoning and complex problem-solving  
C) Single-word classification tasks  
D) Image generation tasks  

**Correct Answer: B**

---

## Question 6
What is a potential drawback of CoT prompting?

A) It reduces model accuracy  
B) It increases token usage and inference time  
C) It only works with GPT-4  
D) It requires model fine-tuning  

**Correct Answer: B**

---

## Question 7
In the context of CoT, what does "reasoning extraction" refer to?

A) Removing unnecessary tokens from the prompt  
B) Parsing and analyzing the intermediate reasoning steps  
C) Extracting the model's training data  
D) Converting reasoning to a different language  

**Correct Answer: B**

---

## Question 8
Which of the following is an example of a CoT prompt trigger phrase?

A) "Answer quickly"  
B) "Let's work through this step by step"  
C) "Give me the answer only"  
D) "Be concise"  

**Correct Answer: B**

---

## Question 9
What is "least-to-most prompting" in CoT?

A) Starting with the hardest problems first  
B) Breaking complex problems into simpler sub-problems solved sequentially  
C) Using the fewest possible tokens  
D) Ranking answers from worst to best  

**Correct Answer: B**

---

## Question 10
When should you prefer Zero-Shot CoT over Few-Shot CoT?

A) When you have many high-quality examples available  
B) When you want to minimize prompt length and don't have good examples  
C) When working with image classification  
D) When the model is too small for reasoning  

**Correct Answer: B**

---

## Scoring Guide
- 9-10 correct: Expert - You have mastered CoT prompting techniques!
- 7-8 correct: Proficient - Strong understanding with minor gaps
- 5-6 correct: Intermediate - Good foundation, review key concepts
- Below 5: Beginner - Review the material and practice more

## Key Takeaways
- CoT prompting makes LLMs show their reasoning explicitly
- Zero-Shot CoT uses trigger phrases like "Let's think step by step"
- Few-Shot CoT provides examples with complete reasoning chains
- Self-consistency improves accuracy by sampling multiple reasoning paths
- CoT is most effective for complex, multi-step reasoning tasks
- Trade-off: Better accuracy vs. increased token usage and latency
