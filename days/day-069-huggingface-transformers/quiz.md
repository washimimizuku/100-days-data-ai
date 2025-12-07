# Day 69: Hugging Face Transformers - Quiz

Test your understanding of Hugging Face Transformers and pre-trained models.

## Questions

### 1. What is the main advantage of using transformers over RNNs?
a) Transformers are smaller in size
b) Transformers can process sequences in parallel using self-attention
c) Transformers don't require training data
d) Transformers only work with text data

**Correct Answer: b**

### 2. Which Hugging Face class automatically loads the correct model architecture?
a) `Model.from_pretrained()`
b) `AutoModel.from_pretrained()`
c) `TransformerModel.load()`
d) `HFModel.auto_load()`

**Correct Answer: b**

### 3. What does the `pipeline()` function provide?
a) A way to chain multiple models together
b) A simple API for common tasks with pre-trained models
c) A data preprocessing tool
d) A model training framework

**Correct Answer: b**

### 4. In text generation, what does the temperature parameter control?
a) The speed of generation
b) The length of generated text
c) The randomness/creativity of outputs
d) The model's memory usage

**Correct Answer: c**

### 5. What is the purpose of tokenization in transformers?
a) To encrypt the text
b) To convert text into numerical representations the model can process
c) To remove stop words
d) To translate text to another language

**Correct Answer: b**

### 6. Which model architecture is best suited for text generation tasks?
a) BERT (encoder-only)
b) GPT (decoder-only)
c) ViT (vision transformer)
d) CLIP (multi-modal)

**Correct Answer: b**

### 7. What does `padding=True` do during tokenization?
a) Adds extra tokens to make all sequences the same length
b) Removes unnecessary tokens
c) Increases the model's accuracy
d) Compresses the input text

**Correct Answer: a**

### 8. Which method is used to save a fine-tuned model?
a) `model.save()`
b) `model.export()`
c) `model.save_pretrained()`
d) `torch.save(model)`

**Correct Answer: c**

### 9. What is the Hugging Face Hub?
a) A training platform for models
b) A repository of pre-trained models and datasets
c) A cloud computing service
d) A model compression tool

**Correct Answer: b**

### 10. In the Trainer API, what does `TrainingArguments` specify?
a) The model architecture
b) The dataset format
c) Training hyperparameters and configuration
d) The tokenizer settings

**Correct Answer: c**

## Scoring Guide
- 9-10 correct: Excellent! You understand Hugging Face Transformers well.
- 7-8 correct: Good job! Review the topics you missed.
- 5-6 correct: Fair. Practice more with the exercises.
- Below 5: Review the material and work through the solutions again.

## Answer Key
1. b - Transformers use self-attention for parallel processing
2. b - AutoModel automatically selects the right architecture
3. b - Pipeline provides simple APIs for common tasks
4. c - Temperature controls randomness in generation
5. b - Tokenization converts text to numerical format
6. b - GPT (decoder-only) is designed for generation
7. a - Padding makes all sequences the same length
8. c - save_pretrained() saves model and config
9. b - Hub is a repository of models and datasets
10. c - TrainingArguments specifies training configuration
