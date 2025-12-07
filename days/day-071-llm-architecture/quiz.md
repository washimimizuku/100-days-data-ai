# Day 71: LLM Architecture - Quiz

Test your understanding of transformer architecture and LLMs.

## Questions

### 1. What is the main advantage of self-attention over RNNs?
a) Self-attention uses less memory
b) Self-attention can process sequences in parallel
c) Self-attention is easier to implement
d) Self-attention requires less training data

**Correct Answer: b**

### 2. In the attention formula Attention(Q, K, V) = softmax(QK^T / √d_k)V, why do we divide by √d_k?
a) To make computation faster
b) To prevent gradients from vanishing
c) To scale the dot products and prevent softmax saturation
d) To normalize the output

**Correct Answer: c**

### 3. What is the purpose of positional encoding in transformers?
a) To reduce model size
b) To add information about token positions in the sequence
c) To improve training speed
d) To prevent overfitting

**Correct Answer: b**

### 4. Which model architecture is best suited for text generation tasks?
a) Encoder-only (BERT)
b) Decoder-only (GPT)
c) Encoder-decoder (T5)
d) All are equally suitable

**Correct Answer: b**

### 5. What is causal masking used for in decoder models?
a) To reduce memory usage
b) To prevent the model from attending to future tokens
c) To improve training speed
d) To handle variable-length sequences

**Correct Answer: b**

### 6. In multi-head attention, why do we use multiple attention heads?
a) To increase model size
b) To allow the model to focus on different aspects of relationships
c) To make training faster
d) To reduce overfitting

**Correct Answer: b**

### 7. What is the typical ratio of feed-forward dimension to embedding dimension?
a) 1:1
b) 2:1
c) 4:1
d) 8:1

**Correct Answer: c**

### 8. Which normalization approach is more stable for deep transformers?
a) Post-norm (normalize after residual)
b) Pre-norm (normalize before sublayer)
c) Batch normalization
d) No normalization

**Correct Answer: b**

### 9. Approximately how much GPU memory does a 7B parameter model require in fp16?
a) ~7 GB
b) ~14 GB
c) ~28 GB
d) ~56 GB

**Correct Answer: b**

### 10. What is the main difference between BERT and GPT architectures?
a) BERT uses encoder-only, GPT uses decoder-only
b) BERT is larger than GPT
c) BERT uses different activation functions
d) BERT doesn't use attention

**Correct Answer: a**

## Scoring Guide
- 9-10 correct: Excellent! You understand LLM architecture deeply.
- 7-8 correct: Good job! Review the topics you missed.
- 5-6 correct: Fair. Revisit the README and practice more.
- Below 5: Review the material and work through the exercises again.

## Answer Key
1. b - Self-attention processes all tokens in parallel, unlike sequential RNNs
2. c - Scaling prevents large dot products that cause softmax saturation
3. b - Positional encoding adds sequence order information
4. b - Decoder-only (GPT) with causal masking is designed for generation
5. b - Causal masking prevents attending to future tokens during training
6. b - Multiple heads capture different types of relationships
7. c - Typically ff_dim = 4 × embed_dim
8. b - Pre-norm is more stable for training deep networks
9. b - ~2 bytes per parameter in fp16: 7B × 2 ≈ 14 GB
10. a - BERT is encoder-only (bidirectional), GPT is decoder-only (causal)
