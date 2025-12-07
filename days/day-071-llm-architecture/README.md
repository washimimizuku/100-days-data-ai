# Day 71: LLM Architecture

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand transformer architecture in depth
- Learn how attention mechanisms work
- Differentiate between encoder, decoder, and encoder-decoder models
- Understand model scaling and parameter counts
- Explore popular LLM architectures (GPT, BERT, T5)
- Learn about positional encodings and embeddings

---

## Transformer Architecture Overview

### The Transformer Revolution

Transformers replaced RNNs/LSTMs as the dominant architecture for sequence modeling:

**Key Advantages**:
- Parallel processing (no sequential dependency)
- Long-range dependencies via attention
- Scalable to billions of parameters
- Transfer learning capabilities

**Core Components**:
1. **Self-Attention**: Weighs importance of tokens
2. **Feed-Forward Networks**: Token-wise transformations
3. **Layer Normalization**: Stabilizes training
4. **Residual Connections**: Enables deep networks

---

## Self-Attention Mechanism

### How Attention Works

Attention allows each token to "attend to" other tokens in the sequence:

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)  # (batch, seq_len, 3*embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.out(attn_output)
        
        return output, attn_weights
```

### Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (Query): What we're looking for
- K (Key): What each token offers
- V (Value): The actual information
- d_k: Dimension of keys (for scaling)
```

---

## Multi-Head Attention

Multiple attention heads allow the model to focus on different aspects (syntax, semantics, position, entities):

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # Linear projections and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: scores = softmax(QK^T / √d_k) V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out_linear(output)
```

---

## Positional Encoding

### Why Positional Information?

Transformers have no inherent notion of sequence order. Positional encodings add position information:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

```python
# Sinusoidal (original): PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d))
# Learned (GPT, BERT): self.pos_embedding = nn.Embedding(max_len, embed_dim)
```

---

## Encoder-Only Models (BERT)

**Architecture**: Input → Embeddings → [Encoder Block] × N → Output  
**Encoder Block**: Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm  
**Use Cases**: Text classification, NER, question answering, sentence similarity

```python
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
    
    def forward(self, x, mask=None):
        x = self.norm1(x + self.attention(x, x, x, mask))  # Self-attention with residual
        x = self.norm2(x + self.ff(x))  # Feed-forward with residual
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        for layer in self.layers: x = layer(x, mask)
        return x
```

---

## Decoder-Only Models (GPT)

**Architecture**: Input → Embeddings → [Decoder Block] × N → Output  
**Decoder Block**: Masked Multi-Head Self-Attention (causal) → Add & Norm → Feed-Forward → Add & Norm  
**Use Cases**: Text generation, code completion, dialogue systems, creative writing

```python
def create_causal_mask(seq_len):
    """Prevent attending to future tokens"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # [[1,0,0,0], [1,1,0,0], [1,1,1,0], [1,1,1,1]]
```

---

## Encoder-Decoder Models (T5)

**Architecture**: Encoder (bidirectional) + Decoder (causal + cross-attention to encoder)  
**Cross-Attention**: Query from decoder, Key/Value from encoder output  
**Use Cases**: Translation, summarization, question answering, text-to-text tasks

---

## Model Scaling

### Parameter Counts

```python
# Typical LLM sizes:
Small:   125M - 350M parameters
Base:    350M - 1B parameters
Large:   1B - 7B parameters
XL:      7B - 70B parameters
XXL:     70B+ parameters

# GPT-3: 175B parameters
# GPT-4: ~1.7T parameters (estimated)
# LLaMA-2: 7B, 13B, 70B variants
```

### Scaling Laws

```python
# Performance scales with:
1. Model size (parameters)
2. Dataset size (tokens)
3. Compute (FLOPs)

# Chinchilla scaling:
# Optimal: N parameters trained on ~20N tokens
```

### Memory Requirements

```python
def estimate_memory(num_params_billions, precision='fp16'):
    """Estimate GPU memory for inference"""
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    params = num_params_billions * 1e9
    memory_gb = (params * bytes_per_param[precision]) / 1e9
    
    # Add ~20% overhead for activations
    return memory_gb * 1.2

# Examples:
# 7B model in fp16: ~14 GB
# 13B model in fp16: ~26 GB
# 70B model in fp16: ~140 GB
```

---

## Popular LLM Architectures

| Model | Type | Pre-training | Architecture | Sizes |
|-------|------|--------------|--------------|-------|
| **GPT** | Decoder-only | Next token prediction | 96 layers, 12,288 hidden, 96 heads | GPT-2 (1.5B), GPT-3 (175B), GPT-4 |
| **BERT** | Encoder-only | Masked LM + NSP | Base: 12L/768H/12H (110M), Large: 24L/1024H/16H (340M) | Context: 512 |
| **T5** | Encoder-decoder | Span corruption | Text-to-text framework | 60M, 220M, 770M, 11B |
| **LLaMA** | Decoder-only | Next token prediction | Grouped-query attention, RMSNorm | 7B, 13B, 70B (4096 context) |

---

## Feed-Forward Networks & Layer Normalization

```python
# Position-wise FFN (typical ratio: ff_dim = 4 * embed_dim)
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

# Pre-Norm (GPT, modern LLMs - more stable) vs Post-Norm (original Transformer)
# Pre-Norm:  x = x + attention(norm(x)); x = x + ffn(norm(x))
# Post-Norm: x = norm(x + attention(x)); x = norm(x + ffn(x))
```

---

## 💻 Exercises (40 min)

Practice implementing transformer components in `exercise.py`:

### Exercise 1: Scaled Dot-Product Attention
Implement the core attention mechanism from scratch.

### Exercise 2: Positional Encoding
Create sinusoidal positional encodings for sequences.

### Exercise 3: Causal Mask
Generate causal attention masks for autoregressive models.

### Exercise 4: Transformer Block
Build a complete transformer encoder block.

### Exercise 5: Parameter Counting
Calculate total parameters in a transformer model.

---

## ✅ Quiz

Test your understanding of LLM architecture in `quiz.md`.

---

## 🎯 Key Takeaways

- **Self-attention** allows tokens to attend to each other in parallel
- **Multi-head attention** captures different types of relationships
- **Positional encoding** adds sequence order information
- **Encoder-only** (BERT) for understanding tasks
- **Decoder-only** (GPT) for generation tasks
- **Encoder-decoder** (T5) for sequence-to-sequence tasks
- **Scaling laws** govern model performance vs size/data/compute
- **Pre-norm** architecture is more stable than post-norm

---

## 📚 Resources

- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [Scaling Laws Paper](https://arxiv.org/abs/2001.08361)

---

## Tomorrow: Day 72 - Tokenization & Embeddings

Learn how text is converted to tokens and how embeddings represent meaning in vector space.
