# Day 72: Tokenization & Embeddings

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand different tokenization algorithms (BPE, WordPiece, SentencePiece)
- Learn how subword tokenization works
- Explore word embeddings and vector representations
- Understand embedding spaces and semantic similarity
- Use pre-trained tokenizers and embeddings
- Visualize embedding spaces

---

## What is Tokenization?

### From Text to Tokens

Tokenization converts text into discrete units (tokens) that models can process:

```python
# Character-level
"hello" → ['h', 'e', 'l', 'l', 'o']

# Word-level
"hello world" → ['hello', 'world']

# Subword-level (BPE)
"unhappiness" → ['un', 'happiness']
```

### Why Subword Tokenization?

**Problems with word-level**:
- Huge vocabulary (millions of words)
- Out-of-vocabulary (OOV) words
- No morphological understanding

**Subword advantages**:
- Smaller vocabulary (30k-50k tokens)
- Handles rare/new words
- Captures morphology (prefix, suffix)

---

## Byte Pair Encoding (BPE)

BPE iteratively merges the most frequent character pairs:

```python
# Algorithm: "low low low lower lowest"
# Iteration 1: Merge 'l' + 'o' → 'lo'
# Iteration 2: Merge 'lo' + 'w' → 'low'
# Continue until desired vocabulary size

# Using BPE Tokenizer (GPT-2)
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Tokenization is fundamental"
tokens = tokenizer.tokenize(text)  # ['Token', 'ization', 'Ġis', 'Ġfundamental']
token_ids = tokenizer.encode(text)
decoded = tokenizer.decode(token_ids)
```

---

## WordPiece & SentencePiece

```python
# WordPiece (BERT): likelihood-based merging, ## prefix for continuations
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("Tokenization is fundamental")
# ['token', '##ization', 'is', 'fundamental']

# Special tokens: [CLS] (start), [SEP] (separator), [PAD], [UNK], [MASK]
encoded = tokenizer.encode("Hello world", add_special_tokens=True)
# ['[CLS]', 'hello', 'world', '[SEP]']

# SentencePiece (T5): language-agnostic, treats text as raw bytes, reversible
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
tokens = tokenizer.tokenize("Tokenization is fundamental")
# ['▁Token', 'ization', '▁is', '▁fundamental']  (▁ = space)
```

---

## Word Embeddings

### From Tokens to Vectors

Embeddings map discrete tokens to continuous vectors:

```python
import torch
import torch.nn as nn

# Create embedding layer
vocab_size = 10000
embed_dim = 300

embedding = nn.Embedding(vocab_size, embed_dim)

# Token IDs to embeddings
token_ids = torch.tensor([42, 123, 456])
embeddings = embedding(token_ids)

print(f"Shape: {embeddings.shape}")  # (3, 300)
print(f"Token 42 embedding: {embeddings[0][:5]}")
```

### Embedding Properties

```python
# Similar words have similar embeddings
# king - man + woman ≈ queen
# paris - france + germany ≈ berlin

# Cosine similarity
def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
```

---

## Pre-trained Embeddings

```python
# Word2Vec: CBOW (predict word from context) or Skip-gram (predict context from word)
# GloVe: Global co-occurrence statistics (dimensions: 50, 100, 200, 300)

# Contextual Embeddings (BERT, GPT): embeddings vary by context
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "The bank by the river"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

# "bank" has different embedding based on context
```

---

## Embedding Spaces

```python
# Semantic similarity with mean pooling
def get_sentence_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Compare sentences
text1, text2, text3 = "The cat sits on the mat", "A feline rests on the rug", "Python is a programming language"
emb1, emb2, emb3 = get_sentence_embedding(text1, tokenizer, model), get_sentence_embedding(text2, tokenizer, model), get_sentence_embedding(text3, tokenizer, model)

sim_12 = cosine_similarity(emb1[0], emb2[0])  # High (similar meaning)
sim_13 = cosine_similarity(emb1[0], emb3[0])  # Low (different topics)

# Dimensionality reduction for visualization
from sklearn.decomposition import PCA
embeddings_2d = PCA(n_components=2).fit_transform(embeddings.numpy())
```

---

## Tokenizer Training

### Train Custom BPE Tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train
trainer = BpeTrainer(
    vocab_size=5000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

files = ["corpus.txt"]
tokenizer.train(files, trainer)

# Save
tokenizer.save("my_tokenizer.json")
```

---

## Practical Considerations

```python
# Vocabulary size trade-offs: Small (1k-5k) fast but many UNK, Medium (30k-50k) balanced, Large (100k+) fewer splits but slower
# GPT-2: 50,257 | BERT: 30,522 | T5: 32,000

# Padding and truncation
inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')  # Dynamic padding

# Context limits: GPT-2 (1024), GPT-3 (2048-4096), GPT-4 (8192-32768), Claude (100k), BERT (512)
# Handling long text: truncation, sliding window, hierarchical processing, summarization
```

---

## 💻 Exercises (40 min)

Practice tokenization and embeddings in `exercise.py`:

### Exercise 1: Compare Tokenizers
Compare BPE, WordPiece, and SentencePiece on the same text.

### Exercise 2: Subword Analysis
Analyze how rare words are tokenized into subwords.

### Exercise 3: Embedding Similarity
Compute semantic similarity between sentences using embeddings.

### Exercise 4: Vocabulary Analysis
Analyze tokenizer vocabulary and token frequency.

### Exercise 5: Custom Tokenizer
Train a simple BPE tokenizer on custom text.

---

## ✅ Quiz

Test your understanding of tokenization and embeddings in `quiz.md`.

---

## 🎯 Key Takeaways

- **Subword tokenization** balances vocabulary size and coverage
- **BPE** merges frequent character pairs iteratively
- **WordPiece** uses likelihood-based merging (BERT)
- **SentencePiece** is language-agnostic and reversible
- **Embeddings** map tokens to continuous vector spaces
- **Contextual embeddings** (BERT, GPT) vary by context
- **Cosine similarity** measures semantic similarity
- **Vocabulary size** affects model size and performance
- **Special tokens** mark boundaries and special positions

---

## 📚 Resources

- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers)
- [BPE Paper](https://arxiv.org/abs/1508.07909)
- [WordPiece Paper](https://arxiv.org/abs/1609.08144)
- [SentencePiece](https://github.com/google/sentencepiece)
- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [BERT Embeddings](https://jalammar.github.io/illustrated-bert/)

---

## Tomorrow: Day 73 - Prompt Engineering

Learn how to craft effective prompts to get the best results from large language models.
