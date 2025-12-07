# Day 69: Hugging Face Transformers

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand transformer architecture fundamentals
- Use pre-trained models from Hugging Face Hub
- Perform text classification and generation tasks
- Fine-tune models on custom datasets
- Apply transformers to NLP and vision tasks
- Use pipelines for quick inference

---

## Transformer Architecture Basics

### What are Transformers?

Transformers are neural network architectures based on self-attention mechanisms. They excel at:
- Natural language processing (NLP)
- Computer vision
- Speech recognition
- Multi-modal tasks

**Key Components**:
- **Self-Attention**: Weighs importance of different parts of input
- **Encoder**: Processes input into representations
- **Decoder**: Generates output from representations
- **Positional Encoding**: Captures sequence order

### Popular Transformer Models

```python
# BERT: Bidirectional encoder (understanding)
# GPT: Unidirectional decoder (generation)
# T5: Encoder-decoder (translation, summarization)
# ViT: Vision transformer (image classification)
```

---

## Hugging Face Ecosystem

### Installation

```python
# Install transformers library
# pip install transformers torch

from transformers import pipeline, AutoTokenizer, AutoModel
```

### The Hub

Hugging Face Hub hosts thousands of pre-trained models:
- Text: BERT, GPT-2, RoBERTa, T5
- Vision: ViT, CLIP, DETR
- Audio: Wav2Vec2, Whisper
- Multi-modal: CLIP, LayoutLM

---

## Using Pipelines

```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=30)

# Question answering
qa = pipeline("question-answering")
result = qa(question="What is the capital of France?", context="Paris is the capital of France.")

# Available tasks: text-classification, token-classification (NER), question-answering,
# summarization, translation, text-generation, fill-mask, image-classification,
# object-detection, image-segmentation, automatic-speech-recognition, audio-classification
```

---

## Text Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Single or batch prediction
texts = ["This movie is amazing!", "This is terrible.", "It's okay, I guess."]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)

for text, pred in zip(texts, predictions):
    print(f"{text}: Positive={pred[1]:.4f}, Negative={pred[0]:.4f}")
```

---

## Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Artificial intelligence will"
inputs = tokenizer(prompt, return_tensors="pt")

# Generation parameters:
# temperature: randomness (0.1=conservative, 1.0=creative)
# top_k: sample from top k tokens, top_p: nucleus sampling
# num_beams: beam search, repetition_penalty: penalize repeats
outputs = model.generate(
    inputs.input_ids,
    max_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    num_beams=5,
    do_sample=True
)

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
```

---

## Fine-tuning Models

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Prepare dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.labels = labels
    
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

texts = ["Great product!", "Terrible service.", "It's okay."]
labels = [1, 0, 1]  # 1=positive, 0=negative
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = TextDataset(texts, labels, tokenizer)

# Fine-tune with Trainer API (recommended)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3, 
                                  per_device_train_batch_size=8, learning_rate=5e-5)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

---

## Named Entity Recognition (NER)

```python
# Using NER pipeline (easiest)
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = ner(text)
for entity in entities:
    print(f"{entity['word']}: {entity['entity']} ({entity['score']:.4f})")

# Custom NER model
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

text = "John works at Google in New York."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    predictions = torch.argmax(model(**inputs).logits, dim=2)

# Map predictions to labels
labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
for token, pred in zip(tokens, predictions[0]):
    if token not in ['[CLS]', '[SEP]', '[PAD]']:
        print(f"{token}: {labels[pred]}")
```

---

## Vision Transformers

### Image Classification

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Load image
image = Image.open('cat.jpg')

# Process and predict
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get prediction
predicted_class = logits.argmax(-1).item()
print(f"Predicted class: {model.config.id2label[predicted_class]}")
```

---

## Model Saving and Loading

```python
# Save model and tokenizer locally
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load from local directory or Hub
model = AutoModelForSequenceClassification.from_pretrained("./my_model")  # Local
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")  # Hub

# Push to Hugging Face Hub (requires login)
# model.push_to_hub("my-username/my-model")
```

---

## Best Practices

```python
# Memory management
model = AutoModel.from_pretrained("distilbert-base-uncased")  # Use smaller models
model = model.half()  # Half precision (FP16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Efficient tokenization
tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)  # Fast tokenizers

# Error handling
try:
    model = AutoModel.from_pretrained("model-name")
except Exception as e:
    print(f"Error: {e}")
    model = AutoModel.from_pretrained("bert-base-uncased")  # Fallback
```

---

## 💻 Exercises (40 min)

Test your understanding with these hands-on exercises in `exercise.py`:

### Exercise 1: Sentiment Analysis Pipeline
Use a pre-trained sentiment analysis model to classify multiple texts.

### Exercise 2: Text Generation
Generate creative text using GPT-2 with different temperature settings.

### Exercise 3: Custom Classification
Fine-tune a BERT model on a custom text classification dataset.

### Exercise 4: Named Entity Recognition
Extract entities from text using a pre-trained NER model.

### Exercise 5: Batch Inference
Process multiple texts efficiently with batching and proper tokenization.

---

## ✅ Quiz

Test your understanding of Hugging Face Transformers in `quiz.md`.

---

## 🎯 Key Takeaways

- **Transformers** use self-attention for sequence processing
- **Pipelines** provide simple APIs for common tasks
- **AutoModel** and **AutoTokenizer** automatically load correct architectures
- **Fine-tuning** adapts pre-trained models to custom tasks
- **Trainer API** simplifies training with built-in features
- Use **padding** and **truncation** for variable-length inputs
- **Temperature** controls generation randomness
- Models can be saved locally or pushed to Hugging Face Hub

---

## 📚 Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Hugging Face Hub](https://huggingface.co/models)
- [Transformers Course](https://huggingface.co/course)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Pipeline Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Model Hub Search](https://huggingface.co/models)

---

## Tomorrow: Day 70 - Mini Project: Image Classifier

Build a complete image classification system using Vision Transformers and PyTorch.
