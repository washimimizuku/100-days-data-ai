# Day 94: NLP Tasks

## Learning Objectives

**Time**: 1 hour

- Understand common NLP tasks and applications
- Learn text classification and sentiment analysis
- Implement named entity recognition (NER)
- Apply text generation and summarization

## Theory (15 minutes)

### What is NLP?

Natural Language Processing enables machines to understand, interpret, and generate human language.

**Core Tasks**:
- Text classification
- Named entity recognition
- Sentiment analysis
- Text generation
- Machine translation
- Question answering

### Text Preprocessing

**Tokenization**:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Hello, world!")
# ['hello', ',', 'world', '!']

# Encode to IDs
input_ids = tokenizer.encode("Hello, world!")
```

**Cleaning**:
```python
import re

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text
```

### Text Classification

**Sentiment Analysis**:
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Multi-class Classification**:
```python
classifier = pipeline("text-classification", 
                     model="distilbert-base-uncased-finetuned-sst-2-english")

texts = ["This is great!", "This is terrible!"]
results = classifier(texts)
```

### Named Entity Recognition

**Extract Entities**:
```python
ner = pipeline("ner", grouped_entities=True)

text = "Apple Inc. was founded by Steve Jobs in Cupertino."
entities = ner(text)

# [{'entity_group': 'ORG', 'word': 'Apple Inc.'},
#  {'entity_group': 'PER', 'word': 'Steve Jobs'},
#  {'entity_group': 'LOC', 'word': 'Cupertino'}]
```

**Entity Types**:
- PER: Person
- ORG: Organization
- LOC: Location
- DATE: Date
- MISC: Miscellaneous

### Text Generation

**Using GPT Models**:
```python
generator = pipeline("text-generation", model="gpt2")

prompt = "Once upon a time"
result = generator(prompt, max_length=50, num_return_sequences=1)
```

**Controlled Generation**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

text = tokenizer.decode(outputs[0])
```

### Text Summarization

**Extractive Summarization**:
```python
summarizer = pipeline("summarization")

article = "Long article text here..."
summary = summarizer(article, max_length=130, min_length=30)
```

**Abstractive Summarization**:
```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

### Question Answering

**Extractive QA**:
```python
qa = pipeline("question-answering")

context = "The Eiffel Tower is in Paris, France."
question = "Where is the Eiffel Tower?"

result = qa(question=question, context=context)
# {'answer': 'Paris, France', 'score': 0.98}
```

### Text Similarity

**Semantic Similarity**:
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ["I love pizza", "Pizza is my favorite food"]
embeddings = model.encode(sentences)

similarity = util.cos_sim(embeddings[0], embeddings[1])
```

### Language Detection

**Detect Language**:
```python
from langdetect import detect

text = "Bonjour le monde"
language = detect(text)  # 'fr'
```

### Text Cleaning

**Remove Stopwords**:
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
words = word_tokenize(text)
filtered = [w for w in words if w.lower() not in stop_words]
```

**Lemmatization**:
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
word = lemmatizer.lemmatize("running", pos='v')  # 'run'
```

### Topic Modeling

**LDA**:
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1000)
doc_term_matrix = vectorizer.fit_transform(documents)

lda = LatentDirichletAllocation(n_components=5)
lda.fit(doc_term_matrix)
```

### Evaluation Metrics

**Classification**:
- Accuracy
- Precision, Recall, F1
- Confusion Matrix

**Generation**:
- BLEU: Translation quality
- ROUGE: Summarization quality
- Perplexity: Language model quality

**NER**:
- Entity-level F1
- Precision and Recall per entity type

### Common Architectures

**Transformers**:
- BERT: Bidirectional encoding
- GPT: Autoregressive generation
- T5: Text-to-text framework
- BART: Denoising autoencoder

**Specialized Models**:
- RoBERTa: Optimized BERT
- DistilBERT: Smaller, faster BERT
- ALBERT: Parameter-efficient BERT

### Use Cases

**Customer Service**:
- Sentiment analysis
- Intent classification
- Chatbots

**Content Analysis**:
- Topic modeling
- Document classification
- Information extraction

**Content Generation**:
- Article writing
- Email composition
- Code generation

**Search & Retrieval**:
- Semantic search
- Document ranking
- Question answering

### Best Practices

1. **Preprocessing**: Clean and normalize text
2. **Tokenization**: Use appropriate tokenizer
3. **Fine-tuning**: Adapt pre-trained models
4. **Evaluation**: Test on diverse data
5. **Error Analysis**: Understand failures
6. **Optimization**: Balance accuracy and speed

### Why This Matters

NLP powers chatbots, search engines, translation services, and content generation. Understanding NLP tasks enables building intelligent text-based applications that can understand and generate human language.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **Text Classification**: Classify text sentiment
2. **Named Entity Recognition**: Extract entities
3. **Text Generation**: Generate text continuations
4. **Summarization**: Summarize long text
5. **Question Answering**: Answer questions from context

## Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/)
- [NLP Course](https://huggingface.co/course)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 95: Audio AI with Whisper

Tomorrow you'll learn about audio AI including speech recognition, transcription, and audio processing with OpenAI's Whisper model.
