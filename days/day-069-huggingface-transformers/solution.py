"""
Day 69: Hugging Face Transformers - Solutions
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader


def exercise_1_sentiment_analysis():
    """Exercise 1: Sentiment Analysis Pipeline"""
    
    texts = [
        "This is the best day ever!",
        "I'm really disappointed with the service.",
        "The weather is nice today.",
        "This product exceeded my expectations!",
        "I don't like this at all."
    ]
    
    # Create sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis")
    
    print("Sentiment Analysis Results:")
    for text in texts:
        result = classifier(text)[0]
        print(f"\nText: {text}")
        print(f"Label: {result['label']}, Score: {result['score']:.4f}")


def exercise_2_text_generation():
    """Exercise 2: Text Generation"""
    
    prompt = "The future of artificial intelligence is"
    
    # Load GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate with temperature=0.3 (conservative)
    print("Conservative Generation (temperature=0.3):")
    outputs_conservative = model.generate(
        inputs.input_ids,
        max_length=50,
        temperature=0.3,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    text_conservative = tokenizer.decode(outputs_conservative[0], skip_special_tokens=True)
    print(text_conservative)
    
    # Generate with temperature=1.0 (creative)
    print("\nCreative Generation (temperature=1.0):")
    outputs_creative = model.generate(
        inputs.input_ids,
        max_length=50,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    text_creative = tokenizer.decode(outputs_creative[0], skip_special_tokens=True)
    print(text_creative)


def exercise_3_custom_classification():
    """Exercise 3: Custom Classification"""
    
    # Create small dataset
    train_texts = [
        "I love this product!",
        "This is terrible.",
        "Amazing quality!",
        "Very disappointed.",
        "Highly recommend!",
        "Waste of money.",
        "Excellent service!",
        "Poor quality.",
        "Best purchase ever!",
        "Not worth it."
    ]
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Tokenize data
    encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
    labels = torch.tensor(train_labels)
    
    # Create dataset and loader
    dataset = TensorDataset(encodings.input_ids, encodings.attention_mask, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()
    
    print("Training custom classifier...")
    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            input_ids, attention_mask, batch_labels = batch
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/3, Loss: {total_loss/len(loader):.4f}")
    
    # Test on new samples
    test_texts = ["This is fantastic!", "I hate this."]
    model.eval()
    
    print("\nTest Predictions:")
    with torch.no_grad():
        test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")
        outputs = model(**test_encodings)
        predictions = torch.softmax(outputs.logits, dim=1)
        
        for text, pred in zip(test_texts, predictions):
            label = "Positive" if pred[1] > pred[0] else "Negative"
            confidence = max(pred[0], pred[1])
            print(f"{text} -> {label} ({confidence:.4f})")


def exercise_4_named_entity_recognition():
    """Exercise 4: Named Entity Recognition"""
    
    texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino.",
        "Elon Musk leads Tesla and SpaceX in California.",
        "Microsoft CEO Satya Nadella announced new products in Seattle."
    ]
    
    # Create NER pipeline
    ner = pipeline("ner", grouped_entities=True)
    
    print("Named Entity Recognition Results:\n")
    
    for text in texts:
        print(f"Text: {text}")
        entities = ner(text)
        
        # Group by entity type
        entity_groups = {}
        for entity in entities:
            entity_type = entity['entity_group']
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity['word'])
        
        # Print grouped entities
        for entity_type, words in entity_groups.items():
            print(f"  {entity_type}: {', '.join(words)}")
        print()


def exercise_5_batch_inference():
    """Exercise 5: Batch Inference"""
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Create batch of texts
    texts = [
        "I love this!",
        "This is terrible.",
        "Pretty good overall.",
        "Not what I expected.",
        "Absolutely amazing!",
        "Could be better.",
        "Fantastic experience!",
        "Very disappointing.",
        "Highly satisfied!",
        "Not recommended."
    ]
    
    # Tokenize batch
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Batch inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
    
    # Print results
    print("Batch Inference Results:\n")
    labels = ["Negative", "Positive"]
    
    for text, pred in zip(texts, predictions):
        label_idx = pred.argmax().item()
        confidence = pred[label_idx].item()
        print(f"{text:30s} -> {labels[label_idx]:8s} ({confidence:.4f})")


if __name__ == "__main__":
    print("Day 69: Hugging Face Transformers - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Sentiment Analysis Pipeline")
    print("=" * 60)
    exercise_1_sentiment_analysis()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Text Generation")
    print("=" * 60)
    exercise_2_text_generation()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Custom Classification")
    print("=" * 60)
    exercise_3_custom_classification()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Named Entity Recognition")
    print("=" * 60)
    exercise_4_named_entity_recognition()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Batch Inference")
    print("=" * 60)
    exercise_5_batch_inference()
