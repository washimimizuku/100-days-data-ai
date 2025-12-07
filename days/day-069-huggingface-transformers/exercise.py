"""
Day 69: Hugging Face Transformers - Exercises

Practice using pre-trained transformers for NLP tasks.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def exercise_1_sentiment_analysis():
    """
    Exercise 1: Sentiment Analysis Pipeline
    
    Use a pre-trained sentiment analysis model to classify texts:
    - Create a sentiment analysis pipeline
    - Classify 5 different texts
    - Print label and confidence score for each
    
    TODO: Create sentiment analysis pipeline
    TODO: Define list of texts to classify
    TODO: Classify each text and print results
    """
    texts = [
        "This is the best day ever!",
        "I'm really disappointed with the service.",
        "The weather is nice today.",
        "This product exceeded my expectations!",
        "I don't like this at all."
    ]
    
    # TODO: Create pipeline
    
    # TODO: Classify texts
    pass


def exercise_2_text_generation():
    """
    Exercise 2: Text Generation
    
    Generate text using GPT-2 with different temperatures:
    - Load GPT-2 model and tokenizer
    - Generate text with temperature 0.3 (conservative)
    - Generate text with temperature 1.0 (creative)
    - Compare the outputs
    
    TODO: Load GPT-2 model and tokenizer
    TODO: Define a prompt
    TODO: Generate with temperature=0.3
    TODO: Generate with temperature=1.0
    TODO: Print both outputs
    """
    prompt = "The future of artificial intelligence is"
    
    # TODO: Load model and tokenizer
    
    # TODO: Generate with different temperatures
    pass


def exercise_3_custom_classification():
    """
    Exercise 3: Custom Classification
    
    Fine-tune a model on custom data:
    - Create a small dataset (10 samples)
    - Load DistilBERT model
    - Tokenize the data
    - Train for 3 epochs
    - Test on new samples
    
    TODO: Create dataset (texts and labels)
    TODO: Load model and tokenizer
    TODO: Tokenize data
    TODO: Create simple training loop
    TODO: Test on new samples
    """
    # TODO: Create dataset
    # train_texts = [...]
    # train_labels = [...]
    
    # TODO: Load model
    
    # TODO: Training loop
    
    # TODO: Test predictions
    pass


def exercise_4_named_entity_recognition():
    """
    Exercise 4: Named Entity Recognition
    
    Extract entities from text:
    - Use pre-trained NER pipeline
    - Process 3 different texts
    - Extract and print all entities with their types
    - Group entities by type (PERSON, ORGANIZATION, LOCATION)
    
    TODO: Create NER pipeline
    TODO: Define texts with entities
    TODO: Extract entities
    TODO: Group by entity type
    """
    texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino.",
        "Elon Musk leads Tesla and SpaceX in California.",
        "Microsoft CEO Satya Nadella announced new products in Seattle."
    ]
    
    # TODO: Create NER pipeline
    
    # TODO: Process texts and extract entities
    pass


def exercise_5_batch_inference():
    """
    Exercise 5: Batch Inference
    
    Process multiple texts efficiently:
    - Load a classification model
    - Create a batch of 10 texts
    - Tokenize with proper padding and truncation
    - Run batch inference
    - Print predictions for all texts
    
    TODO: Load model and tokenizer
    TODO: Create batch of texts
    TODO: Tokenize batch
    TODO: Run inference
    TODO: Print results
    """
    # TODO: Load model
    
    # TODO: Create batch of texts
    
    # TODO: Tokenize
    
    # TODO: Batch inference
    pass


if __name__ == "__main__":
    print("Day 69: Hugging Face Transformers - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Sentiment Analysis Pipeline")
    print("=" * 60)
    # exercise_1_sentiment_analysis()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Text Generation")
    print("=" * 60)
    # exercise_2_text_generation()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Custom Classification")
    print("=" * 60)
    # exercise_3_custom_classification()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Named Entity Recognition")
    print("=" * 60)
    # exercise_4_named_entity_recognition()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Batch Inference")
    print("=" * 60)
    # exercise_5_batch_inference()
