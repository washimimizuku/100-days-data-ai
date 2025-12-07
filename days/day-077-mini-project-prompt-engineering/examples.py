"""
Example data and templates for prompt engineering toolkit.
"""

# Few-shot examples for different tasks
SENTIMENT_EXAMPLES = [
    {
        "input": "I absolutely love this product! It exceeded all my expectations.",
        "output": "positive",
        "reasoning": "Strong positive words like 'love' and 'exceeded expectations'."
    },
    {
        "input": "This is the worst purchase I've ever made. Complete waste of money.",
        "output": "negative",
        "reasoning": "Strong negative words like 'worst' and 'waste of money'."
    },
    {
        "input": "The product is okay. Nothing special but it works.",
        "output": "neutral",
        "reasoning": "Neutral language with 'okay' and 'nothing special'."
    },
]

TOPIC_EXAMPLES = [
    {
        "input": "The stock market reached new highs today as investors reacted to positive earnings reports.",
        "output": "business",
        "reasoning": "Mentions stock market, investors, and earnings - clear business indicators."
    },
    {
        "input": "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
        "output": "science",
        "reasoning": "Discusses scientific discovery and marine biology."
    },
    {
        "input": "The championship game went into overtime with a final score of 98-95.",
        "output": "sports",
        "reasoning": "References championship game and score - sports context."
    },
]

ENTITY_EXAMPLES = [
    {
        "input": "Apple Inc. announced the iPhone 15 launch in Cupertino on September 12, 2023.",
        "output": {
            "organization": ["Apple Inc."],
            "product": ["iPhone 15"],
            "location": ["Cupertino"],
            "date": ["September 12, 2023"]
        },
        "reasoning": "Apple Inc. is a company, iPhone 15 is a product, Cupertino is a location, and September 12, 2023 is a date."
    },
]

COT_MATH_EXAMPLES = [
    {
        "problem": "If 3 notebooks cost $12, how much do 5 notebooks cost?",
        "reasoning": """Let's solve this step by step:
Step 1: Find the cost per notebook: $12 ÷ 3 = $4 per notebook
Step 2: Calculate cost for 5 notebooks: $4 × 5 = $20
Therefore, 5 notebooks cost $20.""",
        "answer": "$20"
    },
    {
        "problem": "A train travels 180 miles in 3 hours. How far will it travel in 5 hours at the same speed?",
        "reasoning": """Let's solve this step by step:
Step 1: Calculate the speed: 180 miles ÷ 3 hours = 60 miles per hour
Step 2: Calculate distance for 5 hours: 60 mph × 5 hours = 300 miles
Therefore, the train will travel 300 miles in 5 hours.""",
        "answer": "300 miles"
    },
]

# Prompt templates
TEMPLATES = {
    "classification": {
        "system": "You are an expert at text classification. Provide only the label without explanation unless asked.",
        "template": """Classify the following text into one of these categories: {labels}

Text: {text}

Classification:""",
        "example_format": "Text: {input}\nClassification: {output}"
    },
    
    "classification_with_reasoning": {
        "system": "You are an expert at text classification. Explain your reasoning.",
        "template": """Classify the following text into one of these categories: {labels}

Text: {text}

Provide your classification and reasoning:
Classification:
Reasoning:""",
        "example_format": "Text: {input}\nClassification: {output}\nReasoning: {reasoning}"
    },
    
    "extraction": {
        "system": "You are an expert at extracting structured information from text.",
        "template": """Extract the following entity types from the text: {entity_types}

Text: {text}

Provide entities in this format:
- Type: Entity text

Entities:""",
        "example_format": "Text: {input}\nEntities:\n{output}"
    },
    
    "summarization": {
        "system": "You are an expert at creating concise, accurate summaries.",
        "template": """Summarize the following text in {max_length} words or less:

Text: {text}

Summary:""",
        "example_format": "Text: {input}\nSummary: {output}"
    },
    
    "qa": {
        "system": "You are a helpful assistant that answers questions based on the provided context.",
        "template": """Context: {context}

Question: {question}

Answer:""",
        "example_format": "Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    },
    
    "qa_with_cot": {
        "system": "You are a helpful assistant that answers questions with step-by-step reasoning.",
        "template": """Context: {context}

Question: {question}

Let's think through this step by step:""",
        "example_format": "Context: {context}\nQuestion: {question}\nReasoning: {reasoning}\nAnswer: {answer}"
    },
    
    "zero_shot_cot": {
        "system": "You are a helpful assistant that solves problems step by step.",
        "template": """{problem}

Let's think step by step:""",
        "example_format": None
    },
    
    "few_shot_cot": {
        "system": "You are a helpful assistant that solves problems with clear reasoning.",
        "template": """{problem}

Let's solve this step by step:""",
        "example_format": "Problem: {problem}\nReasoning: {reasoning}\nAnswer: {answer}"
    },
}

# Task-specific configurations
TASK_CONFIGS = {
    "sentiment": {
        "labels": ["positive", "negative", "neutral"],
        "template": "classification",
        "examples": SENTIMENT_EXAMPLES,
        "temperature": 0.3,
        "model": "mistral"
    },
    
    "topic": {
        "labels": ["business", "science", "sports", "technology", "politics"],
        "template": "classification",
        "examples": TOPIC_EXAMPLES,
        "temperature": 0.3,
        "model": "mistral"
    },
    
    "entity_extraction": {
        "template": "extraction",
        "examples": ENTITY_EXAMPLES,
        "temperature": 0.2,
        "model": "mistral"
    },
    
    "summarization": {
        "template": "summarization",
        "temperature": 0.5,
        "model": "mistral"
    },
    
    "qa": {
        "template": "qa",
        "temperature": 0.3,
        "model": "mistral"
    },
}
