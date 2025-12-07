"""
Day 73: Prompt Engineering - Solutions
"""


def exercise_1_zero_shot_classification():
    """Exercise 1: Zero-Shot Classification"""
    
    sample_texts = [
        "This product exceeded my expectations!",
        "The Lakers won the championship last night.",
        "URGENT: Server is down, need immediate attention!"
    ]
    
    # Sentiment Analysis Prompt
    sentiment_prompt = """
Classify the sentiment of the following text as positive, negative, or neutral.

Text: "{text}"

Sentiment:
"""
    
    # Topic Classification Prompt
    topic_prompt = """
Classify the following text into one of these categories:
- Technology
- Sports
- Politics
- Entertainment

Text: "{text}"

Category:
"""
    
    # Urgency Detection Prompt
    urgency_prompt = """
Classify the urgency level of the following message:
- Urgent: Requires immediate attention
- Normal: Standard priority
- Low: Can be handled later

Message: "{text}"

Urgency Level:
"""
    
    print("Zero-Shot Classification Prompts:\n")
    
    print("1. Sentiment Analysis:")
    print(sentiment_prompt.format(text=sample_texts[0]))
    print()
    
    print("2. Topic Classification:")
    print(topic_prompt.format(text=sample_texts[1]))
    print()
    
    print("3. Urgency Detection:")
    print(urgency_prompt.format(text=sample_texts[2]))


def exercise_2_few_shot_learning():
    """Exercise 2: Few-Shot Learning"""
    
    # Email Classification
    email_prompt = """
Classify emails as spam or not spam:

Email: "Congratulations! You've won $1,000,000! Click here now!"
Classification: spam

Email: "Hi John, can we schedule a meeting for next Tuesday?"
Classification: not spam

Email: "URGENT: Your account will be closed unless you verify now!"
Classification: spam

Email: "Here's the report you requested. Let me know if you need anything else."
Classification: not spam

Email: "Get rich quick! Limited time offer! Act now!"
Classification: spam

Email: "{new_email}"
Classification:
"""
    
    # Code Language Detection
    code_prompt = """
Identify the programming language:

Code: print("Hello, World!")
Language: Python

Code: console.log("Hello, World!");
Language: JavaScript

Code: System.out.println("Hello, World!");
Language: Java

Code: fmt.Println("Hello, World!")
Language: Go

Code: {new_code}
Language:
"""
    
    # Customer Intent Recognition
    intent_prompt = """
Identify customer intent:

Message: "I want to cancel my subscription"
Intent: cancellation

Message: "How do I reset my password?"
Intent: support

Message: "What are your pricing plans?"
Intent: inquiry

Message: "I'd like to upgrade to premium"
Intent: upgrade

Message: "The app keeps crashing"
Intent: bug_report

Message: "{new_message}"
Intent:
"""
    
    print("Few-Shot Learning Prompts:\n")
    
    print("1. Email Classification:")
    print(email_prompt.format(new_email="Buy now and save 90%!"))
    print()
    
    print("2. Code Language Detection:")
    print(code_prompt.format(new_code='puts "Hello, World!"'))
    print()
    
    print("3. Customer Intent Recognition:")
    print(intent_prompt.format(new_message="I need help with billing"))


def exercise_3_instruction_following():
    """Exercise 3: Instruction Following"""
    
    # Python Function Generation
    python_prompt = """
Generate a Python function with the following requirements:

Task: {task}

Requirements:
- Include type hints for all parameters and return value
- Add a comprehensive docstring with description, parameters, and return value
- Implement error handling for invalid inputs
- Include 2-3 example usages in comments
- Follow PEP 8 style guidelines
- Keep the function under 30 lines

Function:
"""
    
    # SQL Query Generation
    sql_prompt = """
Convert the following natural language request to a SQL query:

Request: {request}

Requirements:
- Use proper SQL syntax
- Include appropriate WHERE clauses
- Add ORDER BY if sorting is mentioned
- Use LIMIT if a specific number is requested
- Format the query for readability (use line breaks)

SQL Query:
"""
    
    # Professional Email
    email_prompt = """
Write a professional email with the following details:

To: {recipient}
Purpose: {purpose}
Key Points: {key_points}

Requirements:
- Professional but friendly tone
- Clear subject line
- Proper greeting and closing
- Concise and well-structured
- Include call-to-action if needed
- Length: 100-150 words

Email:
"""
    
    print("Instruction Following Prompts:\n")
    
    print("1. Python Function Generation:")
    print(python_prompt.format(task="Calculate the factorial of a number"))
    print()
    
    print("2. SQL Query Generation:")
    print(sql_prompt.format(request="Find all active users who signed up in the last 30 days"))
    print()
    
    print("3. Professional Email:")
    print(email_prompt.format(
        recipient="Team",
        purpose="Announce new feature launch",
        key_points="Launch date, key features, training session"
    ))


def exercise_4_prompt_templates():
    """Exercise 4: Prompt Templates"""
    
    # Question Answering Template
    qa_template = """
Context: {context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the answer is not in the context, say "Information not available"
- Be concise but complete
- Cite specific parts of the context

Answer:
"""
    
    # Summarization Template
    summary_template = """
Summarize the following text:

Text: {text}

Parameters:
- Length: {length} (short/medium/long)
- Focus: {focus}
- Format: {format} (paragraph/bullets/numbered)

Summary:
"""
    
    # Data Extraction Template
    extraction_template = """
Extract the following information from the text:

Text: {text}

Extract:
{fields}

Output format: JSON

Extracted Data:
"""
    
    print("Prompt Templates:\n")
    
    print("1. Question Answering Template:")
    print(qa_template.format(
        context="Python is a high-level programming language created by Guido van Rossum in 1991.",
        question="Who created Python?"
    ))
    print()
    
    print("2. Summarization Template:")
    print(summary_template.format(
        text="[Long article text here]",
        length="short",
        focus="main findings",
        format="bullets"
    ))
    print()
    
    print("3. Data Extraction Template:")
    print(extraction_template.format(
        text="John Smith (john@example.com) called on 2024-01-15 regarding invoice #12345.",
        fields="- Name\n- Email\n- Date\n- Invoice Number"
    ))


def exercise_5_prompt_optimization():
    """Exercise 5: Prompt Optimization"""
    
    print("Prompt Optimization - Progressive Improvement:\n")
    
    # Version 1: Basic
    v1 = "Explain machine learning"
    print("Version 1 (Basic):")
    print(v1)
    print()
    
    # Version 2: Add Specificity
    v2 = """
Explain machine learning in simple terms, covering:
- What it is
- How it works
- Common applications
"""
    print("Version 2 (Add Specificity):")
    print(v2)
    print()
    
    # Version 3: Add Format
    v3 = """
Explain machine learning in simple terms.

Structure your explanation as follows:
1. Definition (2-3 sentences)
2. How it works (3-4 sentences)
3. Common applications (3 examples with brief descriptions)

Target audience: Someone with no technical background
Length: 200-250 words
"""
    print("Version 3 (Add Format):")
    print(v3)
    print()
    
    # Version 4: Add Constraints and Examples
    v4 = """
Explain machine learning in simple terms for someone with no technical background.

Structure:
1. Definition (2-3 sentences)
   - Use an analogy to explain the concept
2. How it works (3-4 sentences)
   - Explain the learning process
   - Mention data and patterns
3. Common applications (3 examples)
   - Email spam filtering
   - Recommendation systems
   - Image recognition

Requirements:
- Length: 200-250 words
- Avoid technical jargon
- Use everyday analogies
- Include a concrete example for each application
- End with a sentence about future potential

Tone: Educational but conversational

Explanation:
"""
    print("Version 4 (Add Constraints and Examples):")
    print(v4)


if __name__ == "__main__":
    print("Day 73: Prompt Engineering - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Zero-Shot Classification")
    print("=" * 60)
    exercise_1_zero_shot_classification()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Few-Shot Learning")
    print("=" * 60)
    exercise_2_few_shot_learning()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Instruction Following")
    print("=" * 60)
    exercise_3_instruction_following()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Prompt Templates")
    print("=" * 60)
    exercise_4_prompt_templates()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Prompt Optimization")
    print("=" * 60)
    exercise_5_prompt_optimization()
