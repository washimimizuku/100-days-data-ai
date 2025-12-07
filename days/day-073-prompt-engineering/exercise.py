"""
Day 73: Prompt Engineering - Exercises

Practice designing effective prompts for various tasks.
Note: These exercises focus on prompt design, not actual LLM calls.
"""


def exercise_1_zero_shot_classification():
    """
    Exercise 1: Zero-Shot Classification
    
    Design zero-shot prompts for:
    - Sentiment analysis (positive/negative/neutral)
    - Topic classification (tech/sports/politics/entertainment)
    - Urgency detection (urgent/normal/low priority)
    
    TODO: Create 3 different zero-shot classification prompts
    TODO: Include clear instructions and output format
    TODO: Test with sample texts
    """
    sample_texts = [
        "This product exceeded my expectations!",
        "The Lakers won the championship last night.",
        "URGENT: Server is down, need immediate attention!"
    ]
    
    # TODO: Design prompts
    pass


def exercise_2_few_shot_learning():
    """
    Exercise 2: Few-Shot Learning
    
    Create few-shot prompts for:
    - Email classification (spam/not spam)
    - Code language detection
    - Customer intent recognition
    
    TODO: Design prompts with 3-5 examples each
    TODO: Ensure examples cover different cases
    TODO: Include clear pattern for model to follow
    """
    # TODO: Create few-shot prompts
    pass


def exercise_3_instruction_following():
    """
    Exercise 3: Instruction Following
    
    Write detailed instructions for:
    - Generating Python functions with specific requirements
    - Creating SQL queries from natural language
    - Writing professional emails
    
    TODO: Create instruction prompts with:
    - Clear task description
    - Specific requirements
    - Output format
    - Constraints
    """
    # TODO: Design instruction prompts
    pass


def exercise_4_prompt_templates():
    """
    Exercise 4: Prompt Templates
    
    Create reusable templates for:
    - Question answering with context
    - Text summarization with parameters
    - Data extraction from unstructured text
    
    TODO: Design templates with placeholders
    TODO: Include instructions for each template
    TODO: Show example usage
    """
    # TODO: Create prompt templates
    pass


def exercise_5_prompt_optimization():
    """
    Exercise 5: Prompt Optimization
    
    Take a basic prompt and improve it through iterations:
    - Start with: "Explain machine learning"
    - Add specificity, context, format, constraints
    - Create 4 versions showing progressive improvement
    
    TODO: Create version 1 (basic)
    TODO: Create version 2 (add specificity)
    TODO: Create version 3 (add format)
    TODO: Create version 4 (add constraints and examples)
    """
    # TODO: Create progressive prompt versions
    pass


if __name__ == "__main__":
    print("Day 73: Prompt Engineering - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Zero-Shot Classification")
    print("=" * 60)
    # exercise_1_zero_shot_classification()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Few-Shot Learning")
    print("=" * 60)
    # exercise_2_few_shot_learning()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Instruction Following")
    print("=" * 60)
    # exercise_3_instruction_following()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Prompt Templates")
    print("=" * 60)
    # exercise_4_prompt_templates()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Prompt Optimization")
    print("=" * 60)
    # exercise_5_prompt_optimization()
