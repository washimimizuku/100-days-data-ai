"""
Day 74: Few-Shot Learning - Exercises

Practice designing and optimizing few-shot prompts.
"""


def exercise_1_example_selection():
    """
    Exercise 1: Example Selection
    
    Given a pool of examples, select the best ones for few-shot learning:
    - Task: Classify product reviews (positive/negative/neutral)
    - Pool: 10 example reviews
    - Select: 5 best examples
    - Criteria: Diversity, clarity, representativeness
    
    TODO: Define example pool
    TODO: Implement selection criteria
    TODO: Select best 5 examples
    TODO: Explain selection rationale
    """
    example_pool = [
        {"text": "Amazing product!", "label": "positive"},
        {"text": "Love it!", "label": "positive"},
        {"text": "Terrible quality", "label": "negative"},
        {"text": "Waste of money", "label": "negative"},
        {"text": "It's okay", "label": "neutral"},
        {"text": "Not bad", "label": "neutral"},
        {"text": "Exceeded expectations!", "label": "positive"},
        {"text": "Completely broken", "label": "negative"},
        {"text": "Average product", "label": "neutral"},
        {"text": "Best purchase ever!", "label": "positive"}
    ]
    
    # TODO: Implement selection logic
    pass


def exercise_2_dynamic_selection():
    """
    Exercise 2: Dynamic Selection
    
    Implement similarity-based example selection:
    - Given a query and example pool
    - Calculate similarity between query and examples
    - Select top-k most similar examples
    - Use simple word overlap as similarity metric
    
    TODO: Implement similarity function
    TODO: Implement selection function
    TODO: Test with different queries
    """
    # TODO: Implement dynamic selection
    pass


def exercise_3_format_consistency():
    """
    Exercise 3: Format Consistency
    
    Create few-shot prompts with consistent formatting:
    - Task: Extract structured data from text
    - Format: JSON output
    - Examples: 3 well-formatted examples
    - Ensure consistency in structure
    
    TODO: Design consistent format
    TODO: Create 3 examples
    TODO: Build complete prompt
    """
    # TODO: Create consistent format
    pass


def exercise_4_performance_optimization():
    """
    Exercise 4: Performance Optimization
    
    Optimize a few-shot prompt through iteration:
    - Start with basic 2-example prompt
    - Version 2: Add more examples
    - Version 3: Improve example quality
    - Version 4: Add instructions
    - Compare versions
    
    TODO: Create version 1 (basic)
    TODO: Create version 2 (more examples)
    TODO: Create version 3 (better examples)
    TODO: Create version 4 (with instructions)
    """
    # TODO: Create progressive versions
    pass


def exercise_5_edge_case_handling():
    """
    Exercise 5: Edge Case Handling
    
    Design prompts that handle edge cases:
    - Task: Email validation
    - Include examples for:
      - Valid emails
      - Invalid formats
      - Empty strings
      - Special characters
    
    TODO: Identify edge cases
    TODO: Create examples for each case
    TODO: Build robust prompt
    """
    # TODO: Handle edge cases
    pass


if __name__ == "__main__":
    print("Day 74: Few-Shot Learning - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Example Selection")
    print("=" * 60)
    # exercise_1_example_selection()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Dynamic Selection")
    print("=" * 60)
    # exercise_2_dynamic_selection()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Format Consistency")
    print("=" * 60)
    # exercise_3_format_consistency()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Performance Optimization")
    print("=" * 60)
    # exercise_4_performance_optimization()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Edge Case Handling")
    print("=" * 60)
    # exercise_5_edge_case_handling()
