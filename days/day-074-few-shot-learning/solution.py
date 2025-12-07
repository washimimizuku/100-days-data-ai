"""
Day 74: Few-Shot Learning - Solutions
"""


def exercise_1_example_selection():
    """Exercise 1: Example Selection"""
    
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
    
    # Selection criteria:
    # 1. One clear example per class
    # 2. Diverse language/phrasing
    # 3. Unambiguous labels
    # 4. Different lengths
    # 5. Representative of real reviews
    
    selected = [
        {"text": "Exceeded expectations!", "label": "positive"},  # Enthusiastic, specific
        {"text": "Completely broken", "label": "negative"},       # Clear problem
        {"text": "Average product", "label": "neutral"},          # Balanced
        {"text": "Waste of money", "label": "negative"},          # Strong negative
        {"text": "It's okay", "label": "neutral"}                 # Mild neutral
    ]
    
    print("Selected Examples:\n")
    for i, ex in enumerate(selected, 1):
        print(f"{i}. Text: '{ex['text']}'")
        print(f"   Label: {ex['label']}")
        print()
    
    print("Selection Rationale:")
    print("- Balanced: 1 positive, 2 negative, 2 neutral")
    print("- Diverse: Different phrasings and intensities")
    print("- Clear: Unambiguous sentiment")
    print("- Representative: Typical review language")


def exercise_2_dynamic_selection():
    """Exercise 2: Dynamic Selection"""
    
    def word_overlap_similarity(text1, text2):
        """Calculate similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def select_similar_examples(query, example_pool, k=3):
        """Select k most similar examples to query"""
        similarities = []
        
        for example in example_pool:
            sim = word_overlap_similarity(query, example['text'])
            similarities.append((sim, example))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return top-k
        return [(sim, ex) for sim, ex in similarities[:k]]
    
    # Example pool
    example_pool = [
        {"text": "The product quality is excellent", "label": "positive"},
        {"text": "Shipping was very fast", "label": "positive"},
        {"text": "Poor customer service", "label": "negative"},
        {"text": "Product broke after one week", "label": "negative"},
        {"text": "Average quality for the price", "label": "neutral"}
    ]
    
    # Test queries
    queries = [
        "The product quality is amazing",
        "Terrible customer service experience",
        "Decent product, nothing special"
    ]
    
    print("Dynamic Example Selection:\n")
    
    for query in queries:
        print(f"Query: '{query}'")
        print("Selected examples:")
        
        selected = select_similar_examples(query, example_pool, k=2)
        
        for sim, ex in selected:
            print(f"  Similarity: {sim:.3f}")
            print(f"  Text: '{ex['text']}'")
            print(f"  Label: {ex['label']}")
        print()


def exercise_3_format_consistency():
    """Exercise 3: Format Consistency"""
    
    prompt_template = """
Extract structured information from the text in JSON format.

Example 1:
Text: "John Smith (john@email.com) called on 2024-01-15"
Output: {
  "name": "John Smith",
  "email": "john@email.com",
  "date": "2024-01-15",
  "action": "called"
}

Example 2:
Text: "Meeting with Sarah Johnson at 3pm tomorrow"
Output: {
  "name": "Sarah Johnson",
  "time": "3pm",
  "date": "tomorrow",
  "action": "meeting"
}

Example 3:
Text: "Email from mike@company.com received on Monday"
Output: {
  "email": "mike@company.com",
  "date": "Monday",
  "action": "email received"
}

Now extract from:
Text: "{input_text}"
Output:
"""
    
    print("Format Consistency Example:\n")
    print(prompt_template.format(
        input_text="Call from Alice Brown (alice@test.com) scheduled for Friday"
    ))
    
    print("\nKey Consistency Elements:")
    print("- Same JSON structure for all examples")
    print("- Consistent field names")
    print("- Same formatting style")
    print("- Clear input/output separation")


def exercise_4_performance_optimization():
    """Exercise 4: Performance Optimization"""
    
    print("Progressive Prompt Optimization:\n")
    
    # Version 1: Basic (2 examples)
    v1 = """
Classify sentiment:

"Great product!" → positive
"Terrible quality" → negative

"I love this!" →
"""
    
    print("Version 1 (Basic - 2 examples):")
    print(v1)
    print()
    
    # Version 2: More examples
    v2 = """
Classify sentiment:

"Great product!" → positive
"Terrible quality" → negative
"It's okay" → neutral
"Amazing!" → positive
"Waste of money" → negative

"I love this!" →
"""
    
    print("Version 2 (More examples - 5 total):")
    print(v2)
    print()
    
    # Version 3: Better examples (diverse, clear)
    v3 = """
Classify sentiment:

"Exceeded my expectations, highly recommend!" → positive
"Completely broken, requesting refund" → negative
"Average product, nothing special" → neutral
"Best purchase I've made this year!" → positive
"Poor quality, very disappointed" → negative

"I love this!" →
"""
    
    print("Version 3 (Better quality examples):")
    print(v3)
    print()
    
    # Version 4: With instructions
    v4 = """
Task: Classify the sentiment of product reviews as positive, negative, or neutral.

Instructions:
- Consider the overall tone
- Look for emotion words
- Be consistent with the examples

Examples:

"Exceeded my expectations, highly recommend!" → positive
"Completely broken, requesting refund" → negative
"Average product, nothing special" → neutral
"Best purchase I've made this year!" → positive
"Poor quality, very disappointed" → negative

Now classify:
"I love this!" →
"""
    
    print("Version 4 (With instructions):")
    print(v4)
    
    print("\nImprovements:")
    print("V1→V2: Added more examples and neutral class")
    print("V2→V3: Improved example quality and diversity")
    print("V3→V4: Added clear instructions and structure")


def exercise_5_edge_case_handling():
    """Exercise 5: Edge Case Handling"""
    
    prompt = """
Validate email addresses:

Email: "user@example.com"
Valid: true
Reason: Standard format

Email: "invalid.email"
Valid: false
Reason: Missing @ symbol

Email: ""
Valid: false
Reason: Empty string

Email: "user@domain.co.uk"
Valid: true
Reason: Valid with country code

Email: "user@@example.com"
Valid: false
Reason: Double @ symbol

Email: "user@"
Valid: false
Reason: Missing domain

Email: "@example.com"
Valid: false
Reason: Missing username

Email: "user name@example.com"
Valid: false
Reason: Space in username

Now validate:
Email: "{test_email}"
Valid:
Reason:
"""
    
    print("Edge Case Handling Prompt:\n")
    print(prompt.format(test_email="test.user@company.com"))
    
    print("\nEdge Cases Covered:")
    print("✓ Valid standard format")
    print("✓ Missing @ symbol")
    print("✓ Empty input")
    print("✓ International domains")
    print("✓ Double @ symbol")
    print("✓ Incomplete addresses")
    print("✓ Invalid characters (spaces)")
    
    print("\nBest Practices:")
    print("- Include both valid and invalid examples")
    print("- Cover common error patterns")
    print("- Explain reasoning for each case")
    print("- Test boundary conditions")


if __name__ == "__main__":
    print("Day 74: Few-Shot Learning - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Example Selection")
    print("=" * 60)
    exercise_1_example_selection()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Dynamic Selection")
    print("=" * 60)
    exercise_2_dynamic_selection()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Format Consistency")
    print("=" * 60)
    exercise_3_format_consistency()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Performance Optimization")
    print("=" * 60)
    exercise_4_performance_optimization()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Edge Case Handling")
    print("=" * 60)
    exercise_5_edge_case_handling()
