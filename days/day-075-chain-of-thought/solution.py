"""
Day 75: Chain of Thought - Solutions
"""


def exercise_1_basic_cot():
    """Exercise 1: Basic CoT"""
    
    print("Problem 1: What is 25% of 160?\n")
    solution1 = """
Let's solve this step by step:
Step 1: Convert percentage to decimal: 25% = 0.25
Step 2: Multiply by the number: 0.25 × 160 = 40
Step 3: Verify: 40 is 1/4 of 160 ✓

Answer: 40
"""
    print(solution1)
    
    print("\nProblem 2: A car travels 90 km in 1.5 hours. What's its average speed?\n")
    solution2 = """
Let's solve this step by step:
Step 1: Recall formula: Speed = Distance / Time
Step 2: Identify values: Distance = 90 km, Time = 1.5 hours
Step 3: Calculate: Speed = 90 / 1.5 = 60 km/h
Step 4: Verify: 60 × 1.5 = 90 ✓

Answer: 60 km/h
"""
    print(solution2)
    
    print("\nProblem 3: A circle has radius 7cm. What's its area? (π ≈ 3.14)\n")
    solution3 = """
Let's solve this step by step:
Step 1: Recall formula: Area = π × r²
Step 2: Identify values: r = 7 cm, π ≈ 3.14
Step 3: Calculate r²: 7² = 49
Step 4: Calculate area: 3.14 × 49 = 153.86 cm²
Step 5: Verify: Reasonable for 7cm radius ✓

Answer: 153.86 cm²
"""
    print(solution3)


def exercise_2_self_consistency():
    """Exercise 2: Self-Consistency"""
    
    problem = """
A store offers 20% off all items. Sarah buys a jacket originally 
priced at $80. She also has a $10 coupon. What does she pay?
"""
    
    print(f"Problem: {problem}\n")
    
    print("Approach 1 (Discount first, then coupon):")
    approach1 = """
Step 1: Calculate 20% discount: $80 × 0.20 = $16
Step 2: Subtract discount: $80 - $16 = $64
Step 3: Apply coupon: $64 - $10 = $54
Answer: $54
"""
    print(approach1)
    
    print("Approach 2 (Calculate final percentage):")
    approach2 = """
Step 1: After 20% off: $80 × 0.80 = $64
Step 2: Subtract $10 coupon: $64 - $10 = $54
Answer: $54
"""
    print(approach2)
    
    print("Approach 3 (Break down savings):")
    approach3 = """
Step 1: Original price: $80
Step 2: Discount savings: $16
Step 3: Coupon savings: $10
Step 4: Total savings: $16 + $10 = $26
Step 5: Final price: $80 - $26 = $54
Answer: $54
"""
    print(approach3)
    
    print("Consistency Check:")
    print("All three approaches agree: $54")
    print("Confidence: High (3/3 agreement)")


def exercise_3_complex_reasoning():
    """Exercise 3: Complex Reasoning"""
    
    problem = """
A company has 200 employees. 30% work in engineering, 25% in sales,
and the rest in operations. Engineering gets 15% raise, sales gets 10%,
operations gets 8%. If average salary is $60,000, what's the new average?
"""
    
    print(f"Problem: {problem}\n")
    
    solution = """
Let's break this into subproblems:

Subproblem 1: How many employees in each department?
Step 1: Engineering: 200 × 0.30 = 60 employees
Step 2: Sales: 200 × 0.25 = 50 employees
Step 3: Operations: 200 - 60 - 50 = 90 employees

Subproblem 2: What's the total current salary?
Step 1: Total salary: 200 × $60,000 = $12,000,000

Subproblem 3: What's the salary increase per department?
Step 1: Engineering: 60 × $60,000 × 0.15 = $540,000
Step 2: Sales: 50 × $60,000 × 0.10 = $300,000
Step 3: Operations: 90 × $60,000 × 0.08 = $432,000
Step 4: Total increase: $540,000 + $300,000 + $432,000 = $1,272,000

Subproblem 4: What's the new average salary?
Step 1: New total: $12,000,000 + $1,272,000 = $13,272,000
Step 2: New average: $13,272,000 / 200 = $66,360

Verification:
- Original average: $60,000
- Increase: $6,360 (10.6% overall)
- Reasonable given department-specific raises ✓

Answer: $66,360
"""
    print(solution)


def exercise_4_verification():
    """Exercise 4: Verification"""
    
    problem = "If 3x + 5 = 20, what is x?"
    
    print(f"Problem: {problem}\n")
    
    solution = """
Solution with verification:

Step 1: Subtract 5 from both sides
3x + 5 - 5 = 20 - 5
3x = 15
Verify: 3x + 5 should equal 20 when we're done ✓

Step 2: Divide both sides by 3
3x / 3 = 15 / 3
x = 5
Verify: Is 15 divisible by 3? Yes ✓

Step 3: Check final answer
Substitute x = 5 into original equation:
3(5) + 5 = 15 + 5 = 20 ✓
Verify: Matches right side of equation ✓

Answer: x = 5 (verified)
"""
    print(solution)


def exercise_5_domain_application():
    """Exercise 5: Domain Application"""
    
    print("Domain 1: Code Debugging\n")
    code_debug = """
Code with bug:
def calculate_discount(price, discount_percent):
    discount = price * discount_percent
    return price - discount

result = calculate_discount(100, 20)
# Expected: 80, Got: 1980

Debug step by step:
Step 1: Identify the issue
- Expected 20% discount on $100 = $80 final price
- Got $1980 (clearly wrong)

Step 2: Trace the calculation
- discount = 100 * 20 = 2000 (should be 100 * 0.20 = 20)
- return 100 - 2000 = -1900 (negative!)

Step 3: Find the root cause
- discount_percent should be decimal (0.20) not percentage (20)
- Missing conversion: discount_percent / 100

Step 4: Fix the code
def calculate_discount(price, discount_percent):
    discount = price * (discount_percent / 100)
    return price - discount

Step 5: Verify fix
- calculate_discount(100, 20)
- discount = 100 * (20/100) = 100 * 0.20 = 20
- return 100 - 20 = 80 ✓

Fixed!
"""
    print(code_debug)
    
    print("\n" + "="*60)
    print("Domain 2: Data Analysis\n")
    data_analysis = """
Data: Monthly sales
Jan: $100k, Feb: $120k, Mar: $110k, Apr: $140k

Analyze the trend step by step:

Step 1: Calculate month-over-month changes
- Jan to Feb: +$20k (+20%)
- Feb to Mar: -$10k (-8.3%)
- Mar to Apr: +$30k (+27.3%)

Step 2: Identify patterns
- Overall trend: Upward (Jan $100k → Apr $140k)
- Growth: +40% over 4 months
- Volatility: March dip

Step 3: Calculate average growth
- Average monthly sales: ($100k + $120k + $110k + $140k) / 4 = $117.5k
- Average growth rate: 40% / 3 months = 13.3% per month

Step 4: Investigate anomaly
- March decline: -8.3% (only negative month)
- Possible causes: Seasonal, market conditions, competition

Step 5: Forecast
- If trend continues: May estimate = $140k × 1.133 = $158.6k
- Conservative estimate (accounting for volatility): $150k

Conclusion: Strong growth trend with March anomaly to investigate
"""
    print(data_analysis)


if __name__ == "__main__":
    print("Day 75: Chain of Thought - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Basic CoT")
    print("=" * 60)
    exercise_1_basic_cot()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Self-Consistency")
    print("=" * 60)
    exercise_2_self_consistency()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Complex Reasoning")
    print("=" * 60)
    exercise_3_complex_reasoning()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Verification")
    print("=" * 60)
    exercise_4_verification()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Domain Application")
    print("=" * 60)
    exercise_5_domain_application()
