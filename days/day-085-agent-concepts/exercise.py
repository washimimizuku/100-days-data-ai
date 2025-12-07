"""
Day 85: Agent Concepts - Exercises

Complete each exercise to understand agent fundamentals.
"""

from typing import List, Dict, Any, Optional


# Exercise 1: Basic Agent Loop
def basic_agent_loop(goal: str, max_iterations: int = 5) -> List[str]:
    """
    Implement a basic agent loop that iterates toward a goal.
    
    The loop should:
    1. Start with the goal
    2. Generate a thought about what to do next
    3. Take an action (simulated)
    4. Observe the result
    5. Repeat until goal is achieved or max iterations reached
    
    Args:
        goal: The objective to achieve
        max_iterations: Maximum number of iterations
    
    Returns:
        List of actions taken
    
    Example:
        >>> actions = basic_agent_loop("Find the weather in Seattle")
        >>> len(actions) > 0
        True
    """
    actions = []
    
    # TODO: Implement the agent loop
    # Hint: Use a while loop with iteration counter
    # Hint: Simulate thoughts, actions, and observations
    
    return actions


# Exercise 2: Memory System
class AgentMemory:
    """
    Implement a memory system for agents.
    
    Should support:
    - Short-term memory (recent interactions)
    - Long-term memory (persistent facts)
    - Working memory (current task state)
    """
    
    def __init__(self, short_term_limit: int = 10):
        """Initialize memory systems."""
        self.short_term_limit = short_term_limit
        # TODO: Initialize memory structures
        pass
    
    def add_short_term(self, item: str):
        """Add to short-term memory (limited size)."""
        # TODO: Implement with size limit
        pass
    
    def add_long_term(self, key: str, value: Any):
        """Add to long-term memory (persistent)."""
        # TODO: Implement persistent storage
        pass
    
    def get_short_term(self) -> List[str]:
        """Retrieve recent short-term memories."""
        # TODO: Return recent items
        pass
    
    def get_long_term(self, key: str) -> Optional[Any]:
        """Retrieve from long-term memory."""
        # TODO: Return stored value
        pass
    
    def clear_short_term(self):
        """Clear short-term memory."""
        # TODO: Implement clearing
        pass


# Exercise 3: Tool Integration
class AgentTool:
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        raise NotImplementedError


class Calculator(AgentTool):
    """Calculator tool for mathematical operations."""
    
    def __init__(self):
        super().__init__("calculator", "Performs mathematical calculations")
    
    def execute(self, expression: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Math expression as string
        
        Returns:
            Result of calculation
        
        Example:
            >>> calc = Calculator()
            >>> calc.execute("2 + 2")
            4.0
        """
        # TODO: Implement safe evaluation
        # Hint: Use eval() carefully or implement parser
        pass


class SearchTool(AgentTool):
    """Search tool for information retrieval."""
    
    def __init__(self):
        super().__init__("search", "Searches for information")
    
    def execute(self, query: str) -> str:
        """
        Search for information (simulated).
        
        Args:
            query: Search query
        
        Returns:
            Search results
        """
        # TODO: Implement simulated search
        # Hint: Return mock results based on query
        pass


# Exercise 4: Goal Tracking
class GoalTracker:
    """Track progress toward agent goals."""
    
    def __init__(self, goal: str):
        self.goal = goal
        # TODO: Initialize tracking structures
        pass
    
    def add_step(self, action: str, result: Any):
        """Record a step taken toward the goal."""
        # TODO: Track action and result
        pass
    
    def evaluate_progress(self) -> float:
        """
        Evaluate progress toward goal (0.0 to 1.0).
        
        Returns:
            Progress score between 0 and 1
        """
        # TODO: Calculate progress
        # Hint: Consider number of steps, results quality
        pass
    
    def is_complete(self) -> bool:
        """Check if goal is achieved."""
        # TODO: Determine completion
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of progress."""
        # TODO: Return progress summary
        pass


# Exercise 5: Error Handling
class ResilientAgent:
    """Agent with error handling and retry logic."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        # TODO: Initialize error tracking
        pass
    
    def execute_with_retry(self, action_fn, *args, **kwargs) -> Any:
        """
        Execute action with retry logic.
        
        Args:
            action_fn: Function to execute
            *args, **kwargs: Arguments for function
        
        Returns:
            Result of successful execution
        
        Raises:
            Exception if all retries fail
        """
        # TODO: Implement retry logic
        # Hint: Use try-except with loop
        pass
    
    def validate_result(self, result: Any, expected_type: type) -> bool:
        """
        Validate action result.
        
        Args:
            result: Result to validate
            expected_type: Expected type
        
        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement validation
        pass
    
    def fallback_action(self, error: Exception) -> Any:
        """
        Provide fallback when action fails.
        
        Args:
            error: The error that occurred
        
        Returns:
            Fallback result
        """
        # TODO: Implement fallback strategy
        pass


# Bonus: Simple Agent Implementation
class SimpleAgent:
    """
    A simple agent combining all concepts.
    
    Integrates:
    - Agent loop
    - Memory system
    - Tool use
    - Goal tracking
    - Error handling
    """
    
    def __init__(self, goal: str, tools: List[AgentTool]):
        self.goal = goal
        self.tools = {tool.name: tool for tool in tools}
        self.memory = AgentMemory()
        self.tracker = GoalTracker(goal)
    
    def run(self, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Run the agent to achieve the goal.
        
        Args:
            max_iterations: Maximum iterations
        
        Returns:
            Final result and summary
        """
        # TODO: Implement complete agent loop
        # Hint: Combine all previous exercises
        pass
    
    def select_tool(self, thought: str) -> Optional[AgentTool]:
        """Select appropriate tool based on thought."""
        # TODO: Implement tool selection logic
        pass
    
    def reason(self, observation: str) -> str:
        """Generate reasoning about next action."""
        # TODO: Implement reasoning (can be simple rules)
        pass


if __name__ == "__main__":
    print("Day 85: Agent Concepts - Exercises")
    print("=" * 50)
    
    # Test Exercise 1
    print("\nExercise 1: Basic Agent Loop")
    actions = basic_agent_loop("Test goal")
    print(f"Actions taken: {len(actions)}")
    
    # Test Exercise 2
    print("\nExercise 2: Memory System")
    memory = AgentMemory()
    memory.add_short_term("First memory")
    print(f"Short-term memories: {len(memory.get_short_term())}")
    
    # Test Exercise 3
    print("\nExercise 3: Tool Integration")
    calc = Calculator()
    print(f"Calculator tool: {calc.name}")
    
    # Test Exercise 4
    print("\nExercise 4: Goal Tracking")
    tracker = GoalTracker("Complete exercises")
    tracker.add_step("Start", "initialized")
    print(f"Progress: {tracker.evaluate_progress():.2f}")
    
    # Test Exercise 5
    print("\nExercise 5: Error Handling")
    agent = ResilientAgent(max_retries=3)
    print(f"Max retries: {agent.max_retries}")
    
    print("\n" + "=" * 50)
    print("Complete the TODOs to finish the exercises!")
