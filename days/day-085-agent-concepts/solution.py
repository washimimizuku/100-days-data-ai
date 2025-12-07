"""Day 85: Agent Concepts - Solutions"""

from typing import List, Dict, Any, Optional
from collections import deque


# Exercise 1: Basic Agent Loop
def basic_agent_loop(goal: str, max_iterations: int = 5) -> List[str]:
    """Implement a basic agent loop."""
    actions = []
    iteration = 0
    goal_achieved = False
    
    while iteration < max_iterations and not goal_achieved:
        thought = f"Iteration {iteration + 1}: Working toward '{goal}'"
        action = f"action_{iteration + 1}"
        observation = f"result_{iteration + 1}"
        actions.append(action)
        iteration += 1
        if iteration >= 3:
            goal_achieved = True
    
    return actions


# Exercise 2: Memory System
class AgentMemory:
    """Memory system for agents."""
    
    def __init__(self, short_term_limit: int = 10):
        self.short_term_limit = short_term_limit
        self.short_term = deque(maxlen=short_term_limit)
        self.long_term = {}
        self.working = {}
    
    def add_short_term(self, item: str):
        self.short_term.append(item)
    
    def add_long_term(self, key: str, value: Any):
        self.long_term[key] = value
    
    def get_short_term(self) -> List[str]:
        return list(self.short_term)
    
    def get_long_term(self, key: str) -> Optional[Any]:
        return self.long_term.get(key)
    
    def clear_short_term(self):
        self.short_term.clear()
    
    def set_working(self, key: str, value: Any):
        self.working[key] = value
    
    def get_working(self, key: str) -> Optional[Any]:
        return self.working.get(key)


# Exercise 3: Tool Integration
class AgentTool:
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class Calculator(AgentTool):
    """Calculator tool."""
    
    def __init__(self):
        super().__init__("calculator", "Performs mathematical calculations")
    
    def execute(self, expression: str) -> float:
        try:
            allowed = set('0123456789+-*/().')
            if not all(c in allowed or c.isspace() for c in expression):
                raise ValueError("Invalid characters in expression")
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            return 0.0


class SearchTool(AgentTool):
    """Search tool."""
    
    def __init__(self):
        super().__init__("search", "Searches for information")
        self.knowledge = {
            "weather": "The weather is sunny and 72°F",
            "python": "Python is a high-level programming language",
            "ai": "AI stands for Artificial Intelligence"
        }
    
    def execute(self, query: str) -> str:
        query_lower = query.lower()
        for key, value in self.knowledge.items():
            if key in query_lower:
                return value
        return f"No results found for: {query}"


# Exercise 4: Goal Tracking
class GoalTracker:
    """Track progress toward goals."""
    
    def __init__(self, goal: str):
        self.goal = goal
        self.steps = []
        self.completed = False
    
    def add_step(self, action: str, result: Any):
        self.steps.append({"action": action, "result": result})
    
    def evaluate_progress(self) -> float:
        if not self.steps:
            return 0.0
        if self.completed:
            return 1.0
        return min(len(self.steps) / 5.0, 0.9)
    
    def is_complete(self) -> bool:
        return self.completed or len(self.steps) >= 5
    
    def mark_complete(self):
        self.completed = True
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps_taken": len(self.steps),
            "progress": self.evaluate_progress(),
            "completed": self.is_complete()
        }


# Exercise 5: Error Handling
class ResilientAgent:
    """Agent with error handling."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.errors = []
    
    def execute_with_retry(self, action_fn, *args, **kwargs) -> Any:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = action_fn(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                self.errors.append({"attempt": attempt + 1, "error": str(e)})
        return self.fallback_action(last_error)
    
    def validate_result(self, result: Any, expected_type: type) -> bool:
        return isinstance(result, expected_type)
    
    def fallback_action(self, error: Exception) -> Any:
        return {"error": str(error), "fallback": True}


# Bonus: Simple Agent Implementation
class SimpleAgent:
    """Complete agent implementation."""
    
    def __init__(self, goal: str, tools: List[AgentTool]):
        self.goal = goal
        self.tools = {tool.name: tool for tool in tools}
        self.memory = AgentMemory()
        self.tracker = GoalTracker(goal)
        self.resilient = ResilientAgent()
    
    def run(self, max_iterations: int = 10) -> Dict[str, Any]:
        iteration = 0
        while iteration < max_iterations and not self.tracker.is_complete():
            observation = self.memory.get_working("last_result") or "Starting"
            thought = self.reason(observation)
            self.memory.add_short_term(f"Thought: {thought}")
            tool = self.select_tool(thought)
            if tool:
                result = self.resilient.execute_with_retry(tool.execute, thought)
                self.memory.set_working("last_result", result)
                self.tracker.add_step(f"Used {tool.name}", result)
            else:
                self.tracker.add_step("Reasoning", thought)
            iteration += 1
            if iteration >= 3:
                self.tracker.mark_complete()
        
        return {
            "goal": self.goal,
            "summary": self.tracker.get_summary(),
            "memory": self.memory.get_short_term(),
            "iterations": iteration
        }
    
    def select_tool(self, thought: str) -> Optional[AgentTool]:
        thought_lower = thought.lower()
        if "calculate" in thought_lower or "math" in thought_lower:
            return self.tools.get("calculator")
        elif "search" in thought_lower or "find" in thought_lower:
            return self.tools.get("search")
        return None
    
    def reason(self, observation: str) -> str:
        if "Starting" in observation:
            return f"I need to work on: {self.goal}"
        return f"Based on {observation}, continuing toward goal"


def demo_agent_concepts():
    """Demonstrate agent concepts."""
    print("Day 85: Agent Concepts - Solutions Demo\n" + "=" * 60)
    
    print("\n1. Basic Agent Loop")
    actions = basic_agent_loop("Find information", max_iterations=5)
    print(f"   Actions taken: {actions}")
    
    print("\n2. Memory System")
    memory = AgentMemory(short_term_limit=5)
    for i in range(7):
        memory.add_short_term(f"Memory {i}")
    memory.add_long_term("fact", "Python is awesome")
    print(f"   Short-term (limited to 5): {memory.get_short_term()}")
    print(f"   Long-term fact: {memory.get_long_term('fact')}")
    
    print("\n3. Tool Integration")
    calc = Calculator()
    search = SearchTool()
    print(f"   Calculator: 10 + 5 = {calc.execute('10 + 5')}")
    print(f"   Search: {search.execute('weather')}")
    
    print("\n4. Goal Tracking")
    tracker = GoalTracker("Complete demo")
    tracker.add_step("Initialize", "success")
    tracker.add_step("Process", "success")
    print(f"   Progress: {tracker.evaluate_progress():.1%}")
    print(f"   Summary: {tracker.get_summary()}")
    
    print("\n5. Error Handling")
    agent = ResilientAgent(max_retries=3)
    def failing_action():
        raise ValueError("Simulated error")
    result = agent.execute_with_retry(failing_action)
    print(f"   Fallback result: {result}")
    
    print("\n6. Complete Agent")
    tools = [Calculator(), SearchTool()]
    simple_agent = SimpleAgent("Demonstrate agent capabilities", tools)
    result = simple_agent.run(max_iterations=5)
    print(f"   Goal: {result['goal']}")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Completed: {result['summary']['completed']}")
    print(f"   Memory items: {len(result['memory'])}")
    
    print("\n" + "=" * 60)
    print("All agent concepts demonstrated successfully!")


if __name__ == "__main__":
    demo_agent_concepts()
