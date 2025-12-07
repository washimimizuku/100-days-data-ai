"""AI Agent with ReAct Pattern

Main agent implementation using ReAct reasoning.
"""

from typing import Dict, Any, List
import re
from tools import ToolRegistry


class ReActEngine:
    """ReAct reasoning engine."""
    
    def __init__(self, registry: ToolRegistry, max_iterations: int = 10):
        self.registry = registry
        self.max_iterations = max_iterations
    
    def generate_thought(self, state: Dict[str, Any]) -> str:
        """Generate reasoning step."""
        query = state["query"]
        observations = state["observations"]
        iteration = state["iteration"]
        
        if iteration == 0:
            return f"I need to solve: {query}. Let me think about what tools I need."
        
        last_obs = observations[-1]
        
        # Check if we have a final answer
        if "result" in last_obs.lower() or "analysis" in last_obs.lower():
            return "I have the information needed. Let me formulate the final answer."
        
        return f"Based on the observation: {last_obs}, I should continue processing."
    
    def select_action(self, thought: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select tool and parameters based on thought."""
        query = state["query"].lower()
        thought_lower = thought.lower()
        
        # Calculator patterns
        if any(word in query for word in ["calculate", "compute", "math", "+", "-", "*", "/"]):
            # Extract expression
            expression = self._extract_expression(state["query"])
            return {
                "tool": "calculator",
                "parameters": {"expression": expression}
            }
        
        # Weather patterns
        if "weather" in query:
            location = self._extract_location(state["query"])
            return {
                "tool": "weather",
                "parameters": {"location": location}
            }
        
        # Data analysis patterns
        if "analyze" in query or "statistics" in query or "mean" in query:
            data = self._extract_data(state["query"])
            return {
                "tool": "data_analyzer",
                "parameters": {"data": data}
            }
        
        # Search patterns
        if "search" in query or "find" in query or "look up" in query:
            return {
                "tool": "search",
                "parameters": {"query": state["query"]}
            }
        
        # Default to finish
        return {
            "tool": "finish",
            "parameters": {"answer": "Task complete"}
        }
    
    def _extract_expression(self, query: str) -> str:
        """Extract mathematical expression from query."""
        # Look for numbers and operators
        pattern = r'[\d\s+\-*/().]+(?=\?|$)'
        match = re.search(pattern, query)
        if match:
            return match.group().strip()
        return query
    
    def _extract_location(self, query: str) -> str:
        """Extract location from query."""
        # Simple extraction - look for city names
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ["for", "in", "at"]:
                if i + 1 < len(words):
                    return words[i + 1].strip("?,.")
        return "Seattle"  # Default
    
    def _extract_data(self, query: str) -> List[float]:
        """Extract numerical data from query."""
        # Look for lists of numbers
        numbers = re.findall(r'\d+\.?\d*', query)
        return [float(n) for n in numbers] if numbers else [1, 2, 3, 4, 5]
    
    def process_observation(self, action: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Convert tool result to observation."""
        if result.get("error"):
            return f"Error executing {action['tool']}: {result['error']}"
        
        tool = action["tool"]
        
        if tool == "calculator":
            return f"Calculation result: {result['result']}"
        
        if tool == "weather":
            weather = result["weather"]
            return f"Weather in {weather['location']}: {weather['temperature']}°F, {weather['condition']}"
        
        if tool == "data_analyzer":
            analysis = result["analysis"]
            return f"Data analysis: mean={analysis['mean']:.2f}, median={analysis['median']}"
        
        if tool == "search":
            return f"Found {len(result['results'])} search results"
        
        return str(result)
    
    def is_complete(self, state: Dict[str, Any]) -> bool:
        """Check if task is complete."""
        # Check iteration limit
        if state["iteration"] >= self.max_iterations:
            return True
        
        # Check for finish action
        if state["actions"] and state["actions"][-1]["tool"] == "finish":
            return True
        
        # Check for final answer in thought
        if state["thoughts"]:
            last_thought = state["thoughts"][-1].lower()
            if "final answer" in last_thought or "complete" in last_thought:
                return True
        
        return False
    
    def extract_answer(self, state: Dict[str, Any]) -> str:
        """Extract final answer from state."""
        # Check finish action
        if state["actions"] and state["actions"][-1]["tool"] == "finish":
            return state["actions"][-1]["parameters"]["answer"]
        
        # Use last observation
        if state["observations"]:
            return state["observations"][-1]
        
        return "No answer found"


class AIAgent:
    """AI Agent with ReAct reasoning."""
    
    def __init__(self, max_iterations: int = 10):
        self.registry = ToolRegistry()
        self.engine = ReActEngine(self.registry, max_iterations)
    
    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Run agent on query."""
        # Initialize state
        state = {
            "query": query,
            "thoughts": [],
            "actions": [],
            "observations": [],
            "iteration": 0,
            "complete": False
        }
        
        if verbose:
            print(f"\nQuery: {query}\n" + "=" * 60)
        
        # ReAct loop
        while not state["complete"]:
            # Think
            thought = self.engine.generate_thought(state)
            state["thoughts"].append(thought)
            
            if verbose:
                print(f"\nIteration {state['iteration'] + 1}")
                print(f"Thought: {thought}")
            
            # Act
            action = self.engine.select_action(thought, state)
            state["actions"].append(action)
            
            if verbose:
                print(f"Action: {action['tool']}({action['parameters']})")
            
            # Execute tool
            if action["tool"] == "finish":
                state["complete"] = True
                break
            
            result = self.registry.execute_tool(action["tool"], **action["parameters"])
            
            # Observe
            observation = self.engine.process_observation(action, result)
            state["observations"].append(observation)
            
            if verbose:
                print(f"Observation: {observation}")
            
            # Update iteration
            state["iteration"] += 1
            
            # Check completion
            state["complete"] = self.engine.is_complete(state)
        
        # Extract answer
        answer = self.engine.extract_answer(state)
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"Final Answer: {answer}\n")
        
        return {
            "query": query,
            "answer": answer,
            "iterations": state["iteration"],
            "thoughts": state["thoughts"],
            "actions": state["actions"],
            "observations": state["observations"]
        }


def demo_agent():
    """Demonstrate agent capabilities."""
    print("AI Agent Demo\n" + "=" * 60)
    
    agent = AIAgent()
    
    # Test 1: Simple calculation
    print("\n1. Simple Calculation")
    agent.run("What is 25 * 4 + 10?")
    
    # Test 2: Weather query
    print("\n2. Weather Query")
    agent.run("What is the weather in Seattle?")
    
    # Test 3: Data analysis
    print("\n3. Data Analysis")
    agent.run("Analyze the data: 10, 20, 30, 40, 50")
    
    print("\n" + "=" * 60)
    print("Agent demo complete!")


if __name__ == "__main__":
    demo_agent()
