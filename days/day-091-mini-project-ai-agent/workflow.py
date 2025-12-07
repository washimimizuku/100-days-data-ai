"""LangGraph Workflow for AI Agent

Implements state management and workflow orchestration.
"""

from typing import TypedDict, List, Dict, Any, Literal
from tools import ToolRegistry
from agent import ReActEngine


class AgentState(TypedDict):
    """Agent state structure."""
    query: str
    thoughts: List[str]
    actions: List[Dict[str, Any]]
    observations: List[str]
    iteration: int
    complete: bool
    last_result: Dict[str, Any]


class WorkflowAgent:
    """Agent with LangGraph-style workflow."""
    
    def __init__(self, max_iterations: int = 10):
        self.registry = ToolRegistry()
        self.engine = ReActEngine(self.registry, max_iterations)
        self.max_iterations = max_iterations
    
    def create_initial_state(self, query: str) -> AgentState:
        """Create initial state."""
        return {
            "query": query,
            "thoughts": [],
            "actions": [],
            "observations": [],
            "iteration": 0,
            "complete": False,
            "last_result": {}
        }
    
    def think_node(self, state: AgentState) -> AgentState:
        """Generate thought."""
        thought = self.engine.generate_thought(state)
        state["thoughts"].append(thought)
        return state
    
    def act_node(self, state: AgentState) -> AgentState:
        """Execute action."""
        thought = state["thoughts"][-1]
        action = self.engine.select_action(thought, state)
        state["actions"].append(action)
        
        # Execute tool if not finish
        if action["tool"] != "finish":
            result = self.registry.execute_tool(action["tool"], **action["parameters"])
            state["last_result"] = result
        else:
            state["last_result"] = {"finish": True}
        
        return state
    
    def observe_node(self, state: AgentState) -> AgentState:
        """Process observation."""
        action = state["actions"][-1]
        result = state["last_result"]
        
        if result.get("finish"):
            state["complete"] = True
        else:
            observation = self.engine.process_observation(action, result)
            state["observations"].append(observation)
        
        state["iteration"] += 1
        return state
    
    def decide_node(self, state: AgentState) -> AgentState:
        """Check completion."""
        if not state["complete"]:
            state["complete"] = self.engine.is_complete(state)
        return state
    
    def should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine next step."""
        if state["complete"] or state["iteration"] >= self.max_iterations:
            return "end"
        return "continue"
    
    def run_workflow(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Run workflow on query."""
        state = self.create_initial_state(query)
        
        if verbose:
            print(f"\nQuery: {query}\n" + "=" * 60)
        
        while True:
            # Think
            state = self.think_node(state)
            if verbose:
                print(f"\nIteration {state['iteration'] + 1}")
                print(f"Thought: {state['thoughts'][-1]}")
            
            # Act
            state = self.act_node(state)
            if verbose:
                action = state['actions'][-1]
                print(f"Action: {action['tool']}({action['parameters']})")
            
            # Observe
            state = self.observe_node(state)
            if verbose and state['observations']:
                print(f"Observation: {state['observations'][-1]}")
            
            # Decide
            state = self.decide_node(state)
            
            # Check if should continue
            if self.should_continue(state) == "end":
                break
        
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


class MultiStepWorkflow:
    """Workflow for multi-step tasks."""
    
    def __init__(self):
        self.registry = ToolRegistry()
    
    def decompose_task(self, query: str) -> List[str]:
        """Decompose complex query into steps."""
        # Simple decomposition based on keywords
        steps = []
        
        if "then" in query.lower():
            parts = query.split("then")
            steps = [p.strip() for p in parts]
        elif "and" in query.lower():
            parts = query.split("and")
            steps = [p.strip() for p in parts]
        else:
            steps = [query]
        
        return steps
    
    def execute_step(self, step: str) -> Dict[str, Any]:
        """Execute single step."""
        agent = WorkflowAgent(max_iterations=5)
        return agent.run_workflow(step, verbose=False)
    
    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Run multi-step workflow."""
        if verbose:
            print(f"\nMulti-Step Query: {query}\n" + "=" * 60)
        
        steps = self.decompose_task(query)
        
        if verbose:
            print(f"Decomposed into {len(steps)} steps:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")
        
        results = []
        for i, step in enumerate(steps, 1):
            if verbose:
                print(f"\n--- Step {i} ---")
            result = self.execute_step(step)
            results.append(result)
            if verbose:
                print(f"Step {i} Answer: {result['answer']}")
        
        # Combine results
        final_answer = " | ".join([r["answer"] for r in results])
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"Final Combined Answer: {final_answer}\n")
        
        return {
            "query": query,
            "steps": steps,
            "step_results": results,
            "final_answer": final_answer
        }


def demo_workflow():
    """Demonstrate workflow capabilities."""
    print("Workflow Agent Demo\n" + "=" * 60)
    
    # Test 1: Simple workflow
    print("\n1. Simple Workflow")
    agent = WorkflowAgent()
    agent.run_workflow("Calculate 15 * 3")
    
    # Test 2: Multi-step workflow
    print("\n2. Multi-Step Workflow")
    multi_agent = MultiStepWorkflow()
    multi_agent.run("Calculate 10 + 20 then multiply by 2")
    
    print("\n" + "=" * 60)
    print("Workflow demo complete!")


if __name__ == "__main__":
    demo_workflow()
