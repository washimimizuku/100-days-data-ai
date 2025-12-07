"""Day 88: LangGraph Basics - Solutions"""

from typing import TypedDict, List, Dict, Any, Callable


# Exercise 1: State Definition
class AgentState(TypedDict):
    """State structure for agent."""
    input: str
    output: str
    steps: List[str]
    iteration: int
    complete: bool


def validate_state(state: Dict) -> bool:
    """Validate state has required fields."""
    required = ["input", "output", "steps", "iteration", "complete"]
    return all(field in state for field in required)


# Exercise 2: Node Functions
def input_node(state: AgentState) -> AgentState:
    """Process input and initialize state."""
    state["steps"] = state.get("steps", [])
    state["steps"].append("input_processed")
    state["iteration"] = 0
    state["complete"] = False
    return state


def process_node(state: AgentState) -> AgentState:
    """Process current state."""
    state["steps"].append(f"processing_iteration_{state['iteration']}")
    state["iteration"] += 1
    
    # Simulate processing
    if state["iteration"] >= 3:
        state["output"] = f"Processed: {state['input']}"
    
    return state


def decide_node(state: AgentState) -> AgentState:
    """Make decision about next action."""
    state["steps"].append("decision_made")
    
    # Check if processing is complete
    if state["iteration"] >= 3 or state.get("output"):
        state["complete"] = True
    
    return state


def output_node(state: AgentState) -> AgentState:
    """Generate final output."""
    if not state.get("output"):
        state["output"] = f"Final output for: {state['input']}"
    
    state["steps"].append("output_generated")
    state["complete"] = True
    return state


# Exercise 3: Graph Builder
class SimpleGraph:
    """Simple graph implementation."""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.entry_point = None
    
    def add_node(self, name: str, function: Callable):
        """Add a node to the graph."""
        self.nodes[name] = function
    
    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between nodes."""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
    
    def set_entry_point(self, node_name: str):
        """Set the starting node."""
        self.entry_point = node_name
    
    def run(self, initial_state: Dict) -> Dict:
        """Execute the graph."""
        if not self.entry_point:
            return initial_state
        
        state = initial_state.copy()
        current_node = self.entry_point
        visited = set()
        
        while current_node and current_node not in visited:
            if current_node not in self.nodes:
                break
            
            # Execute node
            node_func = self.nodes[current_node]
            state = node_func(state)
            visited.add(current_node)
            
            # Get next node
            next_nodes = self.edges.get(current_node, [])
            current_node = next_nodes[0] if next_nodes else None
        
        return state


# Exercise 4: Conditional Routing
def route_by_iteration(state: AgentState) -> str:
    """Route based on iteration count."""
    if state["iteration"] < 5:
        return "continue"
    return "end"


def route_by_completion(state: AgentState) -> str:
    """Route based on completion status."""
    if state.get("complete", False):
        return "output"
    return "process"


class ConditionalGraph(SimpleGraph):
    """Graph with conditional routing."""
    
    def __init__(self):
        super().__init__()
        self.conditional_edges = {}
    
    def add_conditional_edge(self, from_node: str, condition: Callable, routes: Dict[str, str]):
        """Add conditional edge."""
        self.conditional_edges[from_node] = {
            "condition": condition,
            "routes": routes
        }
    
    def run(self, initial_state: Dict) -> Dict:
        """Execute graph with conditional routing."""
        if not self.entry_point:
            return initial_state
        
        state = initial_state.copy()
        current_node = self.entry_point
        max_iterations = 20
        iteration = 0
        
        while current_node and iteration < max_iterations:
            if current_node not in self.nodes:
                break
            
            # Execute node
            node_func = self.nodes[current_node]
            state = node_func(state)
            iteration += 1
            
            # Determine next node
            if current_node in self.conditional_edges:
                # Use conditional routing
                cond_info = self.conditional_edges[current_node]
                route_key = cond_info["condition"](state)
                current_node = cond_info["routes"].get(route_key)
            else:
                # Use regular edges
                next_nodes = self.edges.get(current_node, [])
                current_node = next_nodes[0] if next_nodes else None
        
        return state


# Exercise 5: Complete Workflow
class WorkflowGraph:
    """Complete workflow with state management."""
    
    def __init__(self):
        self.graph = ConditionalGraph()
        self.state_history = []
    
    def build_workflow(self):
        """Build a complete workflow."""
        # Add nodes
        self.graph.add_node("input", input_node)
        self.graph.add_node("process", process_node)
        self.graph.add_node("decide", decide_node)
        self.graph.add_node("output", output_node)
        
        # Add edges
        self.graph.add_edge("input", "process")
        self.graph.add_edge("process", "decide")
        
        # Add conditional routing from decide
        self.graph.add_conditional_edge(
            "decide",
            route_by_completion,
            {
                "output": "output",
                "process": "process"
            }
        )
        
        # Set entry point
        self.graph.set_entry_point("input")
    
    def execute(self, input_data: str) -> Dict:
        """Execute workflow."""
        # Create initial state
        initial_state = {
            "input": input_data,
            "output": "",
            "steps": [],
            "iteration": 0,
            "complete": False
        }
        
        # Run graph
        final_state = self.graph.run(initial_state)
        
        # Track history
        self.state_history.append(final_state)
        
        return final_state
    
    def get_history(self) -> List[Dict]:
        """Get state history."""
        return self.state_history


# Bonus: Multi-Agent Graph
class MultiAgentGraph:
    """Graph with multiple specialized agents."""
    
    def __init__(self):
        self.agents = {}
        self.graph = ConditionalGraph()
    
    def add_agent(self, name: str, agent_function: Callable):
        """Add a specialized agent."""
        self.agents[name] = agent_function
        self.graph.add_node(name, agent_function)
    
    def route_to_agent(self, state: AgentState) -> str:
        """Route to appropriate agent based on task."""
        task_type = state.get("input", "").lower()
        
        if "calculate" in task_type or "math" in task_type:
            return "calculator_agent"
        elif "search" in task_type or "find" in task_type:
            return "search_agent"
        else:
            return "general_agent"
    
    def build_multi_agent_workflow(self):
        """Build workflow with multiple agents."""
        # Add supervisor
        def supervisor(state: AgentState) -> AgentState:
            state["steps"].append("supervisor_routing")
            return state
        
        self.graph.add_node("supervisor", supervisor)
        
        # Add worker agents
        def calculator_agent(state: AgentState) -> AgentState:
            state["output"] = "Calculator result"
            state["complete"] = True
            return state
        
        def search_agent(state: AgentState) -> AgentState:
            state["output"] = "Search result"
            state["complete"] = True
            return state
        
        def general_agent(state: AgentState) -> AgentState:
            state["output"] = "General result"
            state["complete"] = True
            return state
        
        self.add_agent("calculator_agent", calculator_agent)
        self.add_agent("search_agent", search_agent)
        self.add_agent("general_agent", general_agent)
        
        # Add routing
        self.graph.add_conditional_edge(
            "supervisor",
            self.route_to_agent,
            {
                "calculator_agent": "calculator_agent",
                "search_agent": "search_agent",
                "general_agent": "general_agent"
            }
        )
        
        self.graph.set_entry_point("supervisor")


def demo_langgraph():
    """Demonstrate LangGraph concepts."""
    print("Day 88: LangGraph Basics - Solutions Demo\n" + "=" * 60)
    
    print("\n1. State Definition")
    state = AgentState(input="test", output="", steps=[], iteration=0, complete=False)
    print(f"   State valid: {validate_state(state)}")
    print(f"   State fields: {list(state.keys())}")
    
    print("\n2. Node Functions")
    test_state = AgentState(input="test input", output="", steps=[], iteration=0, complete=False)
    test_state = input_node(test_state)
    print(f"   After input_node: {test_state['steps']}")
    test_state = process_node(test_state)
    print(f"   After process_node: iteration={test_state['iteration']}")
    
    print("\n3. Simple Graph")
    graph = SimpleGraph()
    graph.add_node("start", input_node)
    graph.add_node("process", process_node)
    graph.add_edge("start", "process")
    graph.set_entry_point("start")
    
    result = graph.run({"input": "test", "output": "", "steps": [], "iteration": 0, "complete": False})
    print(f"   Graph result steps: {result['steps']}")
    
    print("\n4. Conditional Routing")
    state1 = AgentState(input="", output="", steps=[], iteration=2, complete=False)
    state2 = AgentState(input="", output="", steps=[], iteration=6, complete=False)
    print(f"   Route at iteration 2: {route_by_iteration(state1)}")
    print(f"   Route at iteration 6: {route_by_iteration(state2)}")
    
    print("\n5. Complete Workflow")
    workflow = WorkflowGraph()
    workflow.build_workflow()
    result = workflow.execute("Process this input")
    print(f"   Workflow complete: {result['complete']}")
    print(f"   Steps taken: {len(result['steps'])}")
    print(f"   Output: {result['output']}")
    
    print("\n6. Multi-Agent Graph")
    multi = MultiAgentGraph()
    multi.build_multi_agent_workflow()
    test_state = AgentState(input="calculate 2+2", output="", steps=[], iteration=0, complete=False)
    route = multi.route_to_agent(test_state)
    print(f"   Routed to: {route}")
    
    print("\n" + "=" * 60)
    print("All LangGraph concepts demonstrated!")


if __name__ == "__main__":
    demo_langgraph()
