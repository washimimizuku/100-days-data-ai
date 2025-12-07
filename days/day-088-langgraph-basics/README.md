# Day 88: LangGraph Basics

## Learning Objectives

**Time**: 1 hour

- Understand LangGraph for stateful agent workflows
- Implement graph-based agent architectures
- Build multi-step workflows with state management
- Apply LangGraph to complex agent tasks

## Theory (15 minutes)

### What is LangGraph?

LangGraph is a framework for building stateful, multi-actor applications with LLMs. It extends LangChain with graph-based workflow control, enabling complex agent behaviors through explicit state management and conditional routing.

**Key Concept**: Agents as graphs where nodes are actions and edges define flow.

### Core Concepts

**State**: Shared data structure passed between nodes
**Nodes**: Functions that process state
**Edges**: Connections defining flow between nodes
**Conditional Edges**: Dynamic routing based on state
**Cycles**: Loops for iterative processing

### Graph Structure

```
┌─────────┐
│  Start  │
└────┬────┘
     │
┌────▼────┐
│  Node1  │
└────┬────┘
     │
┌────▼────┐     ┌─────────┐
│  Node2  ├────►│  Node3  │
└────┬────┘     └────┬────┘
     │               │
     └───────┬───────┘
             │
        ┌────▼────┐
        │   End   │
        └─────────┘
```

### Basic Graph

```python
from langgraph.graph import StateGraph

# Define state
class AgentState(TypedDict):
    messages: List[str]
    next_action: str

# Create graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("process", process_function)
graph.add_node("decide", decide_function)

# Add edges
graph.add_edge("process", "decide")
graph.add_conditional_edges("decide", route_function)

# Set entry point
graph.set_entry_point("process")

# Compile
app = graph.compile()
```

### State Management

State is a dictionary passed between nodes:

```python
class State(TypedDict):
    input: str
    output: str
    steps: List[str]
    iteration: int

def node_function(state: State) -> State:
    # Read state
    current_input = state["input"]
    
    # Process
    result = process(current_input)
    
    # Update state
    state["output"] = result
    state["steps"].append("processed")
    
    return state
```

### Node Functions

Nodes are functions that take and return state:

```python
def analyze_node(state: State) -> State:
    """Analyze input and update state."""
    state["analysis"] = analyze(state["input"])
    return state

def decide_node(state: State) -> State:
    """Make decision based on analysis."""
    if state["analysis"]["confidence"] > 0.8:
        state["next"] = "execute"
    else:
        state["next"] = "research"
    return state
```

### Conditional Routing

Route based on state:

```python
def route_function(state: State) -> str:
    """Determine next node based on state."""
    if state["next"] == "execute":
        return "execute_node"
    elif state["next"] == "research":
        return "research_node"
    else:
        return "end"

graph.add_conditional_edges(
    "decide_node",
    route_function,
    {
        "execute_node": "execute",
        "research_node": "research",
        "end": END
    }
)
```

### Cycles and Loops

Create iterative workflows:

```python
def should_continue(state: State) -> str:
    """Check if should continue iterating."""
    if state["iteration"] < 5 and not state["complete"]:
        return "continue"
    return "end"

graph.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "process",  # Loop back
        "end": END
    }
)
```

### Message Passing

LangGraph uses messages for LLM interactions:

```python
from langchain.schema import HumanMessage, AIMessage

def llm_node(state: State) -> State:
    """Call LLM and update state."""
    messages = state["messages"]
    response = llm.invoke(messages)
    
    state["messages"].append(AIMessage(content=response))
    return state
```

### Multi-Agent Patterns

**Sequential**: Agents run in order
```python
graph.add_edge("agent1", "agent2")
graph.add_edge("agent2", "agent3")
```

**Parallel**: Agents run concurrently
```python
graph.add_edge("start", "agent1")
graph.add_edge("start", "agent2")
graph.add_conditional_edges("agent1", merge)
graph.add_conditional_edges("agent2", merge)
```

**Hierarchical**: Supervisor coordinates workers
```python
graph.add_conditional_edges("supervisor", route_to_worker)
graph.add_edge("worker1", "supervisor")
graph.add_edge("worker2", "supervisor")
```

### Checkpointing

Save and restore state:

```python
from langgraph.checkpoint import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Run with checkpoint
result = app.invoke(initial_state, config={"configurable": {"thread_id": "1"}})

# Resume from checkpoint
result = app.invoke(None, config={"configurable": {"thread_id": "1"}})
```

### Streaming

Stream intermediate results:

```python
for output in app.stream(initial_state):
    print(output)
```

### Error Handling

Handle errors in nodes:

```python
def safe_node(state: State) -> State:
    try:
        result = risky_operation(state)
        state["result"] = result
    except Exception as e:
        state["error"] = str(e)
        state["next"] = "error_handler"
    return state
```

### Use Cases

**Research Agent**: Search → Analyze → Synthesize → Verify
**Customer Support**: Classify → Route → Resolve → Follow-up
**Data Pipeline**: Extract → Transform → Validate → Load
**Content Creation**: Research → Draft → Review → Publish

### Advantages

**Explicit Control**: Clear workflow definition
**State Management**: Shared state across nodes
**Flexibility**: Dynamic routing and loops
**Debuggability**: Inspect state at each step
**Composability**: Combine multiple agents

### LangGraph vs. LangChain

**LangChain**: Linear chains, implicit state
**LangGraph**: Graph workflows, explicit state, conditional routing

### Best Practices

**Small Nodes**: Each node does one thing
**Clear State**: Well-defined state structure
**Error Handling**: Graceful failure recovery
**Logging**: Track state changes
**Testing**: Test nodes independently

### Why This Matters

LangGraph enables building complex, stateful agent systems with explicit control flow. It's essential for production agents that need reliability, debuggability, and sophisticated multi-step reasoning.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **State Definition**: Define typed state structures
2. **Node Functions**: Implement state-processing nodes
3. **Graph Builder**: Construct graphs with edges
4. **Conditional Routing**: Implement dynamic routing
5. **Complete Workflow**: Build multi-step agent workflow

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [State Management Guide](https://langchain-ai.github.io/langgraph/concepts/#state)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 89: AWS S3 & EC2

Tomorrow you'll learn about AWS services for deploying and scaling AI applications.
