# AI Agent Mini Project - Detailed Specification

## Project Goals

Build a production-ready AI agent that demonstrates:
1. ReAct reasoning pattern
2. Tool use and function calling
3. LangGraph state management
4. Error handling and recovery
5. Execution tracing and logging

## System Architecture

### Components

#### 1. Tool System (`tools.py`)

**Tool Schema**:
```python
{
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "parameters": {
        "expression": "string - mathematical expression"
    }
}
```

**Required Tools**:
- Calculator: Evaluate math expressions
- Search: Simulate web search
- Weather: Get weather information
- DataAnalyzer: Statistical analysis

#### 2. ReAct Engine (`agent.py`)

**Core Loop**:
```
1. Thought: Reason about the task
2. Action: Select and execute tool
3. Observation: Process tool result
4. Repeat until task complete
```

**State Structure**:
```python
{
    "query": str,           # User query
    "thoughts": List[str],  # Reasoning steps
    "actions": List[dict],  # Actions taken
    "observations": List[str],  # Tool results
    "iteration": int,       # Current iteration
    "complete": bool        # Task completion status
}
```

#### 3. LangGraph Workflow (`workflow.py`)

**Nodes**:
- `think`: Generate reasoning step
- `act`: Execute tool
- `observe`: Process result
- `decide`: Check if complete

**Edges**:
- think → act
- act → observe
- observe → decide
- decide → think (if not complete)
- decide → END (if complete)

## Implementation Details

### Tool Implementation

Each tool must implement:

```python
class Tool:
    def __init__(self, name: str, description: str, schema: dict):
        self.name = name
        self.description = description
        self.schema = schema
    
    def execute(self, **kwargs) -> dict:
        """Execute tool with parameters."""
        pass
```

**Calculator Tool**:
```python
def execute(self, expression: str) -> dict:
    try:
        result = eval(expression)  # Safe in controlled environment
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}
```

**Search Tool**:
```python
def execute(self, query: str) -> dict:
    # Simulate search results
    results = [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'",
        f"Result 3 for '{query}'"
    ]
    return {"results": results, "error": None}
```

**Weather Tool**:
```python
def execute(self, location: str) -> dict:
    # Mock weather data
    weather = {
        "location": location,
        "temperature": 72,
        "condition": "Sunny",
        "humidity": 45
    }
    return {"weather": weather, "error": None}
```

**DataAnalyzer Tool**:
```python
def execute(self, data: List[float]) -> dict:
    import statistics
    return {
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "stdev": statistics.stdev(data),
        "error": None
    }
```

### ReAct Engine Implementation

**Thought Generation**:
```python
def generate_thought(self, state: dict) -> str:
    """Generate reasoning step based on current state."""
    query = state["query"]
    observations = state["observations"]
    
    if not observations:
        return f"I need to solve: {query}"
    
    last_obs = observations[-1]
    return f"Based on {last_obs}, I should..."
```

**Action Selection**:
```python
def select_action(self, thought: str, state: dict) -> dict:
    """Select tool and parameters based on thought."""
    # Parse thought to determine tool
    if "calculate" in thought.lower():
        return {
            "tool": "calculator",
            "parameters": {"expression": "..."}
        }
    # ... other tools
```

**Observation Processing**:
```python
def process_observation(self, action: dict, result: dict) -> str:
    """Convert tool result to observation."""
    if result.get("error"):
        return f"Error: {result['error']}"
    return f"Tool {action['tool']} returned: {result}"
```

**Completion Check**:
```python
def is_complete(self, state: dict) -> bool:
    """Check if task is complete."""
    # Check for final answer in observations
    # Check iteration limit
    # Check for explicit completion signal
    return state["iteration"] >= 10 or "final answer" in state["observations"][-1].lower()
```

### LangGraph Workflow Implementation

**State Definition**:
```python
from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    thoughts: List[str]
    actions: List[dict]
    observations: List[str]
    iteration: int
    complete: bool
```

**Node Functions**:
```python
def think_node(state: AgentState) -> AgentState:
    """Generate thought."""
    thought = engine.generate_thought(state)
    state["thoughts"].append(thought)
    return state

def act_node(state: AgentState) -> AgentState:
    """Execute action."""
    thought = state["thoughts"][-1]
    action = engine.select_action(thought, state)
    state["actions"].append(action)
    
    # Execute tool
    tool = registry.get_tool(action["tool"])
    result = tool.execute(**action["parameters"])
    state["_last_result"] = result
    return state

def observe_node(state: AgentState) -> AgentState:
    """Process observation."""
    action = state["actions"][-1]
    result = state["_last_result"]
    observation = engine.process_observation(action, result)
    state["observations"].append(observation)
    state["iteration"] += 1
    return state

def decide_node(state: AgentState) -> AgentState:
    """Check completion."""
    state["complete"] = engine.is_complete(state)
    return state
```

**Conditional Routing**:
```python
def should_continue(state: AgentState) -> str:
    """Route based on completion status."""
    if state["complete"]:
        return "end"
    return "continue"
```

**Graph Construction**:
```python
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("think", think_node)
workflow.add_node("act", act_node)
workflow.add_node("observe", observe_node)
workflow.add_node("decide", decide_node)

# Add edges
workflow.add_edge("think", "act")
workflow.add_edge("act", "observe")
workflow.add_edge("observe", "decide")

# Conditional edge
workflow.add_conditional_edges(
    "decide",
    should_continue,
    {
        "continue": "think",
        "end": END
    }
)

# Set entry point
workflow.set_entry_point("think")

# Compile
app = workflow.compile()
```

## Testing Strategy

### Unit Tests

Test each component independently:

```python
def test_calculator_tool():
    tool = CalculatorTool()
    result = tool.execute(expression="2 + 2")
    assert result["result"] == 4
    assert result["error"] is None

def test_thought_generation():
    state = {"query": "What is 2+2?", "observations": []}
    thought = engine.generate_thought(state)
    assert "solve" in thought.lower()

def test_workflow_execution():
    state = {"query": "Calculate 5*5", ...}
    result = app.invoke(state)
    assert result["complete"] is True
```

### Integration Tests

Test complete agent execution:

```python
def test_simple_calculation():
    agent = AIAgent()
    result = agent.run("What is 10 + 20?")
    assert "30" in result["answer"]

def test_multi_step_task():
    agent = AIAgent()
    result = agent.run("Calculate 5*5, then add 10")
    assert "35" in result["answer"]

def test_error_handling():
    agent = AIAgent()
    result = agent.run("Calculate 1/0")
    assert "error" in result["answer"].lower()
```

## Example Execution Traces

### Example 1: Simple Calculation

```
Query: "What is 25 * 4?"

Iteration 1:
  Thought: I need to calculate 25 * 4
  Action: calculator(expression="25 * 4")
  Observation: Result is 100

Iteration 2:
  Thought: I have the answer: 100
  Action: finish(answer="100")
  Observation: Task complete

Final Answer: 100
```

### Example 2: Multi-Step Task

```
Query: "Get weather for Seattle, then calculate if temperature is above 70"

Iteration 1:
  Thought: First, I need to get weather for Seattle
  Action: weather(location="Seattle")
  Observation: Temperature is 72°F, Sunny

Iteration 2:
  Thought: Now I need to check if 72 > 70
  Action: calculator(expression="72 > 70")
  Observation: Result is True

Iteration 3:
  Thought: The temperature is above 70
  Action: finish(answer="Yes, Seattle temperature (72°F) is above 70°F")
  Observation: Task complete

Final Answer: Yes, Seattle temperature (72°F) is above 70°F
```

## Performance Requirements

- **Max Iterations**: 10 per query
- **Tool Execution**: < 100ms per tool
- **Total Time**: < 2 seconds per query
- **Memory**: < 100MB
- **Success Rate**: > 90% on test queries

## Error Handling

### Tool Errors
```python
try:
    result = tool.execute(**params)
except Exception as e:
    result = {"error": str(e), "result": None}
```

### Iteration Limit
```python
if state["iteration"] >= MAX_ITERATIONS:
    state["complete"] = True
    state["observations"].append("Max iterations reached")
```

### Invalid Actions
```python
if action["tool"] not in registry:
    observation = f"Unknown tool: {action['tool']}"
    state["observations"].append(observation)
```

## Logging

Log all execution steps:

```python
import logging

logger = logging.getLogger("agent")

logger.info(f"Query: {query}")
logger.debug(f"Thought: {thought}")
logger.debug(f"Action: {action}")
logger.debug(f"Observation: {observation}")
logger.info(f"Complete: {complete}")
```

## Deployment Considerations

### AWS Lambda Deployment

**Handler**:
```python
def lambda_handler(event, context):
    agent = AIAgent()
    query = event.get("query", "")
    result = agent.run(query)
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }
```

**Environment Variables**:
- `MAX_ITERATIONS`: Maximum iterations
- `LOG_LEVEL`: Logging level
- `TIMEOUT`: Execution timeout

**Memory**: 512 MB recommended
**Timeout**: 30 seconds

## Success Metrics

- ✅ All tools execute correctly
- ✅ ReAct loop completes successfully
- ✅ Multi-step tasks work
- ✅ Errors handled gracefully
- ✅ Execution traces are clear
- ✅ Performance meets requirements

## Extensions

1. **Memory**: Add conversation history
2. **Planning**: Add explicit planning phase
3. **Reflection**: Add self-correction
4. **Multi-Agent**: Coordinate multiple agents
5. **Streaming**: Stream thoughts in real-time
