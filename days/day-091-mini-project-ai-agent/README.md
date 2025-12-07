# Day 91: Mini Project - AI Agent

## Project Overview

Build a complete AI agent system that combines ReAct reasoning, tool use, LangGraph workflows, and AWS deployment capabilities. This project integrates all concepts from Week 13.

**Time**: 2 hours

## What You'll Build

A production-ready AI agent that:
- Uses ReAct pattern for reasoning and acting
- Executes multiple tools (search, calculator, weather, data analysis)
- Manages state with LangGraph workflows
- Handles errors and retries gracefully
- Logs execution traces
- Can be deployed to AWS Lambda

## Architecture

```
┌─────────────────────────────────────────────────┐
│              AI Agent System                     │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐      ┌──────────────┐        │
│  │   ReAct      │─────▶│  LangGraph   │        │
│  │   Engine     │      │   Workflow   │        │
│  └──────────────┘      └──────────────┘        │
│         │                      │                │
│         ▼                      ▼                │
│  ┌──────────────────────────────────┐          │
│  │        Tool Registry              │          │
│  ├──────────────────────────────────┤          │
│  │ • Calculator  • Weather           │          │
│  │ • Search      • Data Analysis     │          │
│  └──────────────────────────────────┘          │
│         │                                        │
│         ▼                                        │
│  ┌──────────────────────────────────┐          │
│  │      Execution Trace Logger       │          │
│  └──────────────────────────────────┘          │
│                                                  │
└─────────────────────────────────────────────────┘
```

## Features

### Core Components

1. **ReAct Engine**: Implements thought-action-observation loop
2. **Tool Registry**: Manages available tools with schemas
3. **LangGraph Workflow**: Orchestrates agent execution
4. **State Management**: Tracks conversation and execution history
5. **Error Handling**: Graceful failures with retries
6. **Trace Logger**: Detailed execution logging

### Tools Included

- **Calculator**: Mathematical operations
- **Search**: Web search simulation
- **Weather**: Weather information retrieval
- **Data Analysis**: Statistical analysis on datasets

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Project Structure

```
day-091-mini-project-ai-agent/
├── README.md              # This file
├── project.md             # Detailed specification
├── agent.py               # Main agent implementation
├── tools.py               # Tool implementations
├── workflow.py            # LangGraph workflow
├── test_agent.sh          # Test script
└── requirements.txt       # Dependencies
```

## Quick Start

### 1. Run the Agent

```bash
python agent.py
```

### 2. Interactive Mode

```python
from agent import AIAgent

agent = AIAgent()
result = agent.run("What is 25 * 4 + 10?")
print(result)
```

### 3. Run Tests

```bash
./test_agent.sh
```

## Usage Examples

### Example 1: Mathematical Reasoning

```python
agent = AIAgent()
result = agent.run("Calculate the average of 10, 20, 30, 40, 50")
# Agent uses calculator tool and reasoning
```

### Example 2: Multi-Step Task

```python
result = agent.run("""
Get the weather for Seattle, then calculate 
how many days until temperature reaches 70°F
""")
# Agent uses weather tool, then calculator
```

### Example 3: Data Analysis

```python
result = agent.run("""
Analyze this dataset: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Calculate mean, median, and standard deviation
""")
# Agent uses data analysis tool
```

## Implementation Guide

### Step 1: Tool Implementation (30 min)

Implement tools in `tools.py`:
- Define tool schemas
- Implement tool functions
- Add error handling

### Step 2: ReAct Engine (30 min)

Build ReAct engine in `agent.py`:
- Thought generation
- Action selection
- Observation processing
- Loop control

### Step 3: LangGraph Workflow (30 min)

Create workflow in `workflow.py`:
- Define state structure
- Create nodes (think, act, observe)
- Add conditional edges
- Handle termination

### Step 4: Integration & Testing (30 min)

Integrate components and test:
- Connect all components
- Add logging
- Test edge cases
- Verify error handling

## Testing

The test script validates:
- ✅ Tool execution
- ✅ ReAct reasoning loop
- ✅ Multi-step tasks
- ✅ Error handling
- ✅ State management
- ✅ Trace logging

## Deployment (Bonus)

### Deploy to AWS Lambda

```python
# lambda_handler.py
from agent import AIAgent

agent = AIAgent()

def lambda_handler(event, context):
    query = event.get('query', '')
    result = agent.run(query)
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Package for Lambda

```bash
pip install -r requirements.txt -t package/
cp agent.py tools.py workflow.py package/
cd package && zip -r ../agent.zip .
```

## Key Concepts Applied

### From Day 85 (Agent Concepts)
- Agent architecture
- Perception-reasoning-action loop
- Memory management

### From Day 86 (ReAct Pattern)
- Thought-action-observation cycles
- Reasoning traces
- Action selection

### From Day 87 (Tool Use)
- Function schemas
- Tool registry
- Parameter extraction

### From Day 88 (LangGraph)
- State management
- Graph workflows
- Conditional routing

### From Days 89-90 (AWS)
- Serverless deployment
- Event-driven architecture
- Cloud integration

## Success Criteria

Your agent should:
- ✅ Execute tools correctly
- ✅ Show reasoning steps
- ✅ Handle multi-step tasks
- ✅ Recover from errors
- ✅ Log execution traces
- ✅ Complete in < 10 iterations

## Troubleshooting

**Agent loops infinitely**: Add max iteration limit
**Tools fail**: Check tool schemas and error handling
**State not updating**: Verify state management in workflow
**No reasoning shown**: Enable trace logging

## Extensions

1. **Add More Tools**: File operations, API calls, database queries
2. **Improve Reasoning**: Add reflection, planning, self-correction
3. **Multi-Agent**: Coordinate multiple specialized agents
4. **Persistent Memory**: Store conversation history in database
5. **Web Interface**: Build FastAPI frontend for agent

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [AWS Lambda Python](https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html)

## Next Steps

After completing this project:
1. Review your implementation
2. Test with complex queries
3. Optimize performance
4. Consider deployment options
5. Move to Week 14: Advanced AI Topics

Congratulations on completing Week 13! You've built a complete AI agent system integrating reasoning, tools, workflows, and cloud deployment.
