# Day 86: ReAct Pattern

## Learning Objectives

**Time**: 1 hour

- Understand the ReAct (Reasoning + Acting) pattern
- Implement thought-action-observation loops
- Build agents that reason before acting
- Apply ReAct to multi-step problem solving

## Theory (15 minutes)

### What is ReAct?

ReAct (Reasoning + Acting) is an agent pattern that interleaves reasoning traces with task-specific actions. Instead of acting immediately, the agent first reasons about what to do, then acts, then observes the result.

**Key Insight**: Explicit reasoning improves decision-making and makes agent behavior interpretable.

### ReAct Loop

```
1. Thought: Reason about current state and next action
2. Action: Execute the chosen action
3. Observation: Receive and process the result
4. Repeat until task complete
```

### Example ReAct Trace

```
Question: What is the capital of France and its population?

Thought 1: I need to find the capital of France first
Action 1: search("capital of France")
Observation 1: Paris is the capital of France

Thought 2: Now I need to find Paris's population
Action 2: search("population of Paris")
Observation 2: Paris has approximately 2.2 million people

Thought 3: I have all the information needed
Action 3: finish("Paris is the capital with 2.2M people")
```

### ReAct vs. Other Patterns

**Direct Prompting**:
```python
response = llm("What is the capital of France?")
# Single shot, no tools
```

**Chain of Thought**:
```python
response = llm("Let's think step by step...")
# Reasoning only, no actions
```

**ReAct**:
```python
agent.run("What is the capital of France?")
# Thought → Action → Observation → Repeat
```

### Core Components

**1. Thought Generation**: LLM reasons about what to do next
- Analyze current state
- Consider available actions
- Plan next step

**2. Action Selection**: Choose and execute action
- Tool selection
- Parameter extraction
- Execution

**3. Observation Processing**: Interpret results
- Parse tool output
- Update state
- Determine next step

**4. Termination**: Decide when task is complete
- Goal achieved
- Max iterations reached
- Error state

### Action Types

**Search**: Query information sources
```python
Action: search("query")
```

**Calculate**: Perform computations
```python
Action: calculate("2 + 2")
```

**Lookup**: Find specific information
```python
Action: lookup("term", "context")
```

**Finish**: Return final answer
```python
Action: finish("answer")
```

### Thought Patterns

**Analysis**: Understanding the problem
```
Thought: The question asks for X, which requires Y
```

**Planning**: Deciding next steps
```
Thought: I should first do A, then B
```

**Reflection**: Evaluating progress
```
Thought: The previous result shows X, so I need to Y
```

**Conclusion**: Synthesizing answer
```
Thought: Based on observations, the answer is X
```

### Implementation Pattern

```python
class ReActAgent:
    def run(self, question):
        state = {"question": question, "steps": []}
        
        while not self.is_complete(state):
            # Generate thought
            thought = self.think(state)
            state["steps"].append({"thought": thought})
            
            # Select and execute action
            action = self.act(thought)
            observation = self.execute(action)
            state["steps"].append({
                "action": action,
                "observation": observation
            })
            
            # Check termination
            if action.type == "finish":
                return observation
        
        return self.synthesize(state)
```

### Prompt Template

```
Answer the following question using this format:

Thought: [your reasoning about what to do next]
Action: [action to take: search/calculate/lookup/finish]
Observation: [result of the action]
... (repeat Thought/Action/Observation as needed)

Question: {question}
```

### Benefits

**Interpretability**: See agent's reasoning process
- Understand decisions
- Debug failures
- Build trust

**Reliability**: Explicit reasoning reduces errors
- Validate logic
- Catch mistakes
- Improve accuracy

**Flexibility**: Adapt to different tasks
- Multi-step problems
- Tool composition
- Error recovery

**Learning**: Improve through examples
- Few-shot learning
- Pattern recognition
- Strategy refinement

### Challenges

**Prompt Engineering**: Crafting effective prompts
- Balance detail and brevity
- Guide without constraining
- Handle edge cases

**Error Handling**: Managing failures
- Invalid actions
- Tool errors
- Infinite loops

**Cost**: Token usage from reasoning
- Longer prompts
- Multiple LLM calls
- Observation text

**Hallucination**: LLM may invent observations
- Validate tool outputs
- Ground in reality
- Verify facts

### Use Cases

**Question Answering**: Multi-hop reasoning
```
Q: Who is older, the president of France or Germany?
→ Search for both → Compare ages → Answer
```

**Data Analysis**: Sequential queries
```
Q: What's the average sales in Q4?
→ Query database → Calculate average → Format result
```

**Task Planning**: Breaking down complex tasks
```
Q: Book a flight to Paris
→ Search flights → Check prices → Select option → Book
```

**Research**: Information gathering
```
Q: Summarize recent AI developments
→ Search papers → Extract key points → Synthesize
```

### Best Practices

**Clear Thoughts**: Make reasoning explicit and logical

**Specific Actions**: Use precise action names and parameters

**Validate Observations**: Check tool outputs for errors

**Limit Iterations**: Set maximum steps to prevent loops

**Handle Errors**: Gracefully manage tool failures

**Log Traces**: Record full thought-action-observation sequences

### Why This Matters

ReAct is one of the most effective agent patterns because it combines the reasoning capabilities of LLMs with the ability to take actions in the world. It's the foundation for many production agent systems and frameworks like LangChain and AutoGPT.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **Basic ReAct Loop**: Implement thought-action-observation cycle
2. **Action Parser**: Extract actions from LLM output
3. **Tool Executor**: Execute different action types
4. **Trace Logger**: Record and display ReAct traces
5. **Complete Agent**: Build full ReAct agent with tools

## Resources

- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain ReAct](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [ReAct Prompting Guide](https://www.promptingguide.ai/techniques/react)
- [HotPotQA Dataset](https://hotpotqa.github.io/)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 87: Tool Use & Function Calling

Tomorrow you'll learn about tool use and function calling, enabling agents to interact with external systems and APIs.
