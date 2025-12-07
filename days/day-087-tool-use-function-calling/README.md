# Day 87: Tool Use & Function Calling

## Learning Objectives

**Time**: 1 hour

- Understand tool use and function calling in LLMs
- Implement tool schemas and parameter extraction
- Build agents that can call external functions
- Apply function calling to real-world tasks

## Theory (15 minutes)

### What is Function Calling?

Function calling enables LLMs to interact with external systems by generating structured function calls. Instead of just text, the LLM outputs JSON specifying which function to call and with what parameters.

### How It Works

```
1. Define available functions with schemas
2. LLM receives user query + function definitions
3. LLM decides which function(s) to call
4. LLM generates function call with parameters
5. System executes function
6. LLM receives result and continues
```

### Function Schema

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
  }
}
```

### Function Call Format

```json
{
  "name": "get_weather",
  "arguments": {"location": "Paris", "unit": "celsius"}
}
```

### Tool Types

**Information Retrieval**: search(query), get_document(id)
**Computation**: calculate(expression), convert_units(value, from, to)
**External APIs**: get_weather(location), get_stock_price(symbol)
**Data Operations**: create_record(data), update_record(id, data)

### Parameter Extraction

```
User: "What's the weather in Paris in Celsius?"
→ get_weather(location="Paris", unit="celsius")

User: "Convert 100 USD to EUR"
→ convert_currency(amount=100, from="USD", to="EUR")
```

### Multi-Function Calls

```
User: "Compare weather in Paris and London"
→ get_weather(location="Paris")
→ get_weather(location="London")
→ Compare results
```

### Function Calling Flow

```python
# 1. Define tools
tools = [{"name": "search", "description": "Search", "parameters": {...}}]

# 2. Send to LLM
response = llm.chat(messages=[{"role": "user", "content": query}], tools=tools)

# 3. Check for function call
if response.tool_calls:
    for call in response.tool_calls:
        result = execute_function(call.name, call.arguments)

# 4. Get final response
final = llm.chat(messages + function_results)
```

### Tool Selection

LLM chooses based on:
- Function descriptions
- Parameter types
- User intent
- Context

### Error Handling

**Invalid Parameters**: Validate types and required fields
**Function Not Found**: Check if function exists
**Execution Errors**: Catch and return error messages

### Best Practices

**Clear Descriptions**: Help LLM understand when to use each tool
**Type Validation**: Ensure parameters match schema
**Result Formatting**: Return structured, parseable results
**Limit Tool Count**: Keep under 20 tools
**Composability**: Design tools that work together

### OpenAI Function Calling

```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    functions=[{
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }]
)
```

### Local LLM Function Calling

```python
prompt = f"""Available functions: {json.dumps(functions)}
User query: {query}
Call the appropriate function in JSON format."""

response = ollama.generate(model="mistral", prompt=prompt)
function_call = json.loads(response)
```

### Use Cases

**Customer Support**: Query knowledge base, create tickets
**Data Analysis**: Query databases, generate reports
**Task Automation**: Send emails, schedule meetings
**Information Gathering**: Search web, fetch data
**Calculations**: Math, conversions, statistics

### Advantages

**Structured Output**: Reliable, parseable results
**Type Safety**: Parameter validation
**Composability**: Chain multiple functions
**Extensibility**: Easy to add new tools
**Debugging**: Clear function call traces

### Challenges

**Schema Design**: Creating clear schemas
**Parameter Extraction**: LLM may misinterpret
**Error Recovery**: Handling failed calls
**Cost**: Additional tokens
**Latency**: Multiple round trips

### Why This Matters

Function calling transforms LLMs from text generators into action-taking agents. It's the foundation for AI systems that interact with the real world through APIs, databases, and external tools.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **Function Schema**: Define tool schemas
2. **Parameter Extractor**: Extract parameters from text
3. **Tool Registry**: Manage available functions
4. **Function Executor**: Execute functions safely
5. **Complete Agent**: Build agent with function calling

## Resources

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
- [JSON Schema](https://json-schema.org/)
- [Function Calling Best Practices](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 88: LangGraph Basics

Tomorrow you'll learn about LangGraph, a framework for building stateful, multi-actor applications with LLMs.
