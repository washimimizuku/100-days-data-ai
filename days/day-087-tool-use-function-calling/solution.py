"""Day 87: Tool Use & Function Calling - Solutions"""

from typing import List, Dict, Any, Optional, Callable
import json
import re


# Exercise 1: Function Schema
def create_function_schema(name: str, description: str, parameters: Dict) -> Dict:
    """Create a function schema."""
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": [k for k, v in parameters.items() if v.get("required", False)]
        }
    }


def validate_schema(schema: Dict) -> bool:
    """Validate function schema format."""
    required_fields = ["name", "description", "parameters"]
    return all(field in schema for field in required_fields)


# Exercise 2: Parameter Extractor
class ParameterExtractor:
    """Extract function parameters from text."""
    
    def extract(self, text: str, schema: Dict) -> Dict[str, Any]:
        """Extract parameters based on schema."""
        params = {}
        properties = schema.get("properties", {})
        
        for param_name, param_def in properties.items():
            if param_name.lower() in text.lower():
                words = text.split()
                for i, word in enumerate(words):
                    if param_name.lower() in word.lower() and i + 1 < len(words):
                        params[param_name] = words[i + 1].strip('",')
                        break
        
        return params
    
    def validate_parameters(self, params: Dict, schema: Dict) -> bool:
        """Validate parameters against schema."""
        required = schema.get("required", [])
        return all(req in params for req in required)
    
    def fill_defaults(self, params: Dict, schema: Dict) -> Dict:
        """Fill in default values."""
        properties = schema.get("properties", {})
        for param_name, param_def in properties.items():
            if param_name not in params and "default" in param_def:
                params[param_name] = param_def["default"]
        return params


# Exercise 3: Tool Registry
class ToolRegistry:
    """Manage available functions."""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, function: Callable, schema: Dict):
        """Register a new tool."""
        self.tools[name] = {
            "function": function,
            "schema": schema
        }
    
    def get_tool(self, name: str) -> Optional[Dict]:
        """Get tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict]:
        """List all available tools."""
        return [{"name": name, **tool} for name, tool in self.tools.items()]
    
    def get_schemas(self) -> List[Dict]:
        """Get schemas for all tools."""
        return [tool["schema"] for tool in self.tools.values()]


# Exercise 4: Function Executor
class FunctionExecutor:
    """Execute functions safely."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def execute(self, function_name: str, arguments: Dict) -> Any:
        """Execute function with arguments."""
        tool = self.registry.get_tool(function_name)
        if not tool:
            raise ValueError(f"Function {function_name} not found")
        
        function = tool["function"]
        return function(**arguments)
    
    def execute_safe(self, function_name: str, arguments: Dict) -> Dict:
        """Execute with error handling."""
        try:
            result = self.execute(function_name, arguments)
            return {"success": True, "result": result, "error": None}
        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}
    
    def execute_batch(self, calls: List[Dict]) -> List[Dict]:
        """Execute multiple function calls."""
        results = []
        for call in calls:
            result = self.execute_safe(call["name"], call.get("arguments", {}))
            results.append(result)
        return results


# Exercise 5: Complete Agent
class FunctionCallingAgent:
    """Agent with function calling capabilities."""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self.executor = FunctionExecutor(self.registry)
        self.extractor = ParameterExtractor()
    
    def add_tool(self, name: str, function: Callable, schema: Dict):
        """Add a tool to the agent."""
        self.registry.register(name, function, schema)
    
    def parse_function_call(self, text: str) -> Optional[Dict]:
        """Parse function call from text."""
        # Try JSON format first
        try:
            data = json.loads(text)
            if "name" in data and "arguments" in data:
                return data
        except:
            pass
        
        # Try function call format: function_name(arg="value")
        pattern = r'(\w+)\((.*?)\)'
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            arguments = {}
            
            # Parse arguments
            arg_pattern = r'(\w+)=["\'](.*?)["\']'
            for arg_match in re.finditer(arg_pattern, args_str):
                arguments[arg_match.group(1)] = arg_match.group(2)
            
            return {"name": name, "arguments": arguments}
        
        return None
    
    def run(self, query: str) -> str:
        """Run agent on query."""
        # Simple rule-based function detection
        if "calculate" in query.lower() or "compute" in query.lower():
            return "Use calculator function"
        elif "search" in query.lower() or "find" in query.lower():
            return "Use search function"
        elif "weather" in query.lower():
            return "Use weather function"
        return "No function needed"
    
    def run_with_llm(self, query: str, llm_response: str) -> Dict:
        """Run with LLM-generated function call."""
        function_call = self.parse_function_call(llm_response)
        
        if not function_call:
            return {"success": False, "error": "Could not parse function call"}
        
        result = self.executor.execute_safe(
            function_call["name"],
            function_call.get("arguments", {})
        )
        
        return result


# Bonus: Tool Definitions
def create_calculator_tool() -> Dict:
    """Create calculator tool definition."""
    return {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }


def create_search_tool() -> Dict:
    """Create search tool definition."""
    return {
        "name": "search",
        "description": "Search for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }


def create_weather_tool() -> Dict:
    """Create weather tool definition."""
    return {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    }


# Example tool functions
def calculator(expression: str) -> float:
    """Calculate mathematical expression."""
    try:
        allowed = set('0123456789+-*/().')
        if all(c in allowed or c.isspace() for c in expression):
            return eval(expression, {"__builtins__": {}}, {})
    except:
        pass
    return 0.0


def search(query: str) -> str:
    """Search for information."""
    knowledge = {
        "python": "Python is a programming language",
        "ai": "AI stands for Artificial Intelligence",
        "weather": "Weather information service"
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return f"No results for: {query}"


def get_weather(location: str, unit: str = "celsius") -> Dict:
    """Get weather for location."""
    return {
        "location": location,
        "temperature": 72 if unit == "fahrenheit" else 22,
        "unit": unit,
        "condition": "sunny"
    }


def demo_function_calling():
    """Demonstrate function calling."""
    print("Day 87: Tool Use & Function Calling - Solutions Demo\n" + "=" * 60)
    
    print("\n1. Function Schema")
    schema = create_function_schema(
        "test_func",
        "Test function",
        {"param1": {"type": "string", "required": True}}
    )
    print(f"   Schema: {schema['name']}")
    print(f"   Valid: {validate_schema(schema)}")
    
    print("\n2. Parameter Extractor")
    extractor = ParameterExtractor()
    params = extractor.extract(
        "location Paris unit celsius",
        {"properties": {"location": {"type": "string"}, "unit": {"type": "string"}}}
    )
    print(f"   Extracted: {params}")
    
    print("\n3. Tool Registry")
    registry = ToolRegistry()
    registry.register("calculator", calculator, create_calculator_tool())
    registry.register("search", search, create_search_tool())
    print(f"   Registered tools: {len(registry.list_tools())}")
    
    print("\n4. Function Executor")
    executor = FunctionExecutor(registry)
    result = executor.execute_safe("calculator", {"expression": "10 + 5"})
    print(f"   Execution result: {result}")
    
    print("\n5. Function Calling Agent")
    agent = FunctionCallingAgent()
    agent.add_tool("calculator", calculator, create_calculator_tool())
    agent.add_tool("search", search, create_search_tool())
    agent.add_tool("get_weather", get_weather, create_weather_tool())
    
    # Test parsing
    call = agent.parse_function_call('calculator(expression="2+2")')
    print(f"   Parsed call: {call}")
    
    # Test execution
    llm_response = '{"name": "calculator", "arguments": {"expression": "15 * 3"}}'
    result = agent.run_with_llm("Calculate 15 * 3", llm_response)
    print(f"   Agent result: {result}")
    
    print("\n" + "=" * 60)
    print("All function calling concepts demonstrated!")


if __name__ == "__main__":
    demo_function_calling()
