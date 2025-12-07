"""AI Agent Tools

Implements various tools for the AI agent.
"""

from typing import Dict, Any, List
import statistics


class Tool:
    """Base tool class."""
    
    def __init__(self, name: str, description: str, schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.schema = schema
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with parameters."""
        raise NotImplementedError


class CalculatorTool(Tool):
    """Calculator tool for mathematical operations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations",
            schema={
                "parameters": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        )
    
    def execute(self, expression: str) -> Dict[str, Any]:
        """Evaluate mathematical expression."""
        try:
            # Safe evaluation for basic math
            allowed_chars = set("0123456789+-*/()., ")
            if not all(c in allowed_chars for c in expression):
                return {"result": None, "error": "Invalid characters in expression"}
            
            result = eval(expression)
            return {"result": result, "error": None}
        except Exception as e:
            return {"result": None, "error": str(e)}


class SearchTool(Tool):
    """Search tool for web search simulation."""
    
    def __init__(self):
        super().__init__(
            name="search",
            description="Search for information on the web",
            schema={
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        )
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Simulate web search."""
        try:
            # Mock search results
            results = [
                f"Article about {query} - comprehensive guide",
                f"{query} explained - detailed tutorial",
                f"Latest news on {query} - recent updates"
            ]
            return {"results": results, "error": None}
        except Exception as e:
            return {"results": [], "error": str(e)}


class WeatherTool(Tool):
    """Weather tool for weather information."""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get weather information for a location",
            schema={
                "parameters": {
                    "location": {
                        "type": "string",
                        "description": "City or location name"
                    }
                },
                "required": ["location"]
            }
        )
        # Mock weather data
        self.weather_data = {
            "seattle": {"temperature": 72, "condition": "Sunny", "humidity": 45},
            "new york": {"temperature": 68, "condition": "Cloudy", "humidity": 60},
            "san francisco": {"temperature": 65, "condition": "Foggy", "humidity": 70},
            "miami": {"temperature": 85, "condition": "Sunny", "humidity": 80}
        }
    
    def execute(self, location: str) -> Dict[str, Any]:
        """Get weather for location."""
        try:
            location_lower = location.lower()
            weather = self.weather_data.get(
                location_lower,
                {"temperature": 70, "condition": "Clear", "humidity": 50}
            )
            weather["location"] = location
            return {"weather": weather, "error": None}
        except Exception as e:
            return {"weather": None, "error": str(e)}


class DataAnalyzerTool(Tool):
    """Data analyzer tool for statistical analysis."""
    
    def __init__(self):
        super().__init__(
            name="data_analyzer",
            description="Perform statistical analysis on numerical data",
            schema={
                "parameters": {
                    "data": {
                        "type": "array",
                        "description": "List of numbers to analyze"
                    }
                },
                "required": ["data"]
            }
        )
    
    def execute(self, data: List[float]) -> Dict[str, Any]:
        """Analyze numerical data."""
        try:
            if not data:
                return {"analysis": None, "error": "Empty data"}
            
            analysis = {
                "count": len(data),
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "min": min(data),
                "max": max(data)
            }
            
            if len(data) > 1:
                analysis["stdev"] = statistics.stdev(data)
            
            return {"analysis": analysis, "error": None}
        except Exception as e:
            return {"analysis": None, "error": str(e)}


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        self.register(CalculatorTool())
        self.register(SearchTool())
        self.register(WeatherTool())
        self.register(DataAnalyzerTool())
    
    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Tool:
        """Get tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.schema
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool '{name}' not found"}
        return tool.execute(**kwargs)


if __name__ == "__main__":
    print("AI Agent Tools Demo\n" + "=" * 50)
    
    registry = ToolRegistry()
    
    print("\n1. Calculator Tool")
    result = registry.execute_tool("calculator", expression="25 * 4 + 10")
    print(f"   25 * 4 + 10 = {result['result']}")
    
    print("\n2. Search Tool")
    result = registry.execute_tool("search", query="AI agents")
    print(f"   Found {len(result['results'])} results")
    
    print("\n3. Weather Tool")
    result = registry.execute_tool("weather", location="Seattle")
    weather = result['weather']
    print(f"   {weather['location']}: {weather['temperature']}°F, {weather['condition']}")
    
    print("\n4. Data Analyzer Tool")
    result = registry.execute_tool("data_analyzer", data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    analysis = result['analysis']
    print(f"   Mean: {analysis['mean']}, Median: {analysis['median']}")
    
    print("\n" + "=" * 50)
    print("All tools working correctly!")
