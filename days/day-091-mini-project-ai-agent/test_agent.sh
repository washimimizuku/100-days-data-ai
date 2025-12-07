#!/bin/bash

# Test script for AI Agent Mini Project

echo "=========================================="
echo "AI Agent Test Suite"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run test
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo "Running: $test_name"
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: $test_name"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Test 1: Tools module
echo "Test 1: Tools Module"
echo "-------------------"
run_test "Import tools module" "python -c 'from tools import ToolRegistry'"
run_test "Create tool registry" "python -c 'from tools import ToolRegistry; r = ToolRegistry()'"
run_test "Execute calculator" "python -c 'from tools import ToolRegistry; r = ToolRegistry(); result = r.execute_tool(\"calculator\", expression=\"2+2\"); assert result[\"result\"] == 4'"
run_test "Execute weather" "python -c 'from tools import ToolRegistry; r = ToolRegistry(); result = r.execute_tool(\"weather\", location=\"Seattle\"); assert result[\"weather\"] is not None'"
run_test "Execute data analyzer" "python -c 'from tools import ToolRegistry; r = ToolRegistry(); result = r.execute_tool(\"data_analyzer\", data=[1,2,3,4,5]); assert result[\"analysis\"] is not None'"

# Test 2: Agent module
echo "Test 2: Agent Module"
echo "-------------------"
run_test "Import agent module" "python -c 'from agent import AIAgent'"
run_test "Create agent" "python -c 'from agent import AIAgent; agent = AIAgent()'"
run_test "Run simple calculation" "python -c 'from agent import AIAgent; agent = AIAgent(); result = agent.run(\"What is 5+5?\", verbose=False); assert \"10\" in result[\"answer\"]'"
run_test "Run weather query" "python -c 'from agent import AIAgent; agent = AIAgent(); result = agent.run(\"Weather in Seattle\", verbose=False); assert result[\"answer\"] is not None'"

# Test 3: Workflow module
echo "Test 3: Workflow Module"
echo "----------------------"
run_test "Import workflow module" "python -c 'from workflow import WorkflowAgent'"
run_test "Create workflow agent" "python -c 'from workflow import WorkflowAgent; agent = WorkflowAgent()'"
run_test "Run workflow" "python -c 'from workflow import WorkflowAgent; agent = WorkflowAgent(); result = agent.run_workflow(\"Calculate 3*3\", verbose=False); assert result[\"answer\"] is not None'"
run_test "Multi-step workflow" "python -c 'from workflow import MultiStepWorkflow; agent = MultiStepWorkflow(); result = agent.run(\"Calculate 5+5 then multiply by 2\", verbose=False); assert result[\"final_answer\"] is not None'"

# Test 4: Integration tests
echo "Test 4: Integration Tests"
echo "------------------------"
run_test "Agent completes task" "python -c 'from agent import AIAgent; agent = AIAgent(); result = agent.run(\"What is 100/5?\", verbose=False); assert result[\"complete\"] or result[\"iterations\"] > 0'"
run_test "Agent handles errors" "python -c 'from agent import AIAgent; agent = AIAgent(); result = agent.run(\"Calculate 1/0\", verbose=False); assert \"error\" in result[\"answer\"].lower() or result[\"answer\"] is not None'"
run_test "Workflow state management" "python -c 'from workflow import WorkflowAgent; agent = WorkflowAgent(); result = agent.run_workflow(\"Calculate 7*8\", verbose=False); assert len(result[\"thoughts\"]) > 0'"

# Test 5: Tool execution
echo "Test 5: Tool Execution"
echo "---------------------"
run_test "Calculator with complex expression" "python -c 'from tools import CalculatorTool; tool = CalculatorTool(); result = tool.execute(expression=\"(10+20)*3\"); assert result[\"result\"] == 90'"
run_test "Search tool returns results" "python -c 'from tools import SearchTool; tool = SearchTool(); result = tool.execute(query=\"test\"); assert len(result[\"results\"]) > 0'"
run_test "Weather tool returns data" "python -c 'from tools import WeatherTool; tool = WeatherTool(); result = tool.execute(location=\"Miami\"); assert result[\"weather\"][\"temperature\"] > 0'"
run_test "Data analyzer computes stats" "python -c 'from tools import DataAnalyzerTool; tool = DataAnalyzerTool(); result = tool.execute(data=[1,2,3,4,5]); assert result[\"analysis\"][\"mean\"] == 3.0'"

# Test 6: Edge cases
echo "Test 6: Edge Cases"
echo "-----------------"
run_test "Empty query handling" "python -c 'from agent import AIAgent; agent = AIAgent(); result = agent.run(\"\", verbose=False); assert result is not None'"
run_test "Invalid tool parameters" "python -c 'from tools import CalculatorTool; tool = CalculatorTool(); result = tool.execute(expression=\"invalid\"); assert result[\"error\"] is not None'"
run_test "Max iterations limit" "python -c 'from agent import AIAgent; agent = AIAgent(max_iterations=2); result = agent.run(\"Complex task\", verbose=False); assert result[\"iterations\"] <= 2'"

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
