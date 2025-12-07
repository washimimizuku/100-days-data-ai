"""Day 86: ReAct Pattern - Solutions"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class ReActStep:
    """Represents one step in ReAct trace."""
    thought: str
    action: str
    observation: str


# Exercise 1: Basic ReAct Loop
def basic_react_loop(question: str, max_steps: int = 5) -> List[ReActStep]:
    """Implement basic ReAct loop."""
    steps = []
    current_state = "start"
    
    for i in range(max_steps):
        if "capital" in question.lower():
            if i == 0:
                thought = "I need to search for the capital"
                action = "search(capital)"
                observation = "Found: Paris"
            elif i == 1:
                thought = "I have the answer"
                action = "finish(Paris)"
                observation = "Task complete"
            else:
                break
        else:
            thought = f"Step {i+1}: Analyzing question"
            action = f"action_{i+1}"
            observation = f"result_{i+1}"
        
        steps.append(ReActStep(thought, action, observation))
        
        if "finish" in action:
            break
    
    return steps


# Exercise 2: Action Parser
class ActionParser:
    """Parse actions from LLM output."""
    
    def parse(self, text: str) -> Optional[Dict[str, str]]:
        """Extract action from text."""
        pattern = r'Action:\s*(\w+)\("([^"]*)"\)'
        match = re.search(pattern, text)
        if match:
            return {'type': match.group(1), 'input': match.group(2)}
        
        pattern2 = r'Action:\s*(\w+)\(([^)]*)\)'
        match2 = re.search(pattern2, text)
        if match2:
            return {'type': match2.group(1), 'input': match2.group(2).strip('\'"')}
        
        return None
    
    def extract_thought(self, text: str) -> Optional[str]:
        """Extract thought from text."""
        pattern = r'Thought:\s*(.+?)(?=\n|Action:|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def extract_observation(self, text: str) -> Optional[str]:
        """Extract observation from text."""
        pattern = r'Observation:\s*(.+?)(?=\n|Thought:|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None


# Exercise 3: Tool Executor
class ToolExecutor:
    """Execute different action types."""
    
    def __init__(self):
        self.tools = {
            'search': self.search,
            'calculate': self.calculate,
            'lookup': self.lookup,
            'finish': self.finish
        }
        self.knowledge = {
            'capital france': 'Paris',
            'capital germany': 'Berlin',
            'population paris': '2.2 million',
            'python': 'Python is a programming language',
            'ai': 'AI stands for Artificial Intelligence'
        }
    
    def execute(self, action_type: str, action_input: str) -> str:
        """Execute action and return observation."""
        tool = self.tools.get(action_type)
        if tool:
            return tool(action_input)
        return f"Unknown action: {action_type}"
    
    def search(self, query: str) -> str:
        """Simulate search."""
        query_lower = query.lower()
        for key, value in self.knowledge.items():
            if key in query_lower:
                return value
        return f"No results found for: {query}"
    
    def calculate(self, expression: str) -> str:
        """Perform calculation."""
        try:
            allowed = set('0123456789+-*/().')
            if all(c in allowed or c.isspace() for c in expression):
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
        except:
            pass
        return "Calculation error"
    
    def lookup(self, term: str) -> str:
        """Look up information."""
        return self.search(term)
    
    def finish(self, answer: str) -> str:
        """Return final answer."""
        return answer


# Exercise 4: Trace Logger
class TraceLogger:
    """Log and display ReAct traces."""
    
    def __init__(self):
        self.traces = []
    
    def log_step(self, step_num: int, thought: str, action: str, observation: str):
        """Log a ReAct step."""
        self.traces.append({
            'step': step_num,
            'thought': thought,
            'action': action,
            'observation': observation
        })
    
    def display(self):
        """Display formatted trace."""
        for trace in self.traces:
            print(f"\nStep {trace['step']}:")
            print(f"  Thought: {trace['thought']}")
            print(f"  Action: {trace['action']}")
            print(f"  Observation: {trace['observation']}")
    
    def get_trace(self) -> List[Dict]:
        """Get trace as list of dicts."""
        return self.traces.copy()
    
    def clear(self):
        """Clear trace history."""
        self.traces = []


# Exercise 5: Complete ReAct Agent
class ReActAgent:
    """Complete ReAct agent implementation."""
    
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.parser = ActionParser()
        self.executor = ToolExecutor()
        self.logger = TraceLogger()
    
    def run(self, question: str) -> str:
        """Run ReAct agent on question."""
        history = []
        
        for step in range(1, self.max_steps + 1):
            thought = self.generate_thought(question, history)
            action_text = self.generate_action(thought, question)
            action = self.parser.parse(f"Action: {action_text}")
            
            if not action:
                break
            
            observation = self.executor.execute(action['type'], action['input'])
            self.logger.log_step(step, thought, action_text, observation)
            history.append({'thought': thought, 'action': action, 'observation': observation})
            
            if self.should_finish(step, action['type']):
                return observation
        
        return "Max steps reached"
    
    def generate_thought(self, question: str, history: List[Dict]) -> str:
        """Generate reasoning about next action."""
        if not history:
            return f"I need to answer: {question}"
        
        last_obs = history[-1]['observation']
        if "error" in last_obs.lower() or "not found" in last_obs.lower():
            return "The previous action failed, I should try a different approach"
        
        return f"Based on '{last_obs}', I should proceed to the next step"
    
    def generate_action(self, thought: str, question: str) -> str:
        """Generate action based on thought."""
        if "search" in thought.lower() or "find" in thought.lower():
            return f'search("{question}")'
        elif "calculate" in thought.lower() or "compute" in thought.lower():
            return 'calculate("2+2")'
        elif "answer" in thought.lower() or "proceed" in thought.lower():
            return 'finish("Answer based on observations")'
        return 'search("information")'
    
    def should_finish(self, step_count: int, last_action: Optional[str]) -> bool:
        """Determine if agent should stop."""
        return last_action == 'finish' or step_count >= self.max_steps
    
    def format_prompt(self, question: str, history: List[Dict]) -> str:
        """Format prompt for LLM."""
        prompt = f"Question: {question}\n\n"
        for h in history:
            prompt += f"Thought: {h['thought']}\n"
            prompt += f"Action: {h['action']}\n"
            prompt += f"Observation: {h['observation']}\n\n"
        return prompt


# Bonus: Multi-Step Reasoning
class MultiStepReAct(ReActAgent):
    """ReAct agent for multi-step problems."""
    
    def decompose_question(self, question: str) -> List[str]:
        """Break question into sub-questions."""
        if "and" in question.lower():
            parts = question.split(" and ")
            return [p.strip() for p in parts]
        return [question]
    
    def synthesize_answer(self, sub_answers: List[str]) -> str:
        """Combine sub-answers into final answer."""
        return " and ".join(sub_answers)
    
    def run_multi_step(self, question: str) -> Dict[str, Any]:
        """Run with explicit decomposition."""
        sub_questions = self.decompose_question(question)
        sub_answers = []
        
        for sq in sub_questions:
            answer = self.run(sq)
            sub_answers.append(answer)
        
        final_answer = self.synthesize_answer(sub_answers)
        
        return {
            'question': question,
            'sub_questions': sub_questions,
            'sub_answers': sub_answers,
            'final_answer': final_answer,
            'trace': self.logger.get_trace()
        }


def demo_react_pattern():
    """Demonstrate ReAct pattern."""
    print("Day 86: ReAct Pattern - Solutions Demo\n" + "=" * 60)
    
    print("\n1. Basic ReAct Loop")
    steps = basic_react_loop("What is the capital of France?", max_steps=3)
    for i, step in enumerate(steps, 1):
        print(f"  Step {i}:")
        print(f"    Thought: {step.thought}")
        print(f"    Action: {step.action}")
        print(f"    Observation: {step.observation}")
    
    print("\n2. Action Parser")
    parser = ActionParser()
    test_text = 'Thought: I need to search\nAction: search("Paris")\nObservation: Found'
    action = parser.parse(test_text)
    thought = parser.extract_thought(test_text)
    print(f"  Parsed action: {action}")
    print(f"  Extracted thought: {thought}")
    
    print("\n3. Tool Executor")
    executor = ToolExecutor()
    print(f"  Search result: {executor.execute('search', 'capital france')}")
    print(f"  Calculate result: {executor.execute('calculate', '10 + 5')}")
    
    print("\n4. Trace Logger")
    logger = TraceLogger()
    logger.log_step(1, "Need to search", "search(test)", "Found result")
    logger.log_step(2, "Have answer", "finish(result)", "Complete")
    print("  Trace:")
    logger.display()
    
    print("\n5. Complete ReAct Agent")
    agent = ReActAgent(max_steps=5)
    result = agent.run("What is the capital of France?")
    print(f"  Final answer: {result}")
    print("  Full trace:")
    agent.logger.display()
    
    print("\n6. Multi-Step Reasoning")
    multi_agent = MultiStepReAct(max_steps=3)
    result = multi_agent.run_multi_step("What is Python and AI?")
    print(f"  Sub-questions: {result['sub_questions']}")
    print(f"  Final answer: {result['final_answer']}")
    
    print("\n" + "=" * 60)
    print("All ReAct patterns demonstrated successfully!")


if __name__ == "__main__":
    demo_react_pattern()
