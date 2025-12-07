"""
Prompt Engineering Toolkit - Main Implementation

Combines few-shot learning, chain of thought, and template system
for production-ready NLP tasks.
"""

import ollama
import time
from typing import List, Dict, Optional, Any
from examples import TEMPLATES, TASK_CONFIGS, SENTIMENT_EXAMPLES, COT_MATH_EXAMPLES


class FewShotManager:
    def __init__(self):
        self.examples = {}
    
    def add_example(self, task: str, input_text: str, output: str, reasoning: Optional[str] = None, metadata: Optional[Dict] = None):
        if task not in self.examples:
            self.examples[task] = []
        self.examples[task].append({"input": input_text, "output": output, "reasoning": reasoning, "metadata": metadata or {}})
    
    def get_examples(self, task: str, k: int = 3) -> List[Dict]:
        return self.examples.get(task, [])[:k]
    
    def format_examples(self, examples: List[Dict], format_template: str) -> str:
        return "\n\n".join([format_template.format(**ex) for ex in examples])


class ChainOfThoughtEngine:
    def __init__(self, llm_backend):
        self.llm = llm_backend
    
    def zero_shot_cot(self, problem: str, model: str = "mistral") -> Dict:
        prompt = f"{problem}\n\nLet's think step by step:"
        response = self.llm.generate(prompt, model=model, temperature=0.7)
        text = response["response"]
        lines = text.strip().split("\n")
        return {"reasoning": text, "answer": lines[-1] if lines else text, "method": "zero-shot-cot"}
    
    def few_shot_cot(self, problem: str, examples: List[Dict], model: str = "mistral") -> Dict:
        prompt_parts = [f"Problem: {ex['problem']}\nReasoning: {ex['reasoning']}\nAnswer: {ex['answer']}\n" for ex in examples]
        prompt_parts.append(f"Problem: {problem}\nLet's solve this step by step:")
        prompt = "\n".join(prompt_parts)
        response = self.llm.generate(prompt, model=model, temperature=0.7)
        text = response["response"]
        answer = text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip().split("\n")[-1]
        return {"reasoning": text, "answer": answer, "method": "few-shot-cot"}
    
    def self_consistency(self, problem: str, n: int = 3, model: str = "mistral") -> Dict:
        from collections import Counter
        answers, reasonings = [], []
        for _ in range(n):
            result = self.zero_shot_cot(problem, model)
            answers.append(result["answer"])
            reasonings.append(result["reasoning"])
        vote_counts = Counter(answers)
        return {"answer": vote_counts.most_common(1)[0][0], "all_answers": answers, 
                "all_reasonings": reasonings, "votes": dict(vote_counts), "method": "self-consistency"}


class TemplateSystem:
    def __init__(self):
        self.templates = TEMPLATES.copy()
    
    def register_template(self, name: str, template_dict: Dict):
        self.templates[name] = template_dict
    
    def get_template(self, name: str) -> Dict:
        return self.templates.get(name, {})
    
    def render(self, template_name: str, **kwargs) -> str:
        template = self.get_template(template_name)
        return template["template"].format(**kwargs) if template else ""
    
    def compose(self, template_name: str, examples: Optional[List[Dict]] = None, **kwargs) -> str:
        template = self.get_template(template_name)
        if not template:
            return ""
        parts = []
        if "system" in template:
            parts.append(f"System: {template['system']}\n")
        if examples and template.get("example_format"):
            parts.append("Examples:\n")
            for ex in examples:
                parts.append(template["example_format"].format(**ex) + "\n")
            parts.append("")
        parts.append(template["template"].format(**kwargs))
        return "\n".join(parts)


class LLMBackend:
    def __init__(self, default_model: str = "mistral"):
        self.default_model = default_model
        self.task_models = {"simple": "phi", "complex": "mistral"}
    
    def generate(self, prompt: str, model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 200) -> Dict:
        model = model or self.default_model
        try:
            response = ollama.generate(model=model, prompt=prompt, options={"temperature": temperature, "num_predict": max_tokens})
            return {"response": response["response"], "model": model, "success": True}
        except Exception as e:
            return {"response": "", "model": model, "success": False, "error": str(e)}
    
    def select_model(self, task_type: str, complexity: str = "complex") -> str:
        return self.task_models.get(complexity, self.default_model)


class PromptToolkit:
    """Main prompt engineering toolkit."""
    
    def __init__(self, model: str = "mistral"):
        self.few_shot = FewShotManager()
        self.templates = TemplateSystem()
        self.llm = LLMBackend(default_model=model)
        self.cot = ChainOfThoughtEngine(self.llm)
        
        # Load default examples
        self._load_default_examples()
    
    def _load_default_examples(self):
        """Load default examples."""
        for ex in SENTIMENT_EXAMPLES:
            self.few_shot.add_example("sentiment", ex["input"], ex["output"], ex.get("reasoning"))
    
    def classify(self, text: str, task: str = "sentiment", 
                labels: Optional[List[str]] = None, use_few_shot: bool = True,
                num_examples: int = 3) -> Dict:
        """Classify text."""
        start_time = time.time()
        
        # Get task config
        config = TASK_CONFIGS.get(task, {})
        labels = labels or config.get("labels", [])
        template_name = config.get("template", "classification")
        temperature = config.get("temperature", 0.3)
        
        # Get examples if few-shot
        examples = None
        if use_few_shot:
            examples = self.few_shot.get_examples(task, k=num_examples)
        
        # Build prompt
        prompt = self.templates.compose(
            template_name,
            examples=examples,
            text=text,
            labels=", ".join(labels)
        )
        
        # Generate
        response = self.llm.generate(prompt, temperature=temperature)
        
        if not response["success"]:
            return {"success": False, "error": response.get("error")}
        
        # Parse label
        label_text = response["response"].strip().lower()
        label = None
        for l in labels:
            if l.lower() in label_text:
                label = l
                break
        
        return {
            "success": True,
            "result": {
                "label": label or label_text.split()[0],
                "confidence": 0.9 if label else 0.5,
            },
            "metadata": {
                "model": response["model"],
                "temperature": temperature,
                "time_seconds": round(time.time() - start_time, 2)
            }
        }
    
    def extract(self, text: str, entity_types: List[str]) -> Dict:
        """Extract entities from text."""
        start_time = time.time()
        
        prompt = self.templates.compose(
            "extraction",
            text=text,
            entity_types=", ".join(entity_types)
        )
        
        response = self.llm.generate(prompt, temperature=0.2)
        
        if not response["success"]:
            return {"success": False, "error": response.get("error")}
        
        # Parse entities (simple parsing)
        entities = []
        lines = response["response"].strip().split("\n")
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    entity_type = parts[0].strip("- ").strip()
                    entity_text = parts[1].strip()
                    entities.append({"type": entity_type, "text": entity_text})
        
        return {
            "success": True,
            "result": {"entities": entities},
            "metadata": {
                "model": response["model"],
                "time_seconds": round(time.time() - start_time, 2)
            }
        }
    
    def generate(self, text: str, task: str = "summarize", 
                max_length: int = 100) -> Dict:
        """Generate text (summarize, rewrite, etc)."""
        start_time = time.time()
        
        prompt = self.templates.compose(
            "summarization",
            text=text,
            max_length=max_length
        )
        
        response = self.llm.generate(prompt, temperature=0.5, max_tokens=max_length * 2)
        
        if not response["success"]:
            return {"success": False, "error": response.get("error")}
        
        generated = response["response"].strip()
        
        return {
            "success": True,
            "result": {
                "generated_text": generated,
                "length": len(generated.split())
            },
            "metadata": {
                "model": response["model"],
                "time_seconds": round(time.time() - start_time, 2)
            }
        }
    
    def reason(self, problem: str, use_cot: bool = True, 
              method: str = "zero-shot", n_samples: int = 3) -> Dict:
        """Solve problem with reasoning."""
        start_time = time.time()
        
        if not use_cot:
            # Direct answer
            response = self.llm.generate(problem, temperature=0.3)
            return {
                "success": response["success"],
                "result": {
                    "answer": response["response"],
                    "reasoning": None,
                    "method": "direct"
                },
                "metadata": {
                    "model": response["model"],
                    "time_seconds": round(time.time() - start_time, 2)
                }
            }
        
        # Chain of thought
        if method == "zero-shot":
            result = self.cot.zero_shot_cot(problem)
        elif method == "few-shot":
            result = self.cot.few_shot_cot(problem, COT_MATH_EXAMPLES)
        elif method == "self-consistency":
            result = self.cot.self_consistency(problem, n=n_samples)
        else:
            result = self.cot.zero_shot_cot(problem)
        
        return {
            "success": True,
            "result": result,
            "metadata": {
                "model": "mistral",
                "time_seconds": round(time.time() - start_time, 2)
            }
        }


if __name__ == "__main__":
    print("Prompt Engineering Toolkit\n")
    print("=" * 60)
    
    try:
        toolkit = PromptToolkit()
        
        # Test classification
        print("\n1. Sentiment Classification")
        result = toolkit.classify(
            text="This product exceeded my expectations!",
            task="sentiment",
            use_few_shot=True
        )
        if result["success"]:
            print(f"Label: {result['result']['label']}")
            print(f"Confidence: {result['result']['confidence']}")
        
        # Test extraction
        print("\n2. Entity Extraction")
        result = toolkit.extract(
            text="Apple Inc. announced iPhone 15 in Cupertino.",
            entity_types=["organization", "product", "location"]
        )
        if result["success"]:
            for entity in result['result']['entities']:
                print(f"  {entity['type']}: {entity['text']}")
        
        # Test generation
        print("\n3. Text Summarization")
        result = toolkit.generate(
            text="Machine learning is a subset of AI that enables systems to learn from data.",
            task="summarize",
            max_length=20
        )
        if result["success"]:
            print(f"Summary: {result['result']['generated_text']}")
        
        # Test reasoning
        print("\n4. Chain of Thought Reasoning")
        result = toolkit.reason(
            problem="If 3 books cost $15, how much do 5 books cost?",
            use_cot=True,
            method="zero-shot"
        )
        if result["success"]:
            print(f"Answer: {result['result']['answer']}")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Ollama is running:")
        print("1. Install: https://ollama.com/download")
        print("2. Pull model: ollama pull mistral")
        print("3. Start: ollama serve")
