"""
Day 76: Local LLMs with Ollama - Solutions

NOTE: These solutions require Ollama to be installed and running.
Install: https://ollama.com/download
Pull models: ollama pull mistral && ollama pull phi
"""

import ollama
from typing import List, Dict, Optional, Tuple
import time
import json
from datetime import datetime
import numpy as np


# Exercise 1: Model Manager
class ModelManager:
    TASK_MODELS = {'simple': 'phi', 'general': 'mistral', 'code': 'codellama', 'creative': 'llama2'}
    
    def list_models(self) -> List[Dict]:
        try:
            models = ollama.list()
            return [{'name': m['name'], 'size_gb': m['size'] / 1e9, 'modified': m.get('modified_at', 'Unknown')} 
                    for m in models.get('models', [])]
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def is_installed(self, model_name: str) -> bool:
        return any(model_name in m['name'] for m in self.list_models())
    
    def recommend_model(self, task_type: str) -> str:
        return self.TASK_MODELS.get(task_type, 'mistral')
    
    def estimate_memory(self, model_name: str) -> str:
        if 'phi' in model_name or '2b' in model_name: return "~4GB RAM"
        elif '7b' in model_name or 'mistral' in model_name: return "~8GB RAM"
        elif '13b' in model_name: return "~16GB RAM"
        else: return "~8GB+ RAM"


def exercise_1_model_manager():
    print("Exercise 1: Model Manager\n" + "-" * 40)
    manager = ModelManager()
    
    print("\nInstalled Models:")
    for model in manager.list_models():
        print(f"  {model['name']}: {model['size_gb']:.2f} GB")
    
    print("\nModel Availability:")
    for model in ['mistral', 'phi', 'llama2']:
        print(f"  {model}: {'✓' if manager.is_installed(model) else '✗'}")
    
    print("\nRecommendations:")
    for task in ['simple', 'general', 'code', 'creative']:
        print(f"  {task}: {manager.recommend_model(task)} ({manager.estimate_memory(manager.recommend_model(task))})")


# Exercise 2: Smart Q&A System
class SmartQA:
    """Smart Q&A system with model selection."""
    
    def __init__(self):
        self.manager = ModelManager()
    
    def analyze_complexity(self, question: str) -> str:
        """Analyze question complexity."""
        # Simple heuristics
        question_lower = question.lower()
        
        # Factual indicators
        factual_words = ['what', 'when', 'where', 'who', 'define']
        # Complex indicators
        complex_words = ['why', 'how', 'explain', 'compare', 'analyze']
        
        if any(word in question_lower for word in complex_words):
            return 'complex'
        elif any(word in question_lower for word in factual_words):
            return 'simple'
        else:
            return 'general'
    
    def select_model(self, complexity: str) -> str:
        """Select model based on complexity."""
        if complexity == 'simple':
            return 'phi' if self.manager.is_installed('phi') else 'mistral'
        else:
            return 'mistral'
    
    def ask(self, question: str) -> Dict:
        """Ask question and return answer with metadata."""
        start_time = time.time()
        
        # Analyze and select model
        complexity = self.analyze_complexity(question)
        model = self.select_model(complexity)
        
        # Determine temperature
        temperature = 0.3 if complexity == 'simple' else 0.7
        
        # Generate answer
        try:
            response = ollama.generate(
                model=model,
                prompt=f"Question: {question}\nAnswer:",
                options={'temperature': temperature}
            )
            
            answer = response['response'].strip()
            elapsed = time.time() - start_time
            
            return {
                'answer': answer,
                'model': model,
                'complexity': complexity,
                'temperature': temperature,
                'time_seconds': round(elapsed, 2)
            }
        except Exception as e:
            return {
                'answer': f"Error: {e}",
                'model': model,
                'complexity': complexity,
                'error': str(e)
            }


def exercise_2_smart_qa():
    """Exercise 2: Smart Q&A System"""
    print("\nExercise 2: Smart Q&A System")
    print("-" * 40)
    
    qa = SmartQA()
    
    questions = [
        "What is the capital of France?",
        "Why is the sky blue?",
        "How does machine learning work?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = qa.ask(question)
        print(f"A: {result['answer'][:100]}...")
        print(f"Model: {result['model']} | "
              f"Complexity: {result['complexity']} | "
              f"Time: {result['time_seconds']}s")


# Exercise 3: Document Processor
class DocumentProcessor:
    def __init__(self, model: str = 'mistral'):
        self.model = model
    
    def _generate(self, prompt: str) -> str:
        try:
            return ollama.generate(model=self.model, prompt=prompt, options={'temperature': 0.3})['response'].strip()
        except Exception as e:
            return f"Error: {e}"
    
    def summarize(self, text: str, max_words: int = 100) -> str:
        return self._generate(f"Summarize in {max_words} words:\n{text}\nSummary:")
    
    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        response = self._generate(f"Extract {num_points} key points:\n{text}\nKey Points:")
        return [p.strip('- •*').strip() for p in response.split('\n') if p.strip()][:num_points]
    
    def answer_question(self, text: str, question: str) -> str:
        return self._generate(f"Based on:\n{text}\nQuestion: {question}\nAnswer:")


def exercise_3_document_processor():
    print("\nExercise 3: Document Processor\n" + "-" * 40)
    processor = DocumentProcessor()
    
    document = """Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly programmed."""
    
    print(f"\nDocument: {document}")
    print(f"\nSummary: {processor.summarize(document, max_words=20)}")
    print("\nKey Points:")
    for i, point in enumerate(processor.extract_key_points(document, num_points=2), 1):
        print(f"{i}. {point}")
    print(f"\nQ&A: {processor.answer_question(document, 'What is machine learning?')}")


# Exercise 4: Conversational Memory
class ChatBot:
    def __init__(self, model: str = 'mistral', system_prompt: Optional[str] = None, max_context_tokens: int = 2048):
        self.model, self.max_context_tokens, self.messages = model, max_context_tokens, []
        if system_prompt:
            self.messages.append({'role': 'system', 'content': system_prompt})
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def truncate_context(self) -> None:
        total_tokens = sum(self.estimate_tokens(m['content']) for m in self.messages)
        if total_tokens > self.max_context_tokens:
            system_msgs = [m for m in self.messages if m['role'] == 'system']
            other_msgs = [m for m in self.messages if m['role'] != 'system']
            kept_msgs, tokens = [], 0
            for msg in reversed(other_msgs):
                msg_tokens = self.estimate_tokens(msg['content'])
                if tokens + msg_tokens < self.max_context_tokens:
                    kept_msgs.insert(0, msg)
                    tokens += msg_tokens
                else:
                    break
            self.messages = system_msgs + kept_msgs
    
    def chat(self, user_message: str) -> str:
        self.messages.append({'role': 'user', 'content': user_message})
        self.truncate_context()
        try:
            response = ollama.chat(model=self.model, messages=self.messages)
            assistant_message = response['message']['content']
            self.messages.append({'role': 'assistant', 'content': assistant_message})
            return assistant_message
        except Exception as e:
            return f"Error: {e}"
    
    def save_conversation(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump({'model': self.model, 'timestamp': datetime.now().isoformat(), 'messages': self.messages}, f, indent=2)
    
    def load_conversation(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.model, self.messages = data['model'], data['messages']


def exercise_4_conversational_memory():
    print("\nExercise 4: Conversational Memory\n" + "-" * 40)
    bot = ChatBot(system_prompt="You are a helpful AI assistant.", max_context_tokens=1000)
    
    for user_msg in ["Hi, I'm learning about ML.", "What are the main types?", "Tell me about supervised learning."]:
        print(f"\nUser: {user_msg}")
        print(f"Bot: {bot.chat(user_msg)[:100]}...")
    
    print(f"\nMessages: {len(bot.messages)}, Tokens: {sum(bot.estimate_tokens(m['content']) for m in bot.messages)}")


# Exercise 5: Semantic Search
class SemanticSearch:
    """Semantic search using embeddings."""
    
    def __init__(self, model: str = 'mistral'):
        self.model = model
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Add document with embedding."""
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            embedding = response['embedding']
            
            self.documents.append({
                'text': text,
                'metadata': metadata or {},
                'index': len(self.documents)
            })
            self.embeddings.append(embedding)
        except Exception as e:
            print(f"Error adding document: {e}")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar documents."""
        if not self.documents:
            return []
        
        try:
            # Get query embedding
            response = ollama.embeddings(model=self.model, prompt=query)
            query_embedding = response['embedding']
            
            # Calculate similarities
            similarities = [
                self.cosine_similarity(query_embedding, emb)
                for emb in self.embeddings
            ]
            
            # Get top-k
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'text': self.documents[idx]['text'],
                    'metadata': self.documents[idx]['metadata'],
                    'similarity': similarities[idx]
                })
            
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return []


def exercise_5_semantic_search():
    """Exercise 5: Semantic Search"""
    print("\nExercise 5: Semantic Search")
    print("-" * 40)
    
    search = SemanticSearch()
    
    # Add documents
    documents = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of AI.",
        "Neural networks are inspired by the human brain.",
        "Data science involves statistics and programming.",
        "Deep learning uses multiple layers of neural networks."
    ]
    
    print("\nAdding documents...")
    for doc in documents:
        search.add_document(doc, metadata={'source': 'tutorial'})
    
    # Search
    query = "What is artificial intelligence?"
    print(f"\nQuery: {query}")
    print("\nTop 3 Results:")
    
    results = search.search(query, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity']:.3f}")
        print(f"   Text: {result['text']}")


if __name__ == "__main__":
    print("Day 76: Local LLMs with Ollama - Solutions\n")
    print("=" * 60)
    
    try:
        exercise_1_model_manager()
        exercise_2_smart_qa()
        exercise_3_document_processor()
        exercise_4_conversational_memory()
        exercise_5_semantic_search()
        
        print("\n" + "=" * 60)
        print("All exercises completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Ollama is installed and running:")
        print("1. Install: https://ollama.com/download")
        print("2. Pull models: ollama pull mistral && ollama pull phi")
        print("3. Run: ollama serve")
