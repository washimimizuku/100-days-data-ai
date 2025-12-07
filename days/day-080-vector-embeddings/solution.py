"""Day 80: Vector Embeddings - Solutions"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import pickle
import os


# Exercise 1: Embedding Generator
class EmbeddingGenerator:
    """Generate and cache embeddings efficiently."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = '.cache'):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def generate(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)
        return embeddings
    
    def batch_generate(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_emb = self.generate(batch, normalize=normalize)
            all_embeddings.append(batch_emb)
        return np.vstack(all_embeddings)
    
    def generate_with_cache(self, texts: List[str], cache_key: str) -> np.ndarray:
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    if cached['texts'] == texts:
                        print(f"Loaded from cache: {cache_key}")
                        return cached['embeddings']
            except:
                pass
        print(f"Generating embeddings: {cache_key}")
        embeddings = self.generate(texts)
        with open(cache_file, 'wb') as f:
            pickle.dump({'texts': texts, 'embeddings': embeddings}, f)
        return embeddings


def exercise_1_embedding_generator():
    print("Exercise 1: Embedding Generator\n" + "-" * 40)
    generator = EmbeddingGenerator()
    texts = ["Machine learning is a subset of AI", "Deep learning uses neural networks", "Python is a programming language"]
    embeddings = generator.generate(texts)
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    large_texts = [f"Document {i}" for i in range(100)]
    batch_embeddings = generator.batch_generate(large_texts, batch_size=32)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    cached_embeddings = generator.generate_with_cache(texts, "test_cache")
    print(f"Cached embeddings shape: {cached_embeddings.shape}")


# Exercise 2: Similarity Calculator
class SimilarityCalculator:
    """Calculate various similarity metrics."""
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two vectors."""
        return np.linalg.norm(a - b)
    
    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """Dot product (for normalized vectors)."""
        return np.dot(a, b)
    
    def compare_all(self, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        """Compare using all metrics."""
        return {
            'cosine_similarity': self.cosine_similarity(a, b),
            'euclidean_distance': self.euclidean_distance(a, b),
            'dot_product': self.dot_product(a, b)
        }
    
    def pairwise_similarities(self, embeddings: np.ndarray, metric: str = 'cosine') -> np.ndarray:
        n = len(embeddings)
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if metric == 'cosine':
                    similarities[i, j] = self.cosine_similarity(embeddings[i], embeddings[j])
                elif metric == 'euclidean':
                    similarities[i, j] = -self.euclidean_distance(embeddings[i], embeddings[j])
                elif metric == 'dot':
                    similarities[i, j] = self.dot_product(embeddings[i], embeddings[j])
        return similarities


def exercise_2_similarity_calculator():
    print("\nExercise 2: Similarity Calculator\n" + "-" * 40)
    generator = EmbeddingGenerator()
    calculator = SimilarityCalculator()
    texts = ["machine learning algorithms", "ML techniques and methods", "pizza recipe ingredients"]
    embeddings = generator.generate(texts)
    print(f"\nSimilar texts:\n  '{texts[0]}' vs '{texts[1]}'")
    metrics = calculator.compare_all(embeddings[0], embeddings[1])
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.3f}")
    print(f"\nDifferent texts:\n  '{texts[0]}' vs '{texts[2]}'")
    metrics = calculator.compare_all(embeddings[0], embeddings[2])
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.3f}")


# Exercise 3: Embedding Evaluator
class EmbeddingEvaluator:
    """Evaluate embedding quality."""
    
    def __init__(self, model):
        self.model = model
        self.calculator = SimilarityCalculator()
    
    def evaluate_pairs(self, test_pairs: List[Tuple[str, str, float]]) -> Dict:
        results = []
        errors = []
        for text1, text2, expected in test_pairs:
            emb1 = self.model.encode(text1)
            emb2 = self.model.encode(text2)
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
            actual = self.calculator.cosine_similarity(emb1, emb2)
            error = abs(expected - actual)
            results.append({'text1': text1, 'text2': text2, 'expected': expected, 'actual': actual, 'error': error})
            errors.append(error)
        return {'results': results, 'mean_error': np.mean(errors), 'max_error': np.max(errors), 'accuracy': sum(1 for e in errors if e < 0.2) / len(errors)}
    
    def find_failures(self, results: Dict, threshold: float = 0.3) -> List[Dict]:
        """Identify failure cases."""
        failures = [r for r in results['results'] if r['error'] > threshold]
        return failures


def exercise_3_embedding_evaluator():
    print("\nExercise 3: Embedding Evaluator\n" + "-" * 40)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    evaluator = EmbeddingEvaluator(model)
    test_pairs = [("dog", "puppy", 0.9), ("dog", "cat", 0.7), ("dog", "car", 0.1), ("king", "queen", 0.8), ("happy", "joyful", 0.9)]
    results = evaluator.evaluate_pairs(test_pairs)
    print(f"\nEvaluation Results:\n  Mean error: {results['mean_error']:.3f}\n  Max error: {results['max_error']:.3f}\n  Accuracy (error < 0.2): {results['accuracy']:.1%}")
    print("\nSample results:")
    for r in results['results'][:3]:
        print(f"  '{r['text1']}' vs '{r['text2']}':\n    Expected: {r['expected']:.2f}, Actual: {r['actual']:.2f}, Error: {r['error']:.2f}")


# Exercise 4: Dimensionality Reducer
class DimensionalityReducer:
    """Reduce embedding dimensions."""
    
    def __init__(self):
        self.pca = None
    
    def reduce(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Reduce dimensions using PCA."""
        self.pca = PCA(n_components=target_dim)
        reduced = self.pca.fit_transform(embeddings)
        return reduced
    
    def analyze_variance(self) -> Dict:
        """Analyze variance retention."""
        if self.pca is None:
            return {}
        
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_),
            'total_variance_retained': self.pca.explained_variance_ratio_.sum()
        }
    
    def compare_quality(self, original: np.ndarray, reduced: np.ndarray) -> Dict:
        calculator = SimilarityCalculator()
        n = min(10, len(original))
        original_sims = []
        reduced_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                orig_sim = calculator.cosine_similarity(original[i], original[j])
                red_sim = calculator.cosine_similarity(reduced[i], reduced[j])
                original_sims.append(orig_sim)
                reduced_sims.append(red_sim)
        correlation = np.corrcoef(original_sims, reduced_sims)[0, 1]
        return {'similarity_correlation': correlation, 'mean_difference': np.mean(np.abs(np.array(original_sims) - np.array(reduced_sims)))}


def exercise_4_dimensionality_reducer():
    print("\nExercise 4: Dimensionality Reducer\n" + "-" * 40)
    generator = EmbeddingGenerator()
    reducer = DimensionalityReducer()
    texts = [f"Document about topic {i}" for i in range(50)]
    embeddings = generator.generate(texts)
    print(f"\nOriginal dimensions: {embeddings.shape[1]}")
    for target_dim in [128, 64, 32]:
        reduced = reducer.reduce(embeddings, target_dim)
        variance = reducer.analyze_variance()
        quality = reducer.compare_quality(embeddings, reduced)
        print(f"\nReduced to {target_dim} dimensions:\n  Variance retained: {variance['total_variance_retained']:.1%}\n  Similarity correlation: {quality['similarity_correlation']:.3f}\n  Mean difference: {quality['mean_difference']:.3f}")


# Exercise 5: Semantic Search Engine
class SemanticSearch:
    """Simple semantic search engine."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.generator = EmbeddingGenerator(model_name)
        self.calculator = SimilarityCalculator()
        self.documents = []
        self.embeddings = None
    
    def index(self, documents: List[str]):
        """Index documents."""
        self.documents = documents
        self.embeddings = self.generator.generate(documents)
        print(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.embeddings is None:
            return []
        query_emb = self.generator.generate([query])[0]
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = self.calculator.cosine_similarity(query_emb, doc_emb)
            similarities.append((i, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        return [{'document': self.documents[idx], 'similarity': sim, 'rank': rank + 1} for rank, (idx, sim) in enumerate(top_results)]


def exercise_5_semantic_search():
    print("\nExercise 5: Semantic Search Engine\n" + "-" * 40)
    search = SemanticSearch()
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Python is a popular programming language for data science",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret images",
        "Reinforcement learning trains agents through rewards",
        "Data preprocessing is crucial for model performance"
    ]
    search.index(documents)
    query = "How do neural networks work?"
    print(f"\nQuery: {query}\n\nTop 3 Results:")
    results = search.search(query, top_k=3)
    for result in results:
        print(f"\n{result['rank']}. Similarity: {result['similarity']:.3f}\n   {result['document']}")


if __name__ == "__main__":
    print("Day 80: Vector Embeddings - Solutions\n" + "=" * 60)
    try:
        exercise_1_embedding_generator()
        exercise_2_similarity_calculator()
        exercise_3_embedding_evaluator()
        exercise_4_dimensionality_reducer()
        exercise_5_semantic_search()
        print("\n" + "=" * 60 + "\nAll exercises completed!")
    except Exception as e:
        print(f"\nError: {e}\n\nMake sure you have installed:\npip install sentence-transformers numpy scikit-learn")
