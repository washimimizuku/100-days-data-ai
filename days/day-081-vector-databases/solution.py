"""Day 81: Vector Databases - Solutions"""

from typing import List, Dict, Tuple
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
import time


# Exercise 1: FAISS Index
class FAISSIndexManager:
    """Manage different FAISS index types."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.indexes = {}
    
    def create_flat_index(self) -> faiss.Index:
        """Create flat (brute force) index."""
        index = faiss.IndexFlatL2(self.dimension)
        self.indexes['flat'] = index
        return index
    
    def create_ivf_index(self, nlist: int = 100) -> faiss.Index:
        """Create IVF index."""
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        self.indexes['ivf'] = index
        return index
    
    def create_hnsw_index(self, M: int = 32) -> faiss.Index:
        """Create HNSW index."""
        index = faiss.IndexHNSWFlat(self.dimension, M)
        self.indexes['hnsw'] = index
        return index
    
    def add_vectors(self, index_name: str, vectors: np.ndarray):
        index = self.indexes[index_name]
        if index_name == 'ivf' and not index.is_trained:
            print(f"Training {index_name} index...")
            index.train(vectors)
        index.add(vectors)
        print(f"Added {len(vectors)} vectors to {index_name} index")
    
    def search(self, index_name: str, query: np.ndarray, k: int = 5) -> Tuple:
        index = self.indexes[index_name]
        distances, indices = index.search(query, k)
        return distances, indices


def exercise_1_faiss_index():
    print("Exercise 1: FAISS Index\n" + "-" * 40)
    dimension = 128
    n_vectors = 10000
    vectors = np.random.random((n_vectors, dimension)).astype('float32')
    query = np.random.random((1, dimension)).astype('float32')
    manager = FAISSIndexManager(dimension)
    
    for index_type in ['flat', 'ivf', 'hnsw']:
        print(f"\n{index_type.upper()} Index:")
        if index_type == 'flat':
            manager.create_flat_index()
        elif index_type == 'ivf':
            manager.create_ivf_index(nlist=100)
        else:
            manager.create_hnsw_index(M=32)
        
        start = time.time()
        manager.add_vectors(index_type, vectors)
        add_time = time.time() - start
        start = time.time()
        distances, indices = manager.search(index_type, query, k=5)
        search_time = time.time() - start
        print(f"  Add time: {add_time:.3f}s\n  Search time: {search_time*1000:.2f}ms\n  Top result distance: {distances[0][0]:.3f}")


# Exercise 2: Chroma Database
class ChromaManager:
    """Manage Chroma database operations."""
    
    def __init__(self):
        self.client = chromadb.Client()
        self.collections = {}
    
    def create_collection(self, name: str):
        """Create or get collection."""
        try:
            collection = self.client.create_collection(name)
        except:
            collection = self.client.get_collection(name)
        
        self.collections[name] = collection
        return collection
    
    def insert_documents(self, collection_name: str, documents: List[str], metadatas: List[Dict], ids: List[str]):
        collection = self.collections[collection_name]
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Inserted {len(documents)} documents")
    
    def query(self, collection_name: str, query_text: str, n_results: int = 5, where: Dict = None) -> Dict:
        collection = self.collections[collection_name]
        results = collection.query(query_texts=[query_text], n_results=n_results, where=where)
        return results
    
    def update(self, collection_name: str, ids: List[str], documents: List[str] = None, metadatas: List[Dict] = None):
        collection = self.collections[collection_name]
        collection.update(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Updated {len(ids)} documents")
    
    def delete(self, collection_name: str, ids: List[str] = None, where: Dict = None):
        collection = self.collections[collection_name]
        collection.delete(ids=ids, where=where)
        print(f"Deleted documents")


def exercise_2_chroma_database():
    print("\nExercise 2: Chroma Database\n" + "-" * 40)
    manager = ChromaManager()
    manager.create_collection("test_docs")
    
    documents = ["Machine learning is a subset of AI", "Deep learning uses neural networks", "Python is a programming language", "Data science involves statistics"]
    metadatas = [{"category": "ml", "year": 2023}, {"category": "ml", "year": 2024}, {"category": "programming", "year": 2023}, {"category": "data", "year": 2024}]
    ids = [f"doc{i}" for i in range(len(documents))]
    manager.insert_documents("test_docs", documents, metadatas, ids)
    
    print("\nQuery: 'What is ML?'")
    results = manager.query("test_docs", "What is ML?", n_results=2)
    for i, doc in enumerate(results['documents'][0]):
        print(f"  {i+1}. {doc}")
    
    print("\nQuery with filter (category='ml'):")
    results = manager.query("test_docs", "neural networks", n_results=2, where={"category": "ml"})
    for i, doc in enumerate(results['documents'][0]):
        print(f"  {i+1}. {doc}")
    
    manager.update("test_docs", ids=["doc0"], metadatas=[{"category": "ml", "year": 2024, "updated": True}])
    manager.delete("test_docs", ids=["doc3"])


# Exercise 3: Index Comparison
class IndexComparator:
    """Compare different index types."""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.results = {}
    
    def benchmark_index(self, index_type: str, vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> Dict:
        manager = FAISSIndexManager(self.dimension)
        if index_type == 'flat':
            index = manager.create_flat_index()
        elif index_type == 'ivf':
            index = manager.create_ivf_index(nlist=100)
        else:
            index = manager.create_hnsw_index(M=32)
        
        start = time.time()
        manager.add_vectors(index_type, vectors)
        add_time = time.time() - start
        
        query_times = []
        all_indices = []
        for query in queries:
            start = time.time()
            distances, indices = manager.search(index_type, query.reshape(1, -1), k)
            query_times.append(time.time() - start)
            all_indices.append(indices[0])
        
        return {'add_time': add_time, 'avg_query_time': np.mean(query_times), 'indices': all_indices}
    
    def calculate_recall(self, ground_truth: List, results: List, k: int = 10) -> float:
        """Calculate recall@k."""
        recalls = []
        for gt, res in zip(ground_truth, results):
            gt_set = set(gt[:k])
            res_set = set(res[:k])
            recall = len(gt_set & res_set) / k
            recalls.append(recall)
        return np.mean(recalls)


def exercise_3_index_comparison():
    print("\nExercise 3: Index Comparison\n" + "-" * 40)
    dimension = 128
    n_vectors = 5000
    n_queries = 100
    vectors = np.random.random((n_vectors, dimension)).astype('float32')
    queries = np.random.random((n_queries, dimension)).astype('float32')
    comparator = IndexComparator(dimension)
    
    results = {}
    for index_type in ['flat', 'ivf', 'hnsw']:
        print(f"\nBenchmarking {index_type.upper()}...")
        results[index_type] = comparator.benchmark_index(index_type, vectors, queries, k=10)
    
    print(f"\nComparison:\n{'Index':<10} {'Add Time':<12} {'Query Time':<12}\n" + "-" * 40)
    for index_type, result in results.items():
        print(f"{index_type:<10} {result['add_time']:<12.3f} {result['avg_query_time']*1000:<12.2f}ms")
    
    ground_truth = results['flat']['indices']
    for index_type in ['ivf', 'hnsw']:
        recall = comparator.calculate_recall(ground_truth, results[index_type]['indices'], k=10)
        print(f"\n{index_type.upper()} Recall@10: {recall:.2%}")


# Exercise 4: Metadata Filtering
class MetadataFilter:
    """Advanced metadata filtering."""
    
    def __init__(self):
        self.manager = ChromaManager()
        self.collection_name = "filtered_docs"
        self.manager.create_collection(self.collection_name)
    
    def add_documents_with_metadata(self, documents: List[str], metadatas: List[Dict]):
        ids = [f"doc{i}" for i in range(len(documents))]
        self.manager.insert_documents(self.collection_name, documents, metadatas, ids)
    
    def filter_by_equality(self, query: str, field: str, value: any) -> Dict:
        return self.manager.query(self.collection_name, query, where={field: value})
    
    def filter_by_range(self, query: str, field: str, min_val: any, max_val: any) -> Dict:
        return self.manager.query(self.collection_name, query, where={field: {"$gte": min_val, "$lte": max_val}})
    
    def filter_compound(self, query: str, filters: List[Dict]) -> Dict:
        return self.manager.query(self.collection_name, query, where={"$and": filters})


def exercise_4_metadata_filtering():
    print("\nExercise 4: Metadata Filtering\n" + "-" * 40)
    filter_system = MetadataFilter()
    documents = ["Machine learning tutorial", "Deep learning guide", "Python basics", "Advanced ML techniques", "Data science fundamentals"]
    metadatas = [
        {"category": "ml", "level": "beginner", "year": 2023},
        {"category": "ml", "level": "advanced", "year": 2024},
        {"category": "programming", "level": "beginner", "year": 2023},
        {"category": "ml", "level": "advanced", "year": 2024},
        {"category": "data", "level": "beginner", "year": 2024}
    ]
    filter_system.add_documents_with_metadata(documents, metadatas)
    
    print("\nFilter: category='ml'")
    results = filter_system.filter_by_equality("machine learning", "category", "ml")
    for doc in results['documents'][0]:
        print(f"  - {doc}")
    
    print("\nFilter: year >= 2024")
    results = filter_system.filter_by_range("learning", "year", 2024, 2025)
    for doc in results['documents'][0]:
        print(f"  - {doc}")
    
    print("\nFilter: category='ml' AND level='advanced'")
    results = filter_system.filter_compound("deep learning", [{"category": "ml"}, {"level": "advanced"}])
    for doc in results['documents'][0]:
        print(f"  - {doc}")


# Exercise 5: Batch Operations
class BatchProcessor:
    """Optimize with batch operations."""
    
    def __init__(self):
        self.manager = ChromaManager()
    
    def batch_insert(self, collection_name: str, documents: List[str], batch_size: int = 100):
        self.manager.create_collection(collection_name)
        total_time = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = [f"doc{j}" for j in range(i, i + len(batch))]
            metadatas = [{"batch": i // batch_size} for _ in batch]
            start = time.time()
            self.manager.insert_documents(collection_name, batch, metadatas, ids)
            total_time += time.time() - start
        return total_time
    
    def single_insert(self, collection_name: str, documents: List[str]):
        self.manager.create_collection(collection_name + "_single")
        total_time = 0
        for i, doc in enumerate(documents):
            start = time.time()
            self.manager.insert_documents(collection_name + "_single", [doc], [{"index": i}], [f"doc{i}"])
            total_time += time.time() - start
        return total_time


def exercise_5_batch_operations():
    """Exercise 5: Batch Operations"""
    print("\nExercise 5: Batch Operations")
    print("-" * 40)
    
    processor = BatchProcessor()
    
    # Generate test data
    documents = [f"Document {i} about topic {i % 10}" for i in range(500)]
    
    # Batch insert
    print("\nBatch insert (batch_size=100):")
    batch_time = processor.batch_insert("batch_test", documents, batch_size=100)
    print(f"  Total time: {batch_time:.3f}s")
    print(f"  Avg per doc: {batch_time/len(documents)*1000:.2f}ms")
    
    # Single insert (sample)
    print("\nSingle insert (first 50 docs):")
    single_time = processor.single_insert("batch_test", documents[:50])
    print(f"  Total time: {single_time:.3f}s")
    print(f"  Avg per doc: {single_time/50*1000:.2f}ms")
    
    # Speedup
    speedup = (single_time / 50) / (batch_time / len(documents))
    print(f"\nBatch speedup: {speedup:.1f}x faster")


if __name__ == "__main__":
    print("Day 81: Vector Databases - Solutions\n" + "=" * 60)
    try:
        exercise_1_faiss_index()
        exercise_2_chroma_database()
        exercise_3_index_comparison()
        exercise_4_metadata_filtering()
        exercise_5_batch_operations()
        print("\n" + "=" * 60 + "\nAll exercises completed!")
    except Exception as e:
        print(f"\nError: {e}\n\nMake sure you have installed:\npip install faiss-cpu chromadb sentence-transformers")
