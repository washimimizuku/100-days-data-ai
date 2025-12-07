"""Day 97: Model Optimization - Solutions

NOTE: Simplified optimization implementations for learning.
"""

import numpy as np
from typing import Dict, Any, Tuple
import time


# Exercise 1: Quantization
class ModelQuantizer:
    """Quantize model weights."""
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = 0
        self.qmax = 2 ** bits - 1
    
    def quantize_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Quantize weights to lower precision."""
        # Calculate scale and zero point
        wmin, wmax = weights.min(), weights.max()
        scale = (wmax - wmin) / (self.qmax - self.qmin)
        zero_point = self.qmin - wmin / scale
        
        # Quantize
        quantized = np.round(weights / scale + zero_point)
        quantized = np.clip(quantized, self.qmin, self.qmax).astype(np.uint8)
        
        metadata = {
            'scale': scale,
            'zero_point': zero_point,
            'bits': self.bits
        }
        
        return quantized, metadata
    
    def dequantize_weights(self, quantized: np.ndarray, 
                          metadata: Dict) -> np.ndarray:
        """Dequantize weights back to float."""
        scale = metadata['scale']
        zero_point = metadata['zero_point']
        
        dequantized = (quantized.astype(np.float32) - zero_point) * scale
        return dequantized


# Exercise 2: Model Pruning
class ModelPruner:
    """Prune model parameters."""
    
    def __init__(self, pruning_ratio: float = 0.3):
        self.pruning_ratio = pruning_ratio
    
    def magnitude_prune(self, weights: np.ndarray) -> np.ndarray:
        """Prune weights by magnitude."""
        # Calculate threshold
        threshold = np.percentile(np.abs(weights), self.pruning_ratio * 100)
        
        # Prune
        pruned = weights.copy()
        pruned[np.abs(pruned) < threshold] = 0
        
        return pruned
    
    def structured_prune(self, weights: np.ndarray, dim: int = 0) -> np.ndarray:
        """Prune entire rows/columns."""
        # Calculate importance (L2 norm along dimension)
        importance = np.linalg.norm(weights, axis=1-dim)
        
        # Determine number to prune
        num_to_prune = int(len(importance) * self.pruning_ratio)
        
        # Get indices to keep
        indices_to_keep = np.argsort(importance)[num_to_prune:]
        
        # Prune
        if dim == 0:
            pruned = weights[indices_to_keep, :]
        else:
            pruned = weights[:, indices_to_keep]
        
        return pruned
    
    def calculate_sparsity(self, weights: np.ndarray) -> float:
        """Calculate percentage of zero weights."""
        total = weights.size
        zeros = np.sum(weights == 0)
        return zeros / total


# Exercise 3: Knowledge Distillation
class KnowledgeDistiller:
    """Distill knowledge from teacher to student."""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha
    
    def softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature."""
        x = x / temperature
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def distillation_loss(self, student_logits: np.ndarray,
                         teacher_logits: np.ndarray,
                         labels: np.ndarray) -> float:
        """Calculate distillation loss."""
        # Soft targets
        soft_targets = self.softmax(teacher_logits, self.temperature)
        soft_student = self.softmax(student_logits, self.temperature)
        
        # KL divergence (simplified)
        soft_loss = -np.sum(soft_targets * np.log(soft_student + 1e-10))
        soft_loss *= (self.temperature ** 2)
        
        # Hard loss (cross entropy)
        student_probs = self.softmax(student_logits)
        hard_loss = -np.log(student_probs[np.arange(len(labels)), labels] + 1e-10).mean()
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
    
    def train_student(self, student_model: Any, teacher_model: Any,
                     data: np.ndarray, epochs: int = 10) -> Dict:
        """Train student model."""
        history = {'loss': []}
        
        for epoch in range(epochs):
            # Mock training
            loss = np.random.uniform(1.0, 0.1)  # Decreasing loss
            history['loss'].append(loss)
        
        return history


# Exercise 4: Model Benchmarking
class ModelBenchmark:
    """Benchmark model performance."""
    
    def measure_latency(self, model: Any, input_data: np.ndarray,
                       num_runs: int = 100) -> Dict[str, float]:
        """Measure inference latency."""
        times = []
        
        # Warmup
        for _ in range(10):
            _ = np.dot(input_data, input_data.T)  # Mock inference
        
        # Measure
        for _ in range(num_runs):
            start = time.time()
            _ = np.dot(input_data, input_data.T)  # Mock inference
            end = time.time()
            times.append((end - start) * 1000)  # ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times)
        }
    
    def measure_throughput(self, model: Any, batch_size: int = 32,
                          duration: float = 1.0) -> float:
        """Measure throughput (samples/sec)."""
        start = time.time()
        samples = 0
        
        while time.time() - start < duration:
            # Mock inference
            _ = np.random.randn(batch_size, 100)
            samples += batch_size
        
        elapsed = time.time() - start
        return samples / elapsed
    
    def compare_models(self, models: Dict[str, Any],
                      input_data: np.ndarray) -> Dict[str, Dict]:
        """Compare multiple models."""
        results = {}
        
        for name, model in models.items():
            latency = self.measure_latency(model, input_data, num_runs=50)
            results[name] = latency
        
        return results


# Exercise 5: Compression Analysis
class CompressionAnalyzer:
    """Analyze model compression."""
    
    def calculate_model_size(self, weights: np.ndarray) -> float:
        """Calculate model size in MB."""
        # Size in bytes
        size_bytes = weights.nbytes
        # Convert to MB
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    
    def calculate_compression_ratio(self, original_size: float,
                                   compressed_size: float) -> float:
        """Calculate compression ratio."""
        return original_size / compressed_size
    
    def analyze_optimization(self, original_model: Any,
                           optimized_model: Any) -> Dict[str, Any]:
        """Analyze optimization results."""
        # Mock analysis
        original_size = 100.0  # MB
        optimized_size = 25.0  # MB
        
        return {
            'original_size_mb': original_size,
            'optimized_size_mb': optimized_size,
            'compression_ratio': original_size / optimized_size,
            'size_reduction_percent': (1 - optimized_size / original_size) * 100,
            'speedup': 2.5  # Mock speedup
        }


# Bonus: Mixed Precision
class MixedPrecisionOptimizer:
    """Optimize with mixed precision."""
    
    def convert_to_fp16(self, weights: np.ndarray) -> np.ndarray:
        """Convert weights to FP16."""
        return weights.astype(np.float16)
    
    def identify_sensitive_layers(self, model: Any,
                                  accuracy_threshold: float = 0.01) -> list:
        """Identify layers sensitive to quantization."""
        # Mock identification
        sensitive = ['layer1', 'layer5', 'output']
        return sensitive


def demo_model_optimization():
    """Demonstrate model optimization."""
    print("Day 97: Model Optimization - Solutions Demo\n" + "=" * 60)
    
    # Create mock weights
    weights = np.random.randn(1000, 1000).astype(np.float32)
    
    print("\n1. Quantization")
    quantizer = ModelQuantizer(bits=8)
    quantized, metadata = quantizer.quantize_weights(weights)
    dequantized = quantizer.dequantize_weights(quantized, metadata)
    print(f"   Original dtype: {weights.dtype}, size: {weights.nbytes / 1024:.1f} KB")
    print(f"   Quantized dtype: {quantized.dtype}, size: {quantized.nbytes / 1024:.1f} KB")
    print(f"   Compression: {weights.nbytes / quantized.nbytes:.1f}x")
    print(f"   Reconstruction error: {np.mean(np.abs(weights - dequantized)):.6f}")
    
    print("\n2. Pruning")
    pruner = ModelPruner(pruning_ratio=0.3)
    pruned = pruner.magnitude_prune(weights)
    sparsity = pruner.calculate_sparsity(pruned)
    print(f"   Original non-zeros: {np.sum(weights != 0)}")
    print(f"   Pruned non-zeros: {np.sum(pruned != 0)}")
    print(f"   Sparsity: {sparsity:.1%}")
    
    print("\n3. Knowledge Distillation")
    distiller = KnowledgeDistiller(temperature=3.0, alpha=0.5)
    student_logits = np.random.randn(32, 10)
    teacher_logits = np.random.randn(32, 10)
    labels = np.random.randint(0, 10, 32)
    loss = distiller.distillation_loss(student_logits, teacher_logits, labels)
    print(f"   Distillation loss: {loss:.4f}")
    print(f"   Temperature: {distiller.temperature}")
    print(f"   Alpha: {distiller.alpha}")
    
    print("\n4. Benchmarking")
    benchmark = ModelBenchmark()
    input_data = np.random.randn(100, 100)
    latency = benchmark.measure_latency(None, input_data, num_runs=50)
    throughput = benchmark.measure_throughput(None, batch_size=32)
    print(f"   Mean latency: {latency['mean_ms']:.2f} ms")
    print(f"   Std latency: {latency['std_ms']:.2f} ms")
    print(f"   Throughput: {throughput:.0f} samples/sec")
    
    print("\n5. Compression Analysis")
    analyzer = CompressionAnalyzer()
    original_size = analyzer.calculate_model_size(weights)
    quantized_size = analyzer.calculate_model_size(quantized)
    ratio = analyzer.calculate_compression_ratio(original_size, quantized_size)
    print(f"   Original size: {original_size:.2f} MB")
    print(f"   Quantized size: {quantized_size:.2f} MB")
    print(f"   Compression ratio: {ratio:.1f}x")
    
    print("\n6. Mixed Precision")
    mixed_precision = MixedPrecisionOptimizer()
    fp16_weights = mixed_precision.convert_to_fp16(weights)
    print(f"   FP32 size: {weights.nbytes / 1024:.1f} KB")
    print(f"   FP16 size: {fp16_weights.nbytes / 1024:.1f} KB")
    print(f"   Reduction: {weights.nbytes / fp16_weights.nbytes:.1f}x")
    
    print("\n" + "=" * 60)
    print("All optimization techniques demonstrated!")


if __name__ == "__main__":
    demo_model_optimization()
