# Day 97: Model Optimization

## Learning Objectives

**Time**: 1 hour

- Understand model optimization techniques
- Learn quantization, pruning, and distillation
- Implement model compression methods
- Apply optimization for deployment

## Theory (15 minutes)

### What is Model Optimization?

Model optimization reduces model size and improves inference speed while maintaining accuracy, enabling deployment on resource-constrained devices.

**Key Techniques**:
- Quantization
- Pruning
- Knowledge distillation
- Model compression
- Hardware acceleration

### Why Optimize Models?

**Benefits**:
- Smaller model size
- Faster inference
- Lower memory usage
- Reduced energy consumption
- Edge device deployment

**Trade-offs**:
- Slight accuracy loss
- Training complexity
- Hardware compatibility

### Quantization

**What is Quantization?**: Converting high-precision weights (float32) to lower precision (int8, int4).

**Types**:
- Post-training quantization
- Quantization-aware training
- Dynamic quantization

**PyTorch Quantization**:
```python
import torch

# Post-training static quantization
model_fp32 = MyModel()
model_fp32.eval()

# Prepare for quantization
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model_fp32)

# Calibrate with representative data
calibrate(model_prepared, data_loader)

# Convert to quantized model
model_int8 = torch.quantization.convert(model_prepared)
```

**Dynamic Quantization**:
```python
# Simpler, no calibration needed
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Pruning

**What is Pruning?**: Removing unnecessary weights or neurons from the model.

**Types**:
- Unstructured pruning: Remove individual weights
- Structured pruning: Remove entire channels/neurons
- Magnitude-based: Remove smallest weights
- Gradient-based: Remove based on importance

**PyTorch Pruning**:
```python
import torch.nn.utils.prune as prune

# Prune 30% of weights in linear layer
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

# Make pruning permanent
prune.remove(model.fc1, 'weight')
```

**Structured Pruning**:
```python
# Prune entire channels
prune.ln_structured(
    model.conv1, 
    name='weight', 
    amount=0.5, 
    n=2, 
    dim=0
)
```

### Knowledge Distillation

**What is Distillation?**: Training a smaller student model to mimic a larger teacher model.

**Process**:
1. Train large teacher model
2. Use teacher predictions as soft targets
3. Train student model on soft targets + hard labels

**Implementation**:
```python
def distillation_loss(student_logits, teacher_logits, labels, 
                     temperature=3.0, alpha=0.5):
    # Soft targets from teacher
    soft_targets = torch.nn.functional.softmax(
        teacher_logits / temperature, dim=1
    )
    soft_prob = torch.nn.functional.log_softmax(
        student_logits / temperature, dim=1
    )
    
    # KL divergence loss
    soft_loss = torch.nn.functional.kl_div(
        soft_prob, soft_targets, reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard label loss
    hard_loss = torch.nn.functional.cross_entropy(
        student_logits, labels
    )
    
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### Model Compression

**Techniques**:
- Weight sharing
- Low-rank factorization
- Huffman coding
- Mixed precision

**Weight Sharing**:
```python
# Cluster weights and share values
from sklearn.cluster import KMeans

weights = model.fc.weight.data.numpy().flatten()
kmeans = KMeans(n_clusters=16)
clusters = kmeans.fit_predict(weights.reshape(-1, 1))
compressed_weights = kmeans.cluster_centers_[clusters]
```

### ONNX Export

**Export to ONNX**:
```python
import torch.onnx

# Export model
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

**ONNX Runtime**:
```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(
    None,
    {"input": input_data.numpy()}
)
```

### TensorRT Optimization

**Convert to TensorRT**:
```python
import tensorrt as trt

# Build TensorRT engine
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

# Parse ONNX model
with open("model.onnx", 'rb') as f:
    parser.parse(f.read())

# Build optimized engine
engine = builder.build_cuda_engine(network)
```

### Benchmarking

**Measure Performance**:
```python
import time

def benchmark(model, input_data, num_runs=100):
    # Warmup
    for _ in range(10):
        _ = model(input_data)
    
    # Measure
    start = time.time()
    for _ in range(num_runs):
        _ = model(input_data)
    end = time.time()
    
    avg_time = (end - start) / num_runs
    return avg_time * 1000  # ms
```

### Model Size Comparison

**Calculate Size**:
```python
def get_model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / (1024 * 1024)  # MB
    os.remove("temp.pth")
    return size
```

### Optimization Pipeline

**Complete Pipeline**:
```python
def optimize_model(model, data_loader):
    # 1. Quantization
    model_quantized = quantize_model(model, data_loader)
    
    # 2. Pruning
    model_pruned = prune_model(model_quantized, amount=0.3)
    
    # 3. Export to ONNX
    export_onnx(model_pruned, "optimized_model.onnx")
    
    return model_pruned
```

### Evaluation Metrics

**Compare Models**:
- Accuracy: Test set performance
- Size: Model file size (MB)
- Latency: Inference time (ms)
- Throughput: Samples per second
- Memory: Peak memory usage

### Best Practices

1. **Baseline First**: Measure original model
2. **Incremental**: Apply one technique at a time
3. **Validate**: Check accuracy after each step
4. **Profile**: Identify bottlenecks
5. **Hardware-Specific**: Optimize for target device
6. **Test Thoroughly**: Verify on real data

### Use Cases

**Mobile Deployment**:
- Quantization for smaller size
- Pruning for faster inference
- ONNX for cross-platform

**Edge Devices**:
- Aggressive quantization (int4)
- Structured pruning
- Hardware-specific optimization

**Cloud Inference**:
- Batch optimization
- Mixed precision
- GPU acceleration

**Real-time Applications**:
- Low latency critical
- TensorRT optimization
- Model distillation

### Why This Matters

Model optimization enables deploying AI on resource-constrained devices, reduces costs, and improves user experience. Understanding optimization techniques is essential for production AI systems.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **Quantization**: Quantize model weights
2. **Pruning**: Prune model parameters
3. **Distillation**: Train student from teacher
4. **Benchmarking**: Measure model performance
5. **Compression**: Calculate compression ratio

## Resources

- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [ONNX](https://onnx.ai/)
- [TensorRT](https://developer.nvidia.com/tensorrt)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 98: Integration Project

Tomorrow you'll build an integration project that combines multiple AI techniques from the course into a complete application.
