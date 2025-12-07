# Day 67: PyTorch Tensors - Quiz

Test your understanding of PyTorch tensors, operations, and autograd.

## Questions

### 1. What is the primary difference between a PyTorch tensor and a NumPy array?
a) Tensors can only store floating-point numbers
b) Tensors can run on GPUs and support automatic differentiation
c) Tensors are always 2-dimensional
d) Tensors cannot be converted to NumPy arrays

**Correct Answer: b**

### 2. Which method creates a tensor that requires gradient computation?
a) `torch.tensor([1, 2, 3])`
b) `torch.tensor([1, 2, 3], requires_grad=True)`
c) `torch.tensor([1, 2, 3], gradient=True)`
d) `torch.tensor([1, 2, 3], autograd=True)`

**Correct Answer: b**

### 3. What does the `.backward()` method do in PyTorch?
a) Reverses the order of tensor elements
b) Computes gradients using backpropagation
c) Moves the tensor to CPU
d) Converts the tensor to NumPy array

**Correct Answer: b**

### 4. How do you move a tensor to GPU in PyTorch?
a) `tensor.gpu()`
b) `tensor.cuda()`
c) `tensor.to('cuda')`
d) Both b and c are correct

**Correct Answer: d**

### 5. What is broadcasting in PyTorch?
a) Sending tensors across network
b) Automatic expansion of tensor dimensions for operations
c) Converting tensors to different data types
d) Parallelizing operations across multiple GPUs

**Correct Answer: b**

### 6. Which operation is performed element-wise by default in PyTorch?
a) `torch.matmul()`
b) `torch.mm()`
c) `tensor1 * tensor2`
d) `torch.dot()`

**Correct Answer: c**

### 7. What does `tensor.detach()` do?
a) Removes the tensor from memory
b) Creates a new tensor that doesn't require gradients
c) Moves the tensor to CPU
d) Converts the tensor to a Python list

**Correct Answer: b**

### 8. Which method reshapes a tensor without copying data?
a) `tensor.reshape()`
b) `tensor.view()`
c) `tensor.resize()`
d) `tensor.transform()`

**Correct Answer: b**

### 9. What happens when you call `.backward()` on a non-scalar tensor?
a) It automatically sums all elements first
b) It raises an error unless you provide a gradient argument
c) It computes gradients for each element separately
d) It converts the tensor to scalar automatically

**Correct Answer: b**

### 10. Which function performs matrix multiplication in PyTorch?
a) `torch.mul()`
b) `torch.matmul()`
c) `torch.multiply()`
d) `torch.product()`

**Correct Answer: b**

## Scoring Guide
- 9-10 correct: Excellent! You have a strong understanding of PyTorch tensors.
- 7-8 correct: Good job! Review the topics you missed.
- 5-6 correct: Fair. Revisit the README and practice more with the exercises.
- Below 5: Review the material and work through the exercises again.

## Answer Key
1. b - Tensors support GPU acceleration and automatic differentiation
2. b - Use `requires_grad=True` parameter
3. b - Computes gradients via backpropagation
4. d - Both `.cuda()` and `.to('cuda')` work
5. b - Broadcasting automatically expands dimensions
6. c - `*` operator performs element-wise multiplication
7. b - Creates a tensor without gradient tracking
8. b - `.view()` reshapes without copying (when possible)
9. b - Non-scalar tensors need gradient argument
10. b - `torch.matmul()` performs matrix multiplication
