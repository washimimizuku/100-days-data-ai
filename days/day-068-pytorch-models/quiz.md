# Day 68: PyTorch Models - Quiz

Test your understanding of building and training PyTorch models.

## Questions

### 1. What is the purpose of the `forward()` method in nn.Module?
a) To initialize model parameters
b) To define the computation performed at every call
c) To update model weights during training
d) To save the model to disk

**Correct Answer: b**

### 2. Which method must be called before `loss.backward()` in the training loop?
a) `model.train()`
b) `optimizer.step()`
c) `optimizer.zero_grad()`
d) `model.eval()`

**Correct Answer: c**

### 3. What is the difference between `model.train()` and `model.eval()`?
a) train() enables gradient computation, eval() disables it
b) train() activates dropout/batch norm, eval() deactivates them
c) train() saves the model, eval() loads it
d) train() uses GPU, eval() uses CPU

**Correct Answer: b**

### 4. Which loss function is appropriate for multi-class classification?
a) `nn.MSELoss()`
b) `nn.BCELoss()`
c) `nn.CrossEntropyLoss()`
d) `nn.L1Loss()`

**Correct Answer: c**

### 5. What does `torch.no_grad()` do?
a) Removes all gradients from the model
b) Disables gradient computation to save memory
c) Prevents the model from training
d) Deletes the computation graph

**Correct Answer: b**

### 6. How do you save only the model weights (not the entire model)?
a) `torch.save(model, 'model.pth')`
b) `torch.save(model.state_dict(), 'model.pth')`
c) `model.save('model.pth')`
d) `torch.save_weights(model, 'model.pth')`

**Correct Answer: b**

### 7. What is the purpose of `optimizer.step()`?
a) To compute gradients
b) To clear previous gradients
c) To update model parameters based on gradients
d) To move to the next epoch

**Correct Answer: c**

### 8. Which activation function is most commonly used in hidden layers?
a) Sigmoid
b) Softmax
c) ReLU
d) Tanh

**Correct Answer: c**

### 9. What does `nn.Sequential` allow you to do?
a) Train multiple models in sequence
b) Create a model by stacking layers in order
c) Process batches sequentially instead of in parallel
d) Save model checkpoints automatically

**Correct Answer: b**

### 10. When loading a saved model, what must you call to use it for inference?
a) `model.train()`
b) `model.eval()`
c) `model.inference()`
d) `model.predict()`

**Correct Answer: b**

## Scoring Guide
- 9-10 correct: Excellent! You understand PyTorch model development well.
- 7-8 correct: Good job! Review the topics you missed.
- 5-6 correct: Fair. Practice more with the exercises.
- Below 5: Review the material and work through the solutions again.

## Answer Key
1. b - forward() defines the computation graph
2. c - Must clear gradients before backward pass
3. b - train() activates dropout/batch norm, eval() deactivates
4. c - CrossEntropyLoss for multi-class classification
5. b - Disables gradient computation to save memory
6. b - state_dict() saves only weights
7. c - Updates parameters using computed gradients
8. c - ReLU is most common for hidden layers
9. b - Sequential stacks layers in order
10. b - eval() mode for inference (disables dropout/batch norm)
