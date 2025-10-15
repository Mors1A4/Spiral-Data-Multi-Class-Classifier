# Spiral Data Multi-Class Classifier

**Posted on October 15, 2025 | Programming Project**

Building on my previous project, this neural network tackles a more complex, multi-class classification problem. The goal is to teach a model to distinguish between three intertwined spiral arms of data points, visualizing how it learns to create highly non-linear decision boundaries.

* **Tech Stack:** PyTorch (Neural Network & Training), Matplotlib (Visualization)
* 
## Experimentation and Results

Solving this problem required experimentation. An initial attempt using the same architecture as the circle project failed. Below is a comparison of different approaches.

### 1. Baseline: SGD with 10 Neurons

![SGD with 10 Neurons](images/training_progress(SGD-10-6k).gif)

The initial attempt used a simple model (10 hidden neurons) and the basic `SGD` optimizer. The model completely fails to learn the spiral pattern, finding a simple linear boundary and getting stuck. This shows the model is not flexible enough for the task.

### 2. More Power, Same Optimizer: SGD with 20 Neurons

![SGD with 20 Neurons](images/training_progress(SGD-20-6k).gif)

Increasing the model's flexibility to 20 neurons helps slightly, but the `SGD` optimizer is still the bottleneck. It struggles to navigate the complex loss landscape, resulting in a poor fit even after 6,000 epochs.

### 3. The Breakthrough: Adam Optimizer with 20 Neurons

![Adam with 20 Neurons](images/training_progress(ADAM-20-600).gif)

The most significant improvement came from switching to the **Adam optimizer**. Its adaptive learning rate allows it to find the complex solution efficiently. With a sufficiently flexible model (20 neurons), Adam solves the problem in only 600 epochs.

### 4. Diminishing Returns: Adam with 128 Neurons

![Adam with 128 Neurons](images/training_progress(ADAM-128-600).gif)

To test the limits, I increased the model's power significantly to 128 neurons. As the GIF shows, the result is not dramatically better than the 20-neuron model. This demonstrates a key concept: once a model has **sufficient capacity** for a task, adding more neurons provides diminishing returns and increases computational cost. The choice of optimizer was the more impactful change.

---

## Model Architecture

The feedforward network was enhanced with a significantly larger hidden layer to handle the increased complexity of the data.

```python
import torch.nn as nn

class SpiralClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # A wider hidden layer for more flexibility
        self.layer_1 = nn.Linear(2, 128)
        # The output layer must have 3 neurons for 3 classes
        self.layer_2 = nn.Linear(128, 3)

    def forward(self, x):
        out = torch.relu(self.layer_1(x))
        # Output raw logits, as CrossEntropyLoss has Softmax built-in
        out = self.layer_2(out)
        return out
```

## Training process 

The training loop now uses nn.CrossEntropyLoss, which is designed for multi-class classification, and the torch.optim.Adam optimizer.

```python
# The model needs more power
model = SpiralClassifier()

# CrossEntropyLoss is essential for multi-class problems
criterion = nn.CrossEntropyLoss()

# Adam is a more advanced optimizer that helps converge faster
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000 # More epochs for a more complex task

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)

    # Calculate loss
    loss = criterion(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
