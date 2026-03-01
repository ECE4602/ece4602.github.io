# Module 2: Single Perceptron Classification

This module focuses on building a single perceptron for binary classification using PyTorch. The content is extracted from the notebook `2_Single_Perceptron_Classification_Healthy_NotHealthy_torch.ipynb`.

## Topics Covered

1. **Problem Definition**
   - Classify `Healthy` vs `Not Healthy` using synthetic data.

2. **Model Creation**
   - Single `nn.Linear` layer.
   - Explanation of input and output dimensions.

3. **Training**
   - Loss function: Mean Squared Error (MSE).
   - Optimizer: Stochastic Gradient Descent (SGD).
   - Training loop with metrics tracking.

4. **Visualization**
   - Loss and accuracy plots.
   - Decision boundary visualization.

---

### Example Code

```python
import torch

# Define the model
model = torch.nn.Linear(in_features=2, out_features=1)
```

### Visualization

![Decision Boundary](assets/module-2-decision-boundary.png)

---

This module demonstrates the simplest neural network for binary classification and provides insights into training and evaluation.