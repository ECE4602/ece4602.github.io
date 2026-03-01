# Module 3: Single Layer Multiclass Classification

This module focuses on building a single-layer neural network for multi-class classification using PyTorch. The content is extracted from the notebook `3_Single_Layer_Multiclass_CE_Softmax_torch.ipynb`.

## Topics Covered

1. **Problem Definition**
   - Classify individuals into 5 health risk categories using synthetic data.

2. **Model Creation**
   - Single `nn.Linear` layer with 5 output neurons.
   - Explanation of input and output dimensions.

3. **Training**
   - Loss function: Cross-Entropy Loss.
   - Optimizer: Stochastic Gradient Descent (SGD).
   - Training loop with metrics tracking.

4. **Visualization**
   - Loss and accuracy plots.
   - Decision regions visualization.

---

### Example Code

```python
import torch

# Define the model
model = torch.nn.Linear(in_features=2, out_features=5)
```

### Visualization

![Decision Regions](assets/module-3-decision-regions.png)

---

This module demonstrates a simple neural network for multi-class classification and provides insights into training and evaluation.