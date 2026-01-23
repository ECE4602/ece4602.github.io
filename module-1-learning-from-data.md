---
layout: page
mathjax: true
permalink: /classification/
title: "Module 1: Learning From Data (Celsius \u2192 Fahrenheit)"
---

In this lesson, a single-neuron PyTorch model learns to convert Celsius to Fahrenheit from examples, introducing the core workflow: data, model, loss, optimizer, and training loop.

Table of Contents:
- [1. The paradigm shift: from rules to learning](#1-the-paradigm-shift-from-rules-to-learning)
- [2. Crafting the toy dataset](#2-crafting-the-toy-dataset)
- [3. The architecture: the single neuron](#3-the-architecture-the-single-neuron)
- [4. The loss function and the optimizer](#4-the-loss-function-and-the-optimizer)
- [5. The training loop](#5-the-training-loop)
- [6. Visualizing success and inspecting memory](#6-visualizing-success-and-inspecting-memory)
- [7. Scaling up: the data science workflow](#7-scaling-up-the-data-science-workflow)
- [8. The impact of data](#8-the-impact-of-data)

## 1. The paradigm shift: from rules to learning
We begin our journey by rethinking how we program. In traditional computer science, we operate as architects of logic. We possess the input and we design the rule (the algorithm) to produce an output. For example, to convert Celsius to Fahrenheit, we would explicitly program the formula \\(F = 1.8C + 32\\).

In deep learning, we invert this process. We possess the input (Celsius temperatures) and the desired output (Fahrenheit temperatures), but we treat the relationship between them as unknown. Instead of writing the rule, we ask the neural network to discover it for us. Our objective is to build a system that observes raw data samples and learns that physical formula on its own.

<!-- VISUALIZATION: Two block diagrams comparing paradigms.

Traditional: [Input] + [Rule] -> [Box] -> [Output]

Machine Learning: [Input] + [Output] -> [Box] -> [Rule]
-->

## 2. Crafting the toy dataset
Before we build a brain, we need memories to teach it. We manually craft a small, clean dataset using PyTorch tensors.

```python
import torch
import pandas as pd
import matplotlib.pyplot as plt

# We manually define 7 known examples
celsius_q    = torch.tensor([-40, -10,  0,  8, 15, 22,  38], dtype=torch.float32)
fahrenheit_a = torch.tensor([-40,  14, 32, 46, 59, 72, 100], dtype=torch.float32)

for i in range(len(celsius_q)):
    print(f"{celsius_q[i].item()} degrees Celsius = {fahrenheit_a[i].item()} degrees Fahrenheit")
```

<!-- VISUALIZATION: A simple scatter plot with Celsius on the X-axis and Fahrenheit on the Y-axis, showing the 7 data points forming a straight line. -->

The model will never see the formula \\(1.8C + 32\\); it will only see these pairs of numbers.

## 3. The architecture: the single neuron
To solve this, we use the fundamental unit of deep learning: the perceptron (a single neuron). A single neuron performs a linear transformation: it scales an input by a weight and adds a bias.

<!-- VISUALIZATION: A diagram of a single neuron. An input node 'x' connects via a line labeled 'w' (weight) to a summing neuron. Another input labeled '1' connects via 'b' (bias). The neuron sums them (w*x + b) to produce output 'y'. -->

Scalar form (single-value input):
\\[
y = w \\cdot x + b
\\]

Vector form (multiple inputs):
\\[
y = \\mathbf{W}^T\\mathbf{x} + \\mathbf{b}
\\]

In PyTorch, we define this structure using `torch.nn.Linear`:

```python
import torch

# 1 input (Celsius) -> 1 output (Fahrenheit)
# This initializes w and b with random numbers
model = torch.nn.Linear(in_features=1, out_features=1)
```

It is important to note that PyTorch initializes \\(w\\) and \\(b\\) randomly. An untrained model is essentially a random number generator until we train it.

## 4. The loss function and the optimizer
To improve, the network must know how wrong it is. We measure this using a loss function. For regression problems like ours, a standard choice is Mean Squared Error (MSE).

<!-- VISUALIZATION: A plot showing a regression line that doesn't fit the data points well. Vertical residual lines represent error. -->

Single-example loss:
\\[
\\text{Loss} = (\\text{Prediction} - \\text{Actual})^2
\\]

Mean squared error over \\(N\\) examples:
\\[
J(w, b) = \\frac{1}{N}\\sum_{i=1}^{N}(y_i - \\hat{y}_i)^2
\\]

Once we know the error, we need a mechanism to reduce it. This is the optimizer. We use gradient-based optimization (here: Adam).

<!-- VISUALIZATION: A bowl-shaped loss landscape with a ball rolling toward the minimum. -->

Update rule (weight):
\\[
w_{new} = w_{old} - \\alpha\\,\\frac{\\partial J}{\\partial w}
\\]

```python
import torch

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
```

## 5. The training loop
We iterate the cycle of predict -> measure error -> update weights. Each full pass is an epoch. The model expects inputs shaped like `(N, 1)`, so we reshape with `.unsqueeze(1)`.

<!-- VISUALIZATION: A circular flowchart showing: Data -> Model -> Loss -> Gradients -> Optimizer -> Update -> Repeat. -->

```python
import torch

losses = []
epochs = 500

x = celsius_q.unsqueeze(1)     # (N, 1)
y = fahrenheit_a.unsqueeze(1)  # (N, 1)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
```

## 6. Visualizing success and inspecting memory
After training, plot the loss curve and inspect what the model learned.

<!-- VISUALIZATION: Loss vs epoch curve that drops quickly and flattens near zero. -->

```python
import matplotlib.pyplot as plt

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses)
plt.show()
```

```python
with torch.no_grad():
    w = model.weight.item()
    b = model.bias.item()

print("Learned w:", w)
print("Learned b:", b)

# Test with a new value (23 degrees C)
with torch.no_grad():
    pred_f = model(torch.tensor([[23.0]])).item()
print("Prediction at 23C:", pred_f)
```

## 7. Scaling up: the data science workflow
In real projects, we load data from files. Here we read a CSV with columns `Celsius` and `Fahrenheit`.

<!-- VISUALIZATION: CSV icon -> Pandas -> Tensor block -> Model. -->

```python
import pandas as pd
import torch

temp_df = pd.read_csv("Temp_prediction.csv")
celsius_q = torch.tensor(temp_df["Celsius"].values, dtype=torch.float32).unsqueeze(1)
fahrenheit_a = torch.tensor(temp_df["Fahrenheit"].values, dtype=torch.float32).unsqueeze(1)
```

Train the same model as before (same architecture, same loss, same optimizer), but now on a larger dataset.

## 8. The impact of data
Why do we do this twice? To demonstrate a core rule in deep learning: **more (clean) data often improves performance**.

With more examples, gradient-based optimization can estimate a more accurate slope \\(w\\) and intercept \\(b\\), producing predictions closer to the true physical relationship.
