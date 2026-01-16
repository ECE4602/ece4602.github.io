---
layout: page
title: "Module 0: Tensor Shapes & Linear Algebra Refresher"
permalink: /python-numpy-tutorial/
---

Module 0 is short, practical onboarding: read shapes, predict tensor dimensions, and debug shape errors with confidence. This page is code-first with brief explanations and simple image placeholders you can replace later.

Table of Contents:

- [Core data objects and notation](#core-data-objects-and-notation)
  - [Scalars](#scalars)
  - [Vectors](#vectors)
  - [Matrices](#matrices)
  - [Tensors](#tensors)
- [Two core shape rules](#two-core-shape-rules)
- [Code, explained](#code-explained-copy-paste-into-colab)
  - [Install a package and check runtime](#1-install-a-package-and-check-runtime)
  - [Scalars, vectors, matrices, tensors](#2-scalars-vectors-matrices-tensors)
  - [Linear layer shape rule](#3-linear-layer-shape-rule)
  - [Broadcasting check](#4-broadcasting-check)
  - [Image tensor example](#5-image-tensor-example)
- [Quick shape drills](#quick-shape-drills-do-before-running-the-cell)

## Core data objects and notation
### Scalars


A **scalar** is a single real number, such as 2, 6, or 20. Scalars are **0<sup>th</sup>-order quantities**, meaning they have **magnitude only** (no direction).

Geometrically, a scalar can be represented as a **point on a number line**, or as a coordinate along a single axis (e.g., the *x*- or *y*-axis) in a higher-dimensional Euclidean space.

Scalars are typically denoted by **lowercase letters**, such as \\(x\\), \\(y\\), or \\(z\\).

The set of all real-valued scalars is denoted by the **real numbers**, written as \\(R \\).
If \\(x \\) is a real number, we write: \\(x \in R \\).




<div class='fig figcenter fighighlight'>
  <img src='/assets/module-0/scalar.gif' alt='Scalar illustrated as a single point on a number line' class='img-center' style='max-width: 360px;' onerror="this.onerror=null;this.src='/assets/module-0/scalar.png';">
  <div class='figcaption'>Scalar (example): a single value on a number line.</div>
</div>

### Vectors
Write your explanation here.

### Matrices
Write your explanation here.

### Tensors
Write your explanation here.



## Two core shape rules
1. Matrix multiplication for a linear layer: `X(B, d) @ W(d, k) -> Z(B, k)`
2. Broadcasting bias: `Z(B, k) + b(k,) -> (B, k)`

<div class='fig figcenter fighighlight'>
  <img src='/assets/placeholders/placeholder.svg' alt='Matrix multiply shape placeholder' width='85%'>
  <div class='figcaption'>Placeholder: diagram of X(B,d) times W(d,k) producing Z(B,k).</div>
</div>

## Code, explained (copy-paste into Colab)
### 1) Install a package and check runtime
You will repeat this after a runtime restart.
```python
# Install a tiny helper package (pip demo)
!pip -q install torchinfo

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("Device:", device)
```

### 2) Scalars, vectors, matrices, tensors
Create each object and print its shape.
```python
scalar = torch.tensor(3.14)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
tensor = torch.randn(2, 3, 4, 5)  # B, C, H, W

for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix), ("tensor", tensor)]:
    print(f"{name}: shape={tuple(t.shape)}")
```

### 3) Linear layer shape rule
The inner dimensions must match: `d` with `d`. The output keeps the batch and uses `k`.
```python
B, d, k = 4, 5, 3
X = torch.randn(B, d)
W = torch.randn(d, k)
b = torch.randn(k)

Z = X @ W
Y = Z + b

print("X", X.shape, "W", W.shape, "Z", Z.shape)
print("b", b.shape, "Y", Y.shape)
```

<div class='fig figcenter fighighlight'>
  <img src='/assets/placeholders/placeholder.svg' alt='Bias broadcast placeholder' width='85%'>
  <div class='figcaption'>Placeholder: bias broadcasting visual (b added to every row of Z).</div>
</div>

### 4) Broadcasting check
Bias of shape `(k,)` is stretched to match `(B, k)`.
```python
same = torch.allclose(Z + b, Z + b.view(1, k))
print("Broadcasting works:", same)
```

### 5) Image tensor example
Images are `(B, C, H, W)`. Flatten to `(B, C*H*W)` before a linear layer.
```python
B, C, H, W = 2, 3, 32, 32
images = torch.randn(B, C, H, W)

X_img = images.view(B, -1)
W_img = torch.randn(C * H * W, 10)
Z_img = X_img @ W_img

print("images", images.shape)
print("X_img", X_img.shape, "W_img", W_img.shape, "Z_img", Z_img.shape)
```

## Quick shape drills (do before running the cell)
- Predict: If `X` is `(8, 64)` and `W` is `(64, 10)`, what is `Z`?
- Fix the bug: Set `W = torch.randn(d + 1, k)`. Read the error, then fix the shape.
- Broadcasting: Change `b` to shape `(1, k)`. Does `Z + b` still work?
- Images: For `B=4, C=1, H=28, W=28`, what is `X_img.shape`?
