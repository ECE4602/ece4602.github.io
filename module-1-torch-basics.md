---
layout: page
mathjax: true
permalink: /module-0-torch-basics/
title: "Module 0: PyTorch Tensors and Data Representation"
---

# Module 0: PyTorch Tensors and Data Representation

This module is a PyTorch-based walkthrough of the ideas presented in Jay Alammar's excellent [A Visual Intro to NumPy and Data Representation](https://jalammar.github.io/visual-numpy/). Instead of NumPy arrays, we use **PyTorch tensors** (`torch.Tensor`) — the core data structure you will use throughout all deep learning work in this course.

<a href="https://colab.research.google.com/github/ece4602/ece4602.github.io/blob/master/1_Torch_Basics.ipynb" target="_blank" style="display:inline-flex;align-items:center;gap:7px;background:#f9ab00;color:#1a1a1a;font-weight:600;font-size:0.8rem;padding:6px 14px;border-radius:6px;text-decoration:none;letter-spacing:0.01em;"><img src="https://colab.research.google.com/img/colab_favicon_256px.png" alt="" style="width:16px;height:16px;margin:0;">Open in Colab</a>

---

## Setup

PyTorch comes pre-installed in Google Colab. To verify:

```python
import torch
torch.__version__
```

---

## 1. Creating Tensors

### 1D Tensors (Vectors)

The most direct way to create a tensor is to pass a Python list to `torch.tensor()`. This is the PyTorch equivalent of `np.array([...])`.

```python
data = torch.tensor([1, 2, 3])
data
```

![Creating a 1D tensor from a list](/assets/module-0/torch_tensor.png)

PyTorch also provides factory functions that create tensors filled with specific values — no need to list every element:

```python
ones  = torch.ones(3)           # all 1s
zeros = torch.zeros(3)          # all 0s
rand  = torch.rand(3)           # uniform random in [0, 1)

ones, zeros, rand
```

![torch.ones, torch.zeros, torch.rand for 1D tensors](/assets/module-0/torch_ones_1d.png)

---

## 2. Tensor Arithmetic

### Element-wise operations

Create two tensors and perform basic arithmetic. PyTorch applies every operation **element-wise** by default:

```python
data = torch.tensor([1, 2])
ones = torch.ones(2)

data + ones          # addition
```

![Adding two 1D tensors](https://jalammar.github.io/images/numpy/numpy-arrays-adding-1.png)

```python
data - ones          # subtraction
data * data          # multiplication
data / data          # division
```

![Subtract, multiply, divide on 1D tensors](https://jalammar.github.io/images/numpy/numpy-array-subtract-multiply-divide.png)

### Broadcasting with a scalar (vector × scalar)

When you multiply a tensor by a single number, PyTorch **broadcasts** the scalar across every element automatically:

```python
data_miles = torch.tensor([1, 2])
data_miles * 1.6     # convert miles → kilometres
```

![Broadcasting a scalar across a vector](https://jalammar.github.io/images/numpy/numpy-array-broadcast.png)

---

## 3. Indexing

Tensors support the same slice syntax as Python lists:

```python
data = torch.tensor([1, 2, 3])

data[0]       # first element  → 1
data[1]       # second element → 2
data[0:2]     # first two      → tensor([1, 2])
data[1:]      # from index 1   → tensor([2, 3])
```

![Indexing and slicing a 1D tensor](https://jalammar.github.io/images/numpy/numpy-array-slice.png)

---

## 4. Aggregation

```python
data = torch.tensor([1., 2., 3.])

data.min()               # minimum value
data.max()               # maximum value
data.sum()               # sum of all elements
data.mean()              # arithmetic mean
data.prod()              # product of all elements
data.std(unbiased=False) # standard deviation
```

![Aggregation functions on a 1D tensor](https://jalammar.github.io/images/numpy/numpy-array-aggregation.png)

---

## 5. In More Dimensions

### Creating Matrices (2D Tensors)

Pass a list of lists to `torch.tensor()` to create a 2-D matrix:

```python
mat = torch.tensor([[1, 2],
                    [3, 4]])
mat, mat.shape   # shape → torch.Size([2, 2])
```

![Creating a 2D tensor (matrix)](/assets/module-0/torch_tensor_2d.png)

The factory functions accept a **tuple** to specify the shape of higher-dimensional tensors:

```python
ones_mat  = torch.ones((3, 2))
zeros_mat = torch.zeros((3, 2))
rand_mat  = torch.rand((3, 2))
```

![torch.ones, torch.zeros, torch.rand for 2D tensors](/assets/module-0/torch_ones_zeros_2d.png)

### Matrix Arithmetic (element-wise)

```python
data = torch.tensor([[1, 2],
                     [3, 4]])
ones = torch.ones((2, 2))

data + ones
```

![Element-wise addition on matrices](https://jalammar.github.io/images/numpy/numpy-matrix-arithmetic.png)

### Broadcasting across a dimension

A `(2,)` row vector can be broadcast across every row of a `(2×2)` matrix:

```python
data     = torch.tensor([[1., 2.],
                         [3., 4.]])
ones_row = torch.tensor([10., 20.])

data + ones_row          # broadcast: each row is added element-wise
```

![Broadcasting a row vector across a 2D matrix](https://jalammar.github.io/images/numpy/numpy-matrix-broadcast.png)

### Matrix Indexing

Two indices (row, col) let you pick any element or slice entire rows/columns:

```python
data = torch.tensor([[1, 2],
                     [3, 4],
                     [5, 6]])

data[0, 1]    # row 0, col 1 → 2
data[1:3]     # rows 1 and 2
data[0:2, 0]  # rows 0–1, only column 0
```

![Matrix indexing and slicing](https://jalammar.github.io/images/numpy/numpy-matrix-indexing.png)

### Matrix Aggregation (with `dim`)

You can aggregate across the whole tensor, or along a specific dimension using the `dim` argument (equivalent to NumPy's `axis`):

```python
data = torch.tensor([[1, 2],
                     [5, 3],
                     [4, 6]])

data.sum()           # scalar — sum of every element
data.max(dim=0)      # max along rows  → one value per column
data.max(dim=1)      # max along cols  → one value per row
```

![Matrix aggregation along rows and columns](https://jalammar.github.io/images/numpy/numpy-matrix-aggregation-4.png)

---

## 6. Transposing and Reshaping

### Transpose

Swap rows and columns with `.T`:

```python
mat = torch.tensor([[1., 2., 3.],
                    [4., 5., 6.]])
mat.T     # shape changes from (2, 3) → (3, 2)
```

![Transposing a matrix](https://jalammar.github.io/images/numpy/numpy-transpose.png)

### Reshape

`reshape()` re-interprets the same data in a new shape. Use `-1` to let PyTorch infer one dimension automatically:

```python
x  = torch.arange(12)       # tensor([0, 1, ..., 11])

x2 = x.reshape(3, 4)        # explicit shape
x3 = x.reshape(3, -1)       # PyTorch infers the second dim → (3, 4)
```

![Reshaping a tensor](https://jalammar.github.io/images/numpy/numpy-reshape.png)

---

## 7. Yet More Dimensions (3D Tensors)

A 3D tensor adds a third axis. This is directly relevant to how PyTorch represents batches of data such as images `(C, H, W)`.

```python
torch.tensor([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
```

![A 3D tensor](/assets/module-0/torch_tensor_3d.png)

```python
# PyTorch convention: (C, H, W) — Channels first
tensor_a = torch.ones((2, 4, 3))
tensor_b = torch.zeros((2, 4, 3))
tensor_c = torch.randn((2, 4, 3))
```

![3D torch.ones, torch.zeros, torch.randn](/assets/module-0/3d_torch_ones_and_zeros.png)

> **Note:** PyTorch uses `(C, H, W)` ordering by default, which differs from many internet resources that assume `(H, W, C)` (e.g., OpenCV, PIL). Keep this in mind when loading images.

---

## 8. NumPy ↔ PyTorch Cheat Sheet

<style>
.cheat-table { width:100%; border-collapse:collapse; font-size:0.88rem; margin:16px 0; }
.cheat-table thead tr { background: var(--aast-navy, #132b61); color:#fff; }
.cheat-table thead th { padding:10px 14px; text-align:left; font-weight:600; letter-spacing:0.03em; }
.cheat-table tbody tr:nth-child(odd)  { background:#f7f9fc; }
.cheat-table tbody tr:nth-child(even) { background:#fff; }
.cheat-table tbody tr:hover { background:#eef2ff; }
.cheat-table td { padding:8px 14px; border-bottom:1px solid #e5e9f0; vertical-align:middle; }
.cheat-table td:last-child { color:#6b7280; font-style:italic; }
.cheat-table code { background:#eef0f8; border-radius:4px; padding:1px 5px; font-size:0.85em; }
</style>

<table class="cheat-table">
  <thead><tr><th>NumPy</th><th>PyTorch</th><th>Notes</th></tr></thead>
  <tbody>
    <tr><td><code>np.array(...)</code></td><td><code>torch.tensor(...)</code></td><td></td></tr>
    <tr><td><code>np.ones(shape)</code></td><td><code>torch.ones(shape)</code></td><td></td></tr>
    <tr><td><code>np.zeros(shape)</code></td><td><code>torch.zeros(shape)</code></td><td></td></tr>
    <tr><td><code>np.random.random(shape)</code></td><td><code>torch.rand(shape)</code></td><td>uniform [0, 1)</td></tr>
    <tr><td><code>a + b</code>, <code>a * b</code>, …</td><td>same</td><td>element-wise</td></tr>
    <tr><td><code>a @ b</code></td><td><code>a @ b</code> or <code>torch.matmul(a, b)</code></td><td>matrix multiply</td></tr>
    <tr><td><code>x.T</code></td><td><code>x.T</code> (2D) / <code>x.permute(...)</code> (nD)</td><td>transpose</td></tr>
    <tr><td><code>x.reshape(...)</code></td><td><code>x.reshape(...)</code> or <code>x.view(...)</code></td><td>view requires contiguous memory</td></tr>
    <tr><td><code>x.sum(axis=...)</code></td><td><code>x.sum(dim=...)</code></td><td>aggregation along a dimension</td></tr>
    <tr><td><code>x.mean(axis=...)</code></td><td><code>x.mean(dim=...)</code></td><td></td></tr>
  </tbody>
</table>

---

*Based on [A Visual Intro to NumPy and Data Representation](https://jalammar.github.io/visual-numpy/) by Jay Alammar. Images © Jay Alammar (CC BY-NC-SA 4.0).*