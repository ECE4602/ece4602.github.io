---
layout: page
title: "Module 0: Tensor Shapes & Linear Algebra Refresher"
permalink: /module-0-tensor-shapes/
---

Module 0 introduces the tensor view of data used throughout ECE4602. It reviews scalars, vectors, matrices, and higher-order tensors, focusing on order/shape, axis meaning, and simple PyTorch examples so you can read and reason about dimensions with confidence.

Table of Contents:

- [Core data objects and notation](#core-data-objects-and-notation)
  - [Scalars](#scalars)
  - [Vectors](#vectors)
  - [Matrices](#matrices)
  - [Tensors](#tensors)

## Core data objects and notation
### Scalars


Scalars are the simplest mathematical objects and can be viewed as **0th-order tensors**, meaning they have **no axes** (no dimensions). A scalar is an element of \\(R\\) (the set of real numbers) and is typically denoted by a lowercase letter such as \\(x\\) or \\(a\\). Unlike vectors or matrices, scalars represent a single numerical value.

#### Scalars in Euclidean space
Geometrically, a scalar lies in one-dimensional Euclidean space and can be visualized as a point on a number line. When embedded in \\(R^2\\), a scalar may be represented along a single axis (such as the \\(x\\)-axis or \\(y\\)-axis) while having no extent along the other axis. In both cases, the scalar still represents only one degree of freedom.

<div class='fig figcenter fighighlight'>
  <img src='/assets/module-0/scalar.gif' alt='Two scalar examples embedded in R^2, each shown along a single axis' class='img-center' style='max-width: 520px;' onerror="this.onerror=null;this.src='/assets/module-0/scalar.png';">
  <div class='figcaption'>Two scalar examples embedded in \\(R^2\\): each scalar has one degree of freedom and can be placed along a single axis.</div>
</div>

#### Discrete and continuous scalars
Scalars can take continuous values, such as \\(x \\in R\\), or discrete values. For example, a binary scalar may be defined as \\(x \\in \\{0,1\\}\\), which is commonly used to represent logical states, class labels, or on/off conditions in machine learning and computer science.

#### Scalars in practice
In practical applications, scalars represent individual quantities such as price, temperature, weight, or a binary decision variable. They also serve as coefficients that scale vectors and matrices, making them foundational to linear algebra, geometry, and numerical computation.



### Vectors
A **vector** is a first-order tensor: a single ordered list of values. A vector lives in \\(R^n\\), where \\(n\\) is its **dimensionality** (the number of components). For example, \\(R^2\\) is the 2D Euclidean space, while \\(R^n\\) generalizes this idea to higher dimensions. By convention, vectors are often written as bold lowercase letters (e.g., \\(\\mathbf{v}\\)) to distinguish them from scalars.

#### Practical interpretation
In data science and machine learning, vectors often represent structured data. For example, a house can be described by a feature vector

\\[
\\mathbf{v} = [\\text{space},\\ \\text{price},\\ \\text{number of rooms}]
\\]

This representation makes it easy to apply the same mathematical operations (scaling, comparison, optimization) to real-world attributes.

#### Vector instantiation in PyTorch
In PyTorch, vectors are **one-dimensional tensors**. For example, a vector \\(\\mathbf{v} \\in R^3\\):

```python
import torch

v = torch.tensor([120.0, 300000.0, 3.0])  # space, price, rooms
```

#### Vector length (dimensionality)
The dimensionality is the number of elements:

```python
n = v.shape[0]
# or
n = len(v)
```

Both return \\(n = 3\\), confirming \\(\\mathbf{v} \\in R^3\\).

#### Scalar-vector multiplication
Multiplying a vector by a scalar \\(\\alpha \\in R\\) scales every component:

```python
alpha = 0.5
scaled_v = alpha * v
```

Mathematically:

\\[
\\alpha\\mathbf{v} = [\\alpha v_1,\\ \\alpha v_2,\\ \\ldots,\\ \\alpha v_n]
\\]

#### Vector-vector operations
Given another vector \\(\\mathbf{u} \\in R^3\\):

```python
u = torch.tensor([100.0, 250000.0, 2.0])

dot_product = torch.dot(v, u)
elementwise_product = v * u
```

The **dot product** produces a scalar (often used as a similarity measure):

\\[
\\mathbf{v} \\cdot \\mathbf{u} = \\sum_{i=1}^{n} v_i u_i
\\]

The **elementwise (Hadamard) product** produces another vector in \\(R^n\\).

### Matrices
A **matrix** is a second-order tensor that lives in \\(R^{m \\times n}\\), where \\(m\\) is the number of rows and \\(n\\) is the number of columns. Matrices are often written using bold uppercase letters, such as \\(\\mathbf{W}\\). At a basic level, a matrix is an organized table of numbers designed to operate on vectors and produce new vectors in a structured way.

*Visualization idea:* a rectangular grid of numbers with labeled rows and columns.

#### Matrices as linear mappings between spaces
A matrix can be interpreted as a linear mapping between vector spaces:

\\[
\\mathbf{W}: R^n \\to R^m
\\]

When an input vector \\(\\mathbf{x} \\in R^n\\) is multiplied by a matrix \\(\\mathbf{W} \\in R^{m \\times n}\\), the result is an output vector:

\\[
\\mathbf{y} = \\mathbf{W}\\mathbf{x}, \\quad \\mathbf{y} \\in R^m
\\]

This means the matrix systematically transforms vectors from one space into another while preserving linear structure.

*Visualization idea:* an arrow in \\(R^2\\) being transformed into a new arrow.

#### Matrix-vector multiplication as linear combinations
Matrix-vector multiplication can be understood as many linear combinations performed in parallel. Component-wise,

\\[
y_i = \\sum_{j=1}^{n} w_{ij} x_j
\\]

Key ideas:
- Each row of the matrix defines one linear combination.
- All rows act on the same input vector.
- Multiple outputs are produced simultaneously.

*Visualization idea:* rows as separate weighting lines combining the same input features.

#### Geometric interpretation
Geometrically, matrices describe how vectors (and even entire spaces) are transformed. In two dimensions, a matrix may scale vectors, rotate them, reflect them, or shear them. In higher dimensions, the same intuition applies: a matrix reshapes space in a consistent and predictable manner.

*Visualization idea:* a square grid before and after transformation.

#### Columns, basis directions, and span
Another useful interpretation comes from the **columns** of a matrix. Each column represents where a basis vector is sent by the transformation. Because any vector can be expressed as a linear combination of basis vectors, the output of the matrix can be understood as applying the same combination to these transformed basis directions.

The set of all vectors that can be produced this way is determined by the **span** of the matrix columns, which describes the range of possible outputs.

*Visualization idea:* standard basis vectors mapped to new directions.

#### Why matrices matter
Matrices are fundamental because they represent many linear relationships compactly, enable efficient computation in high-dimensional spaces, and provide a unified framework for transforming data.

### Tensors
A **tensor** is a mathematical object that generalizes scalars, vectors, and matrices to higher dimensions. In this framework, a scalar is a 0th-order tensor, a vector is a 1st-order tensor, and a matrix is a 2nd-order tensor. Tensors of order three or higher are referred to as higher-order tensors. In general, a tensor lives in \\(R^{d_1 \\times d_2 \\times \\cdots \\times d_k}\\), where each \\(d_i\\) represents the size along one axis.

Examples:
- A single temperature value -> scalar (0th-order tensor)
- A house feature list `[size, price, rooms]` -> vector
- A table of house data -> matrix
- A batch of images -> 4D tensor (e.g., `(B, C, H, W)`)

*Visualization idea:* scalar -> vector -> matrix -> stacked matrices.

#### Tensor order and shape
The **order** (or rank) of a tensor is the number of axes it has, while the **shape** specifies the size along each axis. For example, a tensor with shape `(3, 4, 5)` is a third-order tensor with three axes.

Examples:
- Shape `(5,)` -> vector with 5 elements
- Shape `(10, 3)` -> 10 samples, 3 features each
- Shape `(32, 3, 64, 64)` -> 32 color images of size `64 x 64`

PyTorch example:
```python
import torch

x = torch.randn(10, 3)
print(x.shape)  # torch.Size([10, 3])
```

*Visualization idea:* axes labeled with sizes on a block diagram.

#### Tensors as a unified data representation
Tensors provide a single representation for all numerical data. Whether the data is a single value, a vector of features, or a batch of structured samples, it can always be represented as a tensor.

```python
import torch

# Scalar
a = torch.tensor(2.5)

# Vector
v = torch.tensor([1.0, 2.0, 3.0])

# Matrix
W = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]])

# 3D tensor
T = torch.randn(4, 5, 6)
```

All of these objects are tensors; the difference lies only in their order and shape.

*Visualization idea:* different tensor orders drawn as increasingly complex containers.

#### Axes, indexing, and interpretation
Each axis of a tensor has a semantic meaning determined by how the data is organized. Accessing elements requires one index per axis, and slicing along axes allows subsets of data to be selected.

Example: for a tensor of shape `(10, 3)`:
- Axis 0 -> samples
- Axis 1 -> features

```python
import torch

x = torch.randn(10, 3)

first_sample = x[0]      # shape: (3,)
second_feature = x[:, 1] # shape: (10,)
```

Understanding axis meaning is crucial for working correctly with tensor operations.

*Visualization idea:* slices highlighted along different axes.

#### Why tensors matter
Tensors make it possible to work with large, structured datasets efficiently. They allow operations to be applied over entire collections of data at once and serve as the common data structure for numerical computation frameworks.

Example: a batch of 100 images can be processed simultaneously because they are stored as a single tensor rather than 100 separate images.
