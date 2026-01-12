# Lesson 5: NumPy Essentials

**Duration:** 1.5 hours | **Difficulty:** Intermediate | **Prerequisites:** Lessons 1-4

## Learning Objectives

By the end of this lesson, you will:

- Understand why NumPy is essential for AI
- Create and manipulate NumPy arrays
- Master array shapes and dimensions
- Perform element-wise operations and broadcasting
- Use NumPy for tensor operations
- Prepare for PyTorch and TensorFlow

## Why This Matters for AI

**NumPy is the foundation of all AI libraries.** PyTorch tensors and TensorFlow tensors work almost identically to NumPy arrays. Understanding NumPy is essential because:

- **Tensors are arrays** - Every neural network uses multi-dimensional arrays
- **Speed** - NumPy is 100x faster than Python lists for numerical ops
- **Broadcasting** - Automatic shape matching for operations
- **Image data** - Images are 3D arrays (height, width, channels)
- **Matrix operations** - Neural networks are matrix multiplications

**Master NumPy = Master 80% of AI programming**

## Interactive Notebook

Launch the hands-on notebook to code along with this lesson:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/05-numpy.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/05-numpy.ipynb)

---

## 1. NumPy vs Python Lists

### Why Not Use Lists?

```python
import numpy as np
import time

# Python list
python_list = list(range(1000000))
start = time.time()
result_list = [x * 2 for x in python_list]
list_time = time.time() - start

# NumPy array
numpy_array = np.arange(1000000)
start = time.time()
result_array = numpy_array * 2
numpy_time = time.time() - start

print(f"Python list time: {list_time:.4f}s")
print(f"NumPy array time: {numpy_time:.4f}s")
print(f"NumPy is {list_time / numpy_time:.1f}x faster!")
```

**Key Differences:**

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Speed | Slow | Fast (100x+) |
| Memory | High | Low |
| Operations | Element by element | Vectorized |
| AI Use | Rare | Universal |

---

## 2. Creating Arrays

### From Python Lists

```python
import numpy as np

# 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print(f"1D array: {arr_1d}")
print(f"Shape: {arr_1d.shape}")  # (5,)
print(f"Type: {type(arr_1d)}")    # <class 'numpy.ndarray'>

# 2D array (matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D array:\n{arr_2d}")
print(f"Shape: {arr_2d.shape}")  # (2, 3) - 2 rows, 3 columns
```

### Special Arrays

```python
# Zeros
zeros = np.zeros((3, 4))  # 3 rows, 4 columns of zeros
print(f"Zeros:\n{zeros}")

# Ones
ones = np.ones((2, 3))
print(f"\nOnes:\n{ones}")

# Identity matrix
identity = np.eye(3)  # 3x3 identity matrix
print(f"\nIdentity:\n{identity}")

# Range
range_array = np.arange(0, 10, 2)  # 0 to 10, step 2
print(f"\nRange: {range_array}")  # [0 2 4 6 8]

# Linspace (evenly spaced)
linspace = np.linspace(0, 1, 5)  # 5 values from 0 to 1
print(f"Linspace: {linspace}")  # [0.   0.25 0.5  0.75 1.  ]
```

### Random Arrays

```python
# Random values between 0 and 1
random = np.random.rand(3, 4)  # 3x4 array
print(f"Random:\n{random}")

# Random integers
random_int = np.random.randint(0, 10, size=(3, 3))  # 0-9
print(f"\nRandom integers:\n{random_int}")

# Random normal distribution
random_normal = np.random.randn(3, 3)  # Mean=0, Std=1
print(f"\nRandom normal:\n{random_normal}")
```

### AI Example: Initialize Model Weights

```python
# Neural network layer: 784 inputs → 128 outputs
input_size = 784
output_size = 128

# Xavier initialization (common in deep learning)
weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
biases = np.zeros(output_size)

print(f"Weights shape: {weights.shape}")  # (784, 128)
print(f"Biases shape: {biases.shape}")    # (128,)
print(f"Weight mean: {weights.mean():.6f}")
print(f"Weight std: {weights.std():.6f}")
```

---

## 3. Array Shapes and Dimensions

### Understanding Shape

```python
# 1D: Vector (1 dimension)
vec = np.array([1, 2, 3])
print(f"Vector shape: {vec.shape}")  # (3,)
print(f"Dimensions: {vec.ndim}")      # 1

# 2D: Matrix (2 dimensions)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nMatrix shape: {matrix.shape}")  # (2, 3)
print(f"Dimensions: {matrix.ndim}")        # 2

# 3D: Tensor (3 dimensions)
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\nTensor shape: {tensor.shape}")  # (2, 2, 2)
print(f"Dimensions: {tensor.ndim}")        # 3
```

### AI Example: Image Data Shapes

```python
# Single grayscale image: (height, width)
gray_image = np.random.rand(28, 28)
print(f"Grayscale image: {gray_image.shape}")  # (28, 28)

# Single RGB image: (height, width, channels)
rgb_image = np.random.rand(224, 224, 3)
print(f"RGB image: {rgb_image.shape}")  # (224, 224, 3)

# Batch of images: (batch_size, height, width, channels)
batch = np.random.rand(32, 224, 224, 3)
print(f"Batch shape: {batch.shape}")  # (32, 224, 224, 3)

# Calculate memory
memory_mb = batch.nbytes / (1024 ** 2)
print(f"Memory: {memory_mb:.2f} MB")
```

### Reshaping Arrays

```python
# Create 1D array
arr = np.arange(12)
print(f"Original: {arr.shape}")  # (12,)

# Reshape to 2D
arr_2d = arr.reshape(3, 4)  # 3 rows, 4 columns
print(f"Reshaped 2D:\n{arr_2d}")
print(f"Shape: {arr_2d.shape}")  # (3, 4)

# Reshape to 3D
arr_3d = arr.reshape(2, 2, 3)
print(f"\nReshaped 3D shape: {arr_3d.shape}")  # (2, 2, 3)

# Flatten back to 1D
flattened = arr_3d.reshape(-1)  # -1 means "figure it out"
print(f"Flattened: {flattened.shape}")  # (12,)
```

### AI Example: Flatten Images for MLP

```python
# Batch of 28x28 grayscale images
images = np.random.rand(64, 28, 28)  # 64 images
print(f"Original shape: {images.shape}")

# Flatten each image for fully connected layer
flattened = images.reshape(64, -1)  # -1 = 28*28 = 784
print(f"Flattened shape: {flattened.shape}")  # (64, 784)
```

---

## 4. Indexing and Slicing

### Basic Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Single element
print(f"First: {arr[0]}")   # 10
print(f"Last: {arr[-1]}")   # 50

# Slicing
print(f"First 3: {arr[:3]}")  # [10 20 30]
print(f"Last 2: {arr[-2:]}")  # [40 50]
```

### 2D Indexing

```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix:\n{matrix}")

# Single element
print(f"\nElement [0, 0]: {matrix[0, 0]}")  # 1
print(f"Element [1, 2]: {matrix[1, 2]}")    # 6

# Row
print(f"First row: {matrix[0]}")      # [1 2 3]
print(f"First row: {matrix[0, :]}")  # [1 2 3] (explicit)

# Column
print(f"First column: {matrix[:, 0]}")  # [1 4 7]

# Subarray
print(f"Top-left 2x2:\n{matrix[:2, :2]}")
```

### Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Boolean mask
mask = arr > 3
print(f"Mask: {mask}")  # [False False False True True True]

# Filter values
filtered = arr[mask]
print(f"Values > 3: {filtered}")  # [4 5 6]

# One-liner
print(f"Values > 3: {arr[arr > 3]}")  # [4 5 6]
```

### AI Example: Filter High-Confidence Predictions

```python
predictions = np.array([0.2, 0.8, 0.9, 0.3, 0.95, 0.6])
threshold = 0.7

high_conf = predictions[predictions > threshold]
print(f"High confidence predictions: {high_conf}")
# Output: [0.8  0.9  0.95]

# Get indices
high_conf_indices = np.where(predictions > threshold)[0]
print(f"Indices: {high_conf_indices}")  # [1 2 4]
```

---

## 5. Array Operations

### Element-wise Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Arithmetic
print(f"a + b = {a + b}")  # [11 22 33 44]
print(f"a * b = {a * b}")  # [10 40 90 160]
print(f"b / a = {b / a}")  # [10. 20. 30. 40.]

# Scalar operations
print(f"a * 2 = {a * 2}")      # [2 4 6 8]
print(f"a + 10 = {a + 10}")    # [11 12 13 14]
print(f"a ** 2 = {a ** 2}")    # [1 4 9 16]
```

### Mathematical Functions

```python
arr = np.array([1, 4, 9, 16, 25])

print(f"Square root: {np.sqrt(arr)}")  # [1. 2. 3. 4. 5.]
print(f"Exponential: {np.exp(arr[:3])}")
print(f"Log: {np.log(arr)}")

# Aggregations
print(f"Sum: {arr.sum()}")      # 55
print(f"Mean: {arr.mean()}")    # 11.0
print(f"Max: {arr.max()}")      # 25
print(f"Min: {arr.min()}")      # 1
print(f"Std: {arr.std():.2f}")  # 8.60
```

### AI Example: Normalize Data

```python
# Raw pixel values (0-255)
pixels = np.array([0, 50, 100, 150, 200, 255])

# Min-max normalization to [0, 1]
normalized = (pixels - pixels.min()) / (pixels.max() - pixels.min())
print(f"Normalized: {normalized}")
# Output: [0.   0.2  0.39 0.59 0.78 1.  ]

# Z-score normalization (mean=0, std=1)
z_normalized = (pixels - pixels.mean()) / pixels.std()
print(f"Z-normalized: {z_normalized}")
```

---

## 6. Broadcasting

### What is Broadcasting?

Broadcasting allows operations on arrays of different shapes.

```python
# Scalar broadcasting
arr = np.array([1, 2, 3])
result = arr + 10  # 10 is "broadcast" to [10, 10, 10]
print(f"arr + 10 = {result}")  # [11 12 13]

# 1D to 2D broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])

result = matrix + vector  # vector is broadcast to each row
print(f"Matrix + vector:\n{result}")
# Output:
# [[11 22 33]
#  [14 25 36]]
```

### AI Example: Add Bias to Batch

```python
# Batch of outputs: (batch_size, num_neurons)
batch_outputs = np.random.randn(32, 10)  # 32 samples, 10 neurons
biases = np.random.randn(10)  # 10 biases

# Add bias to each sample (broadcasting!)
result = batch_outputs + biases
print(f"Batch shape: {batch_outputs.shape}")
print(f"Bias shape: {biases.shape}")
print(f"Result shape: {result.shape}")  # (32, 10)
```

### AI Example: Normalize Batch

```python
# Batch of images: (batch_size, height, width, channels)
batch = np.random.rand(16, 28, 28, 3) * 255  # 0-255 range

# Calculate mean and std per channel
mean = batch.mean(axis=(0, 1, 2), keepdims=True)  # (1, 1, 1, 3)
std = batch.std(axis=(0, 1, 2), keepdims=True)

# Normalize (broadcasting!)
normalized = (batch - mean) / std
print(f"Normalized mean: {normalized.mean():.6f}")  # ~0
print(f"Normalized std: {normalized.std():.6f}")    # ~1
```

---

## 7. Matrix Operations

### Dot Product

```python
# Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot product: {dot}")

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.dot(A, B)  # or A @ B
print(f"Matrix multiplication:\n{C}")
```

### AI Example: Forward Pass

```python
# Inputs: (batch_size, input_features)
X = np.random.randn(64, 784)  # 64 samples, 784 features

# Weights: (input_features, output_features)
W = np.random.randn(784, 128)  # 784 → 128
b = np.random.randn(128)

# Forward pass: Y = X @ W + b
Y = np.dot(X, W) + b  # Broadcasting adds bias
print(f"Input shape: {X.shape}")   # (64, 784)
print(f"Weight shape: {W.shape}")  # (784, 128)
print(f"Output shape: {Y.shape}")  # (64, 128)
```

---

## 8. Hands-On Exercise

### Challenge: Image Preprocessing Pipeline

```python
# TODO: Create a complete image preprocessing pipeline
# 1. Create a batch of 32 random "images" (32, 28, 28, 1)
# 2. Normalize pixel values to [0, 1]
# 3. Calculate mean and std across batch
# 4. Apply z-score normalization
# 5. Reshape for fully connected layer (32, 784)

# Your code here:
```

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np

# 1. Create batch of images
batch = np.random.randint(0, 256, size=(32, 28, 28, 1))
print(f"Original shape: {batch.shape}")
print(f"Value range: [{batch.min()}, {batch.max()}]")

# 2. Normalize to [0, 1]
batch_norm = batch / 255.0
print(f"\nNormalized range: [{batch_norm.min():.2f}, {batch_norm.max():.2f}]")

# 3. Calculate statistics
mean = batch_norm.mean()
std = batch_norm.std()
print(f"Mean: {mean:.4f}, Std: {std:.4f}")

# 4. Z-score normalization
batch_z = (batch_norm - mean) / std
print(f"\nZ-normalized mean: {batch_z.mean():.6f}")
print(f"Z-normalized std: {batch_z.std():.6f}")

# 5. Reshape for MLP
batch_flat = batch_z.reshape(32, -1)
print(f"\nFlattened shape: {batch_flat.shape}")  # (32, 784)
```

</details>

---

## Reference Video

For additional visual learning, watch this curated reference video (optional):

<div class="video-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/QUT1VHiLmmI" title="NumPy Tutorial" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

**Note:** The video provides supplementary content. The primary lesson is on this page and in the Colab notebook.

---

## Key Takeaways

- **NumPy is 100x faster** than Python lists for numerical ops
- **Arrays have shapes**: (batch, height, width, channels)
- **Broadcasting** allows ops on different-shaped arrays
- **Vectorized ops** replace loops: `arr * 2` instead of `[x*2 for x in arr]`
- **Reshape** converts between dimensions: `arr.reshape(new_shape)`
- **All AI frameworks** (PyTorch, TensorFlow) work like NumPy

---

## Quiz

Test your understanding before moving on:

[Take the Lesson 5 Quiz →](quizzes.md#lesson-5)

**Quick Check:**

1. Why is NumPy faster than Python lists?
2. What does array shape (32, 224, 224, 3) represent?
3. What is broadcasting?
4. How do you reshape an array?

---

## What's Next?

Congratulations on completing Lesson 5! You now understand NumPy - the foundation of AI programming.

**Next Lesson:** [Lesson 6 - Functions and Modules →](06-functions.md)

In Lesson 6, you'll learn to write reusable functions and import AI libraries like PyTorch and TensorFlow.

---

**Resources:**

- [NumPy Official Documentation](https://numpy.org/doc/stable/)
- [NumPy for AI (Real Python)](https://realpython.com/numpy-array-programming/)
- [Broadcasting Explained](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

[← Lesson 4: Loops](04-loops.md) | [Course Home](index.md) | [Lesson 6: Functions →](06-functions.md)
