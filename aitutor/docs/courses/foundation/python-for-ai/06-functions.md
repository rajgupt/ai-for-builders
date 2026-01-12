# Lesson 6: Functions and Modules

**Duration:** 1 hour | **Difficulty:** Intermediate | **Prerequisites:** Lessons 1-5

## Learning Objectives

By the end of this lesson, you will:

- Write custom functions for AI tasks
- Use parameters, return values, and default arguments
- Create lambda functions for quick operations
- Import and use AI libraries (PyTorch, TensorFlow)
- Read and understand library documentation
- Write docstrings for your functions

## Why This Matters for AI

Functions are everywhere in AI:

- **Custom loss functions** - Define how models learn
- **Data preprocessing** - Reusable transformation pipelines
- **Model architectures** - Functions that build networks
- **Metrics** - Accuracy, precision, recall calculations
- **Libraries** - PyTorch, TensorFlow, scikit-learn are all functions!

**Writing good functions = Writing production-ready AI code**

## Interactive Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/06-functions.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/06-functions.ipynb)

---

## 1. Defining Functions

### Basic Syntax

```python
def function_name(parameters):
    """Docstring: what the function does"""
    # Function body
    return result
```

### Simple Example

```python
def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

message = greet("Alice")
print(message)  # Hello, Alice!
```

### AI Example: Calculate Accuracy

```python
def calculate_accuracy(predictions, labels):
    """
    Calculate classification accuracy.
    
    Args:
        predictions: List/array of predicted labels
        labels: List/array of true labels
    
    Returns:
        Accuracy as float between 0 and 1
    """
    correct = sum([p == l for p, l in zip(predictions, labels)])
    total = len(labels)
    return correct / total

# Usage
preds = [0, 1, 1, 0, 1]
labels = [0, 1, 0, 0, 1]
acc = calculate_accuracy(preds, labels)
print(f"Accuracy: {acc:.2%}")  # 80.00%
```

---

## 2. Parameters and Return Values

### Multiple Parameters

```python
def train_model(learning_rate, batch_size, num_epochs):
    """Simulate model training."""
    print(f"Training with LR={learning_rate}, Batch={batch_size}, Epochs={num_epochs}")
    return 0.95  # Simulated accuracy

accuracy = train_model(0.001, 32, 10)
print(f"Final accuracy: {accuracy:.2%}")
```

### Default Arguments

```python
def create_config(lr=0.001, batch_size=32, epochs=10, optimizer='adam'):
    """Create training configuration with defaults."""
    return {
        'learning_rate': lr,
        'batch_size': batch_size,
        'num_epochs': epochs,
        'optimizer': optimizer
    }

# Use defaults
config1 = create_config()
print(config1)

# Override specific values
config2 = create_config(lr=0.01, epochs=50)
print(config2)
```

### Multiple Return Values

```python
def get_metrics(predictions, labels):
    """Calculate multiple metrics."""
    correct = sum([p == l for p, l in zip(predictions, labels)])
    total = len(labels)
    accuracy = correct / total
    loss = 1 - accuracy  # Simplified
    
    return accuracy, loss, correct, total

acc, loss, correct, total = get_metrics([0, 1, 1], [0, 1, 0])
print(f"Acc: {acc:.2%}, Loss: {loss:.4f}, Correct: {correct}/{total}")
```

---

## 3. AI Function Examples

### Data Normalization

```python
import numpy as np

def normalize_data(data, method='minmax'):
    """
    Normalize data using different methods.
    
    Args:
        data: NumPy array to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized data
    """
    if method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'zscore':
        return (data - data.mean()) / data.std()
    else:
        raise ValueError(f"Unknown method: {method}")

# Usage
pixels = np.array([0, 50, 100, 150, 200, 255])
normalized = normalize_data(pixels, method='minmax')
print(f"Normalized: {normalized}")
```

### Train/Test Split

```python
def split_dataset(data, train_ratio=0.8):
    """Split dataset into train and test sets."""
    split_idx = int(len(data) * train_ratio)
    train = data[:split_idx]
    test = data[split_idx:]
    return train, test

dataset = list(range(100))
train, test = split_dataset(dataset, train_ratio=0.7)
print(f"Train: {len(train)}, Test: {len(test)}")
```

### Batch Creation

```python
def create_batches(data, batch_size):
    """Create batches from dataset."""
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batches.append(batch)
    return batches

data = list(range(50))
batches = create_batches(data, batch_size=16)
print(f"Created {len(batches)} batches")
```

---

## 4. Lambda Functions

### Basic Lambda

```python
# Regular function
def square(x):
    return x ** 2

# Lambda (anonymous function)
square_lambda = lambda x: x ** 2

print(square(5))        # 25
print(square_lambda(5)) # 25
```

### AI Example: Apply Activation

```python
import numpy as np

# ReLU activation
relu = lambda x: np.maximum(0, x)

# Sigmoid activation
sigmoid = lambda x: 1 / (1 + np.exp(-x))

x = np.array([-2, -1, 0, 1, 2])
print(f"ReLU: {relu(x)}")
print(f"Sigmoid: {sigmoid(x)}")
```

### Use with map and filter

```python
# Normalize a list
data = [1, 2, 3, 4, 5]
normalized = list(map(lambda x: x / max(data), data))
print(f"Normalized: {normalized}")

# Filter high confidence
confidences = [0.2, 0.8, 0.9, 0.3, 0.95]
high_conf = list(filter(lambda x: x > 0.7, confidences))
print(f"High confidence: {high_conf}")
```

---

## 5. Importing Modules

### Built-in Modules

```python
import math
import random

print(f"Pi: {math.pi}")
print(f"Square root of 16: {math.sqrt(16)}")
print(f"Random number: {random.random()}")
```

### AI Libraries

```python
# NumPy
import numpy as np
arr = np.array([1, 2, 3])

# Import specific functions
from numpy import mean, std
print(f"Mean: {mean(arr)}, Std: {std(arr)}")

# Import with alias
import matplotlib.pyplot as plt
# Now use plt.plot(), plt.show(), etc.
```

### AI Example: Using PyTorch

```python
import torch

# Create tensors (like NumPy arrays)
x = torch.randn(3, 4)  # Random tensor
y = torch.zeros(2, 3)  # Zeros

print(f"Tensor shape: {x.shape}")
print(f"Tensor:\n{x}")

# PyTorch works like NumPy!
z = x * 2
print(f"Multiplied:\n{z}")
```

---

## Reference Video

<div class="video-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/NSbOtYzIQI0" title="Python Functions" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **Functions** make code reusable: `def function_name(params):`
- **Docstrings** explain what functions do
- **Default arguments** provide sensible defaults
- **Lambda functions** are quick one-liners
- **Import libraries** to use AI frameworks
- **PyTorch/TensorFlow** work like NumPy

---

## Quiz

[Take the Lesson 6 Quiz →](quizzes.md#lesson-6)

---

## What's Next?

**Next Lesson:** [Lesson 7 - Mini-Project: MNIST Classifier →](07-mini-project.md)

Put everything together by building a complete ML project!

---

[← Lesson 5: NumPy](05-numpy.md) | [Course Home](index.md) | [Lesson 7: Project →](07-mini-project.md)
