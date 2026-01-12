# Lesson 2: Lists and Indexing

**Duration:** 1 hour | **Difficulty:** Beginner | **Prerequisites:** Lesson 1

## Learning Objectives

By the end of this lesson, you will:

- Create and manipulate lists for storing training data
- Use indexing and slicing to access data
- Apply list methods for data transformation
- Use list comprehensions for efficient batch processing
- Work with nested lists (multi-dimensional data)

## Why This Matters for AI

Lists are Python's basic data containers. Before you work with NumPy arrays or PyTorch tensors, you need to understand lists:

- **Training data** - Store examples and labels
- **Batch processing** - Split datasets into batches
- **Train/test splits** - Divide data for evaluation
- **Data preprocessing** - Transform and filter datasets
- **Shape understanding** - Foundation for tensor dimensions

## Interactive Notebook

Launch the hands-on notebook to code along with this lesson:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/02-lists.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/02-lists.ipynb)

---

## 1. Creating Lists

### What is a List?

A list is an ordered collection of items. Think of it as a container that can hold multiple values.

```python
# Empty list
empty_list = []

# List of training accuracies
accuracies = [0.82, 0.85, 0.88, 0.91, 0.93]

# List of epoch numbers
epochs = [1, 2, 3, 4, 5]

# List of model names
models = ["ResNet", "VGG", "MobileNet"]

# Mixed types (not common in AI, but possible)
mixed = [100, 0.95, "model.pth", True]
```

### AI Example: Storing Training Metrics

```python
# Track loss values during training
train_losses = [0.89, 0.76, 0.65, 0.58, 0.52]
val_losses = [0.92, 0.81, 0.72, 0.68, 0.65]

# Store predictions from a model
predictions = [0, 1, 1, 0, 1, 0, 0, 1]  # Binary classification

# Store ground truth labels
labels = [0, 1, 0, 0, 1, 0, 1, 1]
```

---

## 2. Indexing: Accessing Elements

### Positive Indexing (From Start)

Lists use zero-based indexing. The first element is at index 0.

```python
losses = [0.89, 0.76, 0.65, 0.58, 0.52]

# Access individual elements
first_loss = losses[0]    # 0.89
second_loss = losses[1]   # 0.76
last_loss = losses[4]     # 0.52

print(f"First epoch loss: {first_loss}")
print(f"Final loss: {last_loss}")
```

### Negative Indexing (From End)

Negative indices count from the end of the list.

```python
losses = [0.89, 0.76, 0.65, 0.58, 0.52]

# Access from the end
last = losses[-1]      # 0.52 (last element)
second_last = losses[-2]  # 0.58 (second from end)

print(f"Most recent loss: {last}")
```

### AI Use Case: Getting Final Metrics

```python
# Training history
accuracies = [0.65, 0.72, 0.78, 0.82, 0.85, 0.88]

# Get initial and final accuracy
initial_acc = accuracies[0]
final_acc = accuracies[-1]

improvement = final_acc - initial_acc
print(f"Accuracy improved by {improvement:.2%}")
# Output: Accuracy improved by 23.00%
```

---

## 3. Slicing: Extracting Ranges

### Basic Slicing Syntax

`list[start:stop:step]`

- `start` - Index to begin (inclusive)
- `stop` - Index to end (exclusive)
- `step` - Interval between elements (default: 1)

```python
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Get first 5 elements
first_five = data[0:5]  # [0, 1, 2, 3, 4]
first_five = data[:5]   # Same result (start defaults to 0)

# Get last 3 elements
last_three = data[-3:]  # [7, 8, 9]

# Get middle elements
middle = data[3:7]      # [3, 4, 5, 6]

# Get every second element
evens = data[::2]       # [0, 2, 4, 6, 8]

# Reverse a list
reversed_data = data[::-1]  # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

### AI Example: Train/Test Split

```python
# Simulate a dataset with 100 samples
dataset = list(range(100))  # [0, 1, 2, ..., 99]

# Split 80% train, 20% test
split_index = int(0.8 * len(dataset))
train_data = dataset[:split_index]  # First 80 samples
test_data = dataset[split_index:]   # Last 20 samples

print(f"Training samples: {len(train_data)}")   # 80
print(f"Test samples: {len(test_data)}")        # 20
```

### AI Example: Getting Recent History

```python
# Loss history from 100 epochs
loss_history = [1.5 - (i * 0.01) for i in range(100)]

# Get last 10 epochs for plotting
recent_losses = loss_history[-10:]
print(f"Last 10 losses: {recent_losses}")
```

---

## 4. List Methods

### Essential Methods for AI

```python
# Starting with an empty list
losses = []

# append() - Add single element to end
losses.append(0.89)
losses.append(0.76)
print(losses)  # [0.89, 0.76]

# extend() - Add multiple elements
losses.extend([0.65, 0.58, 0.52])
print(losses)  # [0.89, 0.76, 0.65, 0.58, 0.52]

# insert() - Add at specific position
losses.insert(0, 1.20)  # Insert at beginning
print(losses)  # [1.20, 0.89, 0.76, 0.65, 0.58, 0.52]

# remove() - Remove first occurrence
losses.remove(1.20)
print(losses)  # [0.89, 0.76, 0.65, 0.58, 0.52]

# pop() - Remove and return element
last_loss = losses.pop()  # Removes last element
print(f"Removed: {last_loss}")  # 0.52
print(losses)  # [0.89, 0.76, 0.65, 0.58]

# len() - Get list length
num_epochs = len(losses)
print(f"Trained for {num_epochs} epochs")
```

### AI Example: Building Training History

```python
# Initialize empty training log
train_losses = []
val_losses = []

# Simulate training loop (5 epochs)
for epoch in range(5):
    # Simulate decreasing loss
    train_loss = 1.0 / (epoch + 1)
    val_loss = 1.1 / (epoch + 1)

    # Record metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

print(f"\nFinal training history: {train_losses}")
```

---

## 5. List Comprehensions

### Basic Syntax

List comprehensions provide a concise way to create lists.

```python
# Traditional way
squares = []
for i in range(10):
    squares.append(i ** 2)

# List comprehension (more Pythonic)
squares = [i ** 2 for i in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### With Conditions

```python
# Get only even squares
even_squares = [i ** 2 for i in range(10) if i % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]
```

### AI Example: Data Normalization

```python
# Raw pixel values (0-255 range)
pixel_values = [0, 50, 100, 150, 200, 255]

# Normalize to [0, 1] range
normalized = [pixel / 255.0 for pixel in pixel_values]
print(f"Normalized: {normalized}")
# Output: [0.0, 0.196, 0.392, 0.588, 0.784, 1.0]
```

### AI Example: Creating Batches

```python
# Full dataset
dataset = list(range(100))
batch_size = 16

# Create batches using list comprehension
batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

print(f"Number of batches: {len(batches)}")  # 7 batches
print(f"First batch size: {len(batches[0])}")  # 16
print(f"Last batch size: {len(batches[-1])}")  # 4 (remaining samples)
```

### AI Example: Filter Predictions

```python
# Model predictions and confidence scores
predictions = [0, 1, 1, 0, 1, 0, 1, 1]
confidences = [0.92, 0.65, 0.88, 0.95, 0.55, 0.78, 0.82, 0.91]

# Keep only high-confidence predictions (>0.8)
high_conf_preds = [pred for pred, conf in zip(predictions, confidences) if conf > 0.8]
print(f"High confidence predictions: {high_conf_preds}")
# Output: [0, 1, 0, 1, 1]
```

---

## 6. Nested Lists

### 2D Lists (Matrix-like Data)

```python
# 2D list representing a batch of samples
# Shape: [batch_size, features]
batch = [
    [0.5, 0.3, 0.8],  # Sample 1
    [0.2, 0.9, 0.4],  # Sample 2
    [0.7, 0.1, 0.6]   # Sample 3
]

# Access elements
first_sample = batch[0]      # [0.5, 0.3, 0.8]
first_feature = batch[0][0]  # 0.5

print(f"Batch shape: {len(batch)} samples x {len(batch[0])} features")
# Output: Batch shape: 3 samples x 3 features
```

### AI Example: Mini-batch of Images

```python
# Simulate a batch of 28x28 grayscale images (simplified)
# In reality, these would be NumPy arrays
batch_size = 2
height = 28
width = 28

# Create batch (nested lists)
batch = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(batch_size)]

print(f"Batch dimensions: [{len(batch)}, {len(batch[0])}, {len(batch[0][0])}]")
# Output: [2, 28, 28]
```

---

## 7. Hands-On Exercise

### Challenge: Create a Data Pipeline

Complete this exercise in the Colab notebook.

```python
# Dataset of 100 samples (0-99)
full_dataset = list(range(100))

# TODO: Complete the following tasks:
# 1. Split into 70% train, 15% validation, 15% test
# 2. Create batches of size 8 for training data
# 3. Print statistics about each split

# Your code here:
train_split = None
val_split = None
test_split = None

train_batches = None

# Print results
print(f"Train samples: {len(train_split)}")
print(f"Val samples: {len(val_split)}")
print(f"Test samples: {len(test_split)}")
print(f"Number of training batches: {len(train_batches)}")
```

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
# Calculate split indices
train_end = int(0.7 * len(full_dataset))
val_end = train_end + int(0.15 * len(full_dataset))

# Create splits
train_split = full_dataset[:train_end]
val_split = full_dataset[train_end:val_end]
test_split = full_dataset[val_end:]

# Create batches for training data
batch_size = 8
train_batches = [train_split[i:i+batch_size] for i in range(0, len(train_split), batch_size)]

# Print statistics
print(f"Train samples: {len(train_split)}")      # 70
print(f"Val samples: {len(val_split)}")          # 15
print(f"Test samples: {len(test_split)}")        # 15
print(f"Number of training batches: {len(train_batches)}")  # 9
print(f"First batch: {train_batches[0]}")
print(f"Last batch size: {len(train_batches[-1])}")  # 6
```

</details>

---

## Reference Video

For additional visual learning, watch this curated reference video (optional):

<div class="video-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/ohCDWZgNIU0" title="Python Lists Tutorial" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

**Note:** The video provides supplementary content. The primary lesson is on this page and in the Colab notebook.

---

## Key Takeaways

- **Lists** are ordered collections: `[1, 2, 3]`
- **Indexing** starts at 0: `list[0]` gets first element
- **Negative indexing** counts from end: `list[-1]` gets last element
- **Slicing** extracts ranges: `list[start:stop]`
- **List comprehensions** are concise: `[x*2 for x in data]`
- **Nested lists** represent batched data: `[[sample1], [sample2]]`

---

## Quiz

Test your understanding before moving on:

[Take the Lesson 2 Quiz →](quizzes.md#lesson-2)

**Quick Check:**

1. What does `data[3:7]` return?
2. How do you get the last element of a list?
3. What's the difference between `append()` and `extend()`?
4. How would you split a dataset 80/20 using slicing?

---

## What's Next?

Congratulations on completing Lesson 2! You now understand how to work with lists and manipulate training data.

**Next Lesson:** [Lesson 3 - Dictionaries and Data Structures →](03-dictionaries.md)

In Lesson 3, you'll learn about dictionaries (key-value pairs) for storing model configurations, hyperparameters, and training metrics.

---

**Resources:**

- [Python Lists Documentation](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)
- [Real Python: Lists and Tuples](https://realpython.com/python-lists-tuples/)
- [List Comprehensions Guide](https://realpython.com/list-comprehension-python/)

---

[← Lesson 1: Basics](01-basics.md) | [Course Home](index.md) | [Lesson 3: Dictionaries →](03-dictionaries.md)
