# Lesson 1: Python Basics for AI

**Duration:** 1 hour | **Difficulty:** Beginner | **Prerequisites:** None

## Learning Objectives

By the end of this lesson, you will:

- Understand Python variables and data types (int, float, string, bool)
- Use basic operators for calculations
- Format strings for logging training metrics
- Write readable code with comments
- Run your first AI-related Python script

## Why This Matters for AI

Every AI script starts with variables and basic operations. You'll use:

- **Floats** for model weights, loss values, and accuracy scores
- **Integers** for epoch counts, batch sizes, and indexing
- **Strings** for logging experiment results and model names
- **Operators** for calculating metrics and gradients

## Interactive Notebook

Launch the hands-on notebook to code along with this lesson:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/01-basics.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/01-basics.ipynb)

---

## 1. Variables and Data Types

### What is a Variable?

A variable is a named container that stores data. Think of it like a labeled box where you put values.

```python
# Creating variables
learning_rate = 0.001
num_epochs = 10
model_name = "ResNet50"
is_training = True
```

### Python Data Types for AI

| Type | Description | AI Use Case | Example |
|------|-------------|-------------|---------|
| `int` | Whole numbers | Epoch count, batch size | `epochs = 100` |
| `float` | Decimal numbers | Loss, accuracy, learning rate | `loss = 0.345` |
| `str` | Text | Model names, file paths | `path = "model.pth"` |
| `bool` | True/False | Training flags, conditions | `use_gpu = True` |

### Why Floats Matter in Neural Networks

```python
# Integer division vs float division
correct_predictions = 85
total_predictions = 100

# Wrong: Integer division loses precision
accuracy_wrong = correct_predictions // total_predictions
print(f"Integer division: {accuracy_wrong}")  # Output: 0 ❌

# Right: Float division preserves precision
accuracy_correct = correct_predictions / total_predictions
print(f"Float division: {accuracy_correct}")  # Output: 0.85 ✅
```

**Key Insight:** Always use `/` (float division) for metrics, not `//` (integer division).

---

## 2. Basic Operators

### Arithmetic Operators

These are essential for calculating loss, accuracy, and gradients.

```python
# Basic math operations
a = 10
b = 3

print(f"Addition: {a + b}")        # 13
print(f"Subtraction: {a - b}")     # 7
print(f"Multiplication: {a * b}")  # 30
print(f"Division: {a / b}")        # 3.3333...
print(f"Floor Division: {a // b}") # 3
print(f"Modulus: {a % b}")         # 1
print(f"Exponentiation: {a ** b}") # 1000
```

### AI Example: Calculating Model Accuracy

```python
# Training metrics
correct = 85
total = 100

# Calculate accuracy
accuracy = correct / total
print(f"Model Accuracy: {accuracy}")  # 0.85

# Convert to percentage
accuracy_percent = accuracy * 100
print(f"Accuracy: {accuracy_percent}%")  # 85.0%
```

### AI Example: Learning Rate Decay

```python
# Initial learning rate
initial_lr = 0.1
decay_factor = 0.9
epoch = 5

# Calculate decayed learning rate
current_lr = initial_lr * (decay_factor ** epoch)
print(f"Learning rate at epoch {epoch}: {current_lr:.6f}")
# Output: 0.059049
```

---

## 3. String Formatting (F-Strings)

String formatting is crucial for logging training progress and debugging.

### Basic F-Strings

```python
epoch = 5
loss = 0.4523
accuracy = 0.8912

# F-string formatting
print(f"Epoch {epoch}: Loss = {loss}, Accuracy = {accuracy}")
# Output: Epoch 5: Loss = 0.4523, Accuracy = 0.8912
```

### Advanced Formatting for AI Logs

```python
epoch = 10
train_loss = 0.04567
val_loss = 0.05234
accuracy = 0.89123

# Control decimal places
print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {accuracy:.2%}")
# Output: Epoch 10 | Train Loss: 0.0457 | Val Loss: 0.0523 | Acc: 89.12%
```

**Format Specifiers:**

- `:02d` - Integer with 2 digits, zero-padded (e.g., 01, 02, 10)
- `:.4f` - Float with 4 decimal places
- `:.2%` - Percentage with 2 decimal places
- `:.2e` - Scientific notation (e.g., 4.50e-03)

---

## 4. Comments and Code Readability

Comments explain what your code does. Essential for collaboration and debugging.

```python
# This is a single-line comment

# Calculate model accuracy
correct = 85  # Number of correct predictions
total = 100   # Total predictions
accuracy = correct / total  # Accuracy between 0 and 1

# Multi-line comment (using triple quotes)
"""
This function trains a neural network.
It takes epochs and learning rate as parameters.
Returns the trained model and loss history.
"""
```

**Best Practice for AI Code:**

```python
# Good: Explains WHY, not just WHAT
learning_rate = 0.001  # Start with small LR to avoid overshooting

# Bad: States the obvious
learning_rate = 0.001  # Set learning rate to 0.001
```

---

## 5. Hands-On Exercise

Now it's your turn to practice! Complete this exercise in the Colab notebook.

### Challenge: Calculate Training Metrics

```python
# Given data from a training run
epoch = 15
correct_train = 850
total_train = 1000
correct_val = 180
total_val = 200
time_taken = 45.7  # seconds

# TODO: Calculate the following:
# 1. Training accuracy (as a decimal)
# 2. Validation accuracy (as a percentage)
# 3. Average time per epoch (assuming this is epoch 15)

# Your code here:
train_accuracy = None  # Replace None
val_accuracy = None    # Replace None
avg_time = None        # Replace None

# Print results using f-strings with proper formatting
# Expected output format:
# Epoch 15 | Train Acc: 0.8500 | Val Acc: 90.00% | Avg Time: 3.05s
```

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
# Calculate metrics
train_accuracy = correct_train / total_train
val_accuracy = (correct_val / total_val) * 100
avg_time = time_taken / epoch

# Print formatted results
print(f"Epoch {epoch:02d} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.2f}% | Avg Time: {avg_time:.2f}s")
# Output: Epoch 15 | Train Acc: 0.8500 | Val Acc: 90.00% | Avg Time: 3.05s
```

</details>

---

## Reference Video

For additional visual learning, watch this curated reference video (optional):

<div class="video-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/kqtD5dpn9C8" title="Python Variables and Data Types" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

**Note:** The video provides supplementary content. The primary lesson is on this page and in the Colab notebook.

---

## Key Takeaways

- **Variables** store data with descriptive names (`learning_rate`, not `x`)
- **Use `/` not `//`** for accuracy calculations to preserve decimals
- **F-strings** are the modern way to format logs: `f"Epoch {i}: Loss = {loss:.4f}"`
- **Comments** explain WHY, not just WHAT
- **Float precision matters** in AI - always use `float` for metrics

---

## Quiz

Test your understanding before moving on:

[Take the Lesson 1 Quiz →](quizzes.md#lesson-1)

**Quick Check:**

1. What is the output of `10 // 3`?
2. What is the output of `10 / 3`?
3. How do you format a float to 2 decimal places in an f-string?
4. What data type should you use for learning rate?

---

## What's Next?

Congratulations on completing Lesson 1! You now understand the basic building blocks of Python for AI.

**Next Lesson:** [Lesson 2 - Lists and Indexing →](02-lists.md)

In Lesson 2, you'll learn how to work with lists (the precursor to NumPy arrays), understand indexing and slicing, and create batches of training data.

---

**Resources:**

- [Python Official Tutorial](https://docs.python.org/3/tutorial/introduction.html)
- [Real Python: Variables and Data Types](https://realpython.com/python-variables/)
- [Colab Tips and Tricks](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)

---

[← Back to Course Home](index.md) | [Next: Lesson 2 →](02-lists.md)
