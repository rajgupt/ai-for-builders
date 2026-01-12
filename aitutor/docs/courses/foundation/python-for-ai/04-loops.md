# Lesson 4: Loops and Iteration

**Duration:** 1 hour | **Difficulty:** Beginner | **Prerequisites:** Lessons 1-3

## Learning Objectives

By the end of this lesson, you will:

- Write for loops to iterate over datasets
- Use while loops for training until convergence
- Apply `range()`, `enumerate()`, and `zip()` for efficient iteration
- Understand break and continue statements
- Implement training loop patterns

## Why This Matters for AI

Loops are the backbone of AI training:

- **Epoch loops** - Train for multiple passes through data
- **Batch iteration** - Process data in chunks
- **Early stopping** - Stop when validation stops improving
- **Data processing** - Transform datasets efficiently
- **Hyperparameter tuning** - Try multiple configurations

Every neural network training script uses nested loops!

## Interactive Notebook

Launch the hands-on notebook to code along with this lesson:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/04-loops.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/04-loops.ipynb)

---

## 1. For Loops Basics

### Simple For Loop

For loops iterate over sequences (lists, tuples, strings).

```python
# Loop over a list
losses = [0.89, 0.76, 0.65, 0.58, 0.52]

for loss in losses:
    print(f"Loss: {loss:.4f}")
```

**Output:**
```
Loss: 0.8900
Loss: 0.7600
Loss: 0.6500
Loss: 0.5800
Loss: 0.5200
```

### AI Example: Processing Predictions

```python
# Model predictions and labels
predictions = [0, 1, 1, 0, 1, 0, 0, 1]
labels = [0, 1, 0, 0, 1, 0, 1, 1]

# Count correct predictions
correct = 0
for pred, label in zip(predictions, labels):
    if pred == label:
        correct += 1

accuracy = correct / len(predictions)
print(f"Accuracy: {accuracy:.2%}")  # 75.00%
```

---

## 2. Range Function

### Basic Range

`range(stop)` - Numbers from 0 to stop-1

```python
# range(5) creates: 0, 1, 2, 3, 4
for i in range(5):
    print(f"Epoch {i + 1}")
```

### Range with Start and Stop

`range(start, stop)` - Numbers from start to stop-1

```python
# range(1, 6) creates: 1, 2, 3, 4, 5
for epoch in range(1, 6):
    print(f"Epoch {epoch}")
```

### Range with Step

`range(start, stop, step)` - Numbers with custom intervals

```python
# Every 5th epoch
for epoch in range(5, 51, 5):
    print(f"Saving checkpoint at epoch {epoch}")
```

### AI Example: Training Epochs

```python
num_epochs = 10
initial_lr = 0.1

for epoch in range(num_epochs):
    # Simulate learning rate decay
    current_lr = initial_lr * (0.9 ** epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}: Learning rate = {current_lr:.6f}")
```

---

## 3. Enumerate Function

### Why Enumerate?

`enumerate()` gives you both the index and value during iteration.

```python
# Without enumerate (manual counter)
losses = [0.89, 0.76, 0.65]
i = 0
for loss in losses:
    print(f"Epoch {i + 1}: Loss = {loss:.4f}")
    i += 1

# With enumerate (cleaner)
for i, loss in enumerate(losses):
    print(f"Epoch {i + 1}: Loss = {loss:.4f}")

# Start counting from 1
for epoch, loss in enumerate(losses, start=1):
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

### AI Example: Batch Processing

```python
# Simulate batches of data
batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

for batch_idx, batch in enumerate(batches):
    # Simulate processing
    batch_mean = sum(batch) / len(batch)

    print(f"Batch {batch_idx + 1}/{len(batches)}: Mean = {batch_mean:.2f}")
```

---

## 4. Zip Function

### Combining Multiple Lists

`zip()` combines multiple sequences element by element.

```python
# Training metrics from different sources
train_losses = [0.89, 0.76, 0.65]
val_losses = [0.92, 0.81, 0.72]
accuracies = [0.75, 0.82, 0.87]

for epoch, (train_loss, val_loss, acc) in enumerate(zip(train_losses, val_losses, accuracies), 1):
    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={acc:.2%}")
```

### AI Example: Comparing Predictions and Labels

```python
predictions = [0, 1, 1, 0, 1, 0, 0, 1]
labels = [0, 1, 0, 0, 1, 0, 1, 1]
confidences = [0.95, 0.88, 0.72, 0.91, 0.65, 0.89, 0.58, 0.94]

# Analyze predictions
print("Sample | Pred | Label | Conf   | Correct")
print("-" * 45)
for i, (pred, label, conf) in enumerate(zip(predictions, labels, confidences)):
    correct = "✓" if pred == label else "✗"
    print(f"{i:6d} | {pred:4d} | {label:5d} | {conf:.2f} | {correct:>7s}")
```

---

## 5. While Loops

### Basic While Loop

While loops continue until a condition becomes False.

```python
# Countdown
count = 5
while count > 0:
    print(f"Count: {count}")
    count -= 1

print("Done!")
```

### AI Example: Training Until Convergence

```python
# Simulate training until loss is low enough
current_loss = 1.0
target_loss = 0.05
epoch = 0
max_epochs = 100

while current_loss > target_loss and epoch < max_epochs:
    epoch += 1
    # Simulate loss decrease
    current_loss = current_loss * 0.85

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {current_loss:.6f}")

print(f"\nConverged at epoch {epoch} with loss {current_loss:.6f}")
```

### AI Example: Early Stopping

```python
# Early stopping based on validation loss
best_val_loss = float('inf')
patience = 3
patience_counter = 0
epoch = 0

# Simulated validation losses
val_losses = [0.5, 0.4, 0.35, 0.36, 0.37, 0.38, 0.39, 0.32, 0.31]

while patience_counter < patience and epoch < len(val_losses):
    current_val_loss = val_losses[epoch]

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0
        print(f"Epoch {epoch + 1}: New best! Val loss = {current_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"Epoch {epoch + 1}: No improvement (patience: {patience_counter}/{patience})")

    epoch += 1

print(f"\nStopped early at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
```

---

## 6. Break and Continue

### Break Statement

`break` exits the loop immediately.

```python
# Stop when target accuracy is reached
for epoch in range(100):
    # Simulate increasing accuracy
    accuracy = 0.5 + (epoch * 0.01)

    print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.2%}")

    if accuracy >= 0.95:
        print("Target accuracy reached! Stopping training.")
        break
```

### Continue Statement

`continue` skips the rest of the current iteration.

```python
# Skip logging for certain epochs
for epoch in range(1, 11):
    # Only log every 2nd epoch
    if epoch % 2 != 0:
        continue

    print(f"Epoch {epoch}: Performing validation...")
```

### AI Example: Skip Corrupted Batches

```python
# Simulate batch processing with some corrupted data
batch_status = ['good', 'good', 'corrupted', 'good', 'corrupted', 'good']

processed = 0
skipped = 0

for batch_idx, status in enumerate(batch_status):
    if status == 'corrupted':
        print(f"Batch {batch_idx}: Corrupted, skipping...")
        skipped += 1
        continue

    # Process batch
    print(f"Batch {batch_idx}: Processing...")
    processed += 1

print(f"\nProcessed: {processed}, Skipped: {skipped}")
```

---

## 7. Nested Loops

### Basic Nested Loop

Loops inside loops - essential for batch processing.

```python
# Outer loop: epochs
# Inner loop: batches
for epoch in range(3):
    print(f"\nEpoch {epoch + 1}")

    for batch in range(4):
        print(f"  Processing batch {batch + 1}/4")
```

### AI Example: Complete Training Loop

```python
# Training loop structure
num_epochs = 3
num_batches = 5

for epoch in range(num_epochs):
    # Training phase
    epoch_loss = 0

    for batch_idx in range(num_batches):
        # Simulate batch loss
        batch_loss = 1.0 / ((epoch * num_batches + batch_idx) + 1)
        epoch_loss += batch_loss

        # Log every 2 batches
        if (batch_idx + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{num_batches}: "
                  f"Loss = {batch_loss:.4f}")

    # Calculate average loss for epoch
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1} complete: Avg loss = {avg_loss:.4f}\n")
```

---

## 8. List Comprehension with Loops

### Converting Loops to Comprehensions

```python
# Traditional loop
squared = []
for i in range(10):
    squared.append(i ** 2)

# List comprehension (more Pythonic)
squared = [i ** 2 for i in range(10)]

print(f"Squares: {squared}")
```

### AI Example: Batch Creation

```python
# Create batches using loop
dataset = list(range(32))
batch_size = 8
batches = []

for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    batches.append(batch)

# Same with list comprehension (cleaner)
batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

print(f"Number of batches: {len(batches)}")
print(f"First batch: {batches[0]}")
```

---

## 9. Hands-On Exercise

### Challenge: Build a Training Loop

Complete this exercise in the Colab notebook.

```python
# Simulate a training scenario
train_data = list(range(80))  # 80 training samples
batch_size = 16
num_epochs = 5

# TODO: Write a nested loop that:
# 1. Iterates through epochs
# 2. Creates batches from train_data
# 3. Calculates a simulated loss for each batch
# 4. Prints progress every 2 batches
# 5. Calculates and prints average loss per epoch

# Your code here:
```

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    # Create and process batches
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]

        # Simulate loss (decreasing over time)
        batch_loss = 1.0 / (epoch + 1) / (num_batches + 1)
        epoch_loss += batch_loss
        num_batches += 1

        # Log every 2 batches
        if num_batches % 2 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Batch {num_batches}/{(len(train_data) + batch_size - 1) // batch_size}: "
                  f"Loss = {batch_loss:.6f}")

    # Calculate average loss
    avg_loss = epoch_loss / num_batches
    print(f"==> Epoch {epoch + 1} complete: Avg Loss = {avg_loss:.6f}\n")
```

</details>

---

## Reference Video

For additional visual learning, watch this curated reference video (optional):

<div class="video-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/94UHCEmprCY" title="Python Loops Tutorial" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

**Note:** The video provides supplementary content. The primary lesson is on this page and in the Colab notebook.

---

## Key Takeaways

- **For loops** iterate over sequences: `for item in sequence:`
- **range()** generates number sequences: `range(start, stop, step)`
- **enumerate()** provides index and value: `for i, val in enumerate(list):`
- **zip()** combines multiple lists: `zip(list1, list2)`
- **While loops** continue until condition is False
- **break** exits loop early, **continue** skips iteration
- **Nested loops** are essential for batch processing
- Training loops follow pattern: epochs → batches → samples

---

## Quiz

Test your understanding before moving on:

[Take the Lesson 4 Quiz →](quizzes.md#lesson-4)

**Quick Check:**

1. What does `range(5)` generate?
2. How do you get both index and value in a loop?
3. What's the difference between `break` and `continue`?
4. Why are nested loops important in AI?

---

## What's Next?

Congratulations on completing Lesson 4! You now understand how to write loops for AI training.

**Next Lesson:** [Lesson 5 - NumPy Essentials →](05-numpy.md)

In Lesson 5, you'll learn NumPy - the foundation of all numerical computing in AI. This is the most critical lesson for AI work!

---

**Resources:**

- [Python For Loops Documentation](https://docs.python.org/3/tutorial/controlflow.html#for-statements)
- [Real Python: For Loops](https://realpython.com/python-for-loop/)
- [While Loops Guide](https://realpython.com/python-while-loop/)

---

[← Lesson 3: Dictionaries](03-dictionaries.md) | [Course Home](index.md) | [Lesson 5: NumPy →](05-numpy.md)
