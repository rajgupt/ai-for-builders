# Lesson 3: Dictionaries and Data Structures

**Duration:** 1 hour | **Difficulty:** Beginner | **Prerequisites:** Lessons 1-2

## Learning Objectives

By the end of this lesson, you will:

- Create and manipulate dictionaries (key-value pairs)
- Store model configurations and hyperparameters
- Track training metrics efficiently
- Understand tuples and when to use them
- Choose the right data structure for AI tasks

## Why This Matters for AI

Dictionaries are everywhere in AI code:

- **Model configs** - Store hyperparameters like learning rate, batch size
- **Training logs** - Track metrics for each epoch
- **Dataset metadata** - Store information about your data
- **Experiment tracking** - Organize results from multiple runs
- **State dictionaries** - Save and load model weights

## Interactive Notebook

Launch the hands-on notebook to code along with this lesson:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/03-dictionaries.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/03-dictionaries.ipynb)

---

## 1. What are Dictionaries?

### Dictionary Basics

A dictionary stores data as key-value pairs. Think of it like a real dictionary: you look up a word (key) to get its definition (value).

```python
# Empty dictionary
empty_dict = {}

# Model configuration
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 10,
    'optimizer': 'adam'
}

# Access values using keys
print(f"Learning rate: {config['learning_rate']}")
print(f"Optimizer: {config['optimizer']}")
```

### Dictionary vs List

| Feature | List | Dictionary |
|---------|------|------------|
| Access | By position (index) | By key (name) |
| Order | Always ordered | Ordered (Python 3.7+) |
| Use case | Sequential data | Named data |
| Example | `[0.5, 0.3, 0.8]` | `{'loss': 0.5, 'acc': 0.8}` |

**When to use dictionaries in AI:**
- ✅ Store hyperparameters
- ✅ Track metrics with names
- ✅ Save experiment results
- ❌ Store sequences of numbers (use lists/arrays)

---

## 2. Creating Dictionaries

### Different Ways to Create Dictionaries

```python
# Method 1: Literal notation
model_config = {
    'model_name': 'ResNet50',
    'num_layers': 50,
    'pretrained': True,
    'dropout': 0.5
}

# Method 2: dict() constructor
training_config = dict(
    epochs=100,
    batch_size=64,
    learning_rate=0.001
)

# Method 3: From lists using zip
keys = ['train_loss', 'val_loss', 'accuracy']
values = [0.45, 0.52, 0.89]
metrics = dict(zip(keys, values))

print(f"Model config: {model_config}")
print(f"Training config: {training_config}")
print(f"Metrics: {metrics}")
```

### AI Example: Experiment Configuration

```python
# Complete experiment configuration
experiment = {
    'experiment_id': 'exp_001',
    'model': {
        'architecture': 'ResNet',
        'depth': 50,
        'pretrained': True
    },
    'training': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 50,
        'optimizer': 'adam'
    },
    'dataset': {
        'name': 'ImageNet',
        'num_classes': 1000,
        'train_size': 1200000,
        'val_size': 50000
    }
}

# Access nested dictionaries
print(f"Model: {experiment['model']['architecture']}")
print(f"Learning rate: {experiment['training']['learning_rate']}")
print(f"Dataset: {experiment['dataset']['name']}")
```

---

## 3. Accessing and Modifying Dictionaries

### Accessing Values

```python
metrics = {
    'epoch': 10,
    'train_loss': 0.345,
    'val_loss': 0.412,
    'accuracy': 0.89
}

# Method 1: Square brackets (raises error if key doesn't exist)
loss = metrics['train_loss']
print(f"Train loss: {loss}")

# Method 2: get() method (returns None or default if key doesn't exist)
learning_rate = metrics.get('learning_rate')  # None
dropout = metrics.get('dropout', 0.5)  # 0.5 (default)

print(f"Learning rate: {learning_rate}")  # None
print(f"Dropout: {dropout}")  # 0.5
```

### Modifying Values

```python
# Update existing key
metrics['epoch'] = 11
metrics['train_loss'] = 0.320

# Add new key
metrics['test_accuracy'] = 0.87

# Delete key
del metrics['test_accuracy']

print(f"Updated metrics: {metrics}")
```

### AI Example: Tracking Training Progress

```python
# Initialize training state
training_state = {
    'current_epoch': 0,
    'best_val_loss': float('inf'),
    'patience_counter': 0,
    'early_stop': False
}

# Simulate training loop
for epoch in range(5):
    training_state['current_epoch'] = epoch + 1
    current_val_loss = 0.5 / (epoch + 1)  # Simulated decreasing loss

    # Check if this is the best model
    if current_val_loss < training_state['best_val_loss']:
        training_state['best_val_loss'] = current_val_loss
        training_state['patience_counter'] = 0
        print(f"Epoch {epoch + 1}: New best model! Val loss: {current_val_loss:.4f}")
    else:
        training_state['patience_counter'] += 1
        print(f"Epoch {epoch + 1}: No improvement (patience: {training_state['patience_counter']})")

print(f"\nFinal state: {training_state}")
```

---

## 4. Dictionary Methods

### Essential Methods

```python
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 10
}

# keys() - Get all keys
print(f"Keys: {list(config.keys())}")

# values() - Get all values
print(f"Values: {list(config.values())}")

# items() - Get all key-value pairs
print(f"Items: {list(config.items())}")

# Iterate through dictionary
print("\nIteration:")
for key, value in config.items():
    print(f"{key}: {value}")
```

### Checking for Keys

```python
hyperparams = {
    'learning_rate': 0.001,
    'momentum': 0.9
}

# Check if key exists
if 'learning_rate' in hyperparams:
    print(f"Learning rate: {hyperparams['learning_rate']}")

if 'weight_decay' not in hyperparams:
    print("Weight decay not specified, using default: 0.0001")
    hyperparams['weight_decay'] = 0.0001
```

### Merging Dictionaries

```python
# Default configuration
default_config = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'num_epochs': 10,
    'optimizer': 'sgd'
}

# User-specified configuration
user_config = {
    'learning_rate': 0.001,
    'optimizer': 'adam'
}

# Merge (user config overrides defaults)
final_config = {**default_config, **user_config}
print(f"Final config: {final_config}")
# Output: {'learning_rate': 0.001, 'batch_size': 32, 'num_epochs': 10, 'optimizer': 'adam'}
```

---

## 5. AI Use Cases

### Use Case 1: Tracking Metrics per Epoch

```python
# Store metrics for each epoch
training_history = {
    'epochs': [],
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

# Simulate 5 epochs
for epoch in range(1, 6):
    training_history['epochs'].append(epoch)
    training_history['train_loss'].append(1.0 / epoch)
    training_history['val_loss'].append(1.1 / epoch)
    training_history['train_acc'].append(0.5 + (epoch * 0.08))
    training_history['val_acc'].append(0.48 + (epoch * 0.07))

# Display final results
print("Training History:")
for i in range(len(training_history['epochs'])):
    epoch = training_history['epochs'][i]
    train_loss = training_history['train_loss'][i]
    val_loss = training_history['val_loss'][i]
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
```

### Use Case 2: Model Hyperparameter Search

```python
# Define search space
hyperparameter_grid = [
    {'learning_rate': 0.1, 'batch_size': 32, 'result': None},
    {'learning_rate': 0.01, 'batch_size': 32, 'result': None},
    {'learning_rate': 0.001, 'batch_size': 32, 'result': None},
    {'learning_rate': 0.01, 'batch_size': 64, 'result': None},
]

# Simulate hyperparameter search
for config in hyperparameter_grid:
    # Simulate training (better results with smaller LR)
    simulated_accuracy = 0.7 + (0.1 / config['learning_rate']) * 0.01
    config['result'] = simulated_accuracy

    print(f"LR: {config['learning_rate']}, Batch: {config['batch_size']}, "
          f"Accuracy: {config['result']:.4f}")

# Find best configuration
best_config = max(hyperparameter_grid, key=lambda x: x['result'])
print(f"\nBest config: LR={best_config['learning_rate']}, "
      f"Batch={best_config['batch_size']}, Acc={best_config['result']:.4f}")
```

### Use Case 3: Dataset Statistics

```python
# Store dataset information
dataset_info = {
    'name': 'CIFAR-10',
    'type': 'image_classification',
    'num_classes': 10,
    'splits': {
        'train': {
            'num_samples': 50000,
            'images_per_class': 5000
        },
        'test': {
            'num_samples': 10000,
            'images_per_class': 1000
        }
    },
    'image_shape': (32, 32, 3),
    'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
}

# Print dataset summary
print(f"Dataset: {dataset_info['name']}")
print(f"Classes: {dataset_info['num_classes']}")
print(f"Training samples: {dataset_info['splits']['train']['num_samples']}")
print(f"Test samples: {dataset_info['splits']['test']['num_samples']}")
print(f"Image shape: {dataset_info['image_shape']}")
print(f"Classes: {', '.join(dataset_info['class_names'])}")
```

---

## 6. Tuples

### What are Tuples?

Tuples are immutable sequences. Once created, they cannot be changed.

```python
# Creating tuples
image_shape = (224, 224, 3)  # Height, width, channels
coordinates = (10, 20)
single_element = (42,)  # Note the comma

# Accessing elements (same as lists)
height = image_shape[0]
width = image_shape[1]
channels = image_shape[2]

print(f"Image shape: {height}x{width}x{channels}")
```

### Why Use Tuples?

```python
# Tuples are immutable (cannot be changed)
shape = (28, 28)
# shape[0] = 32  # This would raise an error!

# Use tuples for:
# 1. Fixed dimensions
batch_shape = (32, 28, 28, 1)  # (batch, height, width, channels)

# 2. Dictionary keys (lists can't be keys)
model_cache = {
    (224, 224): "model_224.pth",
    (384, 384): "model_384.pth"
}

# 3. Multiple return values
def get_model_info():
    return "ResNet50", 50, 25_000_000  # name, layers, params

name, layers, params = get_model_info()
print(f"{name}: {layers} layers, {params:,} parameters")
```

### Tuples vs Lists

| Feature | List | Tuple |
|---------|------|-------|
| Mutable | Yes | No |
| Syntax | `[1, 2, 3]` | `(1, 2, 3)` |
| Speed | Slower | Faster |
| Use case | Data that changes | Fixed data |

---

## 7. Choosing the Right Data Structure

### Decision Guide

```python
# Lists: Ordered, mutable sequences
training_losses = [0.89, 0.76, 0.65, 0.58]  # ✅ Changes over time

# Dictionaries: Named data with key-value pairs
hyperparams = {'lr': 0.001, 'batch': 32}  # ✅ Named parameters

# Tuples: Immutable, ordered sequences
image_shape = (224, 224, 3)  # ✅ Fixed dimensions

# Sets: Unordered, unique elements (less common in AI)
unique_labels = {0, 1, 2, 3}  # ✅ Unique classes
```

### AI Example: Complete Training Configuration

```python
# Combine all data structures
training_setup = {
    # Basic hyperparameters (dict)
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 50,

    # Fixed dimensions (tuples)
    'input_shape': (224, 224, 3),
    'output_shape': (1000,),

    # Changing metrics (lists)
    'train_losses': [],
    'val_losses': [],

    # Nested configuration (dict of dicts)
    'optimizer': {
        'type': 'adam',
        'beta1': 0.9,
        'beta2': 0.999
    },

    # Multiple values (tuple)
    'image_stats': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }
}

print("Training Setup:")
print(f"Input shape: {training_setup['input_shape']}")
print(f"Optimizer: {training_setup['optimizer']['type']}")
print(f"Image mean: {training_setup['image_stats']['mean']}")
```

---

## 8. Hands-On Exercise

### Challenge: Build an Experiment Tracker

Complete this exercise in the Colab notebook.

```python
# TODO: Create a dictionary to track a training experiment with:
# 1. experiment_id: "exp_001"
# 2. model_name: "ResNet18"
# 3. hyperparameters: dict with lr=0.001, batch_size=64, epochs=30
# 4. results: dict with train_acc=0.92, val_acc=0.88, test_acc=0.86
# 5. image_shape: tuple (224, 224, 3)

# Your code here:
experiment = None

# Print summary
print(f"Experiment: {experiment['experiment_id']}")
print(f"Model: {experiment['model_name']}")
print(f"Learning rate: {experiment['hyperparameters']['lr']}")
print(f"Results: Train={experiment['results']['train_acc']:.2%}, "
      f"Val={experiment['results']['val_acc']:.2%}, "
      f"Test={experiment['results']['test_acc']:.2%}")
```

### Solution

<details>
<summary>Click to reveal solution</summary>

```python
experiment = {
    'experiment_id': 'exp_001',
    'model_name': 'ResNet18',
    'hyperparameters': {
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 30
    },
    'results': {
        'train_acc': 0.92,
        'val_acc': 0.88,
        'test_acc': 0.86
    },
    'image_shape': (224, 224, 3)
}

print(f"Experiment: {experiment['experiment_id']}")
print(f"Model: {experiment['model_name']}")
print(f"Learning rate: {experiment['hyperparameters']['lr']}")
print(f"Results: Train={experiment['results']['train_acc']:.2%}, "
      f"Val={experiment['results']['val_acc']:.2%}, "
      f"Test={experiment['results']['test_acc']:.2%}")
```

</details>

---

## Reference Video

For additional visual learning, watch this curated reference video (optional):

<div class="video-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/daefaLgNkw0" title="Python Dictionaries Tutorial" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

**Note:** The video provides supplementary content. The primary lesson is on this page and in the Colab notebook.

---

## Key Takeaways

- **Dictionaries** store key-value pairs: `{'key': value}`
- **Access with keys**: `config['learning_rate']`
- **Safe access**: Use `.get()` to avoid errors
- **Tuples** are immutable: `(224, 224, 3)`
- **Choose wisely**: Lists for sequences, dicts for named data, tuples for fixed data
- **Nested structures**: Dicts can contain dicts, lists, and tuples

---

## Quiz

Test your understanding before moving on:

[Take the Lesson 3 Quiz →](quizzes.md#lesson-3)

**Quick Check:**

1. How do you access a value in a dictionary?
2. What's the difference between `dict['key']` and `dict.get('key')`?
3. When should you use a tuple instead of a list?
4. How do you add a new key-value pair to a dictionary?

---

## What's Next?

Congratulations on completing Lesson 3! You now understand how to organize data using dictionaries and tuples.

**Next Lesson:** [Lesson 4 - Loops and Iteration →](04-loops.md)

In Lesson 4, you'll learn about loops for training models, processing batches, and iterating through datasets.

---

**Resources:**

- [Python Dictionaries Documentation](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)
- [Real Python: Dictionaries Guide](https://realpython.com/python-dicts/)
- [Tuples vs Lists](https://realpython.com/python-lists-tuples/)

---

[← Lesson 2: Lists](02-lists.md) | [Course Home](index.md) | [Lesson 4: Loops →](04-loops.md)
