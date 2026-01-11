# Task 02: Python for AI Mini Course

## Overview

Create a hands-on, beginner-friendly Python mini course specifically designed for AI/ML engineering. This course focuses exclusively on Python concepts that are **must-haves** for AI work, cutting out general programming topics that aren't immediately relevant to building AI systems.

---

## Target Audience

**Primary:** Complete beginners with zero programming experience who want to:
- Build AI models in PyTorch/TensorFlow quickly
- Understand AI tutorials without getting stuck on Python syntax
- Work with data using NumPy, Pandas, and Matplotlib
- Debug AI code confidently

**Secondary:** Developers from other languages (Java, C++, JavaScript) who need AI-specific Python knowledge

**NOT targeting:** General Python learners, web developers, automation scripters

---

## Learning Objectives

By the end of this course, learners will be able to:

1. **Read and write AI code** - Understand 90% of PyTorch/TensorFlow tutorials
2. **Manipulate data** - Load, clean, and transform datasets
3. **Debug AI scripts** - Fix common errors in notebooks
4. **Use AI libraries** - Import packages, call functions, understand documentation
5. **Run experiments** - Modify hyperparameters, compare results

---

## Course Duration

**Total Time:** 2 weeks (10-12 hours of learning)

**Recommended Schedule:**
- Week 1: Lessons 1-4 (Python basics + Data structures)
- Week 2: Lessons 5-7 (NumPy + Functions + Mini-project)

**Each Lesson Includes:**
- 15-20 min concept explanation (text + visuals)
- 30-40 min interactive Colab notebook
- 10 min quiz (5-7 questions)
- 1 mini-exercise (apply concept to AI problem)

---

## Course Structure

### Lesson 1: Python Basics for AI (Day 1-2)

**Concept Page:** `docs/courses/foundation/python-for-ai/01-basics.md`

**Topics:**
- Variables and data types (int, float, string, bool)
- Basic operators (+, -, *, /, //, %, **)
- Print statements and string formatting (f-strings)
- Comments and code readability

**AI Context:**
- Why floats matter in neural networks
- Integer vs float division in gradient calculations
- String formatting for logging training metrics

**Colab Notebook:** `notebooks/foundation/python-for-ai/01-basics.ipynb`

**Hands-on Exercise:**
```python
# Calculate model accuracy from predictions
correct = 85
total = 100
accuracy = correct / total
print(f"Model accuracy: {accuracy:.2%}")
```

**Learning Outcome:** Understand basic Python syntax used in every AI script

---

### Lesson 2: Lists and Indexing (Day 3-4)

**Concept Page:** `docs/courses/foundation/python-for-ai/02-lists.md`

**Topics:**
- Creating lists (training data, labels)
- Indexing and slicing (array access patterns)
- List methods: append(), extend(), pop()
- List comprehensions (data transformation)
- Nested lists (batches of data)

**AI Context:**
- Lists as containers for training examples
- Slicing for train/test splits
- Batch processing with list comprehensions
- Understanding shape: [batch_size, features]

**Colab Notebook:** `notebooks/foundation/python-for-ai/02-lists.ipynb`

**Hands-on Exercise:**
```python
# Split dataset into train/test sets
dataset = list(range(100))
train_data = dataset[:80]  # First 80% for training
test_data = dataset[80:]   # Last 20% for testing

# Create batches of size 16
batch_size = 16
batches = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
print(f"Number of batches: {len(batches)}")
```

**Learning Outcome:** Manipulate training data and understand batch dimensions

---

### Lesson 3: Dictionaries and Data Structures (Day 5-6)

**Concept Page:** `docs/courses/foundation/python-for-ai/03-dictionaries.md`

**Topics:**
- Dictionaries (key-value pairs)
- Accessing and modifying dict values
- Dict methods: keys(), values(), items()
- Tuples (immutable sequences)
- When to use lists vs dicts vs tuples

**AI Context:**
- Model configurations as dictionaries
- Hyperparameter dictionaries
- Dataset metadata storage
- Training logs and metrics tracking

**Colab Notebook:** `notebooks/foundation/python-for-ai/03-dictionaries.ipynb`

**Hands-on Exercise:**
```python
# Store model hyperparameters
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 10,
    'optimizer': 'adam'
}

# Track training metrics
metrics = {
    'epoch': 1,
    'train_loss': 0.45,
    'val_loss': 0.52,
    'accuracy': 0.89
}

print(f"Epoch {metrics['epoch']}: Loss = {metrics['train_loss']:.4f}")
```

**Learning Outcome:** Organize AI experiment configurations and results

---

### Lesson 4: Loops and Iteration (Day 7-8)

**Concept Page:** `docs/courses/foundation/python-for-ai/04-loops.md`

**Topics:**
- For loops (iterating over datasets)
- While loops (training until convergence)
- Range() function (epoch loops)
- Enumerate() and zip() (parallel iteration)
- Break and continue statements

**AI Context:**
- Epoch loops in training
- Iterating through batches
- Early stopping with break
- Processing multiple datasets with zip()

**Colab Notebook:** `notebooks/foundation/python-for-ai/04-loops.ipynb`

**Hands-on Exercise:**
```python
# Training loop structure
num_epochs = 5
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, batch_data in enumerate(batches):
        # Simulate training step
        loss = 1.0 / (epoch + 1)  # Decreasing loss
        epoch_loss += loss

    avg_loss = epoch_loss / len(batches)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

**Learning Outcome:** Understand training loop patterns in AI code

---

### Lesson 5: NumPy Essentials (Day 9-10)

**Concept Page:** `docs/courses/foundation/python-for-ai/05-numpy.md`

**Topics:**
- Arrays vs lists (why NumPy is essential)
- Creating arrays: np.array(), np.zeros(), np.ones(), np.random
- Array shapes and dimensions (1D, 2D, 3D+)
- Basic operations (element-wise math)
- Indexing and slicing arrays
- Broadcasting (automatic shape matching)
- Common functions: sum(), mean(), max(), reshape()

**AI Context:**
- Tensors are NumPy-like arrays
- Image data as 3D arrays (height, width, channels)
- Weight matrices and gradients
- Batch processing with broadcasting
- Understanding tensor shapes in PyTorch/TF

**Colab Notebook:** `notebooks/foundation/python-for-ai/05-numpy.ipynb`

**Hands-on Exercise:**
```python
import numpy as np

# Create a batch of images (batch_size, height, width, channels)
batch = np.random.rand(32, 28, 28, 1)  # 32 grayscale 28x28 images
print(f"Batch shape: {batch.shape}")

# Normalize pixel values to [0, 1]
batch_normalized = (batch - batch.min()) / (batch.max() - batch.min())

# Calculate mean image across batch
mean_image = batch_normalized.mean(axis=0)
print(f"Mean image shape: {mean_image.shape}")
```

**Learning Outcome:** Work with multi-dimensional arrays (the foundation of tensors)

---

### Lesson 6: Functions and Modules (Day 11-12)

**Concept Page:** `docs/courses/foundation/python-for-ai/06-functions.md`

**Topics:**
- Defining functions (def keyword)
- Parameters and return values
- Default arguments
- Lambda functions (anonymous functions)
- Importing modules (import, from...import)
- Installing packages (pip)
- Understanding library documentation

**AI Context:**
- Custom loss functions
- Data preprocessing pipelines
- Model architecture functions
- Importing PyTorch/TensorFlow
- Reading AI library docs (torch.nn, tf.keras)

**Colab Notebook:** `notebooks/foundation/python-for-ai/06-functions.ipynb`

**Hands-on Exercise:**
```python
import torch

# Custom accuracy function
def calculate_accuracy(predictions, targets):
    """
    Calculate classification accuracy.

    Args:
        predictions: Model output (logits or probabilities)
        targets: True labels

    Returns:
        Accuracy as a float between 0 and 1
    """
    pred_labels = predictions.argmax(dim=1)
    correct = (pred_labels == targets).sum().item()
    total = len(targets)
    return correct / total

# Example usage
preds = torch.randn(10, 5)  # 10 samples, 5 classes
targets = torch.randint(0, 5, (10,))  # Random labels
acc = calculate_accuracy(preds, targets)
print(f"Accuracy: {acc:.2%}")
```

**Learning Outcome:** Write reusable code and use AI libraries effectively

---

### Lesson 7: Mini-Project - MNIST Digit Classifier (Day 13-14)

**Concept Page:** `docs/courses/foundation/python-for-ai/07-mini-project.md`

**Project Goal:** Build a complete ML pipeline using only Python + NumPy + scikit-learn

**Project Structure:**
1. Load MNIST dataset
2. Preprocess images (normalize, flatten)
3. Split into train/test sets
4. Train a simple classifier (Logistic Regression)
5. Evaluate accuracy
6. Visualize results

**Colab Notebook:** `notebooks/foundation/python-for-ai/07-mnist-project.ipynb`

**Starter Code:**
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# TODO: Load dataset
# TODO: Split train/test
# TODO: Train model
# TODO: Evaluate accuracy
# TODO: Visualize predictions
```

**Success Criteria:**
- Achieve >90% test accuracy
- Create confusion matrix visualization
- Write a function to predict new images
- Add proper error handling

**Learning Outcome:** Apply all Python concepts to build a complete AI project

---

## Notebook Template

Every Colab notebook follows this structure:

```python
"""
AI Skills Hub - Python for AI
Lesson X: [Lesson Title]

Learn: [Key concepts]
Build: [Hands-on exercise]

Runtime: ~30 minutes
GPU Required: No (except lesson 7)

License: MIT
"""

# ═══════════════════════════════════════════════════════
# SETUP: Run this cell first
# ═══════════════════════════════════════════════════════
import sys
print(f"Python version: {sys.version}")

# ═══════════════════════════════════════════════════════
# CONCEPT 1: [Topic Name]
# ═══════════════════════════════════════════════════════

# Explanation as markdown cell + code examples

# ═══════════════════════════════════════════════════════
# HANDS-ON EXERCISE
# ═══════════════════════════════════════════════════════

# TODO: Complete the exercise

# ═══════════════════════════════════════════════════════
# QUIZ (Check your understanding)
# ═══════════════════════════════════════════════════════

# Interactive quiz using ipywidgets or assertions
```

---

## SEO & Marketing Strategy

### Target Keywords

**Primary:**
- "python for machine learning beginners"
- "python for ai tutorial"
- "learn python for deep learning"
- "numpy tutorial for ai"

**Secondary:**
- "python basics for pytorch"
- "python for tensorflow beginners"
- "data science python crash course"

### Content Marketing Channels

1. **Reddit Posts:**
   - r/learnmachinelearning: "I created a free Python for AI course (no fluff, just what you need for ML)"
   - r/learnpython: "Python for AI - focused mini-course"
   - r/MachineLearning: Share after getting good feedback

2. **Dev.to Article:**
   - Title: "7 Python Concepts You Need for AI (and Nothing Else)"
   - Include code snippets from lessons
   - Link to full course

3. **Twitter/X Thread:**
   - "Most Python courses teach you web dev stuff you don't need for AI. Here's what you actually need: 🧵"
   - 1 tweet per lesson with key concept

4. **YouTube Short Scripts:**
   - "This is the ONLY Python you need for AI" (30 sec demo)
   - Show training loop example

---

## Dependencies

### Python Packages for Notebooks

```python
# requirements.txt (automatically installed in Colab)
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
jupyter>=1.0.0
ipywidgets>=8.0.0  # For interactive quizzes
```

### MkDocs Plugins

```yaml
# Add to mkdocs.yml
plugins:
  - search
  - tags:
      tags_file: courses/foundation/python-for-ai/tags.md
markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details
```

---

## Quiz Examples

Each lesson includes 5-7 quiz questions. Example format:

**Lesson 1 Quiz:**

```markdown
### Quiz: Python Basics

1. **What is the output of: `print(10 // 3)`?**
   - [ ] 3.33
   - [x] 3
   - [ ] 3.0
   - [ ] 4

2. **Which data type is best for storing model accuracy?**
   - [ ] int
   - [x] float
   - [ ] string
   - [ ] bool

3. **What does this print? `loss = 0.0045; print(f"Loss: {loss:.2e}")`**
   - [ ] Loss: 0.00
   - [x] Loss: 4.50e-03
   - [ ] Loss: 0.0045
   - [ ] Loss: 4.5e-3

[View Answers →](#answers)
```

---

## Visual Assets Needed

### Diagrams to Create:

1. **Python vs NumPy Arrays** (Lesson 5)
   - Side-by-side comparison of list vs ndarray
   - Performance benchmark graph

2. **Tensor Shapes Visual** (Lesson 5)
   - 3D visualization of batch dimensions
   - [batch_size, height, width, channels]

3. **Training Loop Flowchart** (Lesson 4)
   - Nested loop structure (epochs → batches → forward/backward)

4. **Function Anatomy** (Lesson 6)
   - Labeled diagram of function components (def, params, return, docstring)

### Tools:
- Excalidraw (free, easy diagrams)
- Python matplotlib (generate graphs in notebooks)

---

## Success Metrics

### Week 1-2 Targets:

| Metric | Target |
|--------|--------|
| Course enrollments | 200 learners |
| Completion rate | >60% (finish lesson 7) |
| Avg. time per lesson | 45-60 minutes |
| Colab notebook opens | 500+ |
| Quiz scores | >80% correct on first try |

### User Feedback Goals:

- "I can finally understand PyTorch tutorials!" ✓
- "This is way faster than generic Python courses" ✓
- "The AI examples make everything click" ✓

---

## Common Pitfalls to Avoid

1. **Over-teaching:** Don't cover file I/O, classes, decorators, etc. (not needed yet)
2. **Under-contextualization:** Always explain "why this matters for AI"
3. **Stale examples:** Use real libraries (torch, sklearn), not toy examples
4. **Skipping NumPy:** This is the most critical lesson - don't rush it
5. **No exercises:** Every concept needs a hands-on AI application

---

## Next Steps After Completion

**For Learners:**
1. Proceed to "Math Essentials for AI" (linear algebra, calculus)
2. Start "Introduction to ML" course
3. Build a personal project from scratch

**For Course Creators:**
1. Collect feedback via GitHub Discussions
2. Add bonus lesson: "Pandas for AI Data" (optional)
3. Create video walkthroughs for each lesson
4. Translate to other languages (Spanish, Hindi)

---

## File Checklist

### Markdown Content Pages:
- [ ] `docs/courses/foundation/python-for-ai/index.md` (course landing page)
- [ ] `docs/courses/foundation/python-for-ai/01-basics.md`
- [ ] `docs/courses/foundation/python-for-ai/02-lists.md`
- [ ] `docs/courses/foundation/python-for-ai/03-dictionaries.md`
- [ ] `docs/courses/foundation/python-for-ai/04-loops.md`
- [ ] `docs/courses/foundation/python-for-ai/05-numpy.md`
- [ ] `docs/courses/foundation/python-for-ai/06-functions.md`
- [ ] `docs/courses/foundation/python-for-ai/07-mini-project.md`

### Colab Notebooks:
- [ ] `notebooks/foundation/python-for-ai/01-basics.ipynb`
- [ ] `notebooks/foundation/python-for-ai/02-lists.ipynb`
- [ ] `notebooks/foundation/python-for-ai/03-dictionaries.ipynb`
- [ ] `notebooks/foundation/python-for-ai/04-loops.ipynb`
- [ ] `notebooks/foundation/python-for-ai/05-numpy.ipynb`
- [ ] `notebooks/foundation/python-for-ai/06-functions.ipynb`
- [ ] `notebooks/foundation/python-for-ai/07-mnist-project.ipynb`

### Supporting Files:
- [ ] `docs/courses/foundation/python-for-ai/assets/` (images/diagrams)
- [ ] `docs/courses/foundation/python-for-ai/quizzes.md` (all quiz solutions)
- [ ] `docs/courses/foundation/python-for-ai/resources.md` (external links)

---

## Implementation Timeline

**Week 1: Content Creation**
- Day 1-2: Write lessons 1-3 (markdown + notebooks)
- Day 3-4: Write lessons 4-6 (markdown + notebooks)
- Day 5: Write lesson 7 (mini-project)
- Day 6-7: Create diagrams and visual assets

**Week 2: Testing & Polish**
- Day 1-2: Test all notebooks on Colab/Kaggle
- Day 3: Write quizzes and solutions
- Day 4: SEO optimization (meta descriptions, keywords)
- Day 5: Peer review and edits
- Day 6: Marketing materials (Reddit posts, tweets)
- Day 7: Launch and promote

**Total Effort:** 40-50 hours (content creation) + 10 hours (promotion)

---

## Accessibility Considerations

- **Color-blind friendly:** Use patterns + colors in graphs
- **Screen reader compatible:** Alt text on all images
- **Mobile-responsive:** Test notebooks on phone browsers
- **Low bandwidth:** Keep images under 200KB each
- **Offline mode:** Downloadable notebook bundle

---

## Legal & Licensing

- **License:** MIT (all code)
- **Content License:** CC BY 4.0 (all markdown)
- **Dataset License:** MNIST is public domain
- **Attribution:** Cite scikit-learn, NumPy in notebooks

---

**Task Owner:** Rajat Gupta
**Estimated Effort:** 40-50 hours (content) + 10 hours (marketing)
**Priority:** P1 (first course in foundation track)
**Status:** Not Started
**Dependencies:** Task 01 (Landing Page) must be complete
**Blocks:** Math Essentials course, Introduction to ML course
