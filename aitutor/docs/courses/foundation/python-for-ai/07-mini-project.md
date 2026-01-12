# Lesson 7: Mini-Project - MNIST Digit Classifier

**Duration:** 2 hours | **Difficulty:** Beginner-Intermediate | **Prerequisites:** Lessons 1-6

## Project Overview

Build a complete machine learning pipeline from scratch! You'll create a digit classifier that recognizes handwritten numbers (0-9) using the MNIST dataset.

**What You'll Build:**
- Load and preprocess image data
- Train a logistic regression classifier
- Evaluate model performance
- Visualize results

**Technologies:**
- Python (everything you've learned!)
- NumPy (array operations)
- scikit-learn (ML model)
- Matplotlib (visualization)

## Learning Objectives

By the end of this project, you will:

- Apply all Python concepts in a real AI project
- Build a complete ML pipeline
- Achieve >90% accuracy on digit classification
- Create visualizations of model performance
- Write production-quality code

## Interactive Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/07-mnist-project.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/rajgupt/ai-for-builders/blob/main/aitutor/notebooks/foundation/python-for-ai/07-mnist-project.ipynb)

---

## Project Structure

### Phase 1: Load and Explore Data
### Phase 2: Preprocess Images
### Phase 3: Train Model
### Phase 4: Evaluate Performance
### Phase 5: Visualize Results

---

## Phase 1: Load and Explore

```python
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST-like dataset (8x8 images)
digits = load_digits()
X, y = digits.data, digits.target

print(f"Dataset shape: {X.shape}")  # (1797, 64)
print(f"Labels shape: {y.shape}")    # (1797,)
print(f"Classes: {np.unique(y)}")    # [0 1 2 3 4 5 6 7 8 9]

# Visualize samples
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.show()
```

---

## Phase 2: Preprocess

```python
def preprocess_data(X_train, X_test):
    """Normalize pixel values to [0, 1]."""
    X_train_norm = X_train / 16.0  # Max pixel value is 16
    X_test_norm = X_test / 16.0
    return X_train_norm, X_test_norm

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize
X_train_norm, X_test_norm = preprocess_data(X_train, X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

---

## Phase 3: Train Model

```python
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    """Train logistic regression classifier."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train
model = train_model(X_train_norm, y_train)
print("✓ Model trained!")
```

---

## Phase 4: Evaluate

```python
def evaluate_model(model, X_test, y_test):
    """Calculate accuracy and confusion matrix."""
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, cm, y_pred

accuracy, cm, y_pred = evaluate_model(model, X_test_norm, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

---

## Phase 5: Visualize

```python
# Plot confusion matrix
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Show misclassified examples
misclassified = np.where(y_test != y_pred)[0]
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    if i < len(misclassified):
        idx = misclassified[i]
        ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        ax.axis('off')
plt.show()
```

---

## Success Criteria

- [ ] Load dataset successfully
- [ ] Preprocess images (normalization)
- [ ] Train model without errors
- [ ] Achieve >90% test accuracy
- [ ] Create confusion matrix
- [ ] Visualize misclassified examples
- [ ] Write clean, documented code

---

## Reference Video

<div class="video-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/aircAruvnKk" title="Neural Networks Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **You built a complete ML project!**
- **Applied all Python concepts**: lists, loops, functions, NumPy
- **Real-world pipeline**: load → preprocess → train → evaluate
- **Production patterns**: functions, documentation, error handling
- **Visualization**: communicate results effectively

---

## What's Next?

**Congratulations! You've completed Python for AI!**

**Next Steps:**
1. Complete the course quiz
2. Share your project on GitHub
3. Move to "Math Essentials for AI" course
4. Build more projects with PyTorch!

**Resources:**
- [scikit-learn Documentation](https://scikit-learn.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Deep Learning with PyTorch](https://pytorch.org/tutorials/)

---

[← Lesson 6: Functions](06-functions.md) | [Course Home](index.md) | [Next Course →](../../math-essentials/)
