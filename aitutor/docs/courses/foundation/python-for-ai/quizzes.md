# Python for AI - Quizzes and Solutions

Test your understanding of each lesson with these comprehensive quizzes. Try to answer all questions before checking the solutions!

---

## Lesson 1: Python Basics for AI {#lesson-1}

### Quiz Questions

**1. What is the output of `print(10 // 3)`?**

- [ ] A) 3.33
- [ ] B) 3.0
- [ ] C) 3
- [ ] D) 4

**2. Which data type is best for storing model accuracy (e.g., 0.89)?**

- [ ] A) int
- [ ] B) float
- [ ] C) string
- [ ] D) bool

**3. What does this code print?**

```python
loss = 0.0045
print(f"Loss: {loss:.2e}")
```

- [ ] A) Loss: 0.00
- [ ] B) Loss: 4.50e-03
- [ ] C) Loss: 0.0045
- [ ] D) Loss: 4.5e-3

**4. Why should you use `/` instead of `//` for calculating accuracy?**

- [ ] A) `/` is faster
- [ ] B) `//` causes errors
- [ ] C) `/` preserves decimal precision
- [ ] D) `//` only works with floats

**5. What is the correct way to format a float to 2 decimal places in an f-string?**

- [ ] A) `f"{value:2f}"`
- [ ] B) `f"{value:.2f}"`
- [ ] C) `f"{value:.f2}"`
- [ ] D) `f"{value:2.f}"`

**6. What is the output of `2 ** 3`?**

- [ ] A) 5
- [ ] B) 6
- [ ] C) 8
- [ ] D) 9

**7. Which variable name is most appropriate for storing the number of training epochs?**

- [ ] A) `x`
- [ ] B) `num_epochs`
- [ ] C) `EPOCHS`
- [ ] D) `n`

### Answers

<details>
<summary>Click to reveal answers</summary>

1. **C) 3** - `//` is floor division, which returns an integer
2. **B) float** - Accuracy is a decimal between 0 and 1
3. **B) Loss: 4.50e-03** - `.2e` formats in scientific notation with 2 decimal places
4. **C) `/` preserves decimal precision** - `//` performs integer division and loses decimals
5. **B) `f"{value:.2f}"`** - The colon, dot, number, then format specifier
6. **C) 8** - `**` is the exponentiation operator (2³ = 8)
7. **B) `num_epochs`** - Descriptive variable names improve code readability

**Score Guide:**
- 7/7: Excellent! Ready for Lesson 2
- 5-6/7: Good! Review weak areas
- 3-4/7: Re-read Lesson 1 content
- 0-2/7: Re-watch video and practice in Colab

</details>

---

## Lesson 2: Lists and Indexing {#lesson-2}

### Quiz Questions

**1. What is the output of `[1, 2, 3, 4, 5][2]`?**

- [ ] A) 1
- [ ] B) 2
- [ ] C) 3
- [ ] D) 4

**2. How do you get the last element of a list called `data`?**

- [ ] A) `data[last]`
- [ ] B) `data[-1]`
- [ ] C) `data[end]`
- [ ] D) `data[length]`

**3. What does `dataset[:80]` return if `dataset` has 100 elements?**

- [ ] A) Elements 0 to 79
- [ ] B) Elements 0 to 80
- [ ] C) Elements 1 to 80
- [ ] D) Elements 80 to 100

**4. Which list comprehension creates a list of squares from 0 to 4?**

- [ ] A) `[x * x for x in range(5)]`
- [ ] B) `[x ** 2 for x in [0, 1, 2, 3, 4]]`
- [ ] C) Both A and B
- [ ] D) Neither A nor B

**5. What is the output of `len([10, 20, 30])`?**

- [ ] A) 2
- [ ] B) 3
- [ ] C) 60
- [ ] D) 30

**6. How do you create batches of size 16 from a list called `data`?**

- [ ] A) `[data[i:i+16] for i in range(len(data))]`
- [ ] B) `[data[i:i+16] for i in range(0, len(data), 16)]`
- [ ] C) `data.split(16)`
- [ ] D) `data[::16]`

**7. What does `[1, 2] + [3, 4]` produce?**

- [ ] A) `[4, 6]`
- [ ] B) `[1, 2, 3, 4]`
- [ ] C) `[[1, 2], [3, 4]]`
- [ ] D) Error

### Answers

<details>
<summary>Click to reveal answers</summary>

1. **C) 3** - Lists are zero-indexed, so index 2 is the third element
2. **B) `data[-1]`** - Negative indexing starts from the end
3. **A) Elements 0 to 79** - Slicing is exclusive of the end index
4. **C) Both A and B** - Both create the list `[0, 1, 4, 9, 16]`
5. **B) 3** - `len()` returns the number of elements
6. **B) `[data[i:i+16] for i in range(0, len(data), 16)]`** - Steps by 16 to create non-overlapping batches
7. **B) `[1, 2, 3, 4]`** - `+` concatenates lists

**Key Concepts:**
- Indexing starts at 0
- Negative indices count from the end
- Slicing is `[start:stop:step]` (stop is exclusive)
- List comprehensions are faster than loops

</details>

---

## Lesson 3: Dictionaries and Data Structures {#lesson-3}

### Quiz Questions

**1. How do you access the value associated with key `'lr'` in dict `config`?**

- [ ] A) `config.lr`
- [ ] B) `config['lr']`
- [ ] C) `config(lr)`
- [ ] D) `config->lr`

**2. What does `config.get('lr', 0.001)` return if `'lr'` is not in the dict?**

- [ ] A) Error
- [ ] B) None
- [ ] C) 0.001
- [ ] D) 'lr'

**3. Which data structure should you use to store model hyperparameters?**

- [ ] A) List
- [ ] B) Dictionary
- [ ] C) Tuple
- [ ] D) String

**4. What is the difference between a tuple and a list?**

- [ ] A) Tuples are faster
- [ ] B) Tuples are immutable (cannot be changed)
- [ ] C) Tuples can store any type
- [ ] D) No difference

**5. How do you add a new key-value pair to a dictionary?**

```python
config = {'lr': 0.001}
# Add 'batch_size': 32
```

- [ ] A) `config.add('batch_size', 32)`
- [ ] B) `config['batch_size'] = 32`
- [ ] C) `config.insert('batch_size', 32)`
- [ ] D) `config.append({'batch_size': 32})`

**6. What does `config.keys()` return?**

- [ ] A) A list of all values
- [ ] B) A list of all keys
- [ ] C) A view of all keys
- [ ] D) The number of keys

**7. Which is faster for storing training metrics across epochs?**

- [ ] A) Multiple variables
- [ ] B) List of tuples
- [ ] C) Dictionary with epoch as key
- [ ] D) Nested lists

### Answers

<details>
<summary>Click to reveal answers</summary>

1. **B) `config['lr']`** - Square bracket notation for key access
2. **C) 0.001** - `.get()` returns the default value if key doesn't exist
3. **B) Dictionary** - Keys map to values (e.g., `'lr': 0.001`)
4. **B) Tuples are immutable** - Once created, tuples cannot be modified
5. **B) `config['batch_size'] = 32`** - Assign to a new key
6. **C) A view of all keys** - Returns a dict_keys object (iterable)
7. **C) Dictionary with epoch as key** - Fast lookups and organized data

**Pro Tips:**
- Use `.get()` to avoid KeyError
- Dictionaries are unordered (before Python 3.7)
- Tuples are great for fixed configurations (coordinates, RGB values)

</details>

---

## Lesson 4: Loops and Iteration {#lesson-4}

### Quiz Questions

**1. How many times does this loop run?**

```python
for i in range(5):
    print(i)
```

- [ ] A) 4
- [ ] B) 5
- [ ] C) 6
- [ ] D) Infinite

**2. What does `enumerate([10, 20, 30])` provide?**

- [ ] A) Just the values
- [ ] B) Just the indices
- [ ] C) Index-value pairs
- [ ] D) Length of the list

**3. How do you iterate through two lists simultaneously?**

- [ ] A) `for a, b in zip(list1, list2):`
- [ ] B) `for a in list1 and b in list2:`
- [ ] C) `for (a, b) in [list1, list2]:`
- [ ] D) Cannot be done

**4. What does `break` do in a loop?**

- [ ] A) Skips the current iteration
- [ ] B) Exits the loop entirely
- [ ] C) Pauses the loop
- [ ] D) Restarts the loop

**5. What is the output?**

```python
for i in range(3):
    if i == 1:
        continue
    print(i)
```

- [ ] A) 0 1 2
- [ ] B) 0 2
- [ ] C) 1
- [ ] D) 0 1

**6. How do you create a training loop for 10 epochs?**

- [ ] A) `for epoch in range(10):`
- [ ] B) `while epoch < 10:`
- [ ] C) Both A and B
- [ ] D) Neither A nor B

**7. What is the benefit of using `enumerate()` over manual indexing?**

- [ ] A) Faster execution
- [ ] B) Cleaner, more readable code
- [ ] C) Works with more data types
- [ ] D) No benefit

### Answers

<details>
<summary>Click to reveal answers</summary>

1. **B) 5** - `range(5)` produces 0, 1, 2, 3, 4 (5 values)
2. **C) Index-value pairs** - `enumerate()` returns `(index, value)` tuples
3. **A) `for a, b in zip(list1, list2):`** - `zip()` pairs elements from multiple iterables
4. **B) Exits the loop entirely** - Used for early stopping
5. **B) 0 2** - `continue` skips iteration when `i == 1`
6. **C) Both A and B** - Both work, but for loop is more Pythonic
7. **B) Cleaner, more readable code** - Avoids manual index management

**Training Loop Pattern:**
```python
for epoch in range(num_epochs):
    for batch_idx, batch_data in enumerate(dataloader):
        # Training step
        loss = train_step(batch_data)
        if loss < threshold:
            break  # Early stopping
```

</details>

---

## Lesson 5: NumPy Essentials {#lesson-5}

### Quiz Questions

**1. Why is NumPy faster than Python lists?**

- [ ] A) Written in C
- [ ] B) Uses vectorized operations
- [ ] C) Stores data in contiguous memory
- [ ] D) All of the above

**2. What is the shape of `np.zeros((3, 4))`?**

- [ ] A) (3,)
- [ ] B) (4,)
- [ ] C) (3, 4)
- [ ] D) (4, 3)

**3. What does broadcasting mean in NumPy?**

- [ ] A) Copying arrays
- [ ] B) Automatically matching array shapes for operations
- [ ] C) Sending data over network
- [ ] D) Flattening arrays

**4. How do you reshape a 1D array of 12 elements into (3, 4)?**

- [ ] A) `arr.reshape(3, 4)`
- [ ] B) `arr.shape = (3, 4)`
- [ ] C) Both A and B
- [ ] D) Cannot be done

**5. What is the output of `np.array([1, 2, 3]) * 2`?**

- [ ] A) `[1, 2, 3, 1, 2, 3]`
- [ ] B) `[2, 4, 6]`
- [ ] C) `6`
- [ ] D) Error

**6. How do you access the first row of a 2D array `arr`?**

- [ ] A) `arr[0]`
- [ ] B) `arr[0, :]`
- [ ] C) Both A and B
- [ ] D) `arr[:, 0]`

**7. What shape represents a batch of 32 RGB images of size 64x64?**

- [ ] A) (32, 64, 64, 3)
- [ ] B) (64, 64, 3, 32)
- [ ] C) (3, 64, 64, 32)
- [ ] D) (64, 64, 32, 3)

### Answers

<details>
<summary>Click to reveal answers</summary>

1. **D) All of the above** - NumPy is optimized at multiple levels
2. **C) (3, 4)** - 3 rows, 4 columns (row, col format)
3. **B) Automatically matching array shapes** - Allows operations on different-sized arrays
4. **C) Both A and B** - Both methods work for reshaping
5. **B) `[2, 4, 6]`** - Element-wise multiplication (broadcasting)
6. **C) Both A and B** - Both return the first row
7. **A) (32, 64, 64, 3)** - Batch first: (batch_size, height, width, channels)

**Critical Concepts:**
- NumPy is 10-100x faster than Python lists for numerical operations
- Shape format: (batch, height, width, channels) for images
- Broadcasting eliminates explicit loops
- All PyTorch tensors behave like NumPy arrays

</details>

---

## Lesson 6: Functions and Modules {#lesson-6}

### Quiz Questions

**1. How do you define a function that calculates accuracy?**

- [ ] A) `function accuracy(preds, targets):`
- [ ] B) `def accuracy(preds, targets):`
- [ ] C) `func accuracy(preds, targets):`
- [ ] D) `create accuracy(preds, targets):`

**2. What does a function without a `return` statement return?**

- [ ] A) 0
- [ ] B) None
- [ ] C) Error
- [ ] D) Empty string

**3. How do you import only the `nn` module from PyTorch?**

- [ ] A) `import torch.nn`
- [ ] B) `from torch import nn`
- [ ] C) Both A and B
- [ ] D) `import nn from torch`

**4. What is a lambda function?**

- [ ] A) A named function
- [ ] B) An anonymous one-line function
- [ ] C) A class method
- [ ] D) A built-in function

**5. What does `*args` allow in a function definition?**

- [ ] A) Variable number of positional arguments
- [ ] B) Variable number of keyword arguments
- [ ] C) Required arguments
- [ ] D) Default arguments

**6. How do you set a default value for a parameter?**

```python
def train(epochs=10):
    pass
```

- [ ] A) Correct
- [ ] B) `def train(epochs: 10):`
- [ ] C) `def train(epochs -> 10):`
- [ ] D) `def train(epochs ? 10):`

**7. What is the purpose of docstrings?**

- [ ] A) Execute code
- [ ] B) Document function purpose and parameters
- [ ] C) Import modules
- [ ] D) Define variables

### Answers

<details>
<summary>Click to reveal answers</summary>

1. **B) `def accuracy(preds, targets):`** - `def` keyword defines functions
2. **B) None** - Functions return `None` if no explicit return
3. **C) Both A and B** - Both import methods work
4. **B) An anonymous one-line function** - `lambda x: x * 2`
5. **A) Variable number of positional arguments** - `*args` collects extras
6. **A) Correct** - Use `=` to set default values
7. **B) Document function purpose** - Triple-quoted strings after `def`

**Best Practices:**
```python
def calculate_loss(predictions, targets, reduction='mean'):
    """
    Calculate loss between predictions and targets.

    Args:
        predictions: Model outputs
        targets: Ground truth labels
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value as float
    """
    # Implementation here
    pass
```

</details>

---

## Lesson 7: Mini-Project - MNIST {#lesson-7}

### Project Checklist

Use this checklist to verify your MNIST project is complete:

**Data Loading:**
- [ ] Successfully loaded MNIST dataset
- [ ] Verified data shape (8x8 images, 1797 samples)
- [ ] Checked for missing values

**Preprocessing:**
- [ ] Normalized pixel values to [0, 1] range
- [ ] Flattened images from 2D to 1D
- [ ] Split into training and test sets (80/20)

**Model Training:**
- [ ] Initialized Logistic Regression classifier
- [ ] Trained model on training data
- [ ] Achieved >90% test accuracy

**Evaluation:**
- [ ] Calculated accuracy on test set
- [ ] Created confusion matrix
- [ ] Identified most common misclassifications

**Visualization:**
- [ ] Plotted sample digits from dataset
- [ ] Visualized confusion matrix as heatmap
- [ ] Showed examples of correct and incorrect predictions

**Code Quality:**
- [ ] Used descriptive variable names
- [ ] Added comments explaining steps
- [ ] Created reusable functions
- [ ] Included error handling

### Challenge Questions

**1. What accuracy did your model achieve?**

Expected: >90% (typically 95-97% for MNIST digits dataset)

**2. Which digit pairs are most often confused?**

Common: 4 ↔ 9, 3 ↔ 8, 5 ↔ 3, 7 ↔ 1

**3. How can you improve model accuracy?**

- [ ] A) Use more training data
- [ ] B) Try different classifiers (SVM, Random Forest)
- [ ] C) Feature engineering (gradients, edges)
- [ ] D) All of the above

**Answer: D) All of the above**

**4. What is the computational advantage of Logistic Regression?**

- Fast training (seconds vs minutes for deep learning)
- No GPU required
- Simple, interpretable model
- Good baseline for comparison

### Final Assessment

**Score Your Project:**

- **Beginner (50-70%)**: Basic loading and training works
- **Intermediate (70-85%)**: Includes visualization and evaluation
- **Advanced (85-95%)**: Clean code, functions, error handling
- **Expert (95-100%)**: Additional features (cross-validation, hyperparameter tuning)

---

## Overall Course Assessment

### Self-Evaluation Checklist

After completing all 7 lessons, evaluate your skills:

**Python Fundamentals:**
- [ ] I can create and manipulate variables of different types
- [ ] I understand when to use `/` vs `//`
- [ ] I can format strings using f-strings
- [ ] I write clear, commented code

**Data Structures:**
- [ ] I can work with lists, dictionaries, and tuples
- [ ] I understand indexing and slicing
- [ ] I can use list comprehensions
- [ ] I can store model configurations in dictionaries

**Control Flow:**
- [ ] I can write for and while loops
- [ ] I understand enumerate, zip, and range
- [ ] I can use break and continue effectively
- [ ] I can implement training loops

**NumPy:**
- [ ] I understand the difference between lists and arrays
- [ ] I can create and reshape NumPy arrays
- [ ] I understand broadcasting
- [ ] I can work with multi-dimensional arrays

**Functions:**
- [ ] I can define and call functions
- [ ] I can use default arguments
- [ ] I understand lambda functions
- [ ] I can import and use external libraries

**Project Skills:**
- [ ] I can load and preprocess data
- [ ] I can train a machine learning model
- [ ] I can evaluate model performance
- [ ] I can visualize results

### Next Steps Based on Score

**Checked 0-20 boxes (0-33%):**
- Re-watch videos and re-read lessons
- Practice more in Colab notebooks
- Ask for help in GitHub Discussions

**Checked 21-40 boxes (34-66%):**
- Review weak areas
- Complete additional practice exercises
- Build a personal project

**Checked 41-50 boxes (67-83%):**
- Good understanding! Move to next course
- Start exploring PyTorch/TensorFlow
- Contribute to open-source projects

**Checked 51-60 boxes (84-100%):**
- Excellent mastery! You're ready for deep learning
- Begin "Introduction to ML" course
- Share your project on GitHub

---

## Additional Practice Problems

### Challenge 1: Data Preprocessing Pipeline

Create a function that:
1. Takes a list of training samples
2. Normalizes values to [0, 1]
3. Splits into train/val sets
4. Returns both sets as NumPy arrays

### Challenge 2: Training Logger

Build a class that:
1. Stores metrics for each epoch
2. Calculates running averages
3. Prints formatted logs
4. Exports results to CSV

### Challenge 3: Hyperparameter Grid Search

Implement a script that:
1. Defines a grid of hyperparameters
2. Trains models for each combination
3. Tracks best-performing configuration
4. Visualizes results

---

**Need Help?**

- 📖 [Python Official Documentation](https://docs.python.org/3/)
- 💬 [GitHub Discussions](https://github.com/rajgupt/ai-for-builders/discussions)
- 🎥 [Additional Video Resources](resources.md)
- 📧 Contact: Submit an issue on GitHub

---

[← Back to Course Home](index.md) | [Continue to Math Essentials →](../../core/)
