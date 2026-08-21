# Lecture 5: Supervised Learning Part 2 (Classification)
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 1, Lesson 2 (Lecture 5)

## 1. The Core Idea
While **regression** models predict continuous numbers along an infinite numerical spectrum, **Classification** is the second major branch of supervised learning focused on predicting **discrete categories** (a finite set of outputs).

Instead of fitting a line to estimate a continuous value, classification algorithms learn a mathematical **decision boundary** that separates data points into distinct classes or categories (e.g., classifying an email as *Spam* vs. *Not Spam*, or a medical tumor as *Benign* vs. *Malignant*).

---

## 2. Mathematical Foundation & Notation

### Binary Classification Formulation:
In binary classification, the target output $y$ takes on only two possible discrete values, typically encoded as $0$ or $1$:

$$y \in \{0, 1\}$$

* **$y = 0$:** Negative Class (e.g., Benign tumor, Not Spam, Normal transaction).
* **$y = 1$:** Positive Class (e.g., Malignant tumor, Spam email, Fraudulent transaction).

*(Note: The terms "positive" and "negative" indicate the presence or absence of a condition, not a moral judgment).*

---

### Multi-Feature Input Vectors:
When classifying based on multiple features, we group inputs into a feature vector $\vec{x}$:

$$\vec{x} = [x_1, x_2, \dots, x_n]$$

**Example (Medical Diagnosis):**
* $x_1$: Tumor size (in cm)
* $x_2$: Patient age (in years)
* $\vec{x} = [x_1, x_2]$

---

### Step 1: Linear Decision Function
We compute a weighted linear combination of input features plus a bias term:

$$z = \vec{w} \cdot \vec{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

---

### Step 2: Thresholding Decision Boundary
The line or surface where $z = 0$ forms the **Decision Boundary**. The decision logic predicts the discrete output class as follows:

$$\hat{y} = \begin{cases} 1 & \text{if } z = \vec{w} \cdot \vec{x} + b \ge 0 \quad \text{(Malignant)} \\ 0 & \text{if } z = \vec{w} \cdot \vec{x} + b < 0 \quad \text{(Benign)} \end{cases}$$

---

## 3. Key Concepts & The Decision Boundary

### 1. The Medical Tumor Diagnosis Example:
Consider predicting whether a breast tumor is benign or malignant:
* **Single Feature ($x_1$):** If we plot tumor size along a 1D line, the algorithm finds a threshold (e.g., size $\ge 3\text{ cm}$) separating benign tumors ($0$) from malignant tumors ($1$).
* **Two Features ($x_1, x_2$):** By plotting tumor size ($x_1$) against patient age ($x_2$) on a 2D grid, the decision boundary becomes a line separating positive data points (red crosses) from negative data points (blue circles).

### 2. Multi-Class Classification:
Classification is not restricted to binary ($0/1$) outcomes. **Multi-class classification** handles datasets with $K > 2$ categories:
* Diagnosing tumor types into Class $0$ (Benign), Class $1$ (Type A Malignant), or Class $2$ (Type B Malignant).
* Categorizing images into Cat ($0$), Dog ($1$), or Horse ($2$).

### 3. Non-Numeric Input Types:
Classification models can process complex inputs like raw image pixels, audio files, or textual documents by converting them into feature vectors $\vec{x}$.

---

## 4. Comparison: Regression vs. Classification

| Property | Regression | Classification |
| :--- | :--- | :--- |
| **Output Type ($y$)** | Continuous numerical value | Discrete categorical values |
| **Number of Possible Outputs**| Infinitely many values | Finite set of classes (e.g., $\{0, 1\}$ or $\{0, \dots, K-1\}$) |
| **Primary Goal** | Fit a function line/curve through data | Learn a decision boundary separating classes |
| **Example Target Output** | Predicting house price: $\$345,200.00$ | Predicting tumor status: $1$ (Malignant) or $0$ (Benign) |

---

## 5. Code Implementation

Below is a Python implementation showing how a classification decision boundary uses feature weights and thresholding to classify tumor diagnoses.

```python
import numpy as np

def predict_tumor_type(tumor_size, patient_age, w, b):
    """
    Classifies a tumor as Malignant (1) or Benign (0) based on a linear decision boundary.
    
    Args:
      tumor_size (float): Size of tumor in cm (x1)
      patient_age (float): Age of patient in years (x2)
      w (ndarray): Weight parameters [w1, w2]
      b (float): Bias parameter scalar
      
    Returns:
      class_id (int): 1 for Malignant, 0 for Benign
      class_label (str): Human-readable diagnosis label
    """
    # Create feature vector x = [x1, x2]
    x = np.array([tumor_size, patient_age])
    
    # Compute linear score z = w . x + b
    z = np.dot(w, x) + b
    
    # Thresholding logic at z = 0 boundary
    if z >= 0:
        return 1, "Malignant"
    else:
        return 0, "Benign"


# Learned parameters defining the decision boundary line
w_learned = np.array([1.5, 0.4])  # Weights for [tumor_size, patient_age]
b_learned = -8.0                  # Bias term

# Test Case 1: Small tumor (1.5 cm) in a 40-year-old patient
id1, label1 = predict_tumor_type(1.5, 40.0, w_learned, b_learned)
print(f"Patient 1 (Size: 1.5 cm, Age: 40): {label1} (Class {id1})")

# Test Case 2: Large tumor (4.5 cm) in a 50-year-old patient
id2, label2 = predict_tumor_type(4.5, 50.0, w_learned, b_learned)
print(f"Patient 2 (Size: 4.5 cm, Age: 50): {label2} (Class {id2})")
```

---

## 6. Key Terminology & Takeaways

### Key Takeaways:
* **Discrete Category Predictions:** Classification focuses on predicting discrete class labels $y \in \{0, 1\}$ (or multi-class $\{0, 1, \dots, K-1\}$) rather than continuous numbers.
* **Decision Boundary Concept:** Algorithms learn a geometric decision boundary $z = \vec{w} \cdot \vec{x} + b = 0$ separating positive and negative class instances in feature space.
* **Feature Vector Flexibility:** Works across numeric metrics, multi-dimensional feature vectors, and non-numeric data structures like image pixel arrays.
