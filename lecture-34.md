# Lecture 34: Cost Function for Logistic Regression
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 3, Lesson 2, Video 1 (Lecture 34)

## 1. The Core Idea

In linear regression, we evaluated model performance using the **squared error cost function**. However, when transitioning to **logistic regression** for binary classification, using squared error leads to a critical mathematical flaw: the resulting cost surface is **non-convex**, filled with local minima and flat plateaus that prevent gradient descent from reliably converging to optimal parameters.

To overcome this, logistic regression requires a specially designed **convex loss function**—the **Log Loss** (also known as **Binary Cross-Entropy Loss**). This lecture examines why squared error fails for logistic regression, derives the log loss function for individual training examples, provides detailed graphical intuition for how it penalizes errors, and defines the overall convex cost function $J(\vec{w}, b)$ for the entire dataset.

---

## 2. Why Squared Error Fails for Logistic Regression (Non-Convexity)

### 1. The Linear Regression Cost Function Recall
In linear regression, the model prediction is $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$, and the cost function is defined as:

$$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{2} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)^2$$

Because $f_{\vec{w},b}(\vec{x})$ is a linear function of $\vec{w}$ and $b$, squaring the error term produces a quadratic equation. Plotted against parameters $(\vec{w}, b)$, $J(\vec{w}, b)$ forms a smooth, bowl-shaped **convex surface** with a single global minimum.

### 2. Substituting Logistic Regression Hypothesis into Squared Error
In logistic regression, the hypothesis incorporates the non-linear sigmoid activation function $g(z)$:

$$f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$

If we attempt to plug this logistic hypothesis directly into the squared error cost function:

$$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{2} \left( \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x}^{(i)} + b)}} - y^{(i)} \right)^2$$

The non-linear, exponential nature of the sigmoid function inside the squared difference breaks the quadratic structure. When plotted against $w$ and $b$, the resulting cost surface becomes **non-convex**.

### 3. Convex vs. Non-Convex Surface Comparison

```text
    CONVEX SURFACE (Linear Regression)           NON-CONVEX SURFACE (Logistic + Squared Error)
    
           J(w,b) ^                                     J(w,b) ^
                  |  \       /                                 |  \  /\   /
                  |   \     /                                  |   \/  \ /  \  /\
                  |    \___/                                   |   /\__/ \__/  \/
                  |  (Global Min)                              | (Local Min) (Global Min)
                  +--------------> w                           +--------------------------> w
                  Gradient descent always                      Gradient descent gets stuck in
                  converges to global minimum.                 local minima or flat regions.
```

### Key Consequences of Non-Convexity:
* **Multiple Local Minima:** Gradient descent can get trapped in sub-optimal local minima depending on parameter initialization.
* **Flat Plateaus:** Derivatives in flat regions approach zero, causing gradient descent to stall or update parameters extremely slowly.
* **Lack of Convergence Guarantee:** There is no guarantee that gradient descent will find the parameter set that minimizes classification error.

---

## 3. The Logistic Loss Function (Single Training Example)

To guarantee a **convex** cost surface, we define a specialized loss function $L\left(f_{\vec{w},b}(\vec{x}), y\right)$ that computes the error for a **single** training example $(\vec{x}, y)$.

> **Terminology Note:**
> * **Loss $L\left(f_{\vec{w},b}(\vec{x}), y\right)$:** Measures the prediction error on a **single** training example.
> * **Cost $J(\vec{w}, b)$:** Measures the average loss over the **entire dataset** of $m$ training examples.

### Piecewise Definition of Logistic Loss
$$L\left(f_{\vec{w},b}(\vec{x}), y\right) = \begin{cases} -\log\left(f_{\vec{w},b}(\vec{x})\right) & \text{if } y = 1 \\ -\log\left(1 - f_{\vec{w},b}(\vec{x})\right) & \text{if } y = 0 \end{cases}$$

Where $\log$ refers to the natural logarithm ($\ln$, base $e$).

### Unified Single-Line Expression (Binary Cross-Entropy Loss)
Since $y \in \{0, 1\}$, we can combine both conditional cases into a single mathematical expression:

$$L\left(f_{\vec{w},b}(\vec{x}), y\right) = -y \log\left(f_{\vec{w},b}(\vec{x})\right) - (1 - y) \log\left(1 - f_{\vec{w},b}(\vec{x})\right)$$

* When $y = 1$: The second term $(1 - 1)\log(1 - f) = 0$, leaving $L = -\log(f)$.
* When $y = 0$: The first term $-0 \log(f) = 0$, leaving $L = -\log(1 - f)$.

---

## 4. Graphical Intuition & Behavior Analysis

### Case 1: When Actual Label $y = 1$
For a positive example ($y = 1$), the loss function simplifies to $Loss = -\log\left(f_{\vec{w},b}(\vec{x})\right)$.

* **As $f_{\vec{w},b}(\vec{x}) \to 1$:** The prediction matches the true label ($y=1$). $-\log(1) = 0$, so **$Loss \to 0$**.
* **As $f_{\vec{w},b}(\vec{x}) \to 0$:** The model confidently predicts the wrong class ($\hat{y}=0$). $-\log(f) \to \infty$, so **$Loss \to \infty$**.

```text
    Loss L(f, y=1)
      ^
  Inf | \
      |  \
      |   \
      |    \
      |     \
      |      '.
    0 +--------''-------------------> f_{w,b}(x)
      0.0       0.5             1.0
      (Very wrong prediction:   (Correct prediction:
       Target=1, Model=0)       Target=1, Model=1)
       Loss -> Infinity         Loss = 0
```

### Case 2: When Actual Label $y = 0$
For a negative example ($y = 0$), the loss function simplifies to $Loss = -\log\left(1 - f_{\vec{w},b}(\vec{x})\right)$.

* **As $f_{\vec{w},b}(\vec{x}) \to 0$:** The prediction matches the true label ($y=0$). $-\log(1 - 0) = -\log(1) = 0$, so **$Loss \to 0$**.
* **As $f_{\vec{w},b}(\vec{x}) \to 1$:** The model confidently predicts the wrong class ($\hat{y}=1$). $-\log(1 - 1) = -\log(0) \to \infty$, so **$Loss \to \infty$**.

```text
    Loss L(f, y=0)
      ^
  Inf |                         /
      |                        /
      |                       /
      |                      /
      |                     /
      |                  .'
    0 +----------------''-----------> f_{w,b}(x)
      0.0       0.5             1.0
      (Correct prediction:      (Very wrong prediction:
       Target=0, Model=0)        Target=0, Model=1)
       Loss = 0                  Loss -> Infinity
```

### Summary of Loss Behavior:
1. **Zero Penalty for Accurate Predictions:** If the model predicts $f_{\vec{w},b}(\vec{x}) = y$, the loss is exactly 0.
2. **Infinite Penalty for Confidently Incorrect Predictions:** If $y=1$ but $f_{\vec{w},b}(\vec{x}) \to 0$ (or $y=0$ but $f_{\vec{w},b}(\vec{x}) \to 1$), the loss approaches infinity. This provides an aggressive, steep gradient force that pushes parameters away from catastrophic misclassifications.

---

## 5. The Overall Convex Cost Function $J(\vec{w}, b)$

The total cost function $J(\vec{w}, b)$ across $m$ training examples is the arithmetic mean of individual example losses:

$$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right)$$

Substituting the unified binary cross-entropy loss formula:

$$J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) + (1 - y^{(i)}) \log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right) \right]$$

### Convexity Guarantee & Optimization Advantage
* **Mathematical Property:** By replacing squared error with log loss, the resulting cost function $J(\vec{w}, b)$ is mathematically proven to be **strictly convex** with respect to parameters $\vec{w}$ and $b$.
* **Global Convergence:** Gradient descent run on $J(\vec{w}, b)$ is guaranteed to reach the global minimum (given an appropriate learning rate $\alpha$).

---

## 6. Code Implementations & Visualizations

### 1. NumPy & Matplotlib Implementation: Non-Convex vs. Convex Cost Surfaces & Loss Curves

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))

def squared_error_loss(f, y):
    """Squared error loss per example: 0.5 * (f - y)^2."""
    return 0.5 * (f - y)**2

def logistic_loss(f, y, eps=1e-15):
    """Binary cross-entropy loss per example: -y log(f) - (1-y) log(1-f)."""
    # Clip f to prevent log(0) numerical instability
    f = np.clip(f, eps, 1 - eps)
    return -y * np.log(f) - (1 - y) * np.log(1 - f)

# -------------------------------------------------------------
# 1. Visualizing Loss Functions L(f, y) for y=1 and y=0
# -------------------------------------------------------------
f_vals = np.linspace(0.001, 0.999, 500)

loss_y1 = logistic_loss(f_vals, y=1)
loss_y0 = logistic_loss(f_vals, y=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot y = 1 case
axes[0].plot(f_vals, loss_y1, color='crimson', linewidth=2.5, label=r'$L(f, y=1) = -\log(f)$')
axes[0].set_title('Logistic Loss when True Label $y = 1$', fontsize=12)
axes[0].set_xlabel('Model Output $f_{w,b}(x)$', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(fontsize=11)
axes[0].axvline(x=1.0, color='gray', linestyle=':', label='Target = 1')

# Plot y = 0 case
axes[1].plot(f_vals, loss_y0, color='navy', linewidth=2.5, label=r'$L(f, y=0) = -\log(1-f)$')
axes[1].set_title('Logistic Loss when True Label $y = 0$', fontsize=12)
axes[1].set_xlabel('Model Output $f_{w,b}(x)$', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(fontsize=11)
axes[1].axvline(x=0.0, color='gray', linestyle=':', label='Target = 0')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 2. Comparing Cost Surfaces (Squared Error vs. Binary Cross Entropy)
# -------------------------------------------------------------
# Synthetic 1D binary classification dataset
X = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
y = np.array([0, 0, 0, 1, 1, 1])

w_vals = np.linspace(-3, 5, 300)
b_fixed = 0.0

cost_squared = []
cost_bce = []

for w in w_vals:
    f = sigmoid(w * X + b_fixed)
    # Compute average squared error cost
    c_sq = np.mean(squared_error_loss(f, y))
    # Compute average binary cross-entropy cost
    c_bce = np.mean(logistic_loss(f, y))
    
    cost_squared.append(c_sq)
    cost_bce.append(c_bce)

plt.figure(figsize=(9, 5))
plt.plot(w_vals, cost_squared, color='orange', linewidth=2.5, label='Squared Error Cost (Non-Convex / Multi-local minima)')
plt.plot(w_vals, cost_bce, color='green', linewidth=2.5, label='Logistic Log-Loss Cost (Convex / Smooth Bowl)')
plt.title('Cost Function Surface vs Weight Parameter $w$', fontsize=13)
plt.xlabel('Weight $w$', fontsize=11)
plt.ylabel('Cost $J(w, b=0)$', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.show()
```

---

### 2. Industry Implementation: Loss Computation with Scikit-Learn & PyTorch

```python
from sklearn.metrics import log_loss
import torch
import torch.nn as nn

# -------------------------------------------------------------
# Scikit-Learn Log-Loss Evaluation
# -------------------------------------------------------------
y_true = [1, 0, 1, 0]
y_pred_probs = [0.9, 0.1, 0.8, 0.3]  # Model output probabilities f(x)

sk_loss = log_loss(y_true, y_pred_probs)
print(f"Scikit-Learn Log-Loss: {sk_loss:.4f}")

# -------------------------------------------------------------
# PyTorch Binary Cross-Entropy Loss (BCELoss & BCEWithLogitsLoss)
# -------------------------------------------------------------
# 1. Using nn.BCELoss (Requires probabilities from Sigmoid)
bce_loss_fn = nn.BCELoss()
probabilities = torch.tensor([0.9, 0.1, 0.8, 0.3], dtype=torch.float32)
targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

pytorch_bce = bce_loss_fn(probabilities, targets)
print(f"PyTorch BCELoss: {pytorch_bce.item():.4f}")

# 2. Using nn.BCEWithLogitsLoss (Combines Sigmoid + BCE for numerical stability)
bce_logits_fn = nn.BCEWithLogitsLoss()
logits = torch.tensor([2.197, -2.197, 1.386, -0.847], dtype=torch.float32)  # Raw z = w*x + b values

pytorch_logits_bce = bce_logits_fn(logits, targets)
print(f"PyTorch BCEWithLogitsLoss: {pytorch_logits_bce.item():.4f}")
```

---

## 7. Key Takeaways & Summary Checklist

1. **Squared Error Flaw:** Using squared error with the non-linear sigmoid hypothesis yields a **non-convex cost function** with local minima where gradient descent gets stuck.
2. **Logistic Loss Formula:** For a single example, $L(f, y) = -y \log(f) - (1-y) \log(1-f)$.
3. **Loss Intuition:** 
   * If correct ($f=y$), $Loss = 0$.
   * If wrong ($f \to 1-y$), $Loss \to \infty$, penalizing confident wrong predictions exponentially.
4. **Overall Cost Function:** $J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f^{(i)}) + (1-y^{(i)}) \log(1-f^{(i)}) \right]$.
5. **Convexity Guarantee:** Log loss produces a smooth, bowl-shaped convex cost surface $J(\vec{w}, b)$, guaranteeing gradient descent converges to the global minimum.

---

## 8. Next Steps

In the next lecture (**Lecture 35**), we will examine the **Simplified Cost Function and Gradient Descent for Logistic Regression**, deriving the parameter updates ($\frac{\partial J}{\partial w_j}$ and $\frac{\partial J}{\partial b}$) and discovering their remarkable mathematical similarity to the gradient descent updates in linear regression.
