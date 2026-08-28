# Lecture 35: Simplified Cost Function for Logistic Regression
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 3, Lesson 2, Video 2 (Lecture 35)

## 1. The Core Idea

In the previous lecture, we established why logistic regression requires a non-squared error loss function and introduced a **piece-wise loss function** defined separately for $y=1$ and $y=0$.

While the piece-wise formulation provides clear graphical intuition, writing conditional `if/else` logic in code and optimization equations is mathematically cumbersome and computationally inefficient. 

This lecture derives the **simplified single-line logistic loss function** and aggregates it to form the standard **logistic cost function** $J(\vec{w}, b)$. We explore the statistical origin of this cost function in **Maximum Likelihood Estimation (MLE)**, revisit its guaranteed **convexity**, and visually demonstrate how different decision boundaries (e.g., a poor magenta boundary vs. an optimal blue boundary) directly map to higher or lower cost values.

---

## 2. Derivation of the Simplified Logistic Loss Function

### 1. Recapping the Piece-wise Loss Function
For a single training example $(\vec{x}, y)$ with model prediction $f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b) \in (0, 1)$:

$$L\left(f_{\vec{w},b}(\vec{x}), y\right) = \begin{cases} -\log\left(f_{\vec{w},b}(\vec{x})\right) & \text{if } y = 1 \\ -\log\left(1 - f_{\vec{w},b}(\vec{x})\right) & \text{if } y = 0 \end{cases}$$

### 2. Collapsing into a Single Equation
Because binary classification constrains $y$ to strictly two discrete values ($y \in \{0, 1\}$), we can combine both cases into a single, compact equation using algebraic weighting terms $y$ and $(1 - y)$:

$$L\left(f_{\vec{w},b}(\vec{x}), y\right) = -y \log\left(f_{\vec{w},b}(\vec{x})\right) - (1 - y) \log\left(1 - f_{\vec{w},b}(\vec{x})\right)$$

### 3. Mathematical Proof of Equivalence

#### Case 1: When $y = 1$
Substitute $y = 1$ into the unified formula:
$$L\left(f_{\vec{w},b}(\vec{x}), 1\right) = -(1) \log\left(f_{\vec{w},b}(\vec{x})\right) - (1 - 1) \log\left(1 - f_{\vec{w},b}(\vec{x})\right)$$
$$L\left(f_{\vec{w},b}(\vec{x}), 1\right) = -\log\left(f_{\vec{w},b}(\vec{x})\right) - (0) \log\left(1 - f_{\vec{w},b}(\vec{x})\right)$$
$$L\left(f_{\vec{w},b}(\vec{x}), 1\right) = -\log\left(f_{\vec{w},b}(\vec{x})\right) \quad \checkmark$$

#### Case 2: When $y = 0$
Substitute $y = 0$ into the unified formula:
$$L\left(f_{\vec{w},b}(\vec{x}), 0\right) = -(0) \log\left(f_{\vec{w},b}(\vec{x})\right) - (1 - 0) \log\left(1 - f_{\vec{w},b}(\vec{x})\right)$$
$$L\left(f_{\vec{w},b}(\vec{x}), 0\right) = 0 - (1) \log\left(1 - f_{\vec{w},b}(\vec{x})\right)$$
$$L\left(f_{\vec{w},b}(\vec{x}), 0\right) = -\log\left(1 - f_{\vec{w},b}(\vec{x})\right) \quad \checkmark$$

> **Key Benefit:** This single-line formula enables seamless vectorization in NumPy/PyTorch/TensorFlow without branching statements or `if-else` blocks.

---

## 3. Derivation of the Simplified Cost Function $J(\vec{w}, b)$

The total cost function $J(\vec{w}, b)$ measures the average loss across all $m$ training examples:

$$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right)$$

Substituting our simplified single-example loss function:

$$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \left[ -y^{(i)} \log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) - (1 - y^{(i)}) \log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right) \right]$$

Factoring out the negative sign gives the standard formula used across machine learning literature:

$$J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) + (1 - y^{(i)}) \log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right) \right]$$

---

## 4. Conceptual Context & Maximum Likelihood Estimation (MLE)

### 1. Statistical Origin: Maximum Likelihood Estimation (MLE)
Where does this specific loss function come from? It is not an arbitrary invention; it stems directly from the statistical framework of **Maximum Likelihood Estimation**:

1. In binary logistic regression, the hypothesis represents conditional probability:
   $$P(y=1 \mid \vec{x}) = f_{\vec{w},b}(\vec{x})$$
   $$P(y=0 \mid \vec{x}) = 1 - f_{\vec{w},b}(\vec{x})$$

2. Combining these into a single probability density function (Bernoulli distribution):
   $$P(y \mid \vec{x}) = \left(f_{\vec{w},b}(\vec{x})\right)^{y} \left(1 - f_{\vec{w},b}(\vec{x})\right)^{1 - y}$$

3. Assuming $m$ independent training samples, the likelihood function $L(\vec{w}, b)$ is the product of individual probabilities:
   $$L(\vec{w}, b) = \prod_{i=1}^{m} \left(f_{\vec{w},b}(\vec{x}^{(i)})\right)^{y^{(i)}} \left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right)^{1 - y^{(i)}}$$

4. Taking the negative log likelihood ($-\log L(\vec{w}, b)$) converts the product into a sum of logarithms and scales by $\frac{1}{m}$, yielding precisely our **logistic cost function** $J(\vec{w}, b)$!

> **Takeaway:** Minimizing $J(\vec{w}, b)$ via gradient descent is mathematically identical to finding the maximum likelihood estimate of parameters $\vec{w}$ and $b$.

### 2. Convexity Guarantee
Because $J(\vec{w}, b)$ is derived via MLE for Bernoulli targets with a logistic link function, it is **strictly convex**. Plotted in parameter space $(\vec{w}, b)$, $J(\vec{w}, b)$ is a single smooth, bowl-shaped surface without sub-optimal local minima.

```text
               J(w,b) ^
                      |  \               /
                      |   \             /
                      |    \___________/
                      |    (Global Min)
                      +---------------------> w
                      Convex Cost Surface guarantees
                      Gradient Descent always reaches Global Minimum
```

### 3. Visualizing Decision Boundaries vs. Cost Value
The value of $J(\vec{w}, b)$ directly measures how accurately the decision boundary separates the positive class ($y=1$) from the negative class ($y=0$).

```text
    x_2 ^                                 x_2 ^
        |   o    o     /                      |   o    o     |
        |           /  (Magenta Boundary)     |              |  (Blue Boundary)
        |  o    o  /                          |  o    o      |
        |--------/-------------------         |--------------|--------------------
        |       /   x    x                    |              |    x    x
        |      /                              |              |
        |     /      x    x                   |              |     x    x
        +-----------------------------> x_1    +-----------------------------> x_1
        
        Poor Separating Boundary               Optimal Separating Boundary
        (Misclassifies several points)          (Clearly separates classes)
        --> HIGH COST J(w,b)                    --> LOW COST J(w,b)
```

---

## 5. Code Implementations & Visualizations

### 1. Vectorized NumPy Implementation of Cost Function

```python
import numpy as np

def sigmoid(z):
    """Computes sigmoid function g(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))

def compute_cost_logistic(X, y, w, b, eps=1e-15):
    """
    Computes the simplified logistic cost function J(w, b).

    Args:
      X (ndarray (m,n)): Feature data matrix
      y (ndarray (m,)):  Target vector (0 or 1)
      w (ndarray (n,)):  Weight parameter vector
      b (scalar):        Bias parameter
      eps (float):       Epsilon clipping value to avoid log(0)

    Returns:
      total_cost (float): Cost J(w, b)
    """
    m = X.shape[0]
    
    # Linear combination z = X @ w + b
    z = np.dot(X, w) + b
    
    # Model predictions f(x) = sigmoid(z)
    f = sigmoid(z)
    
    # Clip probabilities to protect against numerical underflow in log()
    f = np.clip(f, eps, 1.0 - eps)
    
    # Vectorized Binary Cross-Entropy Cost calculation
    cost = - (1.0 / m) * np.sum(y * np.log(f) + (1 - y) * np.log(1 - f))
    
    return cost

# -------------------------------------------------------------
# Demo: Comparing Cost for Two Candidate Parameter Sets
# -------------------------------------------------------------
X_train = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Model 1 (Poor parameters - Magenta boundary)
w_poor = np.array([-0.5, -0.5])
b_poor = 0.0

# Model 2 (Good parameters - Blue boundary)
w_good = np.array([1.0, 1.0])
b_good = -3.0

cost_poor = compute_cost_logistic(X_train, y_train, w_poor, b_poor)
cost_good = compute_cost_logistic(X_train, y_train, w_good, b_good)

print(f"Cost for Poor Model (Magenta Boundary): {cost_poor:.4f}")
print(f"Cost for Good Model (Blue Boundary):    {cost_good:.4f}")
```

---

### 2. Industry Implementation: Scikit-Learn & PyTorch

```python
from sklearn.metrics import log_loss
import torch
import torch.nn as nn

# -------------------------------------------------------------
# 1. Scikit-Learn Log-Loss Evaluation
# -------------------------------------------------------------
y_true = np.array([0, 0, 0, 1, 1, 1])
# Model probabilities output
y_pred_probs = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.8])

sk_cost = log_loss(y_true, y_pred_probs)
print(f"Scikit-Learn Log Loss Cost: {sk_cost:.4f}")

# -------------------------------------------------------------
# 2. PyTorch Binary Cross-Entropy Loss
# -------------------------------------------------------------
bce_loss = nn.BCELoss()

preds_tensor = torch.tensor([0.1, 0.2, 0.15, 0.85, 0.9, 0.8], dtype=torch.float32)
targets_tensor = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float32)

torch_cost = bce_loss(preds_tensor, targets_tensor)
print(f"PyTorch BCELoss Cost:       {torch_cost.item():.4f}")
```

---

## 6. Key Takeaways & Summary Checklist

1. **Simplified Loss Formula:** $L(f, y) = -y \log(f) - (1-y) \log(1-f)$ unifies conditional cases into one equation.
2. **Simplified Cost Formula:** $J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) + (1 - y^{(i)}) \log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right) \right]$.
3. **Statistical Derivation:** Cost function originates from **Maximum Likelihood Estimation (MLE)** assuming a Bernoulli distribution.
4. **Convexity Guarantee:** Strictly convex surface guarantees gradient descent converges to the global minimum.
5. **Decision Boundary Link:** A decision boundary that cleanly separates classes yields a lower cost $J(\vec{w}, b)$ than a poor boundary.

---

## 7. Next Steps

Now that we have derived the simplified convex cost function $J(\vec{w}, b)$, the next lecture (**Lecture 36**) will apply **Gradient Descent for Logistic Regression** to derive parameter update rules for $\vec{w}$ and $b$.
