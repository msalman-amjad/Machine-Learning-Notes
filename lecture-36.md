# Lecture 36: Gradient Descent for Logistic Regression

**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

## 1. The Core Idea
This lecture explains the implementation of **gradient descent** for **logistic regression**, focusing on how to minimize the cost function $J(w, b)$ to find optimal model parameters. The goal is to iteratively adjust the parameters $w$ and $b$ to reach the global minimum of the cost function, allowing our logistic regression model to make accurate predictions.

## 2. Mathematical Foundation & Notation
To minimize $J(w, b)$, the algorithm performs **simultaneous updates** for the parameters $w_j$ and $b$.

### Parameter Update Equations:
$$w_j := w_j - \alpha \frac{\partial}{\partial w_j} J(w, b)$$
$$b := b - \alpha \frac{\partial}{\partial b} J(w, b)$$
Where:
- $\alpha$ is the learning rate.
- $w_j$ represents the $j$-th feature weight.
- $b$ is the bias term.

### Derivative Formulas:
For $n$ features and $m$ training examples, the partial derivatives are calculated as follows:
- For $w_j$: $\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})x_j^{(i)}$
- For $b$: $\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})$

### Logistic vs. Linear Regression
While the update equations appear identical to those in linear regression, the algorithms differ fundamentally in the definition of the prediction function $f(x)$:
- **Linear Regression:** $f(x) = wx + b$
- **Logistic Regression:** $f(x) = g(wx + b)$, where $g$ is the **sigmoid function**.
$$g(z) = \frac{1}{1 + e^{-z}}$$

## 3. Key Concepts / Vectorization / Visual Graphs

### Vectorization
While the equations above update parameters one at a time, **vectorized implementations** are used in practice to significantly increase computational speed, operating on the entire vectors $\vec{w}$ and $\vec{x}$ simultaneously.

### Feature Scaling
Scaling features to similar ranges (e.g., between -1 and 1) is recommended to ensure faster convergence. This mirrors the best practices used in linear regression, preventing the gradient descent path from oscillating excessively in skewed contours.

### Visual Graphs & Plots

#### Sigmoid Function Plot
The sigmoid function maps any real-valued number into the range $(0, 1)$.
```mermaid
xychart-beta
    title "Sigmoid Function"
    x-axis "z (Input)" [-10, -5, 0, 5, 10]
    y-axis "g(z) (Output)" 0 --> 1
    line [0.00004, 0.006, 0.5, 0.993, 0.9999]
```

#### Cost Function Contour Plot
The contour plot shows the cost $J$ with respect to parameters. Gradient descent takes orthogonal steps to the contours to reach the center (minimum).
```text
       w
   |    .-"""""-.
   |  .'  .-""-.  '.
   | /   /      \   \
   | |   |  x   |   |  <- Global Minimum
   | \   \      /   /
   |  '.  '-..-'  .'
   |    '-......-'
   +----------------- b
```

#### Learning Curve
The learning curve tracks the reduction of the cost function $J$ over iterations of gradient descent.
```mermaid
xychart-beta
    title "Learning Curve"
    x-axis "Iterations" [0, 100, 200, 300, 400]
    y-axis "Cost J(w,b)" 0.0 --> 1.0
    line [0.9, 0.5, 0.25, 0.15, 0.1]
```

## 4. Code Implementation
Below is a conceptual Python and NumPy snippet illustrating the vectorized implementation of gradient descent for logistic regression.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient_logistic(X, y, w, b):
    """
    Computes the gradient for logistic regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m = X.shape[0]
    
    # Predictions
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    # Error
    error = f_wb - y
    
    # Gradients
    dj_dw = (1/m) * np.dot(X.T, error)
    dj_db = (1/m) * np.sum(error)
    
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        
        # Simultaneous update
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    return w, b
```

## 5. Key Terminology & Pitfalls

- **Simultaneous Updates:** Always update $w$ and $b$ simultaneously. Do not use an updated $w_j$ to calculate the updates for the remaining parameters in the same iteration.
- **Identical Equations, Different Functions:** The gradient update formula looks exactly like linear regression. The crucial difference is that the prediction function $f(x)$ is now wrapped in the sigmoid function $g(z)$.
- **Vectorization is Crucial:** For large datasets, `for` loops are too slow. Rely on NumPy's vectorized operations (`np.dot`).
- **Feature Scaling:** If your features are on vastly different scales (e.g., age 0-100 and income 0-1,000,000), gradient descent will struggle to converge. Always apply feature scaling.
- **scikit-learn:** In practice, libraries like `scikit-learn` are used to efficiently train logistic regression models for classification tasks out-of-the-box.

> [!TIP]
> Ensure all variables ($X$, $y$, $w$, $b$) have compatible dimensions before performing matrix multiplications (e.g., $X$ of shape $(m, n)$ and $w$ of shape $(n, 1)$).
