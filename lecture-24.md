# Lecture 24: Gradient Descent for Multiple Linear Regression
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

## 1. The Core Idea
In previous lectures, we saw how multiple linear regression uses vector notation to compact multiple feature parameters into vector forms. However, to actually train the model and learn optimal parameters from data, we need an efficient optimization algorithm.

**Gradient Descent for Multiple Linear Regression** extends the standard gradient descent algorithm to work seamlessly with parameter vectors. By replacing individual scalar calculations with vectorized dot products and array operations, we can compute prediction errors and update all weights simultaneously across hundreds or thousands of features.

We also explore an analytical alternative known as the **Normal Equation**, examining its trade-offs and understanding why gradient descent remains the algorithm of choice in modern machine learning.

---

## 2. Mathematical Foundation & Notation

To formulate gradient descent for multiple linear regression, let us define our variables clearly:

### Variables & Notation:
* **$m$**: Total number of training examples in the dataset.
* **$n$**: Total number of input features per example.
* **$\vec{x}^{(i)}$**: Feature vector for the $i^{th}$ training example, $\vec{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}, \dots, x_n^{(i)}]$.
* **$x_j^{(i)}$**: Value of feature $j$ in the $i^{th}$ training example.
* **$y^{(i)}$**: Target value (label) for the $i^{th}$ training example.
* **$\vec{w}$**: Weight parameter vector of length $n$, $\vec{w} = [w_1, w_2, \dots, w_n]$.
* **$b$**: Bias parameter (scalar).
* **$\alpha$**: Learning rate hyperparameter controlling step size.

---

### Step 1: Model Prediction Function
Using vector notation, the model prediction for a single training example $\vec{x}^{(i)}$ is given by the dot product of weight vector $\vec{w}$ and feature vector $\vec{x}^{(i)}$, plus the scalar bias $b$:

$$f_{\vec{w}, b}(\vec{x}^{(i)}) = \vec{w} \cdot \vec{x}^{(i)} + b = \sum_{j=1}^{n} w_j x_j^{(i)} + b$$

---

### Step 2: Cost Function $J(\vec{w}, b)$
The cost function measures the mean squared error (MSE) across all $m$ training examples:

$$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)^2$$

---

### Step 3: Gradient Descent Update Rules
To find the values of $\vec{w}$ and $b$ that minimize $J(\vec{w}, b)$, gradient descent repeatedly updates all parameters simultaneously:

$$\text{Repeat until convergence: } \{$$
$$w_j = w_j - \alpha \frac{\partial J(\vec{w}, b)}{\partial w_j} \quad \text{for } j = 1, \dots, n$$
$$b = b - \alpha \frac{\partial J(\vec{w}, b)}{\partial b}$$
$$\}$$

---

### Step 4: Partial Derivatives Derivation
Evaluating the partial derivatives of $J(\vec{w}, b)$ yields:

1. **Derivative with respect to weight $w_j$:**
   $$\frac{\partial J(\vec{w}, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

2. **Derivative with respect to bias $b$:**
   $$\frac{\partial J(\vec{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)$$

---

### Step 5: Vectorized Gradient Update
Instead of evaluating the derivative for each $w_j$ sequentially in a `for` loop, we combine all feature derivatives into a single gradient vector $\vec{d}_{\vec{w}}$:

$$\vec{d}_{\vec{w}} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right) \vec{x}^{(i)}$$

This allows us to update the entire parameter vector $\vec{w}$ simultaneously using a single vectorized subtraction:

$$\vec{w} = \vec{w} - \alpha \vec{d}_{\vec{w}}$$

---

## 3. Key Concepts & The Normal Equation

### What is the Normal Equation?
The **Normal Equation** is an analytical, non-iterative method to solve for the optimal parameter vector $\vec{w}$ and scalar $b$ directly in a single mathematical step using linear algebra libraries, without running gradient descent iterations.

Mathematically, it uses matrix operations to compute:
$$\mathbf{\hat{w}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### Advantages & Disadvantages of the Normal Equation

* **Advantages:**
  * **No Iterations:** Finds the exact minimum in a single step.
  * **No Hyperparameters:** Eliminates the need to select a learning rate $\alpha$ or run feature scaling.

* **Disadvantages:**
  * **High Computational Cost:** Computing the matrix inverse $(\mathbf{X}^T \mathbf{X})^{-1}$ has a time complexity of approximately $O(n^3)$, where $n$ is the number of features. When $n > 10,000$, matrix inversion becomes extremely slow and computationally expensive.
  * **Lack of Generalizability:** The normal equation applies *only* to linear regression. It cannot be used for logistic regression, neural networks, support vector machines, or other non-linear models.

### Comparison: Gradient Descent vs. Normal Equation

| Feature / Aspect | Gradient Descent | Normal Equation |
| :--- | :--- | :--- |
| **Execution Method** | Iterative (requires multiple steps) | Direct analytical solution (one step) |
| **Learning Rate ($\alpha$)** | Required (must choose carefully) | Not required |
| **Feature Scaling** | Necessary for fast convergence | Not required |
| **Time Complexity** | $O(k \cdot n \cdot m)$ (where $k$ is iterations) | $O(n^3)$ due to matrix inversion |
| **Performance for Large $n$** | Scales exceptionally well ($n \ge 10,000+$) | Becomes very slow for large $n$ |
| **Generalizability** | Works across almost all ML algorithms | Works only for linear regression |

> **Context Note:** While most practitioners and modern frameworks rely on gradient descent due to its scalability and versatility, some mature ML libraries (such as `scikit-learn`'s `LinearRegression`) may utilize the normal equation or related direct decomposition methods under the hood for small datasets.

---

## 4. Code Implementation

Below is a complete, modular implementation using Python and NumPy to demonstrate vectorized predictions, cost calculation, gradient updates, and model training.

```python
import numpy as np

def compute_prediction(X, w, b):
    """
    Computes the prediction for multiple linear regression using vectorization.
    
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      w (ndarray (n,)): Model parameters (weights)
      b (scalar): Model parameter (bias)
      
    Returns:
      f_wb (ndarray (m,)): Model predictions for all m examples
    """
    # Vectorized dot product across feature dimensions
    return np.dot(X, w) + b


def compute_cost(X, y, w, b):
    """
    Computes the cost function J(w, b) for multiple linear regression.
    
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)): Target values
      w (ndarray (n,)): Model parameters (weights)
      b (scalar): Model parameter (bias)
      
    Returns:
      cost (float): Total cost J(w, b)
    """
    m = X.shape[0]
    predictions = compute_prediction(X, w, b)
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the partial derivatives dJ/dw and dJ/db for multiple linear regression.
    
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)): Target values
      w (ndarray (n,)): Model parameters (weights)
      b (scalar): Model parameter (bias)
      
    Returns:
      dj_dw (ndarray (n,)): Gradient of cost w.r.t parameters w
      dj_db (scalar): Gradient of cost w.r.t parameter b
    """
    m = X.shape[0]
    
    # Calculate prediction errors for all m examples: shape (m,)
    errors = compute_prediction(X, w, b) - y
    
    # Vectorized gradient calculation
    # dj_dw = (1/m) * X^T dot errors
    dj_dw = np.dot(X.T, errors) / m
    dj_db = np.sum(errors) / m
    
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b.
    
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)): Target values
      w_in (ndarray (n,)): Initial weight parameters
      b_in (scalar): Initial bias parameter
      alpha (float): Learning rate
      num_iters (int): Number of iterations
      
    Returns:
      w (ndarray (n,)): Updated weight parameters
      b (scalar): Updated bias parameter
    """
    w = np.copy(w_in)
    b = b_in
    
    for i in range(num_iters):
        # Calculate gradients
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        
        # Update parameters simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    return w, b
```

---

## 5. Key Terminology & Pitfalls

### Crucial Distinctions:
1. **Multiple Linear Regression vs. Multivariate Regression:**
   * **Multiple Linear Regression:** Model predicting a **single output ($y$)** using **multiple input features ($\vec{x}$)**.
   * **Multivariate Regression:** Advanced statistical technique used to predict **multiple output targets ($\vec{y}$)** simultaneously.

2. **Gradient Descent vs. Normal Equation:**
   * Use **Gradient Descent** when feature count $n$ is large ($n > 10,000$) or when training complex architectures like neural networks.
   * Use **Normal Equation** only for simple linear regression tasks with relatively small feature counts.

### Common Gotchas & Best Practices:
* **Simultaneous Parameter Updates:** Always update weight vector $\vec{w}$ and bias $b$ using the gradients calculated from the *same* iteration step. Never use partially updated weights to calculate the bias gradient within the same iteration.
* **Feature Scaling Preview:** When features differ vastly in scale (e.g., house size in sq ft: $1,000–5,000$ vs. number of bedrooms: $1–5$), gradient descent can take a long time to converge due to elongated cost contours. Scaling features ensures uniform gradient steps.
* **Learning Rate Selection Preview:** Choosing an appropriate $\alpha$ is critical. If $\alpha$ is too small, convergence is slow; if $\alpha$ is too large, gradient descent may overshoot and diverge.

### Summary Checklist:
- [x] Model prediction formula: $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$
- [x] Cost function evaluated using Mean Squared Error over $m$ examples.
- [x] Parameter update rule vectorized: $\vec{w} = \vec{w} - \alpha \vec{d}_{\vec{w}}$.
- [x] Normal Equation provides direct one-step math, but scales poorly ($O(n^3)$) and lacks algorithm versatility.
