# Lecture 30: Polynomial Regression
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 2, Lesson 2 (Lecture 30)

## 1. The Core Idea
Standard linear regression fits straight lines ($f_{w,b}(x) = wx + b$) to datasets. However, real-world data—such as house prices as a function of size—frequently follows non-linear, curved patterns where a straight line fails to capture the true trend (underfitting).

**Polynomial Regression** extends multiple linear regression to model curved relationships by using **Feature Engineering** to create higher-order powers ($x^2, x^3, \dots$) or non-linear transformations ($\sqrt{x}$) of input features.

> **Crucial Insight:** Even though the function curve is non-linear with respect to the input feature $x$, the model remains **linear with respect to the parameters** $w_1, w_2, w_3, b$. Consequently, standard multiple linear regression mechanisms—including gradient descent optimization and cost function formulas—apply directly without modifying the core training algorithm.

---

## 2. Mathematical Foundation & Model Transformations

### 1. Baseline Linear Model (Straight Line):
$$f_{w,b}(x) = w x + b$$
* **Limitation:** Constrained to a constant slope. Cannot model curvature or plateauing behavior.

---

### 2. Quadratic Model (Adding $x^2$):
$$f_{w,b}(x) = w_1 x + w_2 x^2 + b$$

* **Feature Engineering Mapping:** Define $x_1 = x$ and $x_2 = x^2$.
* **Equivalent Multiple Linear Regression Form:**
  $$f_{\vec{w},b}(\vec{x}) = w_1 x_1 + w_2 x_2 + b$$
* **Physical Intuition & Limitation:** A quadratic term allows the model to curve. However, parabolic functions are symmetric; if $w_2 < 0$, the parabola turns downward after reaching a peak. In real estate, this would unrealistically predict that house prices drop as house size grows beyond a certain point.

---

### 3. Cubic Model (Adding $x^2$ and $x^3$):
$$f_{w,b}(x) = w_1 x + w_2 x^2 + w_3 x^3 + b$$

* **Feature Engineering Mapping:** Define $x_1 = x$, $x_2 = x^2$, and $x_3 = x^3$.
* **Equivalent Multiple Linear Regression Form:**
  $$f_{\vec{w},b}(\vec{x}) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$
* **Physical Intuition:** Adding $x^3$ gives the curve greater flexibility. It can capture an initial upward curve that flattens out and continues rising at larger house sizes rather than bending downward.

---

### 4. Non-Polynomial Transformations (Square Root $\sqrt{x}$):
Alternative feature transformations can be crafted to match physical constraints:

$$f_{w,b}(x) = w_1 x + w_2 \sqrt{x} + b$$

* **Feature Engineering Mapping:** Define $x_1 = x$ and $x_2 = \sqrt{x}$.
* **Physical Intuition:** The square root function $\sqrt{x}$ increases monotonically but at a decreasing rate (diminishing returns). This accurately models real estate where price increases with size, but each additional square foot adds slightly less incremental value than the last, without ever decreasing.

---

## 3. Visual Trends, Graphs & Feature Scaling Impact

### Diagram 1: Model Curves Comparison (Linear vs. Quadratic vs. Cubic vs. Square Root)

```text
 Price ($)
   ^
   |                                 * Cubic (x, x^2, x^3) / Sqrt (x, sqrt(x))
   |                             . '   (Monotonic realistic rise)
   |                         . '
   |     Quadratic (x, x^2) . ' 
   |         . - - - .    . '
   |       .           . '
   |     .           . '
   |   .       . '  * Linear (x) (Underfitting straight line)
   | .   . '
   +--------------------------------------------------------> Size (sq ft)
```

---

### Feature Scale Explosion & Mandatory Feature Scaling

When creating polynomial features, input numerical ranges expand exponentially:

| Feature | Expression | Input Value Range ($x$) | Output Value Range |
| :--- | :--- | :--- | :--- |
| **Original Feature** | $x_1 = x$ | $1$ to $1,000$ | $1$ to $10^3$ |
| **Quadratic Feature** | $x_2 = x^2$ | $1^2$ to $1,000^2$ | $1$ to $10^6$ ($1,000,000$) |
| **Cubic Feature** | $x_3 = x^3$ | $1^3$ to $1,000^3$ | $1$ to $10^9$ ($1,000,000,000$) |

#### Why Feature Scaling is Essential:
Because $x$, $x^2$, and $x^3$ operate on vastly different scales ($10^3$ vs $10^6$ vs $10^9$), the cost function $J(\vec{w}, b)$ contours become extremely elongated, narrow ellipses. 

* Without scaling, gradient descent will oscillate wildly across dimensions with massive feature ranges ($x^3$) while making negligible progress on features with small ranges ($x$).
* **Solution:** Applying **Feature Scaling** (such as Z-score Normalization) is **mandatory** when using polynomial features to ensure smooth, fast gradient descent convergence.

---

## 4. Code Implementation

### Implementation 1: Scratch Implementation using NumPy & Z-Score Normalization

```python
import numpy as np
import matplotlib.pyplot as plt

def create_polynomial_features(x):
    """
    Transforms 1D input x into a 3D matrix [x, x^2, x^3].
    """
    x = x.reshape(-1, 1)
    X_poly = np.hstack((x, x**2, x**3))
    return X_poly


def zscore_normalize_features(X):
    """
    Computes mean and std to normalize features to mean=0, std=1.
    Crucial for polynomial features!
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost(X, y, w, b):
    """Computes MSE cost J(w, b)."""
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    return np.sum((predictions - y) ** 2) / (2 * m)


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """Performs gradient descent to fit parameters w and b."""
    m = X.shape[0]
    w = np.copy(w_in)
    b = b_in
    J_history = []
    
    for i in range(num_iters):
        predictions = np.dot(X, w) + b
        errors = predictions - y
        
        dj_dw = np.dot(X.T, errors) / m
        dj_db = np.sum(errors) / m
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        J_history.append(compute_cost(X, y, w, b))
        
    return w, b, J_history


# Workflow Execution
if __name__ == "__main__":
    # Synthetic Dataset: House sizes (100 to 1000 sq ft) & Prices
    x_raw = np.linspace(100, 1000, 50)
    # Target function with non-linear trend + noise
    y_raw = 50 + 0.05 * x_raw + 0.0003 * (x_raw ** 2) + np.random.normal(0, 10, size=50)
    
    # Step 1: Create Polynomial Features [x, x^2, x^3]
    X_poly = create_polynomial_features(x_raw)
    
    # Step 2: Normalize Polynomial Features (MANDATORY)
    X_norm, mu, sigma = zscore_normalize_features(X_poly)
    
    # Step 3: Run Gradient Descent
    initial_w = np.zeros(3)
    initial_b = 0.0
    alpha = 0.1
    iterations = 1000
    
    w_final, b_final, J_history = gradient_descent(X_norm, y_raw, initial_w, initial_b, alpha, iterations)
    
    print(f"Learned Weights w: {w_final}")
    print(f"Learned Bias b: {b_final:.4f}")
    print(f"Final Cost: {J_history[-1]:.4f}")

    # Plot Fitted Curve
    predictions = np.dot(X_norm, w_final) + b_final
    plt.figure(figsize=(8, 5))
    plt.scatter(x_raw, y_raw, color='red', label='Training Data')
    plt.plot(x_raw, predictions, color='blue', linewidth=2, label='Cubic Polynomial Fit')
    plt.xlabel('House Size (sq ft)')
    plt.ylabel('Price ($k)')
    plt.title('Polynomial Regression Fit with Feature Scaling')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

### Implementation 2: Industry Implementation using Scikit-Learn

In industry, professional toolkits like `scikit-learn` streamline feature transformations, scaling, and linear regression:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Synthetic Dataset
x_data = np.linspace(100, 1000, 50).reshape(-1, 1)
y_data = 50 + 0.05 * x_data.ravel() + 0.0003 * (x_data.ravel() ** 2) + np.random.normal(0, 10, size=50)

# Create Pipeline: Polynomial Features -> Feature Scaling -> Linear Regression
model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    LinearRegression()
)

# Fit Model
model.fit(x_data, y_data)
predictions = model.predict(x_data)

print(f"Scikit-Learn R^2 Score: {model.score(x_data, y_data):.4f}")
```

---

## 5. Key Terminology & Pitfalls

### Key Distinctions:
* **Linear vs. Non-linear in Parameters:** Polynomial regression is non-linear with respect to feature inputs $x$, but **linear with respect to weights** $\vec{w}$. Hence, linear optimization algorithms remain valid.
* **Polynomial ($x^2, x^3$) vs. Non-Polynomial ($\sqrt{x}, \log(x)$) Transformations:** Polynomials add flexible curves but may overshoot or bend downwards at extreme values; non-polynomial features (like $\sqrt{x}$) provide monotonic curves with built-in diminishing returns.

### Common Gotchas:
* **Omitting Feature Scaling:** Creating polynomial features causes feature ranges to explode ($10^3 \to 10^9$). Running gradient descent without normalization will cause extreme instability or divergence.
* **Inappropriate High-Degree Polynomials:** Using excessively high polynomial degrees ($x^4, x^5, \dots$) risks severe **overfitting**, creating wild oscillations between data points.
* **Quadratic Drop-off Trap:** Quadratic models ($x^2$) force symmetry. If $w_2 < 0$, the fitted model will predict declining target values past the vertex.

### Key Takeaways:
* **Feature Engineering Enables Curvature:** Transform input features to fit non-linear data while keeping the underlying linear regression algorithm intact.
* **Always Scale Polynomial Features:** Z-score normalization is essential to ensure gradient descent stability.
* **Scikit-Learn Standard:** Use `PolynomialFeatures`, `StandardScaler`, and `LinearRegression` (or pipelines) for efficient production implementations.
