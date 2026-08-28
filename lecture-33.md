# Lecture 33: Decision Boundary
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 3, Lesson 1, Video 3 (Lecture 33)

## 1. The Core Idea

In the previous lecture, we introduced **logistic regression** and established how it maps any real-valued input $z = \vec{w} \cdot \vec{x} + b$ through the sigmoid function $g(z) = \frac{1}{1 + e^{-z}}$ to yield a probability $f_{\vec{w},b}(\vec{x}) \in (0, 1)$.

However, classification tasks ultimately require making concrete binary decisions—predicting whether a sample belongs to class $1$ ($\hat{y} = 1$) or class $0$ ($\hat{y} = 0$).

This lecture explores the **Decision Boundary**: the geometric line, surface, or contour in the feature space that separates regions where the model predicts $\hat{y} = 1$ from regions where it predicts $\hat{y} = 0$. We will analyze how setting a threshold on the output probability translates mathematically to $z = 0$, examine linear decision boundaries, and show how polynomial feature engineering enables logistic regression to fit complex, non-linear decision boundaries.

---

## 2. Thresholding & Mathematical Derivation of the Decision Boundary

### Step 1: Decision Threshold Rule
To convert the continuous output probability $f_{\vec{w},b}(\vec{x}) = P(y=1 \mid \vec{x})$ into a discrete binary class prediction $\hat{y} \in \{0, 1\}$, a standard **threshold of 0.5** is applied:

$$\hat{y} = \begin{cases} 1 & \text{if } f_{\vec{w},b}(\vec{x}) \ge 0.5 \\ 0 & \text{if } f_{\vec{w},b}(\vec{x}) < 0.5 \end{cases}$$

### Step 2: Connection to the Sigmoid Function $g(z)$
Recall the sigmoid curve properties:
* $g(z) \ge 0.5$ whenever $z \ge 0$
* $g(z) < 0.5$ whenever $z < 0$
* $g(z) = 0.5$ exactly when $z = 0$

```text
       g(z)
        1.0 |                               .......
            |                       . '
            |                 . ' 
  g(z) >= 0.5----------------- (z=0, g=0.5) ---------------> Predict y_hat = 1 when z >= 0
            |             . '
            |         . '
  g(z) < 0.5| . '-------------------------------------------> Predict y_hat = 0 when z < 0
        0.0 +--------------------+--------------------+---> z
                                 0
```

### Step 3: Deriving the Condition on Features ($\vec{w} \cdot \vec{x} + b$)
Since $f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b)$, the threshold conditions simplify directly to:

* **Predict $\hat{y} = 1$ when:** $z \ge 0 \implies \vec{w} \cdot \vec{x} + b \ge 0$
* **Predict $\hat{y} = 0$ when:** $z < 0 \implies \vec{w} \cdot \vec{x} + b < 0$

### Definition of Decision Boundary:
The **decision boundary** is the neutral line or surface where the model's prediction is exactly at the threshold $0.5$ ($50\%$ probability of class 1, $50\%$ probability of class 0). Mathematically, it occurs exactly when:

$$z = 0 \iff \vec{w} \cdot \vec{x} + b = 0$$

> **Key Distinction:** The decision boundary is defined entirely by the trained parameters $\vec{w}$ and $b$ (and the feature representation chosen). While the training dataset is used to learn these parameters, the decision boundary itself is a property of the hypothesis model, not the dataset.

---

## 3. Linear Decision Boundaries

When the input features are raw linear variables ($x_1, x_2, \dots, x_n$), the equation $\vec{w} \cdot \vec{x} + b = 0$ forms a linear hyper-plane. In a 2-dimensional feature space $(x_1, x_2)$, this represents a **straight line**.

### Concrete 2D Linear Example
Suppose we have a dataset with two features $x_1$ and $x_2$, and a trained logistic regression model with parameters:
$$w_1 = 1, \quad w_2 = 1, \quad b = -3$$

#### 1. Linear combination formula:
$$z = w_1 x_1 + w_2 x_2 + b = x_1 + x_2 - 3$$

#### 2. Decision boundary equation ($z = 0$):
$$x_1 + x_2 - 3 = 0 \implies x_1 + x_2 = 3 \implies x_2 = 3 - x_1$$

#### 3. Prediction region classification:
* **Predict $\hat{y} = 1$:** $x_1 + x_2 - 3 \ge 0 \implies x_1 + x_2 \ge 3$ (Region above/to the right of line)
* **Predict $\hat{y} = 0$:** $x_1 + x_2 - 3 < 0 \implies x_1 + x_2 < 3$ (Region below/to the left of line)

### Graphical Visualization:

```text
    x_2 ^
        |
      4 |       o          x          x
        |               \
      3 |       o        \ (Boundary: x_1 + x_2 = 3)
        |                 \
      2 |       o          \          x
        |                   \
      1 |       o            \        x
        |                     \
      0 +----------------------+----------------------> x_1
        0       1     2     3  |   4      5
                               v
                       Region: y_hat = 1 (x_1 + x_2 >= 3)
```

> **Rule:** Without higher-order polynomial features, logistic regression can **only fit straight-line (or planar) decision boundaries**.

---

## 4. Non-Linear Decision Boundaries (Polynomial Features)

When data points are not linearly separable (e.g. positive samples clustered in a circle surrounded by negative samples), standard linear features fail. By incorporating **polynomial terms** (e.g., $x_1^2, x_2^2, x_1 x_2$), logistic regression can fit complex, non-linear boundaries.

### Example 1: Circular Decision Boundary
Consider a model using quadratic features $x_1^2$ and $x_2^2$:
$$z = w_1 x_1^2 + w_2 x_2^2 + b$$

Suppose training yields the parameters:
$$w_1 = 1, \quad w_2 = 1, \quad b = -1$$

#### 1. Decision boundary equation ($z = 0$):
$$x_1^2 + x_2^2 - 1 = 0 \implies x_1^2 + x_2^2 = 1$$

This is the equation of a **unit circle** centered at the origin $(0,0)$ with radius $r = 1$.

#### 2. Prediction regions:
* **Predict $\hat{y} = 1$:** $x_1^2 + x_2^2 - 1 \ge 0 \implies x_1^2 + x_2^2 \ge 1$ (All points **outside or on** the circle)
* **Predict $\hat{y} = 0$:** $x_1^2 + x_2^2 - 1 < 0 \implies x_1^2 + x_2^2 < 1$ (All points **inside** the circle)

```text
       x_2 ^
           |       x         x         x
           |           . - - - .
           |         '   o   o   '
           |        (     o o     )  <- Circle x_1^2 + x_2^2 = 1
           |         '   o   o   '
           |           ' - - - '
           |       x         x         x
     ------+-----------------------------------> x_1
           |
```

### Example 2: Higher-Order / Complex Shapes
By adding higher-degree polynomial combinations:
$$z = w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2 + w_6 x_1^3 + \dots + b$$

The decision boundary $\vec{w} \cdot \vec{f}(\vec{x}) + b = 0$ can take arbitrarily complex geometric forms, such as:
* **Ellipses:** $w_1 x_1^2 + w_2 x_2^2 = c$ (when coefficients $w_1 \neq w_2$)
* **Parabolas / Hyperbolas:** Features like $x_2 - x_1^2 = 0$
* **Irregular Closed / Open Curves:** High-degree polynomials creating multi-lobed boundaries to partition complex datasets.

---

## 5. Code Implementations & Visualizations

### 1. NumPy & Matplotlib Implementation: Plotting Linear vs Circular Boundaries

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))

# -------------------------------------------------------------
# 1. Linear Decision Boundary Visualization (x1 + x2 = 3)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Generate synthetic 2D data points for Linear Example
np.random.seed(42)
x1_lin = np.random.uniform(0, 5, 40)
x2_lin = np.random.uniform(0, 5, 40)

# Classify based on true boundary x1 + x2 >= 3
labels_lin = (x1_lin + x2_lin - 3 >= 0).astype(int)

# Boundary line equation: x2 = 3 - x1
x1_line = np.linspace(0, 5, 100)
x2_line = 3 - x1_line

axes[0].scatter(x1_lin[labels_lin == 1], x2_lin[labels_lin == 1], color='red', marker='x', label='y=1 (x1+x2 >= 3)')
axes[0].scatter(x1_lin[labels_lin == 0], x2_lin[labels_lin == 0], color='blue', marker='o', label='y=0 (x1+x2 < 3)')
axes[0].plot(x1_line, x2_line, color='green', linewidth=2, label=r'Boundary: $x_1 + x_2 - 3 = 0$')
axes[0].set_title('Linear Decision Boundary')
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')
axes[0].set_xlim(0, 5)
axes[0].set_ylim(0, 5)
axes[0].legend()
axes[0].grid(True)

# -------------------------------------------------------------
# 2. Circular Decision Boundary Visualization (x1^2 + x2^2 = 1)
# -------------------------------------------------------------
x1_circ = np.random.uniform(-1.8, 1.8, 60)
x2_circ = np.random.uniform(-1.8, 1.8, 60)

# Classify based on true boundary x1^2 + x2^2 >= 1
labels_circ = (x1_circ**2 + x2_circ**2 - 1 >= 0).astype(int)

# Circle parametric equation: x1 = cos(theta), x2 = sin(theta)
theta = np.linspace(0, 2*np.pi, 200)
x1_circle_pts = np.cos(theta)
x2_circle_pts = np.sin(theta)

axes[1].scatter(x1_circ[labels_circ == 1], x2_circ[labels_circ == 1], color='red', marker='x', label='y=1 (Outside Circle)')
axes[1].scatter(x1_circ[labels_circ == 0], x2_circ[labels_circ == 0], color='blue', marker='o', label='y=0 (Inside Circle)')
axes[1].plot(x1_circle_pts, x2_circle_pts, color='purple', linewidth=2, label=r'Boundary: $x_1^2 + x_2^2 - 1 = 0$')
axes[1].set_title('Non-Linear Circular Decision Boundary')
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_xlim(-2, 2)
axes[1].set_ylim(-2, 2)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

---

### 2. Industry Implementation: Non-Linear Boundary with Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np

# Training dataset: 2D points inside (0) vs outside (1) circle of radius 1
np.random.seed(0)
X = np.random.uniform(-1.5, 1.5, (100, 2))
y = ((X[:, 0]**2 + X[:, 1]**2) >= 1.0).astype(int)

# Create a machine learning pipeline: Polynomial Feature Mapping -> Logistic Regression
model_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('log_reg', LogisticRegression())
])

# Fit non-linear decision boundary model
model_pipeline.fit(X, y)

# Predict for new test samples
X_test = np.array([
    [0.2, 0.2],   # Inside circle -> should predict 0
    [1.2, 1.2]    # Outside circle -> should predict 1
])

predictions = model_pipeline.predict(X_test)
probabilities = model_pipeline.predict_proba(X_test)

for sample, pred, prob in zip(X_test, predictions, probabilities):
    print(f"Sample {sample}: Class Prediction = {pred}, Probabilities [P(y=0), P(y=1)] = {prob.round(4)}")
```

---

## 6. Key Takeaways & Summary Checklist

1. **Threshold & Boundary Link:** Setting output probability threshold $f_{\vec{w},b}(\vec{x}) \ge 0.5$ is mathematically equivalent to predicting $\hat{y} = 1$ whenever $z = \vec{w} \cdot \vec{x} + b \ge 0$.
2. **Decision Boundary Equation:** The boundary separates predicted classes and occurs precisely where $z = 0$, or $\vec{w} \cdot \vec{x} + b = 0$.
3. **Model vs. Data Property:** The decision boundary is a property of the model (its parameters $\vec{w}, b$ and feature representation), not of the dataset directly.
4. **Linear Limit:** Raw linear features produce straight-line, planar, or hyper-planar decision boundaries.
5. **Polynomial Flexibility:** Introducing polynomial terms ($x_1^2, x_2^2, x_1 x_2, \dots$) allows logistic regression to fit circular, elliptical, or arbitrarily complex non-linear decision boundaries.

---

## 7. Next Steps

In the upcoming lecture, we will explore the **Cost Function for Logistic Regression**—explaining why squared error cost function (used in linear regression) produces a non-convex surface for logistic regression and introducing the convex **Log Loss (Binary Cross-Entropy)** cost function.
