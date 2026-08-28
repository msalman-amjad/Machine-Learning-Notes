# Lecture 32: Logistic Regression
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 3, Lesson 1, Video 2 (Lecture 32)

## 1. The Core Idea
In the previous lecture, we established that standard **linear regression** is unsuitable for binary classification because straight lines can produce outputs outside $[0, 1]$ and decision boundaries shift wildly in the presence of outliers.

**Logistic Regression** is one of the most fundamental and widely used algorithms for binary classification. Despite having "regression" in its name, it is explicitly designed for classification tasks where target labels are binary ($y \in \{0, 1\}$).

Instead of fitting a straight line, logistic regression maps the linear output $z = \vec{w} \cdot \vec{x} + b$ through an **S-shaped curve** (the **Sigmoid Function** or **Logistic Function**). This constrains all model predictions to strictly lie within the range $(0, 1)$, allowing predictions to be interpreted directly as **conditional probabilities** $P(y=1|\vec{x})$.

---

## 2. Mathematical Foundation: The Sigmoid (Logistic) Function

The **Sigmoid Function** (also known as the **Logistic Function**), denoted as $g(z)$, is defined mathematically as:

$$g(z) = \frac{1}{1 + e^{-z}}$$

where $z$ is any real-valued number ($z \in \mathbb{R}$) and $e \approx 2.71828$ is Euler's number.

### Key Mathematical Properties & Asymptotic Behavior:

1. **Upper Asymptote ($z \to +\infty$):**
   As $z$ becomes a very large positive number, $e^{-z} = \frac{1}{e^z} \to 0$. Therefore:
   $$\lim_{z \to +\infty} g(z) = \frac{1}{1 + 0} = 1$$

2. **Lower Asymptote ($z \to -\infty$):**
   As $z$ becomes a very large negative number, $e^{-z} = e^{|z|} \to +\infty$. Therefore:
   $$\lim_{z \to -\infty} g(z) = \frac{1}{1 + \infty} = 0$$

3. **Midpoint / Inflection Point ($z = 0$):**
   When $z = 0$, $e^{-0} = 1$. Therefore:
   $$g(0) = \frac{1}{1 + 1} = \frac{1}{2} = 0.5$$

### Value Transformation Table:

| Input $z$ | $e^{-z}$ | Sigmoid Output $g(z)$ | Interpretation |
| :---: | :---: | :---: | :--- |
| $-5.0$ | $e^5 \approx 148.41$ | $\frac{1}{1 + 148.41} \approx 0.0067$ | Nearly $0$ |
| $-2.0$ | $e^2 \approx 7.389$ | $\frac{1}{1 + 7.389} \approx 0.1192$ | Low probability |
| **$0.0$** | **$e^0 = 1.0$** | **$\frac{1}{1 + 1} = 0.5000$** | **Threshold Midpoint** |
| $+2.0$ | $e^{-2} \approx 0.1353$ | $\frac{1}{1 + 0.1353} \approx 0.8808$ | High probability |
| $+5.0$ | $e^{-5} \approx 0.0067$ | $\frac{1}{1 + 0.0067} \approx 0.9933$ | Nearly $1$ |

---

## 3. Visualizing the Sigmoid S-Curve

The graph below illustrates the smooth, monotonic S-shape of $g(z)$ bounded between horizontal asymptotes at $y=0$ and $y=1$:

```text
  g(z)
   1.0 |                                      . - - - - - Asymptote g(z) = 1
       |                                   . '
   0.88|                             . ' (z=2, g=0.88)
       |                         . '
   0.5 |--------------------- . (z=0, g=0.5) ----------------- Midpoint
       |                 . '
   0.12|             . ' (z=-2, g=0.12)
       |         . '
   0.0 | - - - ' - - - - - - - - - - - - - - - - - - - - - - Asymptote g(z) = 0
       +-----------------------+-----------------------+---> z
              -5               0                       +5
```

---

## 4. The Logistic Regression Model Architecture

Building a logistic regression model is a **two-step composite process**:

### Step 1: Compute Linear Combination ($z$)
First, compute the standard linear regression combination of weights, features, and bias:

$$z = \vec{w} \cdot \vec{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

### Step 2: Apply the Sigmoid Activation Function ($g(z)$)
Next, feed $z$ into the sigmoid function $g(z)$ to obtain the final hypothesis prediction $f_{\vec{w},b}(\vec{x})$:

$$f_{\vec{w},b}(\vec{x}) = g(z) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$

```text
  [ Input Features x ] 
           │
           ▼
 [ Linear Combination ] ──► z = w · x + b
           │
           ▼
 [ Sigmoid Activation ] ──► g(z) = 1 / (1 + e^-z)
           │
           ▼
 [ Bounded Output f(x) ] ──► f_w,b(x) ∈ (0, 1)
```

---

## 5. Probabilistic Interpretation & Notation

The output $f_{\vec{w},b}(\vec{x})$ of logistic regression is interpreted as the **conditional probability** that the target label $y$ equals $1$, given input features $\vec{x}$ under model parameters $\vec{w}$ and $b$.

### Formal Mathematical Notation:
$$f_{\vec{w},b}(\vec{x}) = P(y = 1 \mid \vec{x}; \vec{w}, b)$$

* **Read as:** "The probability that $y=1$, given input $\vec{x}$, parameterized by $\vec{w}$ and $b$."

### Complement Probability ($P(y=0 \mid \vec{x})$):
Since the target $y$ can only take one of two values ($0$ or $1$), the sum of probabilities for the two classes must equal $1$:

$$P(y = 0 \mid \vec{x}; \vec{w}, b) + P(y = 1 \mid \vec{x}; \vec{w}, b) = 1$$

Therefore, the probability that the sample belongs to the negative class ($y=0$) is simply:

$$P(y = 0 \mid \vec{x}; \vec{w}, b) = 1 - f_{\vec{w},b}(\vec{x})$$

### Concrete Medical Example (Tumor Classification):
Suppose we evaluate a tumor size feature $x$ in our trained logistic regression model:
* Model output: $f_{\vec{w},b}(x) = 0.7$
* **Interpretation:** There is a **70% probability** ($0.7$) that the tumor is malignant ($y=1$).
* **Complementary Inference:** There is a **30% probability** ($1 - 0.7 = 0.3$) that the tumor is benign ($y=0$).

---

## 6. Code Implementation & Visualization

### 1. Scratch Implementation using NumPy & Matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Computes the sigmoid function g(z) = 1 / (1 + exp(-z)).
    Works for scalars, vectors, or NumPy matrices.
    """
    return 1.0 / (1.0 + np.exp(-z))

def predict_logistic(X, w, b):
    """
    Computes the logistic regression hypothesis f_w,b(X) = g(w * X + b).
    
    Parameters:
    X : ndarray (m, n) -> Input data matrix
    w : ndarray (n,)   -> Weight vector
    b : float          -> Bias term
    
    Returns:
    f_wb : ndarray (m,) -> Probabilities P(y=1|X)
    """
    z = np.dot(X, w) + b
    return sigmoid(z)

# --- Demonstration & Visualization ---
if __name__ == "__main__":
    # Test Sigmoid Function across range z in [-10, 10]
    z_vals = np.linspace(-10, 10, 200)
    g_vals = sigmoid(z_vals)
    
    print(f"Sigmoid at z=0:  {sigmoid(0):.4f}")
    print(f"Sigmoid at z=2:  {sigmoid(2):.4f}")
    print(f"Sigmoid at z=-2: {sigmoid(-2):.4f}")
    
    # Synthetic 1D Classification Example (Tumor size x)
    x_tumor = np.linspace(1, 8, 50).reshape(-1, 1)
    w_learned = np.array([1.5])
    b_learned = -5.0  # z = 1.5*x - 5.0 => z=0 when x = 3.33 cm
    
    probabilities = predict_logistic(x_tumor, w_learned, b_learned)
    
    # Plot Sigmoid curve and predictions
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(z_vals, g_vals, color='blue', linewidth=2.5, label=r'$g(z) = \frac{1}{1 + e^{-z}}$')
    plt.axhline(0.5, color='gray', linestyle=':', label='Midpoint (0.5)')
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    plt.axhline(0.0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(0.0, color='gray', linestyle=':')
    plt.title('Pure Sigmoid Function g(z)')
    plt.xlabel('z')
    plt.ylabel('g(z)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_tumor, probabilities, color='green', linewidth=2.5, label=r'$f_{w,b}(x) = P(y=1|x)$')
    plt.axhline(0.5, color='gray', linestyle=':', label='Threshold = 0.5')
    plt.axvline(3.33, color='orange', linestyle='--', label=r'Decision boundary ($z=0 \implies x=3.33$)')
    plt.title('Logistic Regression Fit (Tumor Size -> P(y=1))')
    plt.xlabel('Tumor Size (cm)')
    plt.ylabel('Predicted Probability')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

---

### 2. Industry Implementation using Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Training dataset: Tumor sizes and binary labels (0 = Benign, 1 = Malignant)
X_train = np.array([[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [4.0], [4.5], [5.0], [6.0]])
y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Initialize and fit Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Inspect learned parameters
print(f"Learned Weight (w): {model.coef_[0][0]:.4f}")
print(f"Learned Bias (b):   {model.intercept_[0]:.4f}")

# Predict probabilities for new patient tumor sizes (e.g., 2.8 cm and 4.2 cm)
X_new = np.array([[2.8], [4.2]])
probs = model.predict_proba(X_new)

for i, x in enumerate(X_new.ravel()):
    print(f"Tumor Size: {x} cm -> P(y=0): {probs[i][0]:.4f}, P(y=1): {probs[i][1]:.4f}")
```

---

## 7. Key Terminology, Practical Notes & Next Steps

### Key Distinctions:
* **Linear vs. Logistic Regression:** Linear regression computes continuous predictions $f(x) = wx+b \in (-\infty, +\infty)$. Logistic regression feeds $wx+b$ through sigmoid $g(z)$ to bound outputs in $(0, 1)$.
* **Sigmoid vs. Linear Output:** The linear component $z = \vec{w} \cdot \vec{x} + b$ serves as the input argument to the non-linear sigmoid activation $g(z)$.

### Industry Context:
* **Internet Advertising & Recommendation Engines:** Historically, variants of logistic regression were critical for computing **Click-Through Rate (CTR)**—predicting the probability $P(\text{click}=1 \mid \text{user}, \text{ad})$ that a user will click on an ad.

### Summary Checklist:
1. **Model Formula:** $f_{\vec{w},b}(\vec{x}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$
2. **Output Bounds:** $0 < f_{\vec{w},b}(\vec{x}) < 1$
3. **Probabilistic Meaning:** $f_{\vec{w},b}(\vec{x}) = P(y=1 \mid \vec{x}; \vec{w}, b)$
4. **Complement Rule:** $P(y=0 \mid \vec{x}) = 1 - P(y=1 \mid \vec{x})$

### Next Steps:
In the upcoming lecture, we will discuss **Decision Boundaries**—examining how setting threshold criteria (e.g., predicting $\hat{y} = 1$ when $f_{\vec{w},b}(\vec{x}) \ge 0.5$) defines geometric surfaces in feature space that partition samples into binary classes.
