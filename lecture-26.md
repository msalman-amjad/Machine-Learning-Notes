# Lecture 26: Feature Scaling Part 2 (Scaling Methods & Practical Rules)
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 2, Lesson 2 (Lecture 26)

## 1. The Core Idea
When features have vastly different ranges (e.g., house size ranging from $300$ to $2,000$ square feet versus number of bedrooms ranging from $0$ to $5$), the cost function contours become elongated, narrow ellipses. This forces gradient descent to oscillate inefficiently back and forth, resulting in slow convergence.

**Feature Scaling** transforms feature values into comparable numerical ranges (ideally centered around $-1$ to $1$). By normalizing feature scales, gradient descent takes a direct path toward the global minimum, drastically improving optimization speed and numerical stability.

---

## 2. Mathematical Foundation & Notation

To scale features effectively, we apply mathematical transformations across individual feature columns in our dataset matrix $\mathbf{X}$.

### Variables & Notation:
* **$m$**: Total number of training examples in the dataset.
* **$n$**: Total number of input features per training example.
* **$x_i$**: Vector containing all values for feature $i$ across $m$ training examples.
* **$x_i^{(k)}$**: The value of feature $i$ for the $k^{th}$ training example.
* **$\mu_i$**: Mean (average) of feature $i$ across all $m$ training examples:
  $$\mu_i = \frac{1}{m} \sum_{k=1}^{m} x_i^{(k)}$$
* **$\sigma_i$**: Standard deviation of feature $i$:
  $$\sigma_i = \sqrt{\frac{1}{m} \sum_{k=1}^{m} \left( x_i^{(k)} - \mu_i \right)^2}$$
* **$\max(x_i)$ & $\min(x_i)$**: Maximum and minimum values of feature $i$.
* **$\text{range}(x_i)$**: Scale range of feature $i$, calculated as $\max(x_i) - \min(x_i)$.

---

### Method 1: Rescaling by Maximum (Divide by Max)
Divides each feature value by the maximum value of that feature, scaling feature values into a normalized range.

$$x_{scaled} = \frac{x_i}{\max(x_i)}$$

* **Target Range:** $[0, 1]$ (assuming non-negative features $x_i \ge 0$).
* **Use Case:** Simple, fast rescaling when feature values are strictly non-negative and zero-bounded.

---

### Method 2: Mean Normalization
Centers feature values around zero by subtracting the mean $\mu_i$ and dividing by the feature range $(\max(x_i) - \min(x_i))$.

$$x_{new} = \frac{x_i - \mu_i}{\max(x_i) - \min(x_i)}$$

* **Target Range:** Approximately $[-0.5, 0.5]$ (centered exactly at $0$).
* **Properties:** Ensures the average value of the transformed feature is zero, balancing negative and positive values.

---

### Method 3: Z-score Normalization (Standardization)
Normalizes features using the sample mean $\mu_i$ and standard deviation $\sigma_i$.

$$x_{new} = \frac{x_i - \mu_i}{\sigma_i}$$

* **Target Range:** Typically spans between $[-3, 3]$ (for normally distributed data, $\sim 99.7\%$ of values fall within $\pm 3\sigma$).
* **Properties:** Re-scales feature distribution to have a mean of $0$ and a standard deviation of $1$. Highly resilient to wide statistical distributions.

---

## 3. Practical Rules of Thumb & Acceptable Feature Ranges

### When to Apply Feature Scaling:
* **Significant Range Imbalance:** Necessary when features differ significantly in magnitude (e.g., $x_1 \in [0, 2000]$ vs. $x_2 \in [0, 5]$).
* **Rule of Caution:** If in doubt, **always apply feature scaling**. It rarely causes harm and almost always speeds up gradient descent convergence.

### Acceptable Feature Ranges (No Extreme Precision Needed):
Feature scaling does not require exact mathematical equality in ranges across all features. The goal is simply to make feature scales roughly comparable.

* **Ideal Target Range:** Approximately $-1 \le x_j \le 1$.
* **Acceptable Boundaries:** Ranges anywhere between $[-3, 3]$ down to $[-0.3, 0.3]$ are completely acceptable and do not require re-scaling.
* **When Re-scaling is Mandatory:**
  * Feature range is too large: e.g., $[-100, 100]$ or $[0, 2000]$.
  * Feature range is too small: e.g., $[-0.0001, 0.0001]$.

---

### Scaling Methods Comparison:

| Scaling Method | Mathematical Formula | Output Range | Mean Centered? | Robustness |
| :--- | :--- | :--- | :--- | :--- |
| **Divide by Max** | $x_{scaled} = \frac{x}{\max(x)}$ | $[0, 1]$ (for $x \ge 0$) | No | Low |
| **Mean Normalization** | $x_{new} = \frac{x - \mu}{\max - \min}$ | Approx. $[-0.5, 0.5]$ | Yes (Mean = $0$) | Moderate |
| **Z-score Normalization** | $x_{new} = \frac{x - \mu}{\sigma}$ | Typically $[-3, 3]$ | Yes (Mean = $0, \sigma = 1$) | High |

---

## 4. Code Implementation

Below are clean Python implementations utilizing NumPy vectorization for mean normalization and Z-score normalization scaling techniques.

```python
import numpy as np

def divide_by_max(x):
    """
    Rescales features by dividing each feature column by its maximum value.
    Formula: x_scaled = x / max(x)
    """
    max_val = np.max(x, axis=0)
    return x / max_val


def mean_normalize(x):
    """
    Centers features around zero using mean normalization.
    Formula: x_new = (x - mu) / (max - min)
    """
    mu = np.mean(x, axis=0)
    rng = np.max(x, axis=0) - np.min(x, axis=0)
    # Avoid division by zero for constant features
    rng = np.where(rng == 0, 1e-8, rng)
    return (x - mu) / rng


def z_score_normalize(x):
    """
    Standardizes features using Z-score normalization (mean = 0, std = 1).
    Formula: x_new = (x - mu) / sigma
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    # Avoid division by zero for constant features
    sigma = np.where(sigma == 0, 1e-8, sigma)
    return (x - mu) / sigma


# Example Dataset: House size (sq ft) vs Number of bedrooms
X_raw = np.array([
    [600.0, 1.0],
    [1200.0, 2.0],
    [1800.0, 3.0],
    [2400.0, 4.0]
])

print("--- Raw Features ---")
print(X_raw)

print("\n--- 1. Divide by Maximum ---")
print(divide_by_max(X_raw))

print("\n--- 2. Mean Normalization ---")
print(mean_normalize(X_raw))

print("\n--- 3. Z-score Normalization ---")
print(z_score_normalize(X_raw))
```

---

## 5. Key Terminology & Pitfalls

### Key Distinctions:
* **Normalization vs. Standardization:** Mean Normalization rescales features bounded around $[-0.5, 0.5]$ relative to the total range ($\max - \min$), while Z-score Standardization rescales relative to variance ($\sigma$), establishing standard deviation $\sigma = 1$.
* **Elongated Cost Contours:** Caused by feature scale imbalance, forcing gradient descent into slow, oscillating paths.

### Common Gotchas:
* **Do Not Scale Target $y$:** Apply feature scaling exclusively to input feature matrix $\mathbf{X}$. Target values $y$ are typically left unscaled unless extreme scale differences exist in multi-target applications.
* **Division by Zero Protection:** Always handle constant features where range ($\max - \min$) or standard deviation ($\sigma$) equals zero to avoid `NaN` or runtime errors.
* **Consistency Across Sets:** Apply the exact training set statistics ($\mu$, $\sigma$, $\max$, $\min$) when scaling validation and test datasets.

### Key Takeaways:
* **Speeds Up Gradient Descent:** Eliminates skewed cost contours, allowing parameter updates to point directly toward the global minimum.
* **Practical Acceptable Ranges:** Extreme precision is unneeded; target feature ranges between $[-3, 3]$ and $[-0.3, 0.3]$ are completely sufficient.
* **General Recommendation:** Default to applying Z-score or mean normalization whenever feature ranges differ significantly.
