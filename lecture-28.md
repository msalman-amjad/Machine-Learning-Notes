# Lecture 28: Choosing the Learning Rate & Gradient Descent Diagnostics
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 2, Lesson 2 (Lecture 28)

## 1. The Core Idea
The **learning rate ($\alpha$)** is one of the most critical hyperparameters in gradient descent optimization. It controls the step size that model parameters take along the steepest slope of the cost function $J(\vec{w}, b)$ during each iteration.

* If $\alpha$ is **too small**, gradient descent takes tiny micro-steps, making training excruciatingly slow and computationally inefficient.
* If $\alpha$ is **too large**, gradient descent takes oversized jumps, causing parameter updates to overshoot the minimum, oscillate wildly, or even diverge upward.

Selecting the optimal learning rate $\alpha$ ensures that gradient descent converges to the global minimum rapidly and reliably without overshooting.

---

## 2. Mathematical Foundation & Debugging Mechanics

### Parameter Update Equation:
For a parameter weight $w_j$, the standard gradient descent update step is defined as:

$$w_j = w_j - \alpha \frac{\partial J(\vec{w}, b)}{\partial w_j}$$

Similarly, for the scalar bias $b$:

$$b = b - \alpha \frac{\partial J(\vec{w}, b)}{\partial b}$$

---

### Mathematical Impact of Learning Rate Choice:

1. **Monotonic Decrease Requirement:**
   When implemented correctly with an appropriately sized $\alpha$, the cost function $J$ must strictly decrease after every single iteration step:

   $$J(\vec{w}^{(k)}, b^{(k)}) < J(\vec{w}^{(k-1)}, b^{(k-1)})$$

2. **Sign Error Implementation Bug ($+$ vs $-$):**
   A common coding mistake is accidentally placing a plus sign ($+$) instead of a minus sign ($-$) in the update rule:

   $$w_j = w_j + \alpha \frac{\partial J(\vec{w}, b)}{\partial w_j} \quad \text{(INCORRECT)}$$

   * **Consequence:** Because $\frac{\partial J}{\partial w_j}$ points in the direction of steepest *increase*, adding $\alpha \cdot \frac{\partial J}{\partial w_j}$ causes the parameters to walk *up* the cost hill, making $J(\vec{w}, b)$ increase exponentially after every iteration.

3. **Isolating Bugs using an Extremely Small $\alpha$:**
   If you are unsure whether an unstable learning curve is caused by a bad $\alpha$ value or an underlying bug in your math/code:
   * **Diagnostic Procedure:** Set $\alpha$ to a tiny value (e.g., $\alpha = 10^{-7}$).
   * **Interpretation:**
     - If $J(\vec{w}, b)$ now decreases slowly and monotonically, your code implementation is correct; the previous issue was simply choosing an oversized $\alpha$.
     - If $J(\vec{w}, b)$ still increases or fluctuates erratically even with a tiny $\alpha$, there is a **bug in your code** (e.g., wrong derivative formula or sign error).

---

## 3. Systematic Strategy & Diagnostic Learning Curves

### Systematic Search Strategy for $\alpha$:
Instead of guessing arbitrary numbers, test a range of $\alpha$ values scaled logarithmically by factors of approximately $3$:

$$\alpha \in \{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0\}$$

#### 3-Step Selection Process:
1. **Run Short Trials:** Run gradient descent for a small number of iterations (e.g., 30–50 iterations) for each candidate $\alpha$.
2. **Plot Learning Curves:** Plot $J(\vec{w}, b)$ vs. Iteration Count for every trial.
3. **Select Optimal $\alpha$:** Choose the largest value of $\alpha$ that produces a fast, consistent decrease in cost without overshooting or flattening too early.

---

### Diagnostic Learning Curve Graphs:

#### Graph 1: Optimal Learning Rate $\alpha$ (Fast, Stable Convergence)
Cost drops rapidly in early iterations and smooths gracefully to the global minimum.

```text
 Cost J(w, b)
   ^
   |  *  (Initial cost)
   |   \
   |    \
   |     *
   |      \__
   |         *----*----*  (Fast Monotonic Convergence)
   +---------------------------------------------> Iterations
```

---

#### Graph 2: Learning Rate $\alpha$ Too Small (Slow Convergence)
Cost decreases monotonically, but the descent is extremely flat, requiring millions of steps to converge.

```text
 Cost J(w, b)
   ^
   |  *
   |   \
   |    *
   |     \
   |      *  (Takes thousands of iterations to reach minimum)
   |       \
   +---------------------------------------------> Iterations
```

---

#### Graph 3: Learning Rate $\alpha$ Too Large (Oscillating / Overshooting)
Parameters jump over the minimum valley, causing $J$ to bounce up and down or diverge upward.

```text
 Cost J(w, b)
   ^           *  (Diverging / Overshooting)
   |          / \
   |    *    /   *
   |   / \  /
   |  *   \/
   +---------------------------------------------> Iterations
```

---

#### Graph 4: Sign Bug ($+$ instead of $-$)
Cost increases monotonically without bound because updates ascend the gradient.

```text
 Cost J(w, b)
   ^             *  (Exponential Increase)
   |            /
   |           *
   |          /
   |         *
   |        /
   +---------------------------------------------> Iterations
```

---

## 4. Code Implementation

Below is a Python script using NumPy and Matplotlib to test multiple learning rates $\alpha$ systematically, log learning histories, and identify the optimal $\alpha$.

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, w, b):
    """Computes Mean Squared Error cost J(w, b)."""
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    return np.sum((predictions - y) ** 2) / (2 * m)


def run_gradient_descent(X, y, alpha, num_iters=50):
    """
    Runs gradient descent for a given alpha and returns cost history.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    J_history = []
    
    for i in range(num_iters):
        predictions = np.dot(X, w) + b
        errors = predictions - y
        
        # Gradients
        dj_dw = np.dot(X.T, errors) / m
        dj_db = np.sum(errors) / m
        
        # Parameter update (Minus sign is critical!)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        cost = compute_cost(X, y, w, b)
        J_history.append(cost)
        
    return J_history


# Systematic Alpha Evaluation Script
if __name__ == "__main__":
    # Synthetic Dataset
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = np.dot(X_train, np.array([1.5, -2.0, 3.0])) + 0.5
    
    # Candidate alphas (scaled by ~3x steps)
    candidate_alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    
    plt.figure(figsize=(10, 6))
    
    for alpha in candidate_alphas:
        history = run_gradient_descent(X_train, y_train, alpha=alpha, num_iters=40)
        plt.plot(history, label=f'alpha = {alpha}')
        
    plt.xlabel('Iterations')
    plt.ylabel('Cost J(w, b)')
    plt.title('Learning Rate Search: J(w, b) vs Iterations')
    plt.yscale('log')  # Log scale highlights scale differences
    plt.grid(True)
    plt.legend()
    plt.show()
```

---

## 5. Key Terminology & Pitfalls

### Key Distinctions:
* **$\alpha$ Too Small vs. $\alpha$ Too Large:** Too small leads to slow convergence without divergence; too large leads to overshooting, oscillation, or unbounded divergence.
* **Code Bug vs. Bad Hyperparameter:** An oversized $\alpha$ causes cost increase that disappears when $\alpha \to 0$; a math/code bug (like a sign error) causes cost increase regardless of how small $\alpha$ is.

### Common Gotchas:
* **The Sign Error Bug:** Writing `w = w + alpha * dj_dw` performs gradient *ascent*, driving parameters away from the minimum.
* **Testing Single $\alpha$ Values:** Never rely on a single default $\alpha = 0.01$. Always test a multiplicative sequence ($0.001, 0.003, 0.01, 0.03, \dots$).
* **Unscaled Features Masking $\alpha$:** If features are unscaled, even a reasonable $\alpha$ might cause oscillation on large-scale feature dimensions while making zero progress on small-scale feature dimensions. Always combine feature scaling with learning rate tuning.

### Key Takeaways:
* **Plot $J$ vs. Iterations:** The learning curve is your primary diagnostic tool for validating $\alpha$.
* **Debug with Tiny $\alpha$:** Use an extremely small $\alpha$ ($10^{-7}$) to isolate software bugs from hyperparameter misconfiguration.
* **Search Grid:** Use logarithmic factors of 3 to quickly pinpoint the optimal learning rate.
