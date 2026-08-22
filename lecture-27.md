# Lecture 27: Checking Gradient Descent for Convergence
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 2, Lesson 2 (Lecture 27)

## 1. The Core Idea
When training machine learning models, how do you know if gradient descent is working properly and when it has finished learning? 

**Checking Gradient Descent for Convergence** involves monitoring the cost function $J(\vec{w}, b)$ over successive iterations of training. By plotting $J(\vec{w}, b)$ against the iteration count—a plot known as a **Learning Curve**—you can visually verify that the cost is decreasing monotonically after every step and detect when the model has converged to the minimum.

---

## 2. Mathematical Foundation & Notation

To track optimization progress, we record the cost function value at each iteration step $k$.

### Variables & Notation:
* **$k$**: Iteration number (step index), $k = 0, 1, 2, \dots$.
* **$\vec{w}^{(k)}, b^{(k)}$**: Parameter values at iteration $k$.
* **$J(\vec{w}^{(k)}, b^{(k)})$**: Cost function value at iteration $k$.
* **$\alpha$**: Learning rate hyperparameter.
* **$\epsilon$ (Epsilon)**: Small positive threshold hyperparameter used for automated convergence tests (e.g., $\epsilon = 10^{-3}$).

---

### Step 1: Monotonic Cost Decrease Criterion
If gradient descent is implemented correctly and the learning rate $\alpha$ is properly chosen, the cost function **must decrease after every single iteration**:

$$J(\vec{w}^{(k)}, b^{(k)}) < J(\vec{w}^{(k-1)}, b^{(k-1)}) \quad \forall \, k \ge 1$$

If $J(\vec{w}, b)$ ever increases after an iteration, it indicates that either:
1. The learning rate $\alpha$ is **too large** (causing parameter updates to overshoot the minimum).
2. There is a **bug in the code** implementation (e.g., incorrect derivative calculation or non-simultaneous parameter updates).

---

### Step 2: Visual Convergence Detection
Convergence occurs when $J(\vec{w}, b)$ **levels off** and flattens out, meaning further iterations yield negligible reduction in cost.

Because the number of iterations required to reach convergence varies dramatically depending on the application (ranging from $30$ iterations for simple linear models up to $100,000+$ iterations for complex features), visual analysis of the learning curve is the most reliable method.

---

### Step 3: Automated Convergence Test ($\epsilon$-Test)
Alternatively, an automated stopping criterion can be implemented using a small threshold $\epsilon$:

$$\Delta J = J(\vec{w}^{(k-1)}, b^{(k-1)}) - J(\vec{w}^{(k)}, b^{(k)}) \le \epsilon$$

* **Target Threshold:** Typically $\epsilon = 10^{-3} (0.001)$ or $\epsilon = 10^{-4}$.
* **Decision Rule:** Declare convergence and terminate training if $\Delta J \le \epsilon$ in a single iteration.
* **Practical Limitation:** Selecting a suitable threshold $\epsilon$ can be difficult because the magnitude of $J$ depends heavily on feature scaling and target ranges. Therefore, visual inspection remains the preferred practice.

---

## 3. Visual Learning Curves & Diagnostic Graphs

### Graph 1: Correct Gradient Descent Behavior (Normal Convergence)
The cost function decreases rapidly during early iterations and asymptotically flattens as it approaches the global minimum.

```text
 Cost J(w, b)
   ^
   |  *  (Initial cost at k=0)
   |   \
   |    \
   |     *  (Cost decreasing)
   |      \
   |       \__
   |          *----*----*----* (Plateau: Converged to Minimum)
   +---------------------------------------------> Iterations (k)
   0   100  200  300  400  500
```

---

### Graph 2: Incorrect Behavior - Learning Rate $\alpha$ Too Large (Diverging Cost)
When $\alpha$ is too large, parameter steps overshoot the minimum valley, causing cost to increase or oscillate wildly upward.

```text
 Cost J(w, b)
   ^                * (Overshooting & Diverging)
   |               /
   |        *     /
   |       / \   /
   |  *   /   * /
   |   \ /     
   +---------------------------------------------> Iterations (k)
   0    1    2    3    4    5
   
   Action Required: Reduce learning rate alpha (e.g., try 0.001 instead of 0.1).
```

---

### Graph 3: Incorrect Behavior - Code Bug or Flawed Derivatives
Erratic, unpredictable spikes up and down indicate an error in the gradient computation or logic.

```text
 Cost J(w, b)
   ^    *        *
   |   / \      / \    *
   |  *   \    /   \  / \
   |       *  *     *    *
   +---------------------------------------------> Iterations (k)
   0    1    2    3    4    5
   
   Action Required: Check partial derivative formulas and simultaneous parameter updates.
```

---

## 4. Code Implementation

Below is a complete Python script demonstrating how to run gradient descent while logging `J_history`, checking for automated $\epsilon$ convergence, and plotting the learning curve using `matplotlib`.

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, w, b):
    """Computes Mean Squared Error cost J(w, b)."""
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    return np.sum((predictions - y) ** 2) / (2 * m)


def gradient_descent_with_monitoring(X, y, w_in, b_in, alpha, num_iters, epsilon=1e-3):
    """
    Performs gradient descent while logging cost history and checking for convergence.
    
    Args:
      X (ndarray): Feature matrix (m, n)
      y (ndarray): Target vector (m,)
      w_in (ndarray): Initial weights (n,)
      b_in (float): Initial bias
      alpha (float): Learning rate
      num_iters (int): Maximum iterations
      epsilon (float): Automated convergence threshold
      
    Returns:
      w, b, J_history (list of cost values)
    """
    m = X.shape[0]
    w = np.copy(w_in)
    b = b_in
    J_history = []
    
    for i in range(num_iters):
        # Compute predictions and errors
        predictions = np.dot(X, w) + b
        errors = predictions - y
        
        # Calculate gradients
        dj_dw = np.dot(X.T, errors) / m
        dj_db = np.sum(errors) / m
        
        # Parameter update
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Record current cost
        current_cost = compute_cost(X, y, w, b)
        J_history.append(current_cost)
        
        # Check automated convergence test (epsilon)
        if i > 0:
            delta_J = J_history[-2] - J_history[-1]
            if 0 <= delta_J <= epsilon:
                print(f"Converged automatically at iteration {i} (delta_J = {delta_J:.6f} <= epsilon)")
                break
            elif delta_J < 0:
                print(f"WARNING: Cost increased at iteration {i}! Reduce learning rate alpha.")
                
    return w, b, J_history


# Example Plotting Function
def plot_learning_curve(J_history):
    """Plots J(w, b) vs Iterations."""
    plt.figure(figsize=(8, 5))
    plt.plot(J_history, color='blue', linewidth=2, label='Cost J(w,b)')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost J(w, b)')
    plt.title('Learning Curve: Checking Gradient Descent Convergence')
    plt.grid(True)
    plt.legend()
    plt.show()
```

---

## 5. Key Terminology & Pitfalls

### Key Distinctions:
* **Learning Curve:** Plot of cost $J(\vec{w}, b)$ versus iteration count $k$, used to assess convergence and diagnose learning rate issues.
* **Visual Inspection vs. Automated Test:** Visual inspection lets you see the overall flattening trend regardless of scale, whereas automated $\epsilon$-tests rely on an arbitrary numeric threshold $\epsilon$.

### Common Gotchas:
* **Cost Must Decrease Monotonically:** If cost increases at any iteration, $\alpha$ is too high or the gradient step is bugged.
* **Wide Variance in Iterations:** Simple datasets converge in $< 100$ iterations; complex datasets may require $> 100,000$ iterations. Never hardcode small iteration limits without checking convergence.

### Key Takeaways:
* **Plot $J$ vs. Iterations:** Always graph the learning curve during model development to confirm learning rate validity.
* **Flat Curve Equals Convergence:** Stop training when the curve levels off completely.
* **Visual Analysis Preferred:** Visual learning curves are generally more practical than automated $\epsilon$-tests due to scale dependency.
