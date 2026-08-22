# Lecture 29: Feature Engineering
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 2, Lesson 2 (Lecture 29)

## 1. The Core Idea
Raw data provided to a machine learning model is rarely in the optimal format for making accurate predictions. **Feature Engineering** is the process of using domain knowledge, human intuition, or mathematical transformations to combine or modify existing features to create *new* features that make it easier for learning algorithms to uncover underlying patterns.

Instead of forcing a linear model to rely solely on raw inputs (such as lot width and lot depth individually), feature engineering allows us to create more relevant composite metrics (such as total land area) that directly correlate with the target variable (such as house price).

---

## 2. Mathematical Foundation & Model Transformation

Consider predicting the price of a house ($y$) given raw measurements of the property lot.

### Raw Input Features:
* **$x_1$**: Frontage (width of the lot in feet).
* **$x_2$**: Depth (length of the lot in feet).

---

### Step 1: Initial Linear Model (Raw Features)
A baseline multiple linear regression model predicts price using width and depth as separate linear components:

$$f_{\vec{w}, b}(\vec{x}) = w_1 x_1 + w_2 x_2 + b$$

* **Limitation:** This model assumes price scales independently with width and depth. It cannot capture the joint geometric relationship that buyers primarily pay for total usable land area.

---

### Step 2: Feature Transformation & Engineering
By applying domain knowledge about real estate geometry, we create a new engineered feature $x_3$ representing total lot area:

$$x_3 = x_1 \times x_2 \quad (\text{Area in sq ft} = \text{Frontage} \times \text{Depth})$$

---

### Step 3: Expanded Model (Engineered Feature)
We update our regression model to include the newly engineered feature $x_3$ alongside original features:

$$f_{\vec{w}, b}(\vec{x}) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$

* **Model Flexibility:** The gradient descent algorithm can now learn weights $w_1, w_2, w_3$ autonomously. If land area $x_3$ is the dominant factor in determining price, the optimization algorithm will automatically assign a large weight to $w_3$ and smaller weights to $w_1$ and $w_2$.

---

## 3. Geometric Intuition & Visual Diagrams

### Diagram 1: Feature Transformation Geometry (Frontage x Depth = Area)

```text
       Frontage (x1) = 50 ft
    +------------------------+
    |                        |
    |                        |  Depth (x2) = 100 ft
    |   Area (x3) = x1 * x2  |
    |      = 5,000 sq ft     |
    |                        |
    +------------------------+
    
   Raw Features:  x1 = 50,  x2 = 100
   Engineered:    x3 = 50 * 100 = 5,000
```

---

### Comparison: Raw vs. Engineered Model Representation

```text
Raw Linear Model:
   Price = (w1 * Frontage) + (w2 * Depth) + b
   --> Assumes additive relationship; fails to model multiplicative spatial area.

Engineered Model:
   Price = (w1 * Frontage) + (w2 * Depth) + (w3 * Land Area) + b
   --> Allows gradient descent to assign appropriate weight to spatial land area.
```

---

## 4. Code Implementation

Below is a Python implementation demonstrating how to transform a raw 2-feature matrix into a 3-feature matrix by engineering a product feature (`x1 * x2`) using NumPy.

```python
import numpy as np

def engineer_area_feature(X_raw):
    """
    Transforms raw [Frontage (x1), Depth (x2)] into [Frontage (x1), Depth (x2), Area (x3)].
    
    Args:
      X_raw (ndarray (m, 2)): Matrix of m examples with [Frontage, Depth]
      
    Returns:
      X_engineered (ndarray (m, 3)): Matrix of m examples with [x1, x2, x3=x1*x2]
    """
    x1 = X_raw[:, 0]
    x2 = X_raw[:, 1]
    
    # Compute engineered feature x3 = x1 * x2
    x3 = (x1 * x2).reshape(-1, 1)
    
    # Combine original features with engineered feature
    X_engineered = np.hstack((X_raw, x3))
    return X_engineered


def predict(X, w, b):
    """Computes vectorized predictions f_wb(X) = X . w + b."""
    return np.dot(X, w) + b


# Example Usage:
if __name__ == "__main__":
    # Raw Dataset: 3 lots with [Frontage (ft), Depth (ft)]
    X_raw_data = np.array([
        [40.0, 100.0],
        [50.0, 120.0],
        [60.0, 150.0]
    ])
    
    print("--- 1. Raw Dataset [Frontage, Depth] ---")
    print(X_raw_data)
    
    # Engineer Feature x3 = Area
    X_engineered_data = engineer_area_feature(X_raw_data)
    
    print("\n--- 2. Engineered Dataset [Frontage, Depth, Area] ---")
    print(X_engineered_data)
    
    # Learned parameters (weights for frontage, depth, area and bias)
    w_learned = np.array([50.0, 30.0, 100.0])  # Note high weight on area (x3)
    b_learned = 50000.0
    
    predictions = predict(X_engineered_data, w_learned, b_learned)
    print("\n--- 3. Predicted House Prices ---")
    for i, p in enumerate(predictions):
        print(f"House {i+1}: Frontage={X_raw_data[i,0]}ft, Depth={X_raw_data[i,1]}ft => Price = ${p:,.2f}")
```

---

## 5. Key Terminology & Pitfalls

### Key Distinctions:
* **Raw Features vs. Engineered Features:** Raw features are direct measurements ($x_1, x_2$); engineered features are mathematical transformations or combinations ($x_3 = x_1 \cdot x_2$, $x_4 = \frac{x_1}{x_2}$, $x_5 = x_1^2$) designed to boost predictive power.
* **Linear vs. Non-linear Feature Representation:** Feature engineering allows standard linear regression algorithms to learn non-linear relationships and curves (e.g., fitting polynomials $x^2, x^3$) without changing the underlying linear optimization code.

### Common Gotchas:
* **Over-engineering & Redundancy:** Creating too many arbitrary feature combinations can increase dimensionality ($n$) unnecessarily and lead to overfitting or computational slowing.
* **Don't Forget Feature Scaling:** Engineered features (like area $x_1 \cdot x_2$ or squared terms $x_1^2$) often produce much larger numerical ranges than raw features. Always apply **Feature Scaling** (e.g., Z-score normalization) after engineering new features!

### Key Takeaways:
* **Drastic Performance Boost:** Thoughtful feature engineering often yields greater accuracy gains than switching to more complex algorithm architectures.
* **Domain Knowledge Matters:** Leverage business or physical intuition to craft feature combinations that mirror real-world dynamics.
* **Gateway to Non-Linear Models:** Feature engineering provides the foundation for fitting curved, polynomial functions to dataset features.
