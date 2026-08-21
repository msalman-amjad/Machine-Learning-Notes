# Lecture 4: Supervised Learning Part 1 (Regression)
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 1, Lesson 2 (Lecture 4)

## 1. The Core Idea
**Supervised Learning** is the most widely deployed and commercially successful branch of machine learning, driving approximately **99% of the economic value** created by AI applications today.

The fundamental objective of supervised learning is to learn a mapping function from an **input $x$** to a target **output $y$**. 

By providing the algorithm with a "supervised" dataset containing correct input-output pairs $(x, y)$, the model learns the underlying relationship. Once trained, the model can generalize to take brand-new, unseen inputs $x$ and predict their correct output $y$.

---

## 2. Mathematical Foundation & Notation

### Variables & Notation:
* **$x$**: The input feature (e.g., house size in sq ft).
* **$y$**: The target output label (e.g., house price in thousands of dollars).
* **$(x^{(i)}, y^{(i)})$**: The $i^{th}$ training example pair in the dataset.
* **$m$**: Total number of training pairs $(x, y)$.
* **$w$**: Weight parameter (slope of the line).
* **$b$**: Bias parameter (y-intercept).

---

### Step 1: Linear vs. Non-Linear Function Fitting

When modeling a relationship between input $x$ and output $y$ (such as housing prices based on square footage):

1. **Linear Regression Model:**
   Fits a straight line through the training data points:
   $$f_{w,b}(x) = w \cdot x + b$$
   * **$w$ (slope):** Rate of change in target $y$ per unit change in feature $x$.
   * **$b$ (y-intercept):** Base value of target $y$ when feature $x = 0$.

2. **Non-Linear Regression Model:**
   If the relationship curves (e.g., housing prices plateau as size increases), we can fit non-linear curves (such as polynomials) to improve fit and accuracy:
   $$f_{\vec{w},b}(x) = w_1 x + w_2 x^2 + b$$

---

### Step 2: Cost Function $J(w,b)$
To evaluate how accurately the fitted line represents the training data, we measure the Mean Squared Error (MSE):

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$

---

### Definition: Regression
In supervised learning, **Regression** specifically refers to predicting a **continuous numerical output**. 

* A continuous output means the algorithm predicts a number from an infinite spectrum of possible numerical values (e.g., predicting a house price of \$250,500, \$312,850.75, etc.).

---

## 3. Common Real-World Applications of Supervised Learning

Supervised learning powers major commercial AI systems by mapping specific inputs $x$ to target outputs $y$:

| Application | Input ($x$) | Output ($y$) | Problem Type |
| :--- | :--- | :--- | :--- |
| **Spam Filtering** | Email text & metadata | Spam / Not Spam (0 or 1) | Classification |
| **Speech Recognition** | Audio waveform snippet | Text transcription string | Sequence Mapping |
| **Machine Translation** | Source language text (English) | Target language text (Spanish) | Sequence Mapping |
| **Online Advertising** | User profile & ad content | Ad Click / No Click (0 or 1) | Classification / Probability |
| **Visual Quality Inspection**| Manufacturing part image | Defect / No Defect (0 or 1) | Classification |
| **Housing Price Prediction**| House size, bedrooms | Price in dollars ($\$$) | **Regression** |

---

## 4. Code Implementation

Below is a Python implementation illustrating how a trained linear regression model calculates price predictions for continuous input values.

```python
import numpy as np

def predict_price(size, w, b):
    """
    Predicts house price given size using learned linear model y = w*x + b.
    
    Args:
      size (float or ndarray): House size in sq ft
      w (float): Learned weight parameter ($k per sq ft)
      b (float): Learned bias parameter ($k base price)
      
    Returns:
      price (float or ndarray): Predicted house price in thousands of dollars ($k)
    """
    return (w * size) + b


# Learned parameters from training data
w_learned = 0.2   # Represents $200 per square foot
b_learned = 50.0  # Represents $50,000 base price

# Predict for a 750 sq ft house
size_input = 750.0
predicted_price = predict_price(size_input, w_learned, b_learned)

print(f"House Size: {size_input} sq ft")
print(f"Predicted Price: ${predicted_price:.1f}k (${predicted_price * 1000:,.2f})")
```

---

## 5. Key Terminology & Takeaways

### Key Concepts:
* **Supervised Learning:** Learning algorithms trained on labeled datasets containing $(x, y)$ input-output pairs.
* **Regression:** Supervised learning tasks where the target output $y$ is a continuous numerical value.
* **Linear vs. Non-Linear Models:** Straight-line models $f(x) = wx + b$ vs. curved polynomial models to capture complex data relationships.
* **Economic Value:** Powers ~99% of current commercial AI applications across online search, advertising, translation, speech recognition, and medical imaging.

### Key Takeaways:
* **Supervised Learning Dominance:** Maps inputs $x$ to labeled targets $y$, powering nearly all modern commercial AI systems.
* **Continuous Value Predictions:** Linear regression fits functions $f_{w,b}(x) = wx + b$ to predict continuous numerical targets.
* **Model Versatility:** Linear and non-linear polynomial regression allow fitting complex real-world data distributions.
