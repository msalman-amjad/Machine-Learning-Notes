# Lecture 31: Motivation (Binary Classification & Linear Regression Limitations)
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 3, Lesson 1 (Lecture 31)

## 1. The Core Idea
Up to this point, our study of supervised learning focused on **regression** problems where the goal is to predict a continuous numerical value (e.g., predicting house prices, temperatures, or revenues). 

In this lecture, we introduce **classification** problems, where the target variable $y$ can only take on discrete categorical values. We begin with **binary classification**, where there are only two possible outcomes (e.g., Yes/No, True/False, Malignant/Benign).

We explore why directly applying **linear regression** to binary classification tasks by thresholding the prediction (e.g., predicting $y=1$ when $f_{w,b}(x) \ge 0.5$) is fundamentally flawed. Specifically, we demonstrate how linear regression decision boundaries are extremely sensitive to outliers, leading to severe misclassifications, and why outputs outside $[0, 1]$ fail to provide sensible probabilistic predictions.

---

## 2. Binary Classification Basics

In binary classification, the target output $y$ is constrained to two discrete classes:

$$y \in \{0, 1\}$$

### Standard Terminology & Conventions:
* **Negative Class ($y = 0$):** Denotes the absence of a condition, property, or event (e.g., False, No, Benign tumor, Non-spam email, Normal transaction).
* **Positive Class ($y = 1$):** Denotes the presence of a condition, property, or event (e.g., True, Yes, Malignant tumor, Spam email, Fraudulent transaction).

> **Note on Naming:** Assigning $y=1$ as "positive" and $y=0$ as "negative" is a standard mathematical convention; "positive" does not necessarily imply a beneficial outcome (e.g., a positive test result for a medical condition indicates presence of disease).

### Common Examples of Binary Classification:
1. **Medical Diagnosis:** Is a tumor malignant ($1$) or benign ($0$)?
2. **Email Filtering:** Is an email spam ($1$) or not spam ($0$)?
3. **Financial Fraud Detection:** Is a transaction fraudulent ($1$) or legitimate ($0$)?
4. **Spam / Defective Product Inspection:** Is a manufactured part defective ($1$) or functional ($0$)?

---

## 3. Linear Regression for Classification: The Thresholding Approach

To adapt linear regression $f_{w,b}(x) = w x + b$ for binary classification:
1. Fit a straight line to the training data.
2. Apply a **threshold value** (typically $0.5$) to classify predictions:

$$\hat{y} = \begin{cases} 1 & \text{if } f_{w,b}(x) \ge 0.5 \\ 0 & \text{if } f_{w,b}(x) < 0.5 \end{cases}$$

### Decision Boundary Concept:
The **decision boundary** is the location in feature space $x$ where the model prediction equals the threshold:

$$f_{w,b}(x) = 0.5 \implies w x + b = 0.5 \implies x = \frac{0.5 - b}{w}$$

* For any $x$ to the right of this decision boundary ($x \ge \frac{0.5 - b}{w}$), $f_{w,b}(x) \ge 0.5$, so we predict $y = 1$.
* For any $x$ to the left ($x < \frac{0.5 - b}{w}$), $f_{w,b}(x) < 0.5$, so we predict $y = 0$.

---

## 4. Why Linear Regression Fails for Classification

### Issue 1: Sensitivity to Outliers & Shift of Decision Boundary

Consider a medical dataset predicting tumor malignancy ($y \in \{0, 1\}$) based on tumor size ($x$).

#### Scenario A: Dataset without Outliers
* Data points for benign tumors ($y=0$) cluster at small sizes ($x = 1, 2, 3$).
* Data points for malignant tumors ($y=1$) cluster at larger sizes ($x = 4, 5, 6$).
* Fitting $f_{w,b}(x) = w x + b$ yields a line that crosses $0.5$ at $x = 3.5$.
* **Decision Boundary:** $x = 3.5$. All tumors with size $x \ge 3.5$ are correctly classified as $y=1$; tumors with $x < 3.5$ are classified as $y=0$.

```text
  y (Malignancy)
   1 |                *   *   *           <-- Malignant (y=1)
     |               /
 0.5 |--------------/------------------  Threshold = 0.5
     |             /|
   0 |  *   *   * / |                     <-- Benign (y=0)
     +------------+-------------------> x (Tumor Size)
                 3.5 (Decision Boundary)
```

#### Scenario B: Adding a Far-Right Outlier
Now add a single extra training example of a very large malignant tumor at $x = 12$ ($y = 1$).

* To minimize the mean squared error (MSE) cost function $\frac{1}{2m}\sum (f(x^{(i)}) - y^{(i)})^2$, the best-fit line rotates downwards to reduce the squared residual for the outlier at $x=12$.
* The slope $w$ flattens, causing the line to cross $y = 0.5$ much farther to the right (e.g., at $x = 5.2$).
* **Shifted Decision Boundary:** $x = 5.2$.
* **Consequence of the Shift:** Tumors of size $x = 4$ and $x = 5$ (which are actually malignant, $y=1$) now lie to the left of the new decision boundary ($x = 5.2$) and are incorrectly classified as benign ($y=0$)!

```text
  y (Malignancy)
   1 |                *   *   *                                      * (Far Outlier x=12, y=1)
     |                     \                                     . '
 0.5 |----------------------\--------------------------------. '  Threshold = 0.5
     |                       \                           . '
   0 |  *   *   *             \                      . '              <-- Benign (y=0)
     +-------------------------+--------------------+-----------------> x (Tumor Size)
                              3.5                  5.2 
                        (Old Boundary)      (NEW Shifted Boundary -> MISCLASSIFIES x=4,5!)
```

### Issue 2: Predictions Unbounded Outside $[0, 1]$

* Linear regression functions $f_{w,b}(x) = w x + b$ range from $-\infty$ to $+\infty$.
* For very small tumor sizes, $f_{w,b}(x)$ can output negative values (e.g., $f(x) = -0.3$).
* For very large tumor sizes, $f_{w,b}(x)$ can output values exceeding $1$ (e.g., $f(x) = 1.8$).
* In classification, target values are strictly binary ($0$ or $1$). Interpreting $f_{w,b}(x)$ as a probability $P(y=1|x)$ makes no sense when predictions fall outside $[0, 1]$.

---

## 5. Mathematical Summary of Linear Regression Limitations

| Metric / Aspect | Linear Regression Approach | Expected / Ideal Classification Behavior |
| :--- | :--- | :--- |
| **Output Range** | $f_{w,b}(x) \in (-\infty, +\infty)$ | Bounded output $P(y=1\|x) \in [0, 1]$ |
| **Loss Function** | Mean Squared Error (MSE) penalizes distance squared | Log Loss / Binary Cross-Entropy penalizes wrong predictions |
| **Robustness to Outliers** | Low (extreme positive points shift linear fit) | High (extreme positive points do not distort boundary) |
| **Decision Boundary** | Dynamic shift with distant outliers | Stable boundary at threshold |

---

## 6. Code Implementation & Visualization

The following Python script demonstrates how fitting linear regression to binary labels leads to decision boundary shifts when an outlier is introduced.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Dataset WITHOUT Outlier
# Tumor sizes (cm) and binary malignancy labels (0 = Benign, 1 = Malignant)
X_clean = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y_clean = np.array([0, 0, 0, 1, 1, 1])

# 2. Dataset WITH Outlier (Large malignant tumor at x=12)
X_outlier = np.array([1, 2, 3, 4, 5, 6, 12]).reshape(-1, 1)
y_outlier = np.array([0, 0, 0, 1, 1, 1, 1])

# Fit Linear Regression Model 1 (Clean)
model_clean = LinearRegression()
model_clean.fit(X_clean, y_clean)

# Fit Linear Regression Model 2 (With Outlier)
model_outlier = LinearRegression()
model_outlier.fit(X_outlier, y_outlier)

# Generate prediction domain for visualization
x_range = np.linspace(0, 13, 200).reshape(-1, 1)
pred_clean = model_clean.predict(x_range)
pred_outlier = model_outlier.predict(x_range)

# Find Decision Boundaries (where f(x) = 0.5)
# f(x) = w*x + b = 0.5 => x = (0.5 - b) / w
db_clean = (0.5 - model_clean.intercept_) / model_clean.coef_[0]
db_outlier = (0.5 - model_outlier.intercept_) / model_outlier.coef_[0]

print(f"Clean Data Decision Boundary (f(x)=0.5): x = {db_clean:.2f}")
print(f"Outlier Data Decision Boundary (f(x)=0.5): x = {db_outlier:.2f}")

# Plotting
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(X_clean[y_clean==0], y_clean[y_clean==0], color='blue', s=80, label='Benign (y=0)', zorder=5)
plt.scatter(X_clean[y_clean==1], y_clean[y_clean==1], color='red', s=80, label='Malignant (y=1)', zorder=5)
plt.scatter([12], [1], color='darkred', s=120, marker='X', label='Outlier (x=12, y=1)', zorder=5)

# Plot fitted lines
plt.plot(x_range, pred_clean, 'g--', linewidth=2, label=f'Fit w/o Outlier (Boundary: {db_clean:.2f})')
plt.plot(x_range, pred_outlier, 'm-', linewidth=2, label=f'Fit w/ Outlier (Boundary: {db_outlier:.2f})')

# Plot Threshold line y = 0.5
plt.axhline(0.5, color='gray', linestyle=':', label='Threshold y=0.5')
plt.axvline(db_clean, color='green', linestyle='--', alpha=0.6)
plt.axvline(db_outlier, color='magenta', linestyle='-', alpha=0.6)

plt.xlabel('Tumor Size (cm)')
plt.ylabel('Malignancy Label / Linear Prediction f(x)')
plt.title('Failure of Linear Regression for Binary Classification (Outlier Sensitivity)')
plt.ylim(-0.2, 1.3)
plt.legend()
plt.grid(True)
plt.show()
```

---

## 7. Key Takeaways & Future Directions

### Core Takeaways:
1. **Binary Classification Targets:** Outputs $y$ belong to discrete sets $\{0, 1\}$ (Negative class $0$, Positive class $1$).
2. **Linear Regression Failure:** Using $f_{w,b}(x) = wx + b$ with a $0.5$ threshold fails for classification because:
   * Adding far-away positive data points shifts the decision boundary unnecessarily, leading to misclassification of valid positive samples.
   * Outputs $f_{w,b}(x)$ can be $<0$ or $>1$, violating probability constraints.

### Next Steps: Logistic Regression
To resolve these issues, the subsequent lessons introduce **Logistic Regression**. 
* Despite containing "regression" in its name, **logistic regression is a classification algorithm**.
* It maps linear outputs through a sigmoid function so that all predictions $f_{\vec{w},b}(\vec{x})$ are strictly constrained between $0$ and $1$:

$$0 \le f_{\vec{w},b}(\vec{x}) \le 1$$

* This provides stable decision boundaries immune to extreme outliers and enables direct interpretation of predictions as probabilities $P(y=1|\vec{x})$.
