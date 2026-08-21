# Lecture 2: Applications of Machine Learning
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

> **Course Context:** Course 1, Week 1, Lesson 1 (Lecture 2)

## 1. The Core Idea
As technology advances, the scope of problems we solve with software has fundamentally shifted. Traditional software engineering relies on **explicit programming**—where human developers manually code step-by-step algorithms and logic rules (for example, computing the shortest path on a map).

However, real-world tasks like speech recognition, analyzing medical X-rays for disease detection, or steering self-driving cars involve far too many variables, edge cases, and noise to be programmed by hand.

**Machine Learning (ML)** overcomes these limits. Instead of hand-coding rules, we feed large amounts of data into machine learning algorithms, allowing the computer to **learn the rules autonomously**.

---

## 2. Mathematical Foundation & Notation

To build ML algorithms capable of learning from data, we express predictive models and their accuracy mathematically.

### Variables & Notation:
* **$m$**: Total number of training examples.
* **$n$**: Number of features per training example.
* **$x^{(i)}$**: Input feature(s) for the $i^{th}$ training example.
* **$y^{(i)}$**: Actual target value (ground truth) for the $i^{th}$ training example.
* **$w$ or $\vec{w}$**: Weight parameter(s) learned by the model.
* **$b$**: Bias scalar parameter learned by the model.
* **$f_{w,b}(x)$**: Model prediction function.

---

### Step 1: Model Prediction Function
For linear regression, model predictions are computed as a linear combination of inputs, parameters, and bias:

$$f_{w,b}(x) = w \cdot x + b$$

When extended to multiple features, vector notation represents this via a dot product:

$$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

---

### Step 2: Cost Function $J(w,b)$
To quantify how accurately our model predicts targets across the dataset, we calculate the Mean Squared Error (MSE) cost function:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$

**Key Components:**
1. **Residual Error ($f_{w,b}(x^{(i)}) - y^{(i)}$):** Difference between prediction and actual target value.
2. **Squared Error ($(\dots)^2$):** Penalizes larger prediction errors and ensures non-negative values.
3. **Average over Dataset ($\frac{1}{2m} \sum$):** Computes mean error across all $m$ examples (with a $\frac{1}{2}$ constant to simplify derivative calculations during optimization).

---

## 3. Industry Impact, Economic Value & AGI

### Real-World AI Systems & Projects:
Andrew Ng shares insights from leading major AI organizations:
* **Google Brain:** Led foundational projects applying deep learning to speech recognition and computer vision.
* **Baidu:** Directed AI efforts in augmented reality (AR), online fraud detection, speech processing, and autonomous driving technology.

### Economic Valuation of AI & ML:
According to a landmark study by **McKinsey & Company**, Artificial Intelligence and Machine Learning are projected to generate an additional **$13 Trillion annually in global economic value by 2030**. This economic transformation impacts diverse global sectors:
* **Healthcare:** Automated diagnostic imaging and personalized treatment planning.
* **Agriculture:** Crop yield forecasting, weed detection, and autonomous farming equipment.
* **Retail & E-commerce:** Personalized recommendation engines and automated demand supply-chain forecasting.
* **Manufacturing:** Predictive machinery maintenance and computer vision quality inspection.

### Narrow AI vs. Artificial General Intelligence (AGI):
* **Narrow AI (Applied AI):** AI models built to perform specific tasks exceptionally well (e.g., web search, medical imaging, game playing). All current commercial AI success belongs to Narrow AI.
* **Artificial General Intelligence (AGI):** The long-term goal of building machines capable of performing *any* intellectual task a human can do. While AGI remains an inspiring vision, true AGI is still a long-term goal that may be decades or centuries away.

---

## 4. Code Implementation

In Python, machine learning computations are implemented efficiently using array math in **NumPy** to perform vectorized operations instead of slow manual loops.

```python
import numpy as np

def predict(x, w, b):
    """
    Computes model predictions using vectorized matrix/vector operations.
    
    Args:
      x (ndarray): Feature inputs, shape (n,) for single example or (m, n) for batch
      w (ndarray): Parameter weights, shape (n,)
      b (float): Parameter bias scalar
      
    Returns:
      prediction (float or ndarray): Model prediction(s)
    """
    return np.dot(x, w) + b


# Example Usage:
# Define a single training example with 3 features (e.g., house size, bedrooms, floors)
X_sample = np.array([1416.0, 3.0, 2.0])

# Initial weight vector for 3 features and bias term
w_sample = np.array([0.15, 10.5, 5.0])
b_sample = 25.0

# Compute vectorized prediction: (1416*0.15 + 3*10.5 + 2*5.0) + 25.0
predicted_price = predict(X_sample, w_sample, b_sample)
print(f"Predicted Output: {predicted_price:.2f}")
```

---

## 5. Key Terminology & Takeaways

### Key Distinctions:
| Concept | Traditional Software | Machine Learning |
| :--- | :--- | :--- |
| **Logic Source** | Explicit, hand-coded rules written by engineers | Autonomous learning from historical data |
| **Best Suited For** | Deterministic tasks (e.g., GPS shortest path) | Complex, perceptual tasks (e.g., vision, speech, diagnosis) |

### Key Takeaways:
* **ML vs. Explicit Programming:** Machine learning enables computers to solve complex perceptual problems (speech, vision, autonomous driving) that are impossible to hardcode manually.
* **Massive Economic Impact:** Projected to contribute $\$13\text{ Trillion}$ annually to the global economy by 2030 across healthcare, retail, manufacturing, and agriculture.
* **Narrow AI Focus:** Practical commercial AI is focused on Narrow AI (task-specific excellence), while Artificial General Intelligence (AGI) remains a long-term goal.
