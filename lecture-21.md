# Lecture 21: Multiple Linear Regression
**Source:** Andrew Ng - Machine Learning Specialization (YouTube)

## 1. The Core Idea
In standard linear regression (univariate), we predict an outcome based on a single input feature (e.g., predicting a house's price based *only* on its size). 

**Multiple Linear Regression** is the evolution of this. In the real world, an outcome rarely depends on just one thing. This model allows us to use **multiple input features** simultaneously (e.g., predicting a house's price based on its size, number of bedrooms, number of floors, and age).

## 2. New Notation for Multiple Features
To handle multiple variables mathematically, we need a clean way to label our data. Let's assume we are predicting house prices:

*   **$n$**: The total number of features. *(Example: size, bedrooms, floors, age = 4 features, so $n = 4$)*.
*   **$x_j$**: The $j^{th}$ specific feature. *(Example: $x_1$ = size, $x_2$ = bedrooms)*.
*   **$\vec{x}^{(i)}$**: All the features for the $i^{th}$ specific training example (represented as a list or "vector"). *(Example: $\vec{x}^{(2)}$ represents all the details of the 2nd house in your dataset).*
*   **$x_j^{(i)}$**: The specific value of feature $j$ in the $i^{th}$ training example. *(Example: $x_2^{(3)}$ means the number of bedrooms in the 3rd house).*

## 3. The Model Representation (Step-by-Step)

**The Old Model (One Feature):**
$$f_{w,b}(x) = wx + b$$

**The Expanded Model (Multiple Features):**
Instead of one weight ($w$), every single feature gets its own weight (parameter). The expanded model looks like this:
$$f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

**Intuition:** 
Think of this as calculating a final price. 
*   **$b$** is the base price (what the house is worth before looking at any features).
*   **$w_1$** is the value added (or subtracted) for every unit of feature $x_1$ (e.g., adding $100 for every extra square foot).
*   **$w_2$** is the value added per unit of $x_2$ (e.g., adding $5000 for an extra bedroom), and so on.

## 4. Vectorization (Simplifying the Math)
Writing out $w_1x_1 + w_2x_2 + \dots$ gets tedious when you have 100 features. To make the math cleaner and the code run much faster, we use **vectors** and a mathematical operation called the **dot product**.

**Step 1: Group the parameters into a vector**
Instead of writing weights individually, pack them into a single column/list called $\vec{w}$:
$$\vec{w} = [w_1, w_2, \dots, w_n]$$

**Step 2: Group the features into a vector**
Do the exact same thing for a single house's features:
$$\vec{x} = [x_1, x_2, \dots, x_n]$$

**Step 3: The Dot Product**
In linear algebra, the "dot product" (written as $\vec{w} \cdot \vec{x}$) simply means: take the first item of $\vec{w}$ and multiply it by the first item of $\vec{x}$, then add it to the second items multiplied together, all the way to the end.

$$\vec{w} \cdot \vec{x} = (w_1 \times x_1) + (w_2 \times x_2) + \dots + (w_n \times x_n)$$

**Step 4: The Final Vectorized Equation**
Because the dot product perfectly replaces that long chain of additions, we can write our entire model beautifully as:

$$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

> **Why do this?** In programming languages (like Python with NumPy), calculating a dot product is highly optimized. It allows the computer to do all those multiplications at the exact same time (in parallel) rather than one by one, massively speeding up your code.

## 5. Important Terminology Clarification
*   This technique is formally called **Multiple Linear Regression** (because there are *multiple* input features).
*   **Do not call it "Multivariate Regression."** In advanced statistics, multivariate regression refers to a different model where you are trying to predict multiple *outputs* at once, not multiple *inputs*.