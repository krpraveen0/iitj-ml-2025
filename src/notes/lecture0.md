
# 📘 Week 0: Math Foundations (Detailed Notes + Problems)

---

## 🔹 1. Linear Algebra

### **1.1 Vectors**
- A vector = ordered list of numbers.  
  $$
  \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
  $$  
  Example: A student’s marks in 3 subjects: [80, 75, 90].

👉 Think of vectors as **arrows** in space pointing from the origin.

---

### **1.2 Vector Operations**
- **Addition**:
  $$
  [1,2] + [3,4] = [4,6]
  $$  

- **Scalar multiplication**:
  $$
  2 \cdot [3,5] = [6,10]
  $$  

- **Dot Product (similarity)**:
  $$
  \mathbf{x}\cdot \mathbf{y} = \sum_{i=1}^n x_i y_i
  $$  
  Example:  
  [1,2] ⋅ [3,4] = (1)(3) + (2)(4) = 11.  

- **Norm (Length of vector)**:
  $$
  \|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
  $$  
  Example: ||[3,4]|| = √(3²+4²) = 5.

---

### **1.3 Matrices**
A matrix is a **table of numbers**.  
$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

- **Matrix multiplication**:  
  Multiply row of first × column of second.  
  $$
  \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
  \cdot
  \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
  =
  \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
  $$

---

### ✅ Solved Example
Compute dot product and norm:  
$$
x = [2,3], \quad y = [4,5]
$$
- Dot product: 2·4 + 3·5 = 23.  
- Norm of x: √(2²+3²) = √13 ≈ 3.61.

---

### 💡 Practice Problems
1. Compute [1,2,3] ⋅ [4,5,6].  
2. Find the length of vector [6,8].  
3. Multiply matrices:  
   $$\begin{bmatrix}1 & 2 \\ 0 & 1\end{bmatrix} \cdot \begin{bmatrix}3 & 4 \\ 5 & 6\end{bmatrix}$$

---

## 🔹 2. Calculus

### **2.1 Derivatives**
The derivative measures **rate of change** (slope of curve).  
$$
f'(x) = \frac{df}{dx}
$$

- Example:  
  If f(x) = x², then f'(x) = 2x.  
  At x=3, slope = 6.

---

### **2.2 Gradients**
For functions with many variables:  
$$
f(x,y) = x^2 + y^2
$$  
Gradient = vector of partial derivatives:  
$$
\nabla f(x,y) = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right] = [2x, 2y]
$$

---

### **2.3 Optimization (Gradient Descent)**
ML uses calculus to **minimize error**.  
Update rule:
$$
\theta := \theta - \alpha \cdot \nabla J(\theta)
$$  
(where α = learning rate)

---

### ✅ Solved Example
Find derivative of:  
$$
f(x) = 3x^2 + 2x
$$  
Solution:  
f'(x) = 6x + 2.  
At x=2: slope = 6(2)+2 = 14.

---

### 💡 Practice Problems
1. Differentiate f(x) = x³ - 5x² + 6.  
2. Compute gradient of f(x,y) = x² + xy + y².  
3. If f(x) = x⁴, find slope at x=2.  

---

## 🔹 3. Probability & Statistics

### **3.1 Basics**
- P(A): probability of event A.  
- P(A ∩ B): probability of A and B.  
- P(A|B): probability of A given B.  

**Bayes’ Rule**:  
$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

---

### **3.2 Expectation & Variance**
- **Expectation**:
  $$
  E[X] = \sum x_i P(x_i)
  $$  
  (average value)  

- **Variance**:
  $$
  Var(X) = E[(X - E[X])^2]
  $$  
  (spread of data)

---

### **3.3 Common Distributions**
- **Bernoulli**: coin flip (0/1).  
- **Gaussian (Normal)**: bell curve.

---

### ✅ Solved Example
Roll a fair dice. What is probability of 6?  
$$
P(6) = \frac{1}{6} ≈ 0.1667
$$  

If you roll dice 600 times, expected number of 6s = 600 × 1/6 = 100.

---

### 💡 Practice Problems
1. Toss a fair coin 3 times. Find probability of getting 2 heads.  
2. Compute mean & variance of dataset: [2, 4, 4, 4, 5, 5, 7, 9].  
3. Apply Bayes’ rule:  
   A patient has a disease with probability 1%. A test is 90% accurate. If test is positive, what is probability that patient has disease?  

---

## 🔹 4. Python Practice

```python
import numpy as np

# Linear Algebra
x = np.array([2,3]); y = np.array([4,5])
print("Dot product:", np.dot(x,y))
print("Norm of x:", np.linalg.norm(x))

# Calculus (sympy)
import sympy as sp
x,y = sp.symbols('x y')
f = 3*x**2 + 2*x
print("Derivative:", sp.diff(f, x))
g = x**2 + x*y + y**2
print("Gradient:", [sp.diff(g, var) for var in (x,y)])

# Probability
data = [2,4,4,4,5,5,7,9]
print("Mean:", np.mean(data))
print("Variance:", np.var(data))
```

---

## 📌 Summary
- **Linear Algebra** = structure of data (vectors/matrices).  
- **Calculus** = optimization (slopes, gradients).  
- **Probability** = uncertainty, distributions, Bayes’ rule.  

---
