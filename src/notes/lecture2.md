# 📘 Lecture 2: ML Paradigms, Bayesian, Decision Trees

---
## 1. ML Paradigms
- **Supervised Learning:** Learn from labeled data (e.g., classification, regression).
- **Unsupervised Learning:** Find patterns in unlabeled data (e.g., clustering, dimensionality reduction).
- **Reinforcement Learning:** Learn by interacting with environment, receiving rewards/penalties.
- **Semi-supervised Learning:** Use both labeled and unlabeled data.
- **Self-supervised Learning:** Generate labels from data itself (e.g., predicting next word in a sentence).

## 2. Bayesian Learning
- **Bayes Theorem:**
   $$P(H|D) = \frac{P(D|H)P(H)}{P(D)}$$
   - $P(H|D)$: Posterior, $P(D|H)$: Likelihood, $P(H)$: Prior, $P(D)$: Evidence
- **Naive Bayes Classifier:** Assumes features are conditionally independent given the class.
- **MAP (Maximum a Posteriori):** Choose hypothesis with highest posterior probability.
- **ML (Maximum Likelihood):** Choose hypothesis that maximizes likelihood of data.

## 3. Decision Trees
- **ID3:** Uses information gain (entropy) to split nodes.
- **C4.5:** Extension of ID3, handles continuous features, pruning.
- **CART:** Uses Gini index, supports classification and regression.
- **Entropy:** Measure of impurity/uncertainty.
- **Information Gain:** Reduction in entropy after split.
- **Gini Index:** Alternative impurity measure.

## 4. Applications & Examples
- **Email Spam Detection:** Naive Bayes for classifying emails.
- **Medical Diagnosis:** Decision trees for predicting diseases.
- **Customer Segmentation:** Unsupervised clustering.

## 5. Summary
- ML paradigms define how learning is structured.
- Bayesian learning uses probability and prior knowledge.
- Decision trees are interpretable, widely used for classification.

---
# 📘 Lecture 2: Regression (Supervised Learning – Part 1)

---

## 🔹 1. Intuition

Regression = **predicting numbers**.  
Think of it like:  
- Hours studied → Exam score.  
- Size of house → Price.  
- Temperature → Ice cream sales.  

We try to fit a **line** (or curve) through the data to make predictions.

---

## 🔹 2. Linear Regression

### **Model**
For one variable (simple linear regression):  
$$
h_\theta(x) = \theta_0 + \theta_1 x
$$  

- x: input (hours studied).  
- y: output (marks).  
- θ0: intercept (baseline).  
- θ1: slope (effect of study hours).  

---

### **Cost Function**
We want predictions close to actual values.  
Mean Squared Error (MSE):  
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$  

---

### **Optimization (Gradient Descent)**
Update rule:  
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$  

- α: learning rate.  
- Repeat until convergence.  

---

## 🔹 3. Multiple Linear Regression
When there are many features:  
$$
h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n
$$  

Example: House price = f(size, bedrooms, location).

---

## 🔹 4. Logistic Regression (Classification Preview)
- For classification (Yes/No).  
- Uses **sigmoid function**:  
$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
$$  

---

## 🔹 5. Python Example: Predict Exam Scores

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: hours studied vs exam marks
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([40, 50, 65, 70, 85])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Print results
print("Intercept:", model.intercept_)
print("Slope:", model.coef_)

# Plot
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.xlabel("Hours studied")
plt.ylabel("Exam Marks")
plt.legend()
plt.show()
```

---

## 🎯 Practice Tasks

1. Fit a linear regression line for data:  
   Hours studied = [2, 4, 6, 8]  
   Marks = [81, 93, 91, 97].  

2. Compute cost function J(θ) for predictions [50, 60, 70] vs actual [52, 58, 75].  

3. Extend to multiple regression: Predict house price using area (sqft) + bedrooms.  

4. Experiment with learning rate α in gradient descent. What happens if it’s too large? Too small?  

5. Plot a logistic regression curve for “exam passed” (Yes/No) given hours studied.  

---

## 📝 Summary

- Regression predicts **numbers** (continuous values).  
- Linear regression uses **straight line**: y = θ0 + θ1x.  
- Cost function = MSE.  
- Optimized using **gradient descent**.  
- Multiple regression → many features.  
- Logistic regression → classification (0/1).  

---
