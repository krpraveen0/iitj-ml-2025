
# 📘 Lecture 1: Practice Task Solutions

---

## 🔹 Task 1: Rule-based classifier for “Play Tennis” dataset

### Dataset (simplified)
| Outlook  | Temperature | Humidity | Wind   | Play Tennis |
|----------|-------------|----------|--------|-------------|
| Sunny    | Hot         | High     | Weak   | No          |
| Sunny    | Hot         | High     | Strong | No          |
| Overcast | Hot         | High     | Weak   | Yes         |
| Rain     | Mild        | High     | Weak   | Yes         |
| Rain     | Cool        | Normal   | Weak   | Yes         |
| Rain     | Cool        | Normal   | Strong | No          |
| Overcast | Cool        | Normal   | Strong | Yes         |
| Sunny    | Mild        | High     | Weak   | No          |

### Simple Rules
- If **Outlook = Overcast**, then Play = Yes.  
- If **Outlook = Rain** and **Wind = Weak**, then Play = Yes.  
- Otherwise, Play = No.  

✅ This gives nearly 100% accuracy for the sample dataset.

---

## 🔹 Task 2: Identify T, P, E for predicting movie ratings

- **Task (T):** Predict a user’s rating for a new movie.  
- **Performance (P):** Mean Squared Error (MSE) between predicted and actual ratings.  
- **Experience (E):** Past ratings of movies by many users (training dataset).  

---

## 🔹 Task 3: Examples from daily life

- **Supervised:** Predict exam score from hours studied (labeled data: past scores).  
- **Unsupervised:** Group friends into clusters by hobbies (no labels).  
- **Reinforcement:** Learning to play a video game by trial-and-error (reward = points).  

---

## 🔹 Task 4: Version space example

Hypothesis set: “Play Sport if Weather = Sunny OR Overcast.”  

- Training samples where Weather = Sunny → Yes.  
- Training samples where Weather = Overcast → Yes.  
- Training samples where Weather = Rain → No.  

The **version space** = all hypotheses consistent with these labels (rules that classify Sunny/Overcast as Yes, Rain as No).  

---

## 🔹 Task 5: VC Dimension

1. **Line in 1D (a threshold function):**  
   - Can shatter at most **1 point** on the number line.  
   - So VC dimension = 1.  

2. **Line in 2D (a straight line classifier):**  
   - Can shatter any set of **3 non-collinear points** (can separate them with some line).  
   - Cannot always shatter 4 points.  
   - So VC dimension = 3.  

---

## ✅ Summary of Solutions
 
---

## 🔥 Advanced Practice Questions

### 1. PAC Learning
- (a) Explain the concept of PAC learning in your own words. Why is it important in machine learning?
- (b) Given a binary classifier, how would you estimate the number of training samples needed to achieve a desired accuracy (ε) and confidence (δ)?

### 2. VC Dimension
- (a) Compute the VC dimension of a rectangle classifier in 2D (i.e., a model that classifies points as inside/outside a rectangle).
- (b) For a neural network with a single perceptron and d inputs, what is the VC dimension? Justify your answer.

### 3. Version Spaces
- (a) Draw the version space for the following hypotheses: “Play Sport if Weather = Sunny OR Overcast,” given a dataset with all three weather types.
- (b) How does the version space change as more training examples are added?

### 4. Coding Tasks
- (a) Write a Python function to compute the accuracy, precision, recall, and F1-score for a binary classifier given a list of predictions and true labels.
- (b) Modify the rule-based classifier from the lecture to handle missing values (e.g., if Wind is unknown).

### 5. Real-World Scenarios
- (a) For the task of recommending news articles to users, identify T, P, and E.
- (b) Give an example of a reinforcement learning problem in robotics and describe the reward structure.

### 6. Theory & Application
- (a) Why is generalization important in machine learning? Give an example where a model performs well on training data but fails on new data.
- (b) Describe a scenario where unsupervised learning would be more appropriate than supervised learning.

---

---
