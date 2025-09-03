
# 📘 Lecture 1: Introduction to Machine Learning

---

## 🔹 1. What is Machine Learning?

### **Definition (Tom Mitchell, 1997)**  
A computer program is said to **learn** from experience **E** with respect to some class of tasks **T** and performance measure **P**, if its performance at tasks in T, as measured by P, improves with experience E.

👉 Example:  
- **Task (T)**: Predict if tomorrow will be sunny or rainy.  
- **Performance (P)**: Accuracy of predictions.  
- **Experience (E)**: Past weather data (temperature, humidity, etc).  

---

## 🔹 2. Components of ML

1. **Data Storage** – Collect data (tables, images, text).  
   - Example: Students’ hours studied vs. exam marks.  

2. **Abstraction** – Represent data with models.  
   - Example: Fit a line y = mx + b.  

3. **Generalization** – Perform well on **new unseen data**, not just training data.  
   - Example: Model predicts marks of a new student.  

4. **Evaluation** – Measure performance (accuracy, error rate, precision, recall, F1).  

---

## 🔹 3. Types of Learning

### **1. Supervised Learning**
- Data has **labels**.  
- Goal: Learn mapping from inputs → outputs.  
- Examples:  
  - Predict house prices (Regression).  
  - Spam email detection (Classification).  

### **2. Unsupervised Learning**
- Data has **no labels**.  
- Goal: Find structure/patterns.  
- Examples:  
  - Group customers by spending (Clustering).  
  - Reduce image dimensions (PCA).  

### **3. Reinforcement Learning**
- Agent interacts with **environment**.  
- Learns by rewards and penalties.  
- Examples:  
  - Training robots to walk.  
  - AlphaGo playing chess/Go.  

---

## 🔹 4. Theoretical Perspectives

### **Version Spaces**
- All hypotheses consistent with training data.  
- Example: If hypothesis = “Play Tennis when sunny,” then version space = all rules that fit given samples.

---

### **PAC Learning (Probably Approximately Correct)**
- Introduced by Valiant (1984).  
- Idea: An algorithm is PAC-learnable if, with high probability, it outputs a hypothesis close to the true function, given enough training examples.  
- Formal: For any ε (accuracy) and δ (confidence), the learner finds h such that:  
  $$
  P(error(h) \leq \epsilon) \geq 1 - \delta
  $$  

---

### **VC Dimension (Vapnik–Chervonenkis)**
- Measures **capacity/complexity** of a model.  
- Higher VC = more complex model (can fit more patterns).  
- Example:  
  - A line in 2D can shatter (perfectly classify) at most 3 points → VC dimension = 3.  
  - A perceptron in d dimensions has VC dimension = d+1.  

---

## 🔹 5. Coding Example – “Enjoy Sport” Dataset

We’ll build a **rule-based classifier**.  

### Dataset

| Weather | Temperature | Humidity | Wind | Play Sport |
|---------|-------------|----------|------|------------|
| Sunny   | Hot         | High     | Weak | No         |
| Sunny   | Hot         | High     | Strong | No      |
| Overcast| Hot         | High     | Weak | Yes        |
| Rain    | Mild        | High     | Weak | Yes        |
| Rain    | Cool        | Normal   | Weak | Yes        |
| Rain    | Cool        | Normal   | Strong | No      |
| Overcast| Cool        | Normal   | Strong | Yes      |
| Sunny   | Mild        | High     | Weak | No         |

---

### Python Implementation

```python
# Rule-based classifier for "Enjoy Sport"

# Simple dataset
data = [
    ("Sunny", "Hot", "High", "Weak", "No"),
    ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("Sunny", "Mild", "High", "Weak", "No"),
]

# Build simple rule-based model (enjoy sport if Overcast OR (Rain & Weak Wind))
def rule_based_classifier(weather, temp, humidity, wind):
    if weather == "Overcast":
        return "Yes"
    if weather == "Rain" and wind == "Weak":
        return "Yes"
    return "No"

# Test classifier
correct = 0
for row in data:
    prediction = rule_based_classifier(*row[:-1])
    if prediction == row[-1]:
        correct += 1

print(f"Accuracy: {correct/len(data) * 100:.2f}%")
```

Expected output: **Accuracy ~100%** (since rules match dataset).

---

## 🎯 Practice Tasks
1. Write your own rule-based classifier for “Play Tennis” dataset.  
2. Identify T, P, and E for the task: predicting movie ratings from user history.  
3. Give an example of supervised, unsupervised, and reinforcement learning from **your daily life**.  
4. Draw version space for hypotheses: “Play Sport if Weather = Sunny OR Overcast.”  
5. Compute VC dimension of:  
   - A line in 1D.  
   - A line in 2D.  

---

## 📝 Summary
- **ML definition** = Task, Performance, Experience.  
- **Components** = data storage, abstraction, generalization, evaluation.  
- **Types** = Supervised, Unsupervised, Reinforcement.  
- **Theories** = Version Spaces, PAC learning, VC dimension.  
- **Hands-on**: Built simple rule-based classifier for “Enjoy Sport.”  

---
