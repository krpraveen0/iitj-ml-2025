
# Lecture 3: Bayesian & Decision Trees

## 1. Intuition
Think of a **decision tree** like a flowchart: you start at the root, ask questions (splits), and reach a leaf that gives a decision (class or prediction).  
But a classical tree just picks **the single best split** at each step (like ID3 or CART).  
A **Bayesian Decision Tree** instead says:  
> “There’s uncertainty about which split is truly best. Let’s treat the tree as a probability model and use Bayes’ rule to average over possible trees.”

So instead of one “best” tree, BDTs maintain a **distribution over trees**, capturing uncertainty in structure and parameters.

---

## 2. The Bayesian Framework

### 2.1 Bayes’ Rule Recap
For a hypothesis/model \(h\) and data \(D\):
\[
P(h \mid D) \;=\; \frac{P(D \mid h) \, P(h)}{P(D)}
\]
- \(P(h)\): prior belief about tree structures.  
- \(P(D \mid h)\): likelihood of data given the tree.  
- \(P(h \mid D)\): posterior probability over trees.  
- \(P(D)\): evidence (normalizer).

In BDTs, \(h\) is a particular tree structure with its splitting rules and leaf parameters.

---

### 2.2 Tree Components
- **Structure prior** \(P(T)\): distribution over possible tree shapes (depth, splits). Often favors smaller trees (Occam’s razor).
- **Leaf distributions**: each leaf models class probabilities (e.g., multinomial with Dirichlet prior).
- **Likelihood** \(P(D \mid T)\): probability of data given the splits and leaf distributions.

---

## 3. Bayesian Decision Tree Construction

### 3.1 Priors
1. **Structure prior**: probability of expanding a node at depth \(d\):  
   \[
   P(\text{split at depth } d) = \alpha (1+d)^{-\beta}
   \]  
   with hyperparameters \(\alpha, \beta\) controlling preference for shallow trees.
2. **Leaf prior**: for classification, leaf class probabilities \(\theta \sim \text{Dirichlet}(\gamma)\).

### 3.2 Posterior
Given dataset \(D = \{(x_i, y_i)\}\), the posterior over trees:
\[
P(T \mid D) \;\propto\; P(D \mid T) \, P(T)
\]

### 3.3 Prediction
For a new input \(x\), Bayesian prediction **averages over trees**:
\[
P(y \mid x, D) \;=\; \sum_{T} P(y \mid x, T) \, P(T \mid D)
\]
In practice, the sum over all trees is intractable → we use **sampling (MCMC)** or **variational approximations**.

---

## 4. Learning Bayesian Trees

### 4.1 Posterior Sampling
- **MCMC (Markov Chain Monte Carlo)** is used to sample tree structures proportional to posterior probability.
- Common moves:  
  - Grow a node (split a leaf).  
  - Prune a node (remove a split).  
  - Change a split rule.

### 4.2 Marginal Likelihood at Leaves
For a leaf with class counts \(n_1, \dots, n_K\) and prior \(\text{Dirichlet}(\gamma)\),  
the marginal likelihood is:
\[
P(D_{\text{leaf}}) = \frac{\Gamma(\sum_k \gamma_k)}{\prod_k \Gamma(\gamma_k)} \;
\frac{\prod_k \Gamma(n_k + \gamma_k)}{\Gamma(\sum_k n_k + \sum_k \gamma_k)}
\]

This formula allows computing \(P(D \mid T)\) without explicitly estimating parameters.

---

## 5. Advantages of Bayesian Trees
1. **Uncertainty estimation**: Instead of one tree, BDTs give posterior distributions over trees.
2. **Better generalization**: Avoids overfitting by averaging multiple plausible trees.
3. **Model selection built-in**: Tree depth and splits are controlled probabilistically (priors penalize overly complex trees).

---

## 6. Practical Example

### Dataset: “Enjoy Sport”
Suppose we want to classify if a person enjoys sport (\(y=\{Yes,No\}\)) based on attributes (Sky, Temp, Wind, etc.).

1. **Classical Tree**: Picks the attribute with max information gain at each step.  
2. **Bayesian Tree**: Assigns probabilities to possible splits, samples multiple tree structures, then averages predictions.

So instead of predicting:  
- “Sky = Sunny → Enjoy Sport = Yes” deterministically,  
it says:  
- “There’s a 70% chance Enjoy Sport = Yes, 30% No,” reflecting uncertainty in tree choice.

---

## 7. Coding Snippet (Simplified)

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Trick: approximate Bayesian averaging by bagging many trees
# Each tree gets a bootstrap sample → simulates posterior sampling
bayesian_tree = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=50,   # number of trees ~ posterior samples
    bootstrap=True
)

X = np.array([[1,0],[0,1],[1,1],[0,0]])  # toy features
y = np.array([1,0,1,0])                  # labels

bayesian_tree.fit(X, y)
print(bayesian_tree.predict_proba([[1,0]]))  # predictive distribution
```

While true Bayesian decision trees use MCMC, **bagging approximates the posterior averaging** in practice.

---

## 8. Practice Problems

1. **Conceptual**:  
   - Why might a Bayesian decision tree perform better than a single CART tree in noisy data?  
2. **Math**:  
   - Derive the marginal likelihood of a leaf with counts \(n_1 = 3, n_2 = 2\) under a uniform Dirichlet prior \(\gamma = (1,1)\).  
3. **Coding**:  
   - Use `sklearn`’s `RandomForestClassifier` and interpret it as an approximation to Bayesian averaging over trees. Compare results to a single tree.  

---

## 9. Summary Notes
- **Classical trees**: pick greedy splits → single tree.  
- **Bayesian trees**: use Bayes’ rule → distribution over trees.  
- **Posterior sampling**: MCMC to explore tree space.  
- **Predictions**: average over sampled trees = better calibrated probabilities.  
- **Approximation**: Random forests and bagging ≈ Bayesian averaging.
