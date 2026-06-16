# 📊 Linear Discriminant Analysis (LDA)
> *"Finding the best angle to separate your classes — like adjusting a spotlight to tell friends apart."*

---

## 📌 Summary

**Linear Discriminant Analysis (LDA)** is a supervised dimensionality reduction and classification technique. Unlike PCA (which ignores class labels), LDA uses class information to find a new axis (or axes) that **maximally separates different classes** while keeping data within the same class tightly grouped.

It's used both as a **feature reduction tool** (before classification) and directly as a **linear classifier**.

---

## 1. What is LDA?

### 🔦 The Spotlight Analogy
Imagine you're in a dark room with three groups of people standing in different clusters. You shine a flashlight at the floor to cast their shadows. Most angles create overlapping shadows — you can't tell the groups apart. LDA finds the **one perfect angle** where the shadows of each group are as far apart as possible, with minimal overlap within each group.

That "perfect angle" is the **Linear Discriminant** — a new axis in the data space that maximizes class separability.

**Supervised**: LDA knows the class labels during training. This is the key difference from PCA, which is unsupervised.

---

## 2. Mathematical Idea (No Derivation)

LDA works by optimizing a ratio called the **Fisher Criterion**:

```
Maximize:  Between-Class Scatter (SB)
           ─────────────────────────
           Within-Class Scatter  (SW)
```

- **Between-Class Scatter (SB)**: How far apart are the class centers (means) from each other?
- **Within-Class Scatter (SW)**: How spread out are the data points within each class?

LDA wants the numerator **large** (classes far apart) and denominator **small** (each class tightly packed).

**Maximum number of LDA components**: min(n_classes − 1, n_features)  
→ For 3 classes: at most 2 LDA components (that's why Iris reduces to 2D perfectly)

The solution comes from solving a **generalized eigenvalue problem** on the scatter matrices. The top eigenvectors become the new projection axes.

---

## 3. How LDA Works (Step-by-Step)

1. **Compute class means**: Calculate the mean vector for each class
2. **Compute Within-Class Scatter (SW)**: Measure how spread out data is inside each class
3. **Compute Between-Class Scatter (SB)**: Measure how far class means are from the overall mean
4. **Solve eigenvalue problem**: Find directions that maximize SB / SW
5. **Select top k eigenvectors**: These are your new axes (Linear Discriminants)
6. **Project data**: Multiply original data by the selected eigenvectors → lower-dimensional representation

---

## 4. Key Assumptions

| Assumption | What it means |
|-----------|--------------|
| **Normality** | Each class follows a Gaussian (normal) distribution |
| **Equal covariance** | All classes have roughly the same spread/shape |
| **Independence** | Features are not perfectly correlated |
| **No multicollinearity** | Features should not be linear combinations of each other |
| **Sufficient samples** | Need more samples than features (n > p) |

> ⚠️ LDA is robust in practice even when assumptions are mildly violated, but large violations hurt performance.

---

## 5. When to Use / When Not to Use

| ✅ Use LDA When | ❌ Avoid LDA When |
|----------------|-----------------|
| You have labeled class data | You have no labels (use PCA instead) |
| Classes are approximately Gaussian | Data has complex non-linear boundaries |
| You want both reduction AND classification | Classes have very different covariance shapes |
| You need a fast, interpretable baseline | Data is highly imbalanced |
| Feature count is manageable | You have more features than samples |

---

## 6. Implementation Overview

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Scale features (important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)   # Note: y (labels) required!

# Explained variance ratio
print(lda.explained_variance_ratio_)
```

Full implementation with Iris dataset is in `lda.ipynb`.

---

## 7. Top 5 Interview Questions

1. **What is the key difference between PCA and LDA?**  
   PCA is unsupervised — maximizes total variance. LDA is supervised — maximizes class separability.

2. **What is the maximum number of LDA components for a 4-class problem?**  
   Maximum = n_classes − 1 = 3 components.

3. **What are the main assumptions of LDA?**  
   Gaussian class distributions, equal covariance matrices across classes, no multicollinearity.

4. **Can LDA be used for classification directly?**  
   Yes — LDA finds a decision boundary and can classify new points directly, not just reduce dimensions.

5. **When would you prefer LDA over PCA for dimensionality reduction?**  
   When class labels are available and separating classes is the goal. LDA often gives better downstream classification accuracy because it uses label information.

---

## 8. Quick Reference Table

| Feature | LDA | PCA |
|---------|-----|-----|
| Type | Supervised | Unsupervised |
| Uses Labels? | ✅ Yes | ❌ No |
| Goal | Maximize class separation | Maximize variance |
| Max Components | n_classes − 1 | n_features |
| Assumes Gaussian? | ✅ Yes | ❌ No |
| Also a Classifier? | ✅ Yes | ❌ No |
| Linear? | ✅ Yes | ✅ Yes |

---

## 9. Connections to Other Techniques

- **PCA**: Unsupervised cousin — use PCA when you have no labels; LDA when you do
- **t-SNE / UMAP**: Non-linear dimensionality reduction for visualization only — not generative
- **QDA (Quadratic DA)**: Relaxes the equal-covariance assumption — more flexible, more parameters
- **Logistic Regression**: Also a linear classifier, but doesn't reduce dimensions
- **SVM**: Non-linear classification option when LDA's linear boundary isn't enough

---

## 10. References

- Fisher, R.A. (1936) — *The Use of Multiple Measurements in Taxonomic Problems* (original LDA paper)
- scikit-learn docs: https://scikit-learn.org/stable/modules/lda_qda.html
- Pattern Recognition and Machine Learning — Bishop, Chapter 4
- The Elements of Statistical Learning — Hastie, Tibshirani, Friedman, Chapter 4
- StatQuest LDA video: https://www.youtube.com/watch?v=azXCzI57Yfc
