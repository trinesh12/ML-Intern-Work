# 📊 Principal Component Analysis (PCA)

## 🔹 Tagline

Find the directions where your data "spreads out" the most — and keep only those.

---

## 🔹 Summary

PCA is an unsupervised, linear dimensionality reduction technique that transforms high-dimensional data into a smaller set of uncorrelated variables called **principal components**, each capturing maximum variance.

It works by rotating the feature space to align with directions of highest information, helping in faster computation, removing redundancy, and visualizing complex datasets.

---

## 📌 What is PCA?

Imagine you're photographing a **3D sculpture**, but you can only take one picture. You walk around and choose the angle that shows the most detail. That best angle is your **first principal component**. The second-best angle (perpendicular to the first) is your **second component**.

In real-world data, many features are **correlated**. For example, height and shoe size often increase together. Keeping both adds little new information.

👉 PCA solves this by:

* Combining features
* Removing redundancy
* Creating new features (principal components)

These components:

* Are **uncorrelated**
* Are ordered by importance (variance)

👉 Think of it like compressing a **10-page report into 3 key points**.

---

## 📐 Mathematical Idea

### 1. Mean Centering

[
X_{centered} = X - \mu
]

* (X): original data
* (\mu): mean of each feature

👉 Shifts data so mean = 0 (focus on variation)

---

### 2. Covariance Matrix

[
\Sigma = \frac{1}{n-1} X_{centered}^T X_{centered}
]

* Measures how features move together
* Diagonal = variance
* Off-diagonal = correlation

---

### 3. Eigenvalues & Eigenvectors

[
\Sigma v = \lambda v
]

* (v): direction (principal component)
* (\lambda): importance (variance)

👉 Eigenvectors = directions
👉 Eigenvalues = importance

---

## ⚙️ How It Works

### Step 1 — Standardize Data

Scale features (mean = 0, std = 1)

---

### Step 2 — Compute Covariance Matrix

Find relationships between features

---

### Step 3 — Find Eigenvalues & Eigenvectors

Identify important directions

---

### Step 4 — Select Top Components

Choose top **k components** based on variance

---

### Step 5 — Transform Data

[
Z = X_{centered} \cdot W
]

* (W): selected eigenvectors
* (Z): reduced dataset

---

## 📌 Key Assumptions

| Assumption                | Why It Matters                            |
| ------------------------- | ----------------------------------------- |
| Linearity                 | Only captures linear relationships        |
| High variance = important | Assumes useful info lies in high variance |
| Features scaled           | Prevents dominance of large values        |

---

## ✅ When to Use

* High-dimensional data
* Correlated features
* Data visualization (2D/3D)
* Speeding up models
* Noise reduction

---

## ❌ When NOT to Use

* When interpretability is required
* Non-linear patterns exist
* Very small datasets
* When labels matter (use LDA instead)

---

## ⚙️ Implementation Overview

* Use **StandardScaler** before PCA
* Apply PCA using sklearn
* Use `explained_variance_ratio_` to measure importance
* Use `n_components` to control dimensions

👉 Variants:

* Incremental PCA (large data)
* Kernel PCA (non-linear data)

---

## 🎯 Top Interview Questions

| Question               | Hint                                           |
| ---------------------- | ---------------------------------------------- |
| What is PCA?           | Linear dimensionality reduction                |
| Why scaling is needed? | Prevents feature dominance                     |
| What are eigenvalues?  | Variance captured                              |
| PCA vs t-SNE?          | PCA = linear, fast; t-SNE = non-linear         |
| Limitations of PCA?    | Not interpretable, ignores non-linear patterns |

---

## 📊 Quick Reference Table

| Item       | Detail                     |
| ---------- | -------------------------- |
| Type       | Unsupervised, Linear       |
| Use Case   | Dimensionality Reduction   |
| Input      | Numerical, scaled features |
| Output     | Uncorrelated components    |
| Strength   | Removes redundancy         |
| Limitation | Hard to interpret          |
| Library    | sklearn.decomposition.PCA  |

---

## 📚 References

* Scikit-learn Documentation
* StatQuest PCA (YouTube)
* Towards Data Science Blogs
* Kaggle Datasets
* Jolliffe (2002) – PCA Book

---
