 Principal Component Analysis (PCA)

> *"Compress your data's story into its most important chapters — without losing the plot."*

---

## Summary

PCA is an **unsupervised dimensionality reduction technique** that transforms a high-dimensional dataset into a smaller set of uncorrelated variables called **principal components**, while retaining as much original variance (information) as possible. It is widely used for visualization, noise reduction, and as a preprocessing step before model training.

The core insight: most real-world datasets contain **redundant features** — multiple columns that essentially say the same thing in different units. PCA mathematically identifies and merges this redundancy into compact, uncorrelated axes. Two carefully chosen components can often represent 90%+ of a 50-feature dataset without meaningful information loss.

---

## What is PCA?

### The Simple Idea

Imagine you're a photographer trying to capture a 3D sculpture. You walk around it and realize that **one particular angle captures the most detail** — the shape, the shadows, the texture — far better than any other angle. You take that shot. PCA does exactly this with data: it finds the "angle" from which your dataset shows the most variation, then projects everything onto that view.

More precisely, PCA takes possibly correlated features and creates a new coordinate system. The **first principal component** (PC1) points in the direction of greatest variance. The second (PC2) is perpendicular to PC1 and captures the next most variance, and so on. Each new axis is uncorrelated with all the others — which is the whole point.

### A Concrete Analogy

Think of a shadow on a wall. If you hold a 3D object in front of a light source and rotate it, the shadow changes shape dramatically. Some orientations produce a rich, informative silhouette. Others produce a shapeless blob. **PCA finds the orientation that produces the most informative shadow** — and uses that as your new reduced-dimension space. The shadow (2D projection) loses some depth information, but if chosen well, it captures what matters most.

### Why It Matters in Practice

In a typical ML pipeline, you might have 50–500 features. Many of these are correlated — "number of rooms" and "total area" in a housing dataset, for example, or "temperature" and "heat index" in a weather dataset. Training models on correlated features adds noise, slows computation, and can cause overfitting. PCA resolves this by collapsing correlated groups into a single, clean component — reducing dimensionality while preserving structure.

---

## Mathematical Idea

### Step 1 — Mean Centering

Before finding directions of variance, we reframe all data around the origin:

$$X_{\text{centered}} = X - \bar{X}$$

**Intuition:** We're repositioning the coordinate system at the data's center of gravity so PCA focuses on *spread* and *shape*, not *location*.

---

### Step 2 — Covariance Matrix

We measure how each pair of features varies together:

$$\Sigma = \frac{1}{n-1} \, X_{\text{centered}}^T X_{\text{centered}}$$

| Entry | Meaning |
|---|---|
| **Diagonal** (e.g., σ₁₁) | Variance of individual feature |
| **Off-diagonal** (e.g., σ₁₂) | Covariance between two features |
| **High off-diagonal value** | Features are redundant — carry the same signal |

**Intuition:** A high off-diagonal value means two features rise and fall together — they're saying the same thing. PCA rotates the axes to eliminate this cross-talk.

---

### Step 3 — Eigenvalues and Eigenvectors

We decompose the covariance matrix to find its "natural axes":

$$\Sigma \mathbf{v} = \lambda \mathbf{v}$$

- **v (eigenvector):** A direction in the original feature space → becomes a principal component  
- **λ (eigenvalue):** The variance captured along that direction → larger means more important  
- Eigenvectors are sorted by decreasing eigenvalue: PC1 has the largest λ, PC2 the next, and so on

**Intuition:** Eigenvectors are the axes along which your data cloud is most *stretched*. Eigenvalues measure *how much* stretching occurs in each direction.

---

### Step 4 — Data Projection

Select the top *k* eigenvectors and project the centered data:

$$X_{\text{reduced}} = X_{\text{centered}} \cdot V_k$$

| Symbol | Meaning |
|---|---|
| **V_k** | Matrix of top *k* eigenvectors — shape: (d × k) |
| **X_reduced** | New compressed dataset — shape: (n × k) |
| **k** | Number of components you keep |

> **Implementation note:** `sklearn.decomposition.PCA` uses **Singular Value Decomposition (SVD)** internally, not explicit eigendecomposition. SVD is numerically more stable for large, wide matrices and avoids forming the full covariance matrix — important for high-dimensional data.

---

## How It Works — Step by Step

| Step | Action | Why It Matters |
|---|---|---|
| **1. Standardize** | Scale features to mean=0, std=1 | Prevents large-scale features from dominating |
| **2. Covariance Matrix** | Compute Σ from scaled data | Maps out all feature relationships |
| **3. Eigen Decomposition** | Find eigenvectors + eigenvalues of Σ | Reveals the natural axes of the data's shape |
| **4. Rank Components** | Sort by eigenvalue (descending) | Biggest variance first |
| **5. Select k Components** | Keep enough to explain 90–95% variance | Reduce dimension while retaining signal |
| **6. Project Data** | Multiply centered data by V_k | Create the new, smaller dataset |

---

## Key Assumptions

Understanding these assumptions helps you know *when* PCA will work well and *when* it will fail silently.

| Assumption | What It Means | What Breaks It |
|---|---|---|
| **Linearity** | PCA captures linear combinations of features only | Non-linear manifolds (spiral, Swiss roll) — use Kernel PCA |
| **Variance = Signal** | High-variance directions carry meaningful information | Outliers can inflate variance in meaningless directions |
| **Feature Scaling** | All features must be on the same scale | Forgetting `StandardScaler` — the most common mistake |
| **Continuous Features** | PCA requires numerical, continuous input | Categorical features must be encoded first (risky) |

---

## Common Mistakes

These are the most frequent errors made when applying PCA — even by experienced practitioners.

**1. Skipping StandardScaler**  
This is the single most common mistake. If `sepal_length` ranges 0–8 cm and some other feature ranges 0–1000, PCA will orient PC1 almost entirely along that large-scale feature. The result looks like PCA "worked" but is actually just reflecting measurement scale, not data structure. **Always scale first.**

**2. Choosing n_components Arbitrarily**  
Picking `n_components=2` just for visualization convenience without checking `explained_variance_ratio_` is a mistake. If those 2 components only explain 55% of variance, you've discarded nearly half the information. Use the scree plot to pick a principled threshold (typically 90–95%).

**3. Applying PCA Before Train/Test Split**  
Fitting the scaler *and* PCA on the full dataset (including test data) causes **data leakage**. The correct order: split first → fit scaler on train → transform both → fit PCA on train → transform both.

**4. Treating PCA Components as Interpretable Features**  
PC1, PC2, etc. are linear mixtures of *all* original features. They don't have clean names like "height" or "income." If a stakeholder needs to understand *what* drove a prediction, PCA is the wrong tool — use feature selection instead.

**5. Using PCA for Categorical or Sparse Data**  
PCA assumes continuous numerical data and dense matrices. For text data (high-dimensional, sparse), use **TruncatedSVD (LSA)**. For mixed data types, use **MCA** or **FactorAnalysis**.

---

## Real-World Applications

PCA is not just a classroom tool — it is used in production across many industries.

| Domain | How PCA Is Used |
|---|---|
| **Computer Vision** | Eigenfaces: compress face images from 10,000 pixels to ~100 components for face recognition |
| **Genomics** | Reduce gene expression matrices (20,000+ genes) to explore population structure and ancestry |
| **Finance** | Identify risk factors in asset return matrices; compress correlated stock movements |
| **NLP** | LSA (Latent Semantic Analysis) applies PCA/SVD to document-term matrices for topic modeling |
| **Sensor Data / IoT** | Compress multi-channel time series from industrial sensors for anomaly detection |
| **Medical Imaging** | Reduce fMRI, MRI, or EEG signal dimensions while preserving diagnostic structure |
| **Recommendation Systems** | Matrix factorization (which uses SVD, PCA's mathematical sibling) for collaborative filtering |

---

## When to Use PCA

- You have **many correlated features** (multicollinearity is a known issue)
- You want to **visualize** high-dimensional data in 2D or 3D
- You need to **speed up** downstream model training
- You want to **reduce noise** by discarding very-low-variance components
- You are working on **image compression** or signal processing pipelines
- You need to **remove the curse of dimensionality** before applying distance-based models (KNN, clustering)

## When NOT to Use PCA

- **Interpretability is required** — components are mixtures, not named features
- **Data is non-linear** in structure — use Kernel PCA, t-SNE, or UMAP
- **Data is categorical** — PCA requires continuous numerical input
- **You need class-aware separation** — PCA ignores labels; use LDA instead
- **Very few features** (< 5–10) — PCA adds complexity without meaningful benefit
- **You haven't split train/test yet** — always split before fitting any preprocessor

---

## PCA vs. Other Dimensionality Reduction Methods

| Method | Type | Captures | Best For | Limitation |
|---|---|---|---|---|
| **PCA** | Linear, Unsupervised | Global variance structure | Preprocessing, visualization, noise reduction | Linear only; components uninterpretable |
| **LDA** | Linear, Supervised | Class-discriminative directions | Classification preprocessing | Requires labels; max k = classes−1 |
| **Kernel PCA** | Non-linear, Unsupervised | Non-linear manifolds | Curved or complex data distributions | Computationally expensive; kernel choice matters |
| **t-SNE** | Non-linear, Unsupervised | Local neighborhood structure | 2D/3D visualization only | Non-deterministic; not for preprocessing |
| **UMAP** | Non-linear, Unsupervised | Local + global structure | Fast visualization of large datasets | Less theoretically grounded than t-SNE |
| **Autoencoders** | Non-linear, Unsupervised | Arbitrary complex patterns | High-dimensional image/text embeddings | Requires neural network training |

**Rule of thumb:**  
- Start with PCA. If variance structure is clear → use it.  
- If classes need separating → use LDA.  
- If structure is non-linear → use UMAP for exploration, Kernel PCA for preprocessing.

---

## Limitations of PCA

| Limitation | Explanation |
|---|---|
| **Linear only** | Cannot capture curves, clusters, or complex manifold geometry |
| **Information loss** | Dropping low-variance components always loses *some* data — this may matter for anomaly detection |
| **Not robust to outliers** | Outliers inflate variance and can pull principal components in misleading directions |
| **Components are uninterpretable** | PC1 = 0.52×f1 + 0.37×f2 + ... is not meaningful to a business stakeholder |
| **Assumes stationarity** | On time series, PCA ignores temporal ordering and autocorrelation |
| **Sensitive to scaling** | One unscaled feature ruins the entire decomposition |

---

## Implementation Overview

### Correct Order of Operations

```
Raw Data
  → Train/Test Split
    → StandardScaler (fit on train, transform both)
      → PCA (fit on train, transform both)
        → Model Training
```

### Using sklearn

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # ← transform only, not fit!

# Step 2: Apply PCA
pca = PCA(n_components=0.95)  # Keep enough components for 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)   # ← transform only, not fit!
```

**Key sklearn parameters:**

| Parameter | Description |
|---|---|
| `n_components=k` | Keep exactly *k* components |
| `n_components=0.95` | Auto-select components to retain 95% variance |
| `n_components='mle'` | Use MLE heuristic to estimate optimal k |
| `svd_solver='full'` | Standard SVD (default for small data) |
| `svd_solver='randomized'` | Faster approximation for large matrices |

**Reading `explained_variance_ratio_`:**
```python
pca.explained_variance_ratio_         # Per-component fractions
pca.explained_variance_ratio_.cumsum() # Cumulative — use this for the elbow
```

---

## Top Interview Questions

| # | Question | Strong Answer Hint |
|---|---|---|
| 1 | **What does PCA do geometrically?** | Rotates the coordinate system so axes align with directions of maximum variance; then truncates to keep only the most informative axes |
| 2 | **Why scale before PCA?** | PCA is variance-based — unscaled features with large ranges dominate eigenvectors, making components reflect units, not data structure |
| 3 | **How do you choose n_components?** | Plot the cumulative scree curve; pick the elbow — typically where 90–95% cumulative variance is reached |
| 4 | **PCA vs LDA — when would you choose each?** | PCA = unsupervised (maximize variance); LDA = supervised (maximize class separation). Use LDA when labels are available and class boundaries matter |
| 5 | **What is data leakage in the context of PCA?** | Fitting the scaler or PCA on the full dataset (including test set) allows the model to "see" test data during training. Always fit on train only |

---

## Quick Reference

| Item | Detail |
|---|---|
| **Type** | Unsupervised Dimensionality Reduction |
| **Output** | Principal Components — orthogonal, uncorrelated linear combinations of features |
| **Use Cases** | Visualization, noise reduction, multicollinearity removal, preprocessing |
| **Strengths** | Fast, well-understood, reliable, variance interpretation via scree plot |
| **Limitations** | Linear only; uninterpretable components; sensitive to outliers and scale |
| **sklearn Class** | `sklearn.decomposition.PCA` |
| **Key Parameter** | `n_components` |
| **Must-Do Before** | `StandardScaler` — always |
| **Related Methods** | LDA, Kernel PCA, t-SNE, UMAP, TruncatedSVD |

---

## References

1. **Scikit-learn Documentation — PCA**  
   https://scikit-learn.org/stable/modules/decomposition.html#pca

2. **StatQuest with Josh Starmer — PCA Step-by-Step (YouTube)**  
   https://www.youtube.com/watch?v=FgakZw6K1QQ  
   *(Best visual intuition available — highly recommended for all levels)*

3. **Pattern Recognition and Machine Learning — Bishop (2006)**  
   Chapter 12: Continuous Latent Variables — the mathematical backbone

4. **The Elements of Statistical Learning — Hastie, Tibshirani, Friedman**  
   Chapter 14.5: Principal Components and Derived Input Directions

5. **Towards Data Science — A One-Stop Shop for PCA**  
   https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c

---

