# t-SNE: t-Distributed Stochastic Neighbor Embedding

> *"Making the invisible visible — t-SNE reveals hidden structure in high-dimensional data through the lens of probability."*

A complete, beginner-friendly guide to understanding, implementing, and interpreting t-SNE for dimensionality reduction and data visualization. Designed for ML interns, students, and practitioners looking to build real intuition — not just run a function.

---

## 📌 Table of Contents

- [What is t-SNE?](#what-is-t-sne)
- [Mathematical Formulation](#mathematical-formulation)
- [How It Works (Step-by-Step)](#how-it-works-step-by-step)
- [Key Assumptions](#key-assumptions)
- [When to Use / When NOT to Use](#when-to-use--when-not-to-use)
- [Implementation Overview](#implementation-overview)
- [Top 5 Interview Questions](#top-5-interview-questions)
- [Quick Reference Table](#quick-reference-table)
- [References](#references)

---

## What is t-SNE?

Imagine you have a city map — but it exists in 500 dimensions. t-SNE is like a master cartographer who takes that incomprehensible 500-dimensional map and folds it down into a 2D poster you can actually read, while trying to keep **neighbors close together** and **strangers far apart**.

More precisely, t-SNE (pronounced "tee-snee") is a **non-linear dimensionality reduction algorithm** developed by Laurens van der Maaten and Geoffrey Hinton in 2008. Its primary purpose is **visualization** — it transforms data with hundreds or thousands of features into 2 or 3 dimensions while preserving the local structure (clusters and neighborhoods) of the original data.

What sets t-SNE apart from linear techniques like PCA is its ability to capture **complex, non-linear relationships**. PCA preserves global variance; t-SNE preserves local similarity. Think of it this way: PCA would give you a satellite photo of a city, while t-SNE gives you a hand-drawn map that highlights neighborhoods and communities — the structure that actually matters for many tasks.

The "t" in t-SNE refers to the **Student's t-distribution**, which is used in the low-dimensional space to avoid the "crowding problem" — a phenomenon where all points collapse into the center when projecting from high to low dimensions. By using a heavy-tailed distribution (the t-distribution), t-SNE spreads points out more naturally, creating visually clear and interpretable clusters.

---

## Mathematical Formulation

t-SNE works by converting distances into **probabilities** in both the original high-dimensional space and the new low-dimensional space, then minimizing the difference between these two probability distributions.

### High-Dimensional Space: Gaussian Similarity

$$p_{j|i} = \frac{\exp\left(-\|x_i - x_j\|^2 / 2\sigma_i^2\right)}{\sum_{k \neq i} \exp\left(-\|x_i - x_k\|^2 / 2\sigma_i^2\right)}$$

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$

| Symbol | Meaning |
|--------|---------|
| $x_i, x_j$ | Data points in the original high-dimensional space |
| $\sigma_i$ | Bandwidth of the Gaussian kernel centered on point $i$ (determined by perplexity) |
| $p_{j\|i}$ | Conditional probability that $x_i$ would pick $x_j$ as a neighbor |
| $p_{ij}$ | Symmetrized joint probability (used for stability) |
| $N$ | Total number of data points |

**Intuition:** A high $p_{ij}$ means points $i$ and $j$ are close in the original space and should also be close in the 2D map.

---

### Low-Dimensional Space: Student's t-Distribution

$$q_{ij} = \frac{\left(1 + \|y_i - y_j\|^2\right)^{-1}}{\sum_{k \neq l} \left(1 + \|y_k - y_l\|^2\right)^{-1}}$$

| Symbol | Meaning |
|--------|---------|
| $y_i, y_j$ | Corresponding points in the low-dimensional (2D/3D) map |
| $q_{ij}$ | Similarity between points in the low-dimensional space |

**Why t-distribution?** It has heavier tails than Gaussian. This means dissimilar points get pushed far apart in 2D space, resolving the crowding problem.

---

### Cost Function: Kullback-Leibler Divergence

$$C = KL(P \| Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

| Symbol | Meaning |
|--------|---------|
| $C$ | Cost to minimize (lower = better 2D map) |
| $P$ | Probability distribution in the original space |
| $Q$ | Probability distribution in the 2D map |
| $KL(P\|Q)$ | KL divergence — measures how different $Q$ is from $P$ |

**The goal:** Find positions $y_i$ in 2D space such that $Q$ matches $P$ as closely as possible. This is done via **gradient descent**.

---

## How It Works (Step-by-Step)

```
High-Dimensional Data
       │
       ▼
┌─────────────────────────────────────────┐
│ STEP 1: Compute Pairwise Distances      │
│  Calculate Euclidean distance between   │
│  every pair of points in the original   │
│  high-dimensional space.                │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ STEP 2: Convert to Probabilities (P)    │
│  Use Gaussian kernel to convert         │
│  distances to similarity probabilities. │
│  σ_i is tuned per point using perplexity│
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ STEP 3: Initialize Low-Dim Map          │
│  Randomly place points in 2D/3D space.  │
│  (Often initialized with PCA for speed) │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ STEP 4: Compute Q Probabilities         │
│  Use t-distribution to compute pairwise │
│  similarities in the low-dim space.     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ STEP 5: Minimize KL Divergence          │
│  Use gradient descent to update 2D      │
│  positions until P ≈ Q.                 │
│  (Runs for N iterations)                │
└─────────────────┬───────────────────────┘
                  │
                  ▼
       2D Visualization Output
```

**Key insight at each step:**
- **Step 2** is where perplexity matters — it controls how wide/narrow the Gaussian is, i.e., how many effective neighbors each point considers.
- **Step 5** uses gradient descent with momentum. The algorithm prioritizes keeping similar points close (low $p_{ij}$, high $q_{ij}$ is very costly).

---

## Key Assumptions

Understanding these assumptions prevents misinterpretation of t-SNE plots:

1. **Local structure is more important than global structure.**  
   t-SNE is designed to preserve neighborhoods, not global geometry. The distances *between* clusters in a t-SNE plot are **not meaningful**. Don't conclude that two clusters are more related just because they're closer together on the 2D map.

2. **Cluster sizes in t-SNE plots don't reflect real densities.**  
   t-SNE tends to expand dense clusters and compress sparse ones for visual clarity. A smaller cluster on the plot is not necessarily a smaller cluster in reality.

3. **The result is non-deterministic.**  
   Different random seeds produce different plots. Always set `random_state` for reproducibility.

4. **Perplexity must be smaller than the number of data points.**  
   Typical values: 5–50. Using a perplexity too close to `n_samples` causes issues.

5. **t-SNE is for visualization, not feature engineering.**  
   The 2D output components are not interpretable features. Don't feed t-SNE output directly into a downstream ML model.

---

## When to Use / When NOT to Use

### ✅ Use t-SNE When:

| Scenario | Why t-SNE Helps |
|----------|-----------------|
| Exploring clusters in high-dimensional data | Reveals natural groupings visually |
| Visualizing word embeddings or image features | Shows semantic similarity between items |
| Sanity-checking your learned representations | Helps debug neural network embeddings |
| EDA on datasets with 10–1000+ features | Provides 2D summary of data structure |
| Comparing pre/post preprocessing effects | Visual diff of data distributions |

### ❌ Do NOT Use t-SNE When:

| Scenario | Why It Fails |
|----------|-------------|
| You need to transform new/unseen data | t-SNE has no `transform()` — it must refit from scratch |
| You want interpretable axes | t-SNE axes have no meaning |
| You're doing feature reduction before ML | Distances aren't preserved reliably; use PCA instead |
| Your dataset has > 100K points | Will be very slow; use UMAP or PCA first |
| You need consistent/deterministic results | Non-deterministic by nature |
| Global structure matters | t-SNE sacrifices global geometry for local |

---

## Implementation Overview

### Conceptual Difference: From Scratch vs Scikit-learn

#### From Scratch (What actually happens):
```python
# Pseudocode — conceptual walkthrough
def tsne_from_scratch(X, perplexity=30, n_iter=1000, lr=200):
    # Step 1: Compute pairwise distances
    D = pairwise_euclidean_distances(X)
    
    # Step 2: Compute P (high-dim probabilities)
    # Binary search for sigma_i such that H(P_i) = log(perplexity)
    P = compute_joint_probabilities(D, perplexity)
    
    # Step 3: Random initialization in 2D
    Y = random_init(n_samples=X.shape[0], n_dims=2)
    
    # Step 4-5: Gradient descent loop
    for iteration in range(n_iter):
        Q = compute_q_distribution(Y)         # t-dist in 2D
        grad = compute_kl_gradient(P, Q, Y)   # dC/dY
        Y = Y - lr * grad                      # Update positions
    
    return Y  # Final 2D coordinates
```

**Complexity:** O(N²) time and space — expensive for large N.

#### With Scikit-learn (Production approach):
```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,     # 2D output
    perplexity=30,      # Effective number of neighbors
    learning_rate=200,  # Step size for gradient descent
    n_iter=1000,        # Number of optimization steps
    random_state=42     # For reproducibility
)

X_embedded = tsne.fit_transform(X)  # Shape: (n_samples, 2)
```

Scikit-learn's implementation uses the **Barnes-Hut approximation** (O(N log N)) for large datasets, which is dramatically faster than the naive O(N²) implementation.

#### Key Hyperparameters Explained:

| Hyperparameter | Default | Effect |
|----------------|---------|--------|
| `perplexity` | 30 | Controls number of effective neighbors. Low = local focus. High = global focus. Try: 5–50 |
| `learning_rate` | 200 | Gradient descent step size. Too high = noisy; too low = slow convergence. Try: 10–1000 |
| `n_iter` | 1000 | Total optimization steps. More = better results, slower runtime. Min: 250 |
| `n_components` | 2 | Output dimensions. Usually 2 for visualization, rarely 3 |
| `early_exaggeration` | 12 | Multiplies P values early in optimization to form tight clusters |
| `init` | 'pca' | Initialization method. PCA init is more stable than random |

---

## Top 5 Interview Questions

### Q1: What problem does t-SNE solve, and how is it different from PCA?

**Answer:**  
t-SNE solves the problem of **visualizing high-dimensional data** by reducing it to 2D or 3D while preserving **local neighborhood structure**. PCA is a linear technique that maximizes global variance and finds orthogonal components. t-SNE is non-linear and focuses on keeping similar points close in the low-dimensional space. PCA is deterministic, interpretable, and invertible — ideal for preprocessing. t-SNE is stochastic, non-invertible, and for visualization only. Use PCA when you want to reduce dimensions before training a model; use t-SNE when you want to visually explore structure.

---

### Q2: What is perplexity in t-SNE, and how do you choose it?

**Answer:**  
Perplexity is a measure of the **effective number of neighbors** each point considers. Mathematically, it controls the bandwidth $\sigma_i$ of the Gaussian kernel: a higher perplexity leads to wider kernels (more neighbors considered). Practically, perplexity = 30 means each point is compared against roughly 30 neighbors. Typical values range from 5 to 50. A rule of thumb: perplexity should be smaller than the number of data points and roughly `sqrt(N)` for medium datasets. Always run t-SNE with multiple perplexity values and compare visualizations — structure that appears in all runs is likely real.

---

### Q3: Why does t-SNE use a t-distribution in the low-dimensional space instead of a Gaussian?

**Answer:**  
This is the key innovation of t-SNE over SNE. When projecting from high to low dimensions, there's a **crowding problem**: the area of a 2D ball grows as $r^2$ while a high-dimensional ball grows as $r^d$ — meaning there's not enough space in 2D to faithfully represent all moderate-distance neighbors. Using a **Student's t-distribution** (with 1 degree of freedom, equivalent to a Cauchy distribution) in the low-dimensional space gives the distribution **heavy tails**. This allows moderately dissimilar points to be placed much farther apart in 2D than they would be with a Gaussian, naturally resolving the crowding problem and creating cleaner, more separated clusters.

---

### Q4: Can you use t-SNE output as features for a downstream ML model? Why or why not?

**Answer:**  
Generally **no**, and this is a common mistake. There are several reasons:
1. **No transform method**: t-SNE cannot embed new data points without refitting. Scikit-learn's `TSNE` has no `.transform()` — only `.fit_transform()`.
2. **Non-deterministic**: Different runs give different embeddings, making it impossible to consistently embed train and test sets in the same space.
3. **Distances not preserved globally**: The 2D coordinates don't faithfully encode feature relationships useful for prediction.
4. **Overfitting risk**: t-SNE can create spurious clusters, misleading a downstream model.
If you need to reduce dimensions before ML, use PCA, UMAP (which supports `transform()`), or autoencoders.

---

### Q5: What are common pitfalls when interpreting t-SNE plots?

**Answer:**  
Three major pitfalls, frequently asked at FAANG:
1. **Cluster sizes are meaningless**: t-SNE expands dense clusters and compresses sparse ones for visual clarity. A visually larger cluster is not necessarily a larger cluster in reality.
2. **Inter-cluster distances are meaningless**: The distance between cluster A and cluster B in the 2D plot tells you nothing about their actual dissimilarity in the original space. t-SNE only reliably preserves *local* (within-cluster) structure.
3. **Noise can look like clusters**: With the wrong perplexity or insufficient iterations, random noise can appear as structured clusters. Always validate t-SNE findings with quantitative methods (silhouette score, cluster purity) and try multiple random seeds.

---

## Quick Reference Table

| Property | Value |
|----------|-------|
| **Type** | Non-linear, unsupervised dimensionality reduction |
| **Primary Use** | Data visualization (2D/3D) |
| **Time Complexity** | O(N²) naive; O(N log N) with Barnes-Hut approximation |
| **Space Complexity** | O(N²) |
| **Output** | Low-dimensional coordinates (no feature interpretation) |
| **Invertible?** | No |
| **Handles New Data?** | No (must refit) |
| **Deterministic?** | No (set `random_state` for reproducibility) |
| **Key Hyperparameters** | `perplexity` (5–50), `learning_rate` (10–1000), `n_iter` (≥250) |
| **Ideal Dataset Size** | < 100K samples (use UMAP for larger) |
| **Comparison** | Better than PCA for non-linear structure; slower than UMAP |
| **Paper** | van der Maaten & Hinton, JMLR 2008 |

### t-SNE vs PCA vs UMAP

| Feature | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| Linear? | Yes | No | No |
| Preserves | Global variance | Local structure | Local + some global |
| Speed | Very fast | Slow | Fast |
| New data? | Yes | No | Yes |
| Deterministic? | Yes | No | No (but more stable) |
| Best for | Preprocessing, noise removal | Visualization | Visualization + downstream use |

---

## References

1. **Original Paper**: van der Maaten, L. & Hinton, G. (2008). *Visualizing Data using t-SNE*. Journal of Machine Learning Research, 9, 2579–2605. [Link](https://jmlr.org/papers/v9/vandermaaten08a.html)

2. **Distill.pub Interactive Guide**: Wattenberg, M., Viégas, F., & Johnson, I. (2016). *How to Use t-SNE Effectively*. Distill. [Link](https://distill.pub/2016/misread-tsne/) — **Must read before interpreting any t-SNE plot.**

3. **Scikit-learn Documentation**: *t-SNE User Guide and API Reference*. [Link](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

4. **StatQuest with Josh Starmer**: *t-SNE, Clearly Explained!!!* [YouTube](https://www.youtube.com/watch?v=NEaUSP4YerM) — Best video explanation for beginners.

5. **Comparison with UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. [arXiv](https://arxiv.org/abs/1802.03426)

---


