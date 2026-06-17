# UMAP: Uniform Manifold Approximation and Projection

> *"UMAP finds the shape of your data — not just its spread — and faithfully redraws it in 2D."*

A complete, beginner-to-advanced guide to understanding, implementing, and interpreting UMAP for dimensionality reduction and data visualization. Built for ML interns, students, and practitioners who want real intuition, not just a `fit_transform()` call.

---

## 📌 Table of Contents

- [What is UMAP?](#what-is-umap)
- [Mathematical Formulation](#mathematical-formulation)
- [How It Works (Step-by-Step)](#how-it-works-step-by-step)
- [Key Assumptions](#key-assumptions)
- [When to Use / When NOT to Use](#when-to-use--when-not-to-use)
- [Implementation Overview](#implementation-overview)
- [Top 5 Interview Questions](#top-5-interview-questions)
- [Quick Reference Table](#quick-reference-table)
- [References](#references)

---

## What is UMAP?

Imagine you have a crumpled piece of paper in 3D space. The paper itself is 2D — it has a flat, intrinsic structure — but because it's crumpled, it exists in a 3D world. UMAP is the algorithm that *uncrumples* that paper: it finds the true underlying low-dimensional shape hiding inside a high-dimensional mess. Formally, this underlying shape is called a **manifold**, and UMAP's core insight is that real-world high-dimensional data (images, text embeddings, gene expression) almost always lives on a low-dimensional manifold embedded in a high-dimensional space.

UMAP (Uniform Manifold Approximation and Projection) was introduced by Leland McInnes, John Healy, and James Melville in 2018. It is grounded in **algebraic topology** and **Riemannian geometry**, making it theoretically the most principled dimensionality reduction algorithm commonly used in practice. But you don't need to know topology to use it — UMAP can be summarized as: *"Build a weighted graph capturing who is close to whom in the original space, then find a low-dimensional layout that preserves that graph as faithfully as possible."*

What sets UMAP apart from t-SNE is its speed, scalability, and ability to **preserve both local and global structure**. While t-SNE collapses everything into tight local clusters and destroys global geometry, UMAP maintains a meaningful sense of how clusters relate to each other. UMAP also supports `transform()`, meaning it can embed **new, unseen data points** without refitting — something t-SNE cannot do. These properties make UMAP suitable not just for visualization but also as a preprocessing step before clustering or supervised learning.

---

## 📌 Key Insight
UMAP preserves both local and some global structure better than t-SNE and is much faster, making it suitable for larger datasets.

---
## Mathematical Formulation

UMAP operates in two phases: building a **high-dimensional fuzzy graph**, then finding a **low-dimensional layout** that matches it.

### Phase 1 — High-Dimensional Fuzzy Graph

**Directed edge weight (fuzzy membership):**

$$v_{ij} = \exp\!\left(\frac{-(d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$$

| Symbol | Meaning |
|--------|---------|
| $d(x_i, x_j)$ | Distance between point $i$ and neighbor $j$ in original space |
| $\rho_i$ | Distance to the nearest neighbor of $x_i$ — ensures local connectivity |
| $\sigma_i$ | Per-point bandwidth, calibrated so $\sum_j v_{ij} = \log_2(k)$ |
| $v_{ij}$ | Directed fuzzy probability that $j$ is a neighbor of $i$ |

**Symmetrized edge weight (fuzzy union):**

$$w_{ij} = v_{ij} + v_{ji} - v_{ij} \cdot v_{ji}$$

| Symbol | Meaning |
|--------|---------|
| $w_{ij}$ | Undirected edge weight — probability that at least one direction holds |

**Intuition:** Subtracting $\rho_i$ ensures every point has at least one neighbor at "distance 0," guaranteeing the graph is connected even in sparse regions.

---

### Phase 2 — Low-Dimensional Embedding

**Low-dimensional similarity:**

$$q_{ij} = \left(1 + a \cdot \|y_i - y_j\|^{2b}\right)^{-1}$$

| Symbol | Meaning |
|--------|---------|
| $y_i, y_j$ | Coordinates in the low-dimensional embedding |
| $a, b$ | Shape parameters automatically fit from `min_dist` |
| $q_{ij}$ | Fuzzy similarity in the low-dimensional space |

**Cross-entropy loss (optimization objective):**

$$C = \sum_{(i,j)} \left[ w_{ij} \log\frac{w_{ij}}{q_{ij}} + (1 - w_{ij}) \log\frac{1 - w_{ij}}{1 - q_{ij}} \right]$$

| Symbol | Meaning |
|--------|---------|
| $C$ | Total cross-entropy loss to minimize |
| $w_{ij}$ | Target edge weight from the high-dimensional graph |
| $q_{ij}$ | Predicted edge weight in the low-dimensional embedding |

**Key insight:** Binary cross-entropy penalizes BOTH putting dissimilar points together AND putting similar points apart. t-SNE only penalizes the first case — this is why UMAP preserves global structure better.

---

## How It Works (Step-by-Step)

```
High-Dimensional Data (N samples × D features)
              │
              ▼
┌─────────────────────────────────────────────────┐
│  STEP 1: Find k Nearest Neighbors               │
│  For each of N points, find k closest           │
│  neighbors using approximate NN search          │
│  (PyNNDescent) — O(N^1.14), far faster          │
│  than the naive O(N²) brute-force search.       │
│  Controlled by: n_neighbors                     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 2: Build Fuzzy Simplicial Set             │
│  For each point, compute local bandwidth σᵢ     │
│  via binary search. Convert distances to        │
│  fuzzy weights v_ij. Symmetrize to get          │
│  an undirected weighted graph W.                │
│  This graph encodes the manifold topology.      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 3: Spectral Initialization                │
│  Initialize embedding using Laplacian           │
│  Eigenmaps on the graph W.                      │
│  Advantage over t-SNE's random init:            │
│  global structure is coherent from start.       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 4: Optimize via SGD                       │
│  Minimize cross-entropy C over n_epochs.        │
│  Attractive forces: pull true neighbors close.  │
│  Repulsive forces: push non-neighbors apart.    │
│  Controlled by: min_dist, n_epochs, lr          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
       Low-Dimensional Embedding (N × 2)
       (Can also transform new data!)
```

---

## Key Assumptions

1. **Data lies on a low-dimensional manifold.** If your data is pure noise with no intrinsic low-dimensional structure, UMAP output is arbitrary.

2. **The manifold is locally connected.** The $\rho_i$ correction guarantees each point connects to at least one neighbor, but in very sparse datasets this can create artificial bridges between disconnected clusters.

3. **n_neighbors defines the manifold resolution.** Small values reveal fine local structure (possibly noisy). Large values give a smoother, more global manifold. Neither is correct for all situations.

4. **min_dist affects density, not topology.** It controls how tightly points are packed visually — an aesthetic choice, not a structural one.

5. **Global structure is better than t-SNE, but imperfect.** Inter-cluster distances in UMAP are more meaningful than in t-SNE, but still shouldn't be over-interpreted without quantitative validation.

6. **Metric choice matters.** Euclidean is the default but not always ideal. For text embeddings, use `metric='cosine'`. For count data (RNA-seq), consider `metric='correlation'`.

---

## When to Use / When NOT to Use

### ✅ Use UMAP When:

| Scenario | Why UMAP Excels |
|----------|-----------------|
| Large datasets (>10K samples) | 10–100× faster than t-SNE |
| Need to embed new/unseen data | `transform()` works without refitting |
| Both local and global structure matter | Better global preservation than t-SNE |
| Preprocessing before clustering | UMAP + HDBSCAN is a leading pipeline |
| Single-cell RNA-seq or genomics | Industry standard for bioinformatics EDA |
| 3D visualization | Works cleanly with `n_components=3` |
| Supervised structure (labels available) | Supports `y` labels via `target_metric` |

### ❌ Do NOT Use UMAP When:

| Scenario | Why It Fails |
|----------|-------------|
| You need interpretable axes | UMAP axes have no unit or meaning |
| Tiny datasets (<50 points) | k-NN graph is too sparse to be reliable |
| You need perfectly reproducible results | Stochastic — always set `random_state` |
| You need a linear transformation | Use PCA (faster, invertible, interpretable) |
| You need to explain components to stakeholders | Axes are uninterpretable without labels |

---

## Implementation Overview

### From Scratch (Conceptual):
```python
# Phase 1: Build the high-dimensional fuzzy graph
def build_fuzzy_graph(X, n_neighbors):
    knn_indices, knn_dists = approx_nearest_neighbors(X, n_neighbors)
    rho   = [dists[0] for dists in knn_dists]          # Nearest neighbor dist
    sigma = [binary_search_sigma(dists, rho[i])
             for i, dists in enumerate(knn_dists)]       # Local bandwidth
    V = compute_directed_weights(knn_indices, knn_dists, rho, sigma)
    W = V + V.T - V * V.T                               # Fuzzy union (symmetrize)
    return W

# Phase 2: Optimize low-dimensional layout
def optimize_embedding(W, n_components, min_dist, n_epochs):
    a, b = fit_ab_from_min_dist(min_dist)        # Shape params for q_ij
    Y    = spectral_init(W, n_components)         # Globally coherent start
    for epoch in range(n_epochs):
        for each edge (i,j):
            Y = sgd_step(Y, i, j, W[i,j], a, b)  # Attract + repel
    return Y
```

### With umap-learn (Production):
```python
import umap

reducer = umap.UMAP(
    n_neighbors=15,      # Local neighborhood size
    min_dist=0.1,        # Packing density in embedding
    n_components=2,      # Output dimensions
    metric='euclidean',  # Distance in input space
    random_state=42      # Always set for reproducibility
)
X_embedded  = reducer.fit_transform(X_train)
X_new_embed = reducer.transform(X_test)  # t-SNE cannot do this!
```

### Hyperparameter Guide:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `n_neighbors` | 15 | 5–200 | Manifold resolution: low = local, high = global |
| `min_dist` | 0.1 | 0.0–0.99 | Visual density: low = tight clusters, high = spread |
| `n_components` | 2 | 2–50 | Output dimensions |
| `metric` | 'euclidean' | any sklearn metric | Input space distance |
| `n_epochs` | auto | 200–500 | Optimization quality vs speed |
| `init` | 'spectral' | 'spectral','random','pca' | Initialization strategy |

---

## Top 5 Interview Questions

### Q1: How does UMAP differ from t-SNE? When would you choose one?

**Answer:**
Four key differences: (1) **Speed** — UMAP is 10–100× faster; (2) **Global structure** — UMAP preserves inter-cluster relationships better via symmetric cross-entropy loss; (3) **New data** — UMAP has `transform()`, t-SNE requires refitting; (4) **Theory** — UMAP is grounded in Riemannian geometry, t-SNE in KL divergence minimization. Use t-SNE for small datasets (<10K) with purely local focus. Use UMAP for large datasets, global structure, ML pipelines, or bioinformatics.

---

### Q2: What do n_neighbors and min_dist control?

**Answer:**
`n_neighbors` controls the **scale of manifold estimation** — how many local points define each point's neighborhood. Low values (2–5): fine-grained, noisy, fragmented. High values (50–200): smooth, global, loses local detail. Default 15 is a good balance. `min_dist` is a **visual packing parameter** — it sets the minimum distance allowed between points in the 2D output. Low values (0.0): tight, compact clusters. High values (0.8): spread-out continuous distributions. Think of `n_neighbors` as the structural parameter and `min_dist` as the aesthetic one.

---

### Q3: Why does UMAP preserve global structure better than t-SNE?

**Answer:**
Two reasons. First, **initialization**: UMAP uses spectral initialization (Laplacian Eigenmaps), providing a globally coherent starting layout. t-SNE starts randomly. Second, **loss function**: t-SNE's KL divergence only strongly penalizes placing dissimilar points too close (one-sided). UMAP's cross-entropy penalizes both putting similar points far apart AND putting dissimilar points too close — balanced attractive and repulsive forces that preserve more global geometry.

---

### Q4: Can UMAP output be used for downstream ML tasks?

**Answer:**
Yes — a key advantage over t-SNE. Because UMAP has `transform()`, you can embed training data, then embed test data in the same space. Common uses: UMAP (n_components=10–50) → k-means or HDBSCAN clustering; UMAP → classification with SVM or random forest; UMAP → anomaly detection. Important caveat: for ML tasks, use higher `n_components` (10–50) than for visualization (2–3). Always validate that UMAP + model outperforms simpler approaches like PCA + model.

---

### Q5: What is a fuzzy simplicial set and why does UMAP use it?

**Answer:**
A fuzzy simplicial set is UMAP's mathematical representation of the data's topological structure. It's an undirected weighted graph where edge weights $w_{ij} \in [0,1]$ represent the fuzzy probability that points $i$ and $j$ are mutual neighbors. The "fuzzy" part comes from fuzzy set theory — unlike a crisp binary graph, every pair has a continuous degree of membership. The symmetrization formula $w_{ij} = v_{ij} + v_{ji} - v_{ij}v_{ji}$ implements the **fuzzy union** — the probability at least one directed relationship holds. UMAP uses this because it captures continuous manifold structure (trajectories, gradients) better than any binary graph approach, and gives the algorithm its strong theoretical foundations in algebraic topology.

---

## Quick Reference Table

| Property | Value |
|----------|-------|
| **Full Name** | Uniform Manifold Approximation and Projection |
| **Authors** | McInnes, Healy, Melville (2018) |
| **Type** | Non-linear, unsupervised dimensionality reduction |
| **Primary Uses** | Visualization + ML preprocessing |
| **Theory basis** | Riemannian geometry + fuzzy topology |
| **Time Complexity** | O(N^1.14) NN search + O(N · epochs) SGD |
| **Invertible?** | No (approximate `inverse_transform()` available) |
| **Handles New Data?** | ✅ Yes — `transform()` on unseen data |
| **Deterministic?** | No — always set `random_state` |
| **Install** | `pip install umap-learn` |
| **Key Hyperparameters** | `n_neighbors` (5–200), `min_dist` (0–0.99) |

### UMAP vs t-SNE vs PCA

| Feature | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| Linear? | Yes | No | No |
| Preserves | Global variance | Local clusters | Local + global |
| Speed | Very fast | Slow O(N²) | Fast O(N log N) |
| New data (`transform`)? | ✅ | ❌ | ✅ |
| Deterministic? | ✅ | ❌ | ❌ (stable) |
| Use in ML pipeline? | ✅ | ❌ | ✅ |
| Interpretable axes? | ✅ | ❌ | ❌ |
| Best dataset size | Any | <50K | Any |

---

## References

1. **Original Paper**: McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)

2. **How UMAP Works (by the author)**: McInnes, L. *How UMAP Works* — most accessible technical deep-dive. [Link](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)

3. **Official Documentation**: umap-learn parameter guide, supervised UMAP, metric variants. [Link](https://umap-learn.readthedocs.io/en/latest/)

4. **Becht et al. (2019)**: *Dimensionality reduction for single-cell data using UMAP*. Nature Biotechnology. Real-world benchmark. [Link](https://www.nature.com/articles/nbt.4314)

5. **Distill — How to Use t-SNE Effectively**: Same interpretability lessons apply to UMAP. [Link](https://distill.pub/2016/misread-tsne/)

---



