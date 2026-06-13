# t-SNE — t-Distributed Stochastic Neighbor Embedding

---

## Tagline

> *Map your high-dimensional data onto a 2D canvas — so hidden clusters and patterns finally become visible to the human eye.*

---

## Summary

t-SNE is a non-linear, unsupervised dimensionality reduction technique designed primarily for **visualization**. It converts high-dimensional pairwise similarities between data points into 2D (or 3D) positions, placing similar points close together and dissimilar points far apart. Unlike PCA, which finds global linear directions of variance, t-SNE excels at revealing **local cluster structure** — making it the go-to tool for exploring embeddings, image datasets, and any high-dimensional data where clusters are expected but invisible.

---

## What is t-SNE?

Imagine you manage a massive social network with millions of users. Each user has hundreds of attributes — interests, activity hours, location, content preferences. You want to create a single 2D "friendship map" where users who are genuinely similar end up sitting next to each other, and users who are very different sit far apart. That is exactly what t-SNE produces: a 2D map that honestly reflects the neighborhood structure of high-dimensional data.

The core idea is built on **probability**. In high-dimensional space, t-SNE computes the probability that two points are "neighbors" — close points get high probability, distant points get near-zero probability. It then arranges points in 2D so that those neighborhood probabilities are matched as closely as possible. The algorithm doesn't care about absolute distances or global geometry; it cares only about *who is close to whom*.

The "t" in t-SNE refers to the **Student's t-distribution** used in the low-dimensional space. This is not an arbitrary choice. In 2D, if you used a normal (Gaussian) distribution for similarities, points would crowd toward the center — a problem known as the **crowding problem**. The t-distribution has heavier tails, which gives moderately-distant points more room to spread out, producing cleaner, more separated clusters in the visualization.

---

## Mathematical Idea

t-SNE works with two sets of probability distributions — one in high-dimensional space, one in low-dimensional space — and minimizes the difference between them.

**High-dimensional similarity (Gaussian kernel):**

$$p_{j|i} = \frac{\exp\!\left(-\|x_i - x_j\|^2 / 2\sigma_i^2\right)}{\displaystyle\sum_{k \neq i} \exp\!\left(-\|x_i - x_k\|^2 / 2\sigma_i^2\right)}$$

Symmetrized joint probability:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**Low-dimensional similarity (Student's t-distribution, 1 degree of freedom):**

$$q_{ij} = \frac{\left(1 + \|y_i - y_j\|^2\right)^{-1}}{\displaystyle\sum_{k \neq l}\left(1 + \|y_k - y_l\|^2\right)^{-1}}$$

**Loss function — KL Divergence:**

$$\mathcal{L} = KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

| Symbol | Meaning |
|--------|---------|
| $x_i, x_j$ | Points in high-dimensional space |
| $y_i, y_j$ | Corresponding points in 2D embedding (what we optimize) |
| $\sigma_i$ | Gaussian bandwidth for point $i$ — controlled by `perplexity` |
| $p_{ij}$ | Probability of $i$ and $j$ being neighbors in high-D |
| $q_{ij}$ | Probability of $i$ and $j$ being neighbors in 2D |
| $\mathcal{L}$ | KL divergence — the loss we minimize via gradient descent |

Intuitively: if two points are close in high-D ($p_{ij}$ is high), but far apart in 2D ($q_{ij}$ is low), the loss is large. Gradient descent moves the 2D points to fix this mismatch.

---

## How It Works

**Step 1 — Compute Pairwise Similarities in High-D**
For every pair of points, calculate how similar they are using a Gaussian kernel. The bandwidth $\sigma_i$ is set individually per point using **binary search** so that the effective number of neighbors (controlled by `perplexity`) is consistent across all points.

**Step 2 — Symmetrize the Probabilities**
Convert conditional probabilities $p_{j|i}$ into symmetric joint probabilities $p_{ij}$. This ensures that the relationship between two points is the same regardless of which one you start from.

**Step 3 — Initialize 2D Positions**
Place all points at random 2D positions (or use PCA initialization for stability). These are the starting positions that will be optimized.

**Step 4 — Compute Low-D Similarities**
Using the current 2D positions, compute similarities $q_{ij}$ using the t-distribution (heavier tails than Gaussian — this is the crowding fix).

**Step 5 — Minimize KL Divergence via Gradient Descent**
Compute the gradient of the KL divergence with respect to each 2D point's position. Move the points: similar pairs attract each other, dissimilar pairs repel. Repeat for hundreds of iterations until convergence.

**Step 6 — Read the Final Map**
The final 2D positions form your t-SNE embedding. Nearby clusters represent groups of similar data points in the original high-dimensional space.

---

## Key Assumptions

| Assumption | Why It Matters |
|---|---|
| **Local structure matters most** | t-SNE preserves neighborhoods, not global distances. Two clusters far apart in 2D may or may not be far in high-D. |
| **Perplexity is meaningful** | Perplexity ≈ effective number of neighbors. If set too low, the map is noisy; too high, clusters merge. |
| **Many iterations needed** | t-SNE converges slowly. Fewer than 250 iterations often gives misleading maps. |
| **Not for new data** | t-SNE learns a fixed embedding — it cannot project unseen points without rerunning from scratch. |

---

## When to Use

- Exploring cluster structure in high-dimensional data (images, text embeddings, gene expression)
- Visualizing the output of neural network layers or word/sentence embeddings
- Confirming that clusters found by K-Means or DBSCAN make visual sense
- When your data clearly has non-linear, manifold-like structure

## When NOT to Use

- When you need to project **new data points** into the embedding (use UMAP instead)
- When **global distances matter** — cluster separation in t-SNE plots is not meaningful
- On datasets larger than ~50,000 rows without subsampling (very slow: O(n²))
- As a preprocessing step for machine learning models — t-SNE embeddings don't generalize
- When you need **reproducible results** without a fixed random seed

---

## Implementation Overview

Scikit-learn provides `sklearn.manifold.TSNE` — the standard starting point:

- **Always standardize first** using `StandardScaler`. t-SNE relies on Euclidean distances; unscaled features distort neighborhood structure.
- **Key parameter — `perplexity`**: Controls how many neighbors each point effectively considers. Typical range: 5–50. Default is 30. Try multiple values and compare.
- **`n_iter`**: Number of gradient descent steps. Minimum 250; recommend 1000 for stable results.
- **`init='pca'`**: Initializes 2D positions using PCA — far more stable than random initialization, especially for larger datasets.
- **`learning_rate='auto'`**: Scikit-learn handles this well automatically in recent versions.
- **`random_state=42`**: Always set — t-SNE is non-deterministic and different runs give different layouts.

**Pro tip:** For large datasets, first reduce to 50 dimensions with PCA, then apply t-SNE. This dramatically speeds up the computation without losing meaningful structure.

---

## Top Interview Questions

| Question | Key Hint |
|---|---|
| What is t-SNE and what makes it different from PCA? | t-SNE is non-linear and preserves local structure; PCA is linear and preserves global variance. t-SNE is for visualization only. |
| What is perplexity and how does it affect the result? | Perplexity ≈ effective number of neighbors per point. Low → noisy local clusters; High → clusters may merge. Try 5, 30, 50 and compare. |
| Why does t-SNE use the t-distribution in low-D instead of Gaussian? | To solve the crowding problem. Gaussian in 2D clusters too many moderately-distant points near the center. The t-distribution's heavy tails give them room to spread. |
| Can t-SNE be used for feature engineering or preprocessing? | No. t-SNE embeddings can't project new data and don't generalize. It's purely for visualization/exploration. |
| What are the limitations of t-SNE? | Slow (O(n²)); non-deterministic; cluster sizes and inter-cluster distances are not meaningful; cannot handle new points. |

---

## Quick Reference Table

| Item | Detail |
|------|--------|
| Type | Non-linear, Unsupervised |
| Primary Use | Visualization (2D / 3D only) |
| Preserves | Local neighborhood structure |
| Does NOT Preserve | Global distances, cluster sizes |
| Key Hyperparameter | `perplexity` (5–50) |
| Complexity | O(n²) standard; O(n log n) Barnes-Hut |
| New Data Projection | ❌ Not supported |
| Scikit-learn Class | `sklearn.manifold.TSNE` |
| Strength | Reveals clusters invisible to PCA |
| Limitation | Slow, non-deterministic, visualization only |

---

## References

- [Scikit-learn TSNE Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [van der Maaten & Hinton (2008) — Original t-SNE Paper](https://jmlr.org/papers/v9/vandermaaten08a.html)
- [Distill.pub — How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
- [StatQuest — t-SNE Clearly Explained (YouTube)](https://www.youtube.com/watch?v=NEaUSP4YerM)
- [Kaggle — Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)
