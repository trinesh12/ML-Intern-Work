# t-SNE (t-Distributed Stochastic Neighbor Embedding)

## Tagline

Turning complex data into simple visual clusters.

## Summary

t-SNE is a powerful dimensionality reduction technique used to visualize high-dimensional data in 2D or 3D. It preserves relationships between nearby data points, making it easier to identify clusters and patterns. In simple terms, it converts complicated data into a visual map.

---

## What is t-SNE?

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a technique used to reduce high-dimensional data into lower dimensions, mainly for visualization purposes.

Imagine you have a classroom full of students, and each student has 100 characteristics (marks, hobbies, height, etc.). Now you want to place them on a 2D paper such that:

* Similar students sit close together
* Different students sit far apart

👉 t-SNE does exactly this for data.

It creates a map where:

* Nearby points = similar data
* Distant points = different data

This helps us easily understand patterns and clusters in complex datasets.

---

## Mathematical Idea

t-SNE works by converting distances into probabilities and matching them across dimensions.

### Core Concept:

[
KL(P || Q) = \sum P(i,j) \log \frac{P(i,j)}{Q(i,j)}
]

Where:

* ( P(i,j) ): Similarity between points in high-dimensional space
* ( Q(i,j) ): Similarity between points in low-dimensional space
* KL Divergence: Measures difference between the two distributions

### Key Points:

* High-dimensional space uses **Gaussian distribution**
* Low-dimensional space uses **t-distribution**

👉 Why t-distribution?
It spreads points better and avoids crowding.

---

## How It Works

Step-by-step:

1. Take high-dimensional data
2. Compute similarity between data points
3. Convert similarities into probabilities
4. Place points randomly in 2D/3D space
5. Iteratively adjust positions:

   * Pull similar points closer
   * Push dissimilar points apart
6. Stop when the structure stabilizes

👉 Final result: clear clusters and patterns

---

## Key Assumptions

* **Local structure is important**
  Nearby points should remain close

* **Global structure is less important**
  Overall distances may not be preserved

* **Similarity over exact distance**
  Focus is on relationships, not exact values

---

## When to Use

* Visualizing high-dimensional data
* Exploring clusters in datasets
* Understanding embeddings (like word vectors, image features)
* Data exploration before modeling

## When NOT to Use

* When you need features for ML models
* When dataset is very large (slow computation)
* When exact distances are important
* When reproducibility is critical (results can vary)

---

## Implementation Overview

### Using sklearn

Steps:

* Load dataset (e.g., Iris)
* Scale features
* Apply t-SNE from `sklearn.manifold`
* Visualize results using plots

### Important Parameters:

* `n_components` → Output dimensions (2 or 3)
* `perplexity` → Balance between local & global structure (5–50)
* `random_state` → For reproducibility

---

### From Scratch

t-SNE is complex to implement manually because:

* Requires probability distributions
* Uses KL divergence optimization
* Needs iterative gradient updates

👉 Beginners should use sklearn instead of coding from scratch.

---

## Top Interview Questions

1. Why is t-SNE non-linear?
   → It captures complex relationships, not just straight-line patterns

2. What is the crowding problem?
   → Points get compressed in low dimensions

3. Why use t-distribution instead of Gaussian?
   → To spread points and reduce crowding

4. What happens if perplexity is too high?
   → Clusters become less clear

5. Can t-SNE be used for prediction?
   → No, it is mainly for visualization

---

## Quick Reference Table

| Item       | Detail                              |
| ---------- | ----------------------------------- |
| Type       | Non-linear Dimensionality Reduction |
| Use Case   | Data Visualization                  |
| Strength   | Preserves local structure           |
| Limitation | Slow and non-deterministic          |

---

## References

* Scikit-learn Documentation (t-SNE)
* Original Paper by Laurens van der Maaten
* StatQuest – t-SNE Explained
* Andrew Ng ML Course
* Practical ML Tutorials

---
