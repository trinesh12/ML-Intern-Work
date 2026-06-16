📌 Topic: UMAP

Tagline: Faster and smarter visualization of complex data
Summary:
UMAP is a modern dimensionality reduction technique that converts high-dimensional data into lower dimensions while preserving both local and global structure efficiently.

🧠 What is UMAP (with analogy)

Imagine a large social network where:

Each person is connected to their close friends
Some groups are tightly connected (friend circles)

Now you want to draw this network on paper:

Keep friends close
Keep different groups separate

👉 UMAP builds a “friendship map” and then redraws it in 2D.

📐 Mathematical Idea (simple)

UMAP works in 3 main ideas:

Build a graph
Each point connects to its nearest neighbors
Assign strengths to connections
Strong connection = very similar
Weak connection = less similar
Optimize in low dimension
Try to preserve these connections

👉 In simple terms:

High dimension → build graph
Low dimension → recreate same graph
⚙️ How it Works (step-by-step)
Take high-dimensional data
Find nearest neighbors
Build a weighted graph
Initialize points in 2D
Optimize positions:
Pull connected points closer
Push unrelated points apart
Final stable embedding
⚠️ Key Assumptions
Data lies on a hidden structure (manifold)
Local relationships are important
Graph structure represents data well
✅ When to Use / ❌ When Not to Use
✅ Use when:
Large datasets
Need faster alternative to t-SNE
Want both local + global structure
❌ Avoid when:
Very small datasets (overkill)
Need exact reproducibility
Interpret exact distances
🛠️ Implementation Overview

Steps:

Install library:

pip install umap-learn
Load dataset
Scale data
Apply UMAP
Plot results

Important parameters:

n_neighbors → controls local/global balance
min_dist → controls cluster tightness
n_components → output dimensions
🎯 Interview Questions
Why is UMAP faster than t-SNE?
What is the role of n_neighbors?
Difference between UMAP and PCA?
What is manifold learning?
Can UMAP be used before clustering?
📊 Quick Reference Table
Feature	UMAP
Type	Non-linear
Speed	Fast
Preserves	Local + Global
Scalable	Yes
Output	2D / 3D
📚 References
UMAP official documentation
Research paper by McInnes et al.
Scikit-learn ecosystem tutorials