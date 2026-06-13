 Principal Component Analysis (PCA)Principal Component Analysis (PCA)

Tagline

Find the directions where your data "spreads out" the most — and keep only those.


Summary
PCA is an unsupervised, linear dimensionality reduction technique that transforms high-dimensional data into a smaller set of uncorrelated variables called principal components, each capturing as much variance as possible. It achieves this by rotating the original feature axes to align with the directions of maximum spread in the data. PCA is widely used for speeding up machine learning pipelines, removing multicollinearity, and visualizing complex datasets in 2D or 3D.

What is PCA?
Imagine you're photographing a 3D sculpture for a museum catalog. You can only take one photo — so you walk around the sculpture looking for the angle that shows the most detail, the most shape, the most "information." That single best angle is your first principal component. The second-best angle, perpendicular to the first, is your second principal component. PCA finds those angles for your data automatically.
In practical terms, real-world datasets often contain dozens or hundreds of features — many of which are correlated with each other. For example, a person's height and shoe size tend to move together. Storing both features gives you little extra information compared to storing just one. PCA identifies and removes this redundancy by creating new features (principal components) that are linear combinations of the originals, ordered from most to least informative.
The result is a compressed representation of your data that retains the maximum possible amount of information. You lose some detail — but you gain speed, simplicity, and the ability to visualize. Think of it like summarizing a 10-page report into 3 bullet points: you lose some specifics, but the core message survives.

Mathematical Idea
PCA operates on three key mathematical concepts:
1. Mean Centering
Xcentered=X−μX_{\text{centered}} = X - \muXcentered​=X−μ
Each feature is shifted so its mean becomes zero. This ensures PCA focuses on variance (spread), not on where the data happens to be located in space. Without this step, the first component would simply point toward the data's average position rather than its most variable direction.
2. Covariance Matrix
Σ=1n−1XcenteredTXcentered\Sigma = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}Σ=n−11​XcenteredT​Xcentered​
The covariance matrix is a square matrix (features × features) that captures how much every pair of features moves together. A high value between features A and B means they're correlated — and PCA will combine them. The diagonal holds each feature's own variance.
3. Eigenvalues and Eigenvectors
Σv=λv\Sigma \mathbf{v} = \lambda \mathbf{v}Σv=λv

v\mathbf{v}
v (eigenvector): A direction in feature space — this becomes a principal component axis
λ\lambda
λ (eigenvalue): How much variance exists along that direction

Intuitively: eigenvectors are the "new camera angles," and eigenvalues tell you how informative each angle is. Sort by eigenvalue descending → first component captures the most variance, second captures the next most, and so on.

How It Works
Step 1 — Standardize the Data

Scale each feature to zero mean and unit variance using StandardScaler. Without this, features measured in large units (e.g., salary in thousands) will dominate over features in small units (e.g., age in years), giving misleading components.
Step 2 — Compute the Covariance Matrix

Calculate how correlated each pair of features is. This reveals the internal structure — which features are redundant and which carry unique information.
Step 3 — Find Eigenvalues and Eigenvectors

Decompose the covariance matrix. Each eigenvector points in a direction of variance; each eigenvalue scores how much variance that direction contains.
Step 4 — Select the Top k Components

Sort eigenvectors by their eigenvalue (largest first). Keep the top k. Use a scree plot or the 95% cumulative explained variance rule to pick k — the point where adding more components gives diminishing returns.
Step 5 — Transform the DataProject the original data onto the selected eigenvectors:


Z=Xcentered⋅WZ = X_{\text{centered}} \cdot WZ=Xcentered​⋅W
where WW
W is the matrix of top-k eigenvectors. ZZ
Z is your compressed dataset — same rows, fewer columns.

Key Assumptions
AssumptionWhy It MattersLinearityPCA only finds linear combinations of features. Non-linear patterns (spirals, clusters) are invisible to it.Large variance = importantPCA treats high-variance directions as signal. If noise happens to have high variance, PCA will retain it.Features must be scaledUnscaled features with large magnitudes dominate the covariance matrix unfairly. Always standardize first.

When to Use

Data has many correlated features (multicollinearity)
You need to visualize high-dimensional data in 2D or 3D
You want to speed up a downstream model (fewer input features = faster training)
You need to remove noise from data before modeling

When NOT to Use

When feature interpretability is critical (PCA components are combinations of all features — not easy to explain to stakeholders)
When data has meaningful non-linear structure (use t-SNE, UMAP, or Kernel PCA)
When your dataset is very small — PCA can overfit to noise
When the target variable matters for reduction (use LDA instead — it's supervised)


Implementation Overview
In scikit-learn, PCA is straightforward:

Always apply StandardScaler first — PCA is scale-sensitive; unscaled data produces biased components
Instantiate with PCA(n_components=k) — set k as an integer (fixed components) or a float like 0.95 (retain 95% variance automatically)
Check explained_variance_ratio_ — this array tells you what fraction of total variance each component captures; cumulative sum helps choose k
Use fit_transform(X_scaled) — fits the PCA model and returns the projected data in one step

For very large datasets, IncrementalPCA processes data in batches. For non-linear data, KernelPCA applies a kernel trick before decomposition.

Top Interview Questions
QuestionKey HintWhat is PCA and what problem does it solve?Unsupervised linear DR; removes redundancy; finds max-variance directionsWhy must we scale features before PCA?Covariance matrix is scale-sensitive; large-magnitude features dominate eigenvectors unfairlyWhat do eigenvalues represent in PCA?Amount of variance captured along the corresponding eigenvector (principal component)PCA vs t-SNE — when to use which?PCA: fast, linear, preserves global structure; t-SNE: slow, non-linear, better for cluster visualizationWhat are the main limitations of PCA?Assumes linearity; components are uninterpretable; sensitive to outliers; drops low-variance (but possibly important) features

Quick Reference Table
ItemDetailTypeUnsupervised, LinearUse CaseDimensionality reduction, visualization, noise removalInputNumerical, standardized featuresOutputk uncorrelated principal componentsKey Hyperparametern_componentsStrengthRemoves multicollinearity, speeds up trainingLimitationComponents are hard to interpret; fails on non-linear dataScikit-learn Classsklearn.decomposition.PCA

References

Scikit-learn PCA Documentation
StatQuest — PCA Step-by-Step (YouTube)
Towards Data Science — PCA Explained Visually
Kaggle — Breast Cancer Wisconsin Dataset
Jolliffe, I.T. (2002). Principal Component Analysis, 2nd ed. Springer.
