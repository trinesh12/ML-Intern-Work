# PCA (Principal Component Analysis)

## 📌 What is PCA?

PCA (Principal Component Analysis) is a technique used to reduce the number of features (columns) in a dataset without changing the most important information.


It converts many variables into few variables without affecting much data.

---

## 📌 Why do we need PCA?

Sometimes datasets have:
- Too many features
      Example-You have:
          Math marks
          Physics marks
          Chemistry marks
          
          without PCA -3 features
          With PCS 1 new feature → “Science Score”
          
          👉 These are related
          👉 Instead of remembering 100 small details
          👉 PCA finds main ideas
          - 
- Redundant (repeated) information
-  Example- Height (cm) -->> Height (m)
        👉 PCA creates: PC1 (combined feature)
          
- High complexities
           What is complexity :Many features can be  hard to process

Problems:
- Model becomes slow
- Overfitting may happen(Overfitting means Model is learning too much than required)
- Hard to visualize data

👉 PCA solves this by reducing dimensions.
✔ You can draw graphs
✔ You can see patterns
✔ You can identify clusters

📊 Example

Before PCA:

10 features ❌ (cannot visualize)

After PCA:

2 features (PC1, PC2) ✅ (can plot)

---

## 📌 Intuition (Easy Understanding)

Imagine you have 3D data (length, width, height).

👉 PCA tries to:
- Find the direction where data varies the most
- Project data onto that direction

So instead of 3 features → you may get 2 or 1 important features.

---

## 📌 Key Idea

👉 PCA keeps the directions with **maximum variance**
Varience means how much the data is spread
High variance:
10, 50, 90

👉 Big differences → more information ✅

Low variance:
50, 51, 49

👉 Almost same → less information ❌
-
🧠 Simple intuition

Imagine students:

Feature 1: Marks vary a lot
Feature 2: Marks almost same for everyone

👉 Which is useful?

✔ Feature 1 (high variance)
❌ Feature 2 (low variance)
---

## 📌 Steps in PCA (Simple)

1. Normalize the data  --Make all features come to the same scale->So we convert them to similar range ✅
2. Find mean of each feature  --To understand the “center” of data 
3. Compute covariance matrix  --Check how features are related
4. Find eigenvalues and eigenvectors Eigenvectors → directions and Eigenvalues → importance of those directions
6. Select top components (based on highest eigenvalues)  
7. Transform the data  

---

## 📌 Mathematical Idea (Simple Form)

PCA transforms data using:

New Data = Original Data × Eigenvectors

---

## 📌 Applications of PCA

- Data visualization (reduce to 2D/3D)  
- Noise reduction  
- Image compression  
- Feature reduction before ML models  

---

## 📌 Advantages

- Reduces complexity  
- Improves speed of models  
- Removes noise  (Noise = unwanted or random information in data)
- Helps in visualization  

---

## 📌 Limitations

- Loss of some information  
- Hard to interpret new features  
- Works best for linear data  
- Sensitive to scaling -PCA gives wrong importance if features are in different scales
You have 2 features:

Age → values like 20, 25, 30
Salary → values like 50,000, 80,000, 100,000
❌ Problem

👉 Salary values are much bigger

So PCA thinks:

✔ Salary = very important
❌ Age = not important

👉 Even if age is actually useful!
---

## 📌 When NOT to use PCA

- When interpretability is important  -Interpretability = ability to understand what each feature means
- When data is already small  
- When relationships are non-linear  

---


## 📌 Summary

- PCA reduces dimensions  
- Keeps important information  
- Uses variance to decide importance  
- Helps in faster and better ML models  
