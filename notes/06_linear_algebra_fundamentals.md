# Lecture 6 Notes
**Topic: Linear Algebra Fundamentals — Vectors, Dot Products, Transformations, Projections, PCA (Deep Dive)**

---

## 1. What is Linear Algebra?

> **Linear algebra is the study of vectors, matrices, and the corresponding operations on them.**

It is the mathematical foundation for everything in ML — data representation, model weights, PCA, loss functions, and neural networks.

---

## 2. Vectors — Three Views (Revisited with Depth)

### Physics Student
A vector is an arrow with **magnitude and direction**.

### CS Student
A vector is a **list of numbers** — implemented as a NumPy array.

### Math Student
A vector is an object that supports two operations:
1. **Scalar multiplication** — `3 × [1, 2] = [3, 6]`
2. **Vector addition** — `[1,2] + [3,4] = [4,6]`

---

## 3. Vector Addition — The Geometric Meaning

Why is vector addition element-wise? Because of the "walk" interpretation:

```
A = [2, 1]    B = [1, 3]

Walk along A: reach x=2, y=1
Then walk along B: add x=1, y=3

Result C = [3, 4]
```

You move B so it starts where A ends (parallel displacement). Where you end up is C — the resultant vector. This is why element-wise addition works — you're summing the intercepts on each axis independently.

---

## 4. Basis Vectors

Every vector in a vector space can be defined using **basis vectors** — the reference axes of that space.

Standard basis vectors:
```
î (i-hat) = [1, 0]    ← unit vector along x-axis
ĵ (j-hat) = [0, 1]    ← unit vector along y-axis
```

Any vector can be written as a scalar combination of basis vectors:
```
[3, 2] = 3 × [1,0] + 2 × [0,1]
       = 3î + 2ĵ
```

### Why Orthogonal Bases?

Basis vectors must be **orthogonal** (perpendicular) to each other. If two basis vectors are parallel (point in the same direction), you lose the ability to define the second dimension — you can only move in one direction. Orthogonal bases give you **independent dimensions** to span the full space.

> This connects directly to why correlated features are a problem in ML — they are like non-orthogonal basis vectors. They don't independently add information.

---

## 5. The Dot Product — Three Views

### Why Do We Need It?

Scalar multiplication lets us scale a vector. Addition lets us combine two vectors. But we need a way to measure **alignment** between two vectors.

### Physics View
```
A · B = |A| × |B| × cos(θ)
```
- `|A|` = magnitude of A
- `|B|` = magnitude of B
- `cos(θ)` = cosine of angle between them
- Result is a **scalar** (a single number, not a vector)

### Math/Geometric View
The dot product equals the **length of the projection of A onto B** multiplied by the length of B:
```
A · B = (length of shadow of A onto B) × |B|
```

### CS View (Implementation)
```python
A = [2, 1]    B = [1, 3]
A · B = (2×1) + (1×3) = 5
```
Element-wise multiply then sum.

```python
# Implementation
dot_product = np.sum(A * B)      # element-wise multiply then sum
# or equivalently
dot_product = np.dot(A, B)       # fine for vectors only
```

> **Key rule:** `np.dot` is built for vectors. For matrices and tensors, use `@` or `np.matmul`.

### What Does the Dot Product Tell Us?

| Dot product value | Meaning |
|---|---|
| Large positive | Vectors point in similar direction — strongly aligned |
| Zero | Vectors are perpendicular — completely independent |
| Negative | Vectors point in opposite directions |

---

## 6. Dot Product vs Matrix Multiplication — The Critical Distinction

**Dot product** — between two **vectors** only. Measures alignment. Returns a scalar.

**Matrix multiplication** — between **matrices**. Applies transformations. Returns a matrix.

> "In ML we'll keep calling it dot product between vectors, between matrices, between a vector and a matrix. But essentially dot product exists between vectors only. For matrix multiplication, even if we call it dot product, it is actually matrix multiplication."

**Matrix multiplication = dot products of row vectors with column vectors:**
```
C[i,j] = row_i(A) · col_j(B)    (dot product of row vector with column vector)
```

---

## 7. Three Types of Multiplication in Linear Algebra

### 7.1 Vector × Vector (Dot Product)

Measures alignment — returns a **scalar**.

```python
A = np.array([2, 1])
B = np.array([1, 3])
scalar = np.dot(A, B)    # = 5
```

### 7.2 Vector × Matrix (Linear Transformation)

Transforms the vector into a **new vector** in the same space.

```
v = [3, 4]    M = [[1,2],[3,4]]

Result = 3×[1,2] + 4×[3,4]     ← 3 × (first basis) + 4 × (second basis)
       = [3,6] + [12,16]
       = [15, 22]
```

**Geometric meaning:** You are expressing the vector in terms of the new basis vectors defined by the columns of the matrix. The result is where your original vector ends up after the transformation.

### 7.3 Matrix × Matrix

A composition of two transformations — apply one transformation then another.

```python
# Shape rule: (M×N) @ (N×P) → (M×P)
C = A @ B     # use @ always for matrices
```

---

## 8. Linear Transformations

A **linear transformation** is a transformation applied to a vector space that satisfies two properties:

1. **Grid lines remain parallel and equidistant** — the space stretches/rotates uniformly
2. **The origin stays fixed** — the transformation doesn't shift the whole space

**Examples of linear transformations:**
- Rotation (e.g. 90° rotation of the entire plane)
- Scaling (multiplying by a scalar)
- Reflection
- Addition of another vector (shifting along a direction)

**Non-linear transformation example:** Bending the grid lines (e.g. applying sin(x)) — the result is no longer a straight line.

### Why Matrix Multiplication = Linear Transformation

Every vector is just a combination of basis vectors:
```
[3, 2] = 3î + 2ĵ
```

When you multiply by a matrix M, you're moving the basis vectors to new positions (the columns of M), and the vector follows:
```
New position = 3×(new î) + 2×(new ĵ)
```

So matrix × vector = "where does this vector go after the transformation?"

**Rotation by 90° example:**
```
Original basis: î=[1,0], ĵ=[0,1]
After 90° rotation: new_î=[0,1], new_ĵ=[-1,0]

Vector [1,1]:
New position = 1×[0,1] + 1×[-1,0] = [-1,1]
```

---

## 9. Cosine Similarity

Cosine similarity is a **normalized dot product** — only cares about direction, not magnitude.

```
cosine_similarity(A, B) = A·B / (|A| × |B|)
```

| Value | Meaning |
|---|---|
| 1.0 | Identical direction |
| 0.9 | Very similar |
| 0.0 | Completely unrelated (orthogonal) |
| -1.0 | Opposite directions |

**Where it's used:** Embeddings in LLMs, semantic search, recommendation systems. "Going to school" vs "embarking on a journey to school" — their embedding vectors will have high cosine similarity.

---

## 10. Covariance as a Dot Product

Three views connect here:

```
Physics view:    A·B = |A||B|cos(θ)
                 → if two vectors are aligned (θ=0), cos(θ)=1 → maximum dot product
                 → if perpendicular (θ=90°), cos(θ)=0 → dot product = 0

Math view:       projection of A onto B
                 → if fully correlated: A projects completely onto B (full shadow)
                 → if independent: A projects nothing onto B (zero shadow)

ML view:         covariance = mean-centered dot product
```

> **Mean-centered dot product = covariance.**

This is why when computing the covariance matrix in PCA, we:
1. Center the data (subtract column means)
2. Take `Xᵀ @ X` (dot products between features)

If two features have high covariance (dot product), they are aligned — redundant. If covariance is zero, they're orthogonal — independent, adding unique information.

```python
# Covariance matrix
Xc = X - X.mean(axis=0)          # center each feature
Cov = (Xc.T @ Xc) / (n - 1)     # feature × feature matrix (n×n)
```

---

## 11. Norms — Distance in Linear Algebra

A **norm** is a measure of distance in linear algebra.

**Magnitude** is a special case — the norm between a vector and the origin.

### L2 Norm (Euclidean) — default in ML

```
||x|| = √(x₁² + x₂² + ... + xₙ²)
```

```python
# Three equivalent implementations
norm = np.sqrt(np.sum(x**2))
norm = np.sqrt(np.dot(x, x))    # dot product = sum of squares for same vector
norm = np.linalg.norm(x)        # under the hood: same thing
```

> Note: `np.dot(x, x)` = `np.sum(x * x)` = `x · x`. These are all identical — the dot product of a vector with itself is the sum of its squared components.

### L1 Norm (Manhattan)

```
||x||₁ = |x₁| + |x₂| + ... + |xₙ|
```

### Distance Between Two Vectors

```python
# Euclidean distance between vector a and vector b
dist = np.linalg.norm(a - b)
```

---

## 12. Projections — The Building Block of Linear Regression Error

### Scalar Projection (length of shadow)

How much of A falls along B:
```
proj_length = (A · B) / |B|
```

### Vector Projection

The actual projection vector (shadow of A onto B):
```
proj_vector = ((A · B) / (B · B)) × B
```

```python
def project(u, v):
    """Project vector u onto vector v"""
    denom = np.dot(v, v)
    if denom == 0:
        raise ValueError("Cannot project onto zero vector")
    return (np.dot(u, v) / denom) * v
```

### Why Projections Appear in Linear Regression

In linear regression, you have:
- `ŷ` = predicted target vector (what your model produces)
- `y` = true target vector (actual values)

The **error** is the perpendicular distance from `y` to `ŷ`. To find this closest point, you need the projection of `y` onto the prediction space.

```
Projection = closest point on ŷ to y
Error = y - projection(y onto ŷ)
```

> **FAANG interview question:** "Why does projection show up in linear regression?"
> **Answer:** To find the closest predicted value to the true target — the projection of y onto the predicted vector gives the minimum-distance error, which is what least squares minimizes.

**Label/target error** — difference between true and predicted targets (uses projection)
**Model error** — difference between true weights and predicted weights

---

## 13. Solving Linear Models — Closed Form Solution

### The Normal Equations

For the linear model `y = Xw`, we want to find `w`.

Multiply both sides by `Xᵀ`:
```
Xᵀy = Xᵀ(Xw) = (XᵀX)w
```

Rearrange:
```
w = (XᵀX)⁻¹ Xᵀy
```

In NumPy (using LU decomposition, not explicit inverse):
```python
XtX = X.T @ X          # (features × features) — covariance of features with features
XtY = X.T @ y          # (features × 1)        — covariance of features with target

w = np.linalg.solve(XtX, XtY)   # solves (XᵀX)w = Xᵀy
```

### Deep Meaning of XᵀX and XᵀY

| Matrix | Shape | Meaning |
|---|---|---|
| `XᵀX` | (features × features) | Covariance of features with each other — captures redundancy |
| `XᵀY` | (features × 1) | How each feature co-varies with the target — captures signal |

> `XᵀX` is literally the covariance matrix (without centering). It answers: "which features are redundant with each other?"
> `XᵀY` answers: "which features contribute to predicting the target?"

The solution `w = (XᵀX)⁻¹ XᵀY` is called the **closed form solution** or **ordinary least squares (OLS)**.

```python
# Full example
n, d = 50, 3
X = np.random.randn(n, d)
w_true = np.array([1.5, -2.0, 0.5])
y = X @ w_true + 0.1 * np.random.randn(n)   # add noise

# Solve for weights
XtX = X.T @ X
XtY = X.T @ y
w_pred = np.linalg.solve(XtX, XtY)

print(w_true)   # [1.5, -2.0, 0.5]
print(w_pred)   # very close: [1.51, -1.98, 0.49]
```

### Closed Form vs Gradient Descent

| Method | When to use | Why |
|---|---|---|
| Closed form (`np.linalg.solve`) | Small-medium data (fits in RAM) | Single operation, exact solution |
| Gradient Descent | Large-scale data (millions of rows) | LU decomposition too slow at scale |

> "All the models you built with gradient descent taking 2,000 epochs — try building them with closed form and you get the solution instantly."

---

## 14. Orthogonal Matrices

A matrix V is **orthogonal** if all its column vectors are perpendicular to each other (orthogonal vectors).

**Key property:**
```
Vᵀ × V = I    (identity matrix)
```

Therefore: **V⁻¹ = Vᵀ** — the inverse equals the transpose.

This is enormously useful computationally — transposing is O(n²), inverting is O(n³).

```python
# Verify orthogonality
V = np.array([[1, -1], [1, 1]])
is_orthogonal = np.allclose(V.T @ V, 2 * np.eye(2))  # True (scaled by 2)
```

**Why eigenvectors of the covariance matrix are orthogonal:**
- They represent independent directions in feature space
- Orthogonality = independence = no redundancy
- Each captures unique variance that the others don't

---

## 15. Eigendecomposition — The Deep Explanation

### What Are Eigenvectors?

By definition: eigenvectors are vectors that **do not change direction** when a transformation is applied. They only scale.

```
A × v = λ × v
```
- `v` = eigenvector (direction stays the same)
- `λ` = eigenvalue (the scaling factor)

**Rotation example:** Rotate the entire XY plane 180° along the Z-axis. The Y-axis doesn't change direction (it was pointing up, still points up). The Y-axis is an eigenvector of this rotation.

### Eigendecomposition Formula

Any square matrix A can be decomposed:
```
A = V × Λ × V⁻¹ = V × Λ × Vᵀ   (since V is orthogonal → V⁻¹ = Vᵀ)
```

Where:
- `V` = matrix of eigenvectors (columns are eigenvectors)
- `Λ` = diagonal matrix of eigenvalues
- `Vᵀ` = inverse transformation (back to original basis)

### The Geometric Story of Eigendecomposition

1. **Vᵀ** — change basis from original to eigenvector basis (rotate to align with eigenvectors)
2. **Λ** — apply transformation (just scale along each eigenvector axis — easy diagonal operation)
3. **V** — change basis back to original space

> "Rather than doing a complex transformation directly, change the basis to eigenvectors (where transformation is trivial — just scaling), do the scaling, then change back."

**Why diagonal (Λ) transformation is fast:**
```
Complex: [[-1,1],[-1,2],[3,2],[1,0],[5,3]] → hard
Diagonal: [[3,0],[0,1]] → just scale each vector by its eigenvalue → trivial
```

---

## 16. PCA — Complete Picture

### Full Pipeline

```python
# 1. Standardize (center and scale)
mu    = X.mean(axis=0)
sigma = X.std(axis=0)
Xc    = (X - mu) / sigma          # shape: (n, d)

# 2. Covariance matrix
Cov = (Xc.T @ Xc) / (n - 1)      # shape: (d, d) — mean-centered dot product

# 3. Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(Cov)

# 4. Sort by eigenvalue (descending = most variance first)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. Choose K (e.g. 95% variance explained)
cumulative_var = np.cumsum(eigenvalues) / eigenvalues.sum()
K = np.argmax(cumulative_var >= 0.95) + 1

# 6. Select top K eigenvectors
W = eigenvectors[:, :K]           # shape: (d, K)

# 7. Transform data
Z = Xc @ W                        # shape: (n, K) ← new input matrix

# 8. Reconstruction (to check quality)
X_reconstructed = Z @ W.T
```

### PCA via SVD (Alternative — used by sklearn)

```python
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
W = Vt[:K].T          # top K eigenvectors
Z = Xc @ W            # transformed data
```

SVD does not require computing the covariance matrix explicitly — more numerically stable for large dimensions.

### Why Eigenvectors Are the Principal Components

- Eigenvectors of the covariance matrix are orthogonal → independent directions
- Each captures a different source of variance
- Eigenvalue = how much variance that direction captures
- Taking top K eigenvectors = keeping the K most informative directions

---

## 17. Key Connections — Everything Ties Together

```
Dot product
    ├── Alignment measure (cosine similarity in embeddings)
    ├── Mean-centered = covariance (feature redundancy detection in EDA)
    └── Building block of:
            ├── Matrix multiplication (row · column)
            ├── Norm (x · x = ||x||²)
            ├── Projection (A·B / B·B × B)
            └── Covariance matrix (Xᵀ @ X)

Covariance matrix
    └── Eigendecomposition
            └── Eigenvectors = principal components (PCA)
                    └── Dimensionality reduction → better models

Projection
    └── Error in linear regression (distance from y to ŷ)
            └── Least squares solution
                    └── Normal equations: w = (XᵀX)⁻¹ Xᵀy

Orthogonal matrices (eigenvectors)
    └── V⁻¹ = Vᵀ → computationally efficient eigendecomposition
```

---

## 18. Interview Questions Covered This Lecture

| Question | Answer |
|---|---|
| What is the dot product geometrically? | Length of projection of A onto B times length of B; measures alignment |
| Why is dot product a scalar? | It's a magnitude × magnitude × cos(θ) — all scalars |
| What is cosine similarity? | Normalized dot product — direction only, magnitude-free |
| What is covariance geometrically? | Mean-centered dot product between two feature vectors |
| Why do we need projection in linear regression? | To find the closest point on the prediction space to the true target — defines the error |
| Why are eigenvectors orthogonal? | Because the covariance matrix is symmetric — symmetric matrices always have orthogonal eigenvectors |
| What is eigendecomposition geometrically? | Change to eigenbasis → scale by eigenvalues → change back |
| Why does V⁻¹ = Vᵀ for eigenvectors? | Because eigenvectors are orthogonal → Vᵀ @ V = I |
| What does XᵀX represent? | Covariance of features with each other |
| What does XᵀY represent? | How each feature co-varies with the target |
| Closed form vs gradient descent? | Closed form exact + fast for small data; gradient descent scales to millions of rows |

---

## 19. Key Terms

| Term | Definition |
|---|---|
| **Basis vectors** | Reference axes that span a vector space; all vectors expressed as combinations of them |
| **Linear transformation** | Grid lines stay parallel, origin stays fixed; implemented as matrix multiplication |
| **Dot product** | Scalar measure of alignment between two vectors: `Σ(aᵢ×bᵢ)` |
| **Cosine similarity** | Normalized dot product: `A·B / (|A||B|)` |
| **Norm** | Distance measure; L2 = Euclidean, L1 = Manhattan |
| **Projection** | Shadow of one vector onto another; `(A·B / B·B) × B` |
| **Covariance** | Mean-centered dot product; how two features co-vary |
| **Covariance matrix** | `XᵀX / (n-1)` — pairwise covariances between all features |
| **Orthogonal matrix** | Matrix whose columns are mutually perpendicular; `Vᵀ = V⁻¹` |
| **Eigenvector** | Direction that doesn't change under a transformation — only scales |
| **Eigenvalue** | The scaling factor applied to an eigenvector |
| **Closed form solution** | `w = (XᵀX)⁻¹ Xᵀy` — exact weight solution via linear algebra |
| **Normal equations** | The matrix form of the least squares solution |
| **SVD** | Singular Value Decomposition — alternative to eigendecomposition, more stable |

---

## 20. Coming Up Next

- **Calculus** — derivatives, chain rule, partial derivatives
- **Gradient descent in depth** — how ∂L/∂w drives weight updates
- **Linear regression from scratch** — building the full model using closed form and gradient descent
- **Logistic regression** — sigmoid function, binary classification
