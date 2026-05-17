# Lecture 5 Notes
**Topics: Matrix Multiplication, Linear Models, Solving Linear Equations, In-place Operations, PCA**

---

## 1. Matrix Multiplication

### Shape Rules

For matrix multiplication A × B to work, the **inner dimensions must match**.

```
A: (M × N)  ×  B: (N × P)  →  Result: (M × P)
         ↑___↑
         must match
```

```python
# Examples
A: (3, 4)  ×  B: (4, 3)  →  Result: (3, 3)  ✓
A: (3, 4)  ×  B: (3, 4)  →  Error!            ✗ (4 ≠ 3)
```

### For Tensors (3D+)

The **depth (batch) dimensions must match** first, then the inner matrix dimensions must match.

```
A: (2, 3, 4)  ×  B: (2, 4, 5)  →  Result: (2, 3, 5)  ✓
A: (2, 3, 4)  ×  B: (3, 4, 5)  →  Error!               ✗ (depth 2 ≠ 3)
```

### Three Ways to Do Matrix Multiplication in NumPy

```python
import numpy as np

A = np.arange(6).reshape(2, 3)
B = np.arange(12).reshape(3, 4)

# Method 1: np.matmul — preferred for matrices
C = np.matmul(A, B)          # shape: (2, 4)

# Method 2: @ operator — syntactic sugar for matmul
C = A @ B                    # same as np.matmul

# Method 3: np.dot — legacy, avoid for tensors
C = np.dot(A, B)             # works for 2D but behaves unexpectedly on 3D+
```

> **Rule:** Always use `@` or `np.matmul` for matrix multiplication. `np.dot` was built for vector dot products and was extended to matrices — it behaves unexpectedly with tensors.

### How Matrix Multiplication Works

```
A = [[a, b],    B = [[e, f],
     [c, d]]         [g, h]]

Result[0,0] = a×e + b×g   (row 0 of A · col 0 of B)
Result[0,1] = a×f + b×h   (row 0 of A · col 1 of B)
Result[1,0] = c×e + d×g   (row 1 of A · col 0 of B)
Result[1,1] = c×f + d×h   (row 1 of A · col 1 of B)
```

---

## 2. Linear Models — The Big Picture

### What is a Linear Model?

A linear model assumes the target Y is a **linear combination** of input features X, weighted by W:

```
Y = Wᵀ · X + bias + noise
```

Where:
- **X** = feature matrix (samples × features) — what we know
- **W** = weight vector — what we are trying to find
- **bias** = intercept (shifts the line up/down)
- **noise** = random error (Gaussian, Bernoulli, etc.)

```python
# Conceptual representation
y_predicted = w1*x1 + w2*x2 + w3*x3 + ... + bias
```

**The goal of training a linear model is to find the weights W.**

### Why Weights Matter

Different features have different impact on the target. The weights tell us how much to "trust" each feature:

```
house_price = 3 × num_rooms + 4 × distance_from_school + bias
              ↑              ↑
              w1=3           w2=4
```

---

## 3. Three Ways to Solve a Linear System

As data scales up, different methods become necessary.

### Problem Setup

A linear model is a **system of linear equations**:

```
w1·x1 + w2·x2 = y₁
w1·x1 + w2·x2 = y₂
```

In matrix form: `A · W = B` (find W given A and B)

---

### Method 1: Gaussian Elimination (manual / small scale)

For simple systems with 2–3 variables. Multiply rows and subtract to isolate variables.

```
3w1 + 4w2 = 25
 w1 +  w2 =  7

Multiply row 2 by 4:  4w1 + 4w2 = 28
Subtract row 1:        w1         = 3   → w2 = 4
```

**Fails when:** One row is a scalar multiple of another (the system is not uniquely solvable).

---

### Method 2: Linear Algebra — Matrix Inverse / LU Decomposition (medium scale)

If `A · W = B`, then `W = A⁻¹ · B` (if A is invertible).

```python
# Using numpy — preferred method for medium-sized data
A = np.random.randn(5, 5)
A += 0.5 * np.eye(5)   # add to diagonal to reduce singularity risk
b = np.random.randn(5)

# np.linalg.solve uses LU decomposition under the hood
# More compute-efficient than computing inverse explicitly
W = np.linalg.solve(A, b)

# Alternatively (less efficient):
W = np.linalg.inv(A) @ b
```

**Why `solve` over `inv`?** LU decomposition is faster and more numerically stable than computing the full matrix inverse.

**Fails when:** Matrix is singular (determinant = 0, system not solvable).

---

### Method 3: Gradient Descent (large scale)

When data is too large to fit in RAM, or has too many features for linear algebra to be practical.

**Intuition:**
1. Start with random weights W
2. Compute predictions: `ŷ = W · X`
3. Compute loss (how wrong we are): e.g. Mean Squared Error
4. Compute gradient: `∂Loss/∂W` — which direction reduces loss?
5. Update weights: `W = W - α × gradient`
6. Repeat until loss converges to minimum

```python
# Conceptual pseudocode
W = np.random.randn(n_features)
for iteration in range(max_iters):
    y_pred = X @ W
    loss = np.mean((y_pred - y_actual) ** 2)  # MSE
    gradient = X.T @ (y_pred - y_actual) / n
    W = W - learning_rate * gradient
```

> **Initial weights don't matter.** Even random initialization converges — gradient descent takes large steps when far from minimum, smaller steps as it approaches.

**Why MSE is preferred over MAE:** MSE is a smooth quadratic curve — no kinks, always differentiable. MAE has a kink at 0, making gradient computation harder.

---

### When to Use Each Method

| Method | When to use | Speed |
|---|---|---|
| Gaussian Elimination | 2–5 variables, by hand | N/A |
| `np.linalg.solve` | Data fits in RAM | Fast for small-medium data |
| Gradient Descent | Large scale, can't fit in RAM | Slow for small data, fast for large |

---

## 4. Key Concepts in Linear Algebra

### Full Rank Matrix

A matrix is **full rank** if:
- All rows are linearly independent (no row is a combination of other rows)
- After Gaussian elimination, no row becomes all zeros

**Why it matters:** A full-rank square matrix is invertible. A non-full-rank matrix means the system of linear equations has no unique solution.

```python
# Check rank
rank = np.linalg.matrix_rank(A)
```

### Singular Matrix

A matrix is **singular** (non-invertible) when:
- Its determinant = 0
- `det(A) = a11×a22 − a12×a21 = 0` (for 2×2)

```python
det = np.linalg.det(A)
# det ≈ 0 → singular → system not solvable
```

**Examples of singular matrices:**
```
[1, 2]    → det = 1×4 - 2×2 = 0  → singular
[2, 4]

[1, 2]    → det = 1×1 - 2×2 = -3 → invertible
[2, 1]
```

### Adding to Diagonal for Stability

```python
# Add small value to diagonal to reduce chance of singularity
A += 0.5 * np.eye(n)
# The diagonal elements contribute to the determinant
# Adding to them reduces the probability of det = 0
```

This is a regularization technique — related to **Ridge regression** (which adds λI to the matrix before solving).

### LU Decomposition

Decomposes a matrix into:
- **L** = lower triangular matrix
- **U** = upper triangular matrix

```
A = L × U
```

Essentially formalizes Gaussian elimination into matrix operations. More efficient than computing the full inverse, especially for large matrices.

---

## 5. In-Place vs Out-of-Place Operations

A subtle but important performance difference:

```python
x = np.arange(10)

# Out-of-place — creates NEW memory allocation
x_out = x + 1
# x.id ≠ x_out.id → different memory addresses

# In-place — modifies existing memory, no new allocation
x += 1
# x.id stays the same before and after
```

```python
# Verify with id()
print(id(x))          # e.g. 1234
x += 1
print(id(x))          # still 1234 → same memory

x_out = x + 1
print(id(x_out))      # different number → new memory
```

> **Prefer in-place operations** (`+=`, `-=`, `*=`) when you don't need to keep the original. They avoid allocating new memory, which matters at scale.

---

## 6. Strides — Revisited

Strides tell NumPy how many bytes to skip to get to the next element in each dimension.

```python
x = np.arange(20)
x_reshaped = x.reshape(10, 2)

print(x.strides)          # (4,)   — 4 bytes per element (int32)
print(x_reshaped.strides) # (8, 4) — 8 bytes per row (2 ints), 4 bytes per col

# Slicing every other row doubles the row stride
x_slice = x_reshaped[::2]
print(x_slice.strides)    # (16, 4) — skip 2 rows at a time = 16 bytes
```

No data is copied — only strides metadata changes. This is what makes NumPy's slicing and reshaping memory efficient.

---

## 7. Principal Component Analysis (PCA)

### What is PCA?

PCA is a **dimensionality reduction** technique. It transforms high-dimensional data into fewer dimensions (principal components) while retaining as much of the data's variance (spread) as possible.

> **Common misconception:** PCA is NOT selecting the top K features. It creates NEW axes (principal components) that are linear combinations of all original features.

> **Common misconception 2:** PCA does not give you a prediction — it gives you transformed data. You still need a downstream model (logistic regression, KNN, etc.).

### Why Reduce Dimensions?

**Two problems with high dimensionality:**

1. **Compute** — 1000 features × 1000 features covariance matrix = 1M operations, before even fitting a model

2. **Curse of dimensionality** — In very high dimensions, all points appear equally distant from each other. Distance-based algorithms (like KNN) break down completely.

---

### PCA Step-by-Step

```
Input: X matrix (2000 samples × 40 features)
Goal:  Z matrix (2000 samples × 5 principal components)
```

**Step 1: Center and standardize the data**
```python
mu    = X.mean(axis=0)    # mean per feature
sigma = X.std(axis=0)     # std per feature
Xc    = (X - mu) / sigma  # shape: (2000, 40)
```

**Step 2: Compute the covariance matrix**
```python
# Covariance: how much does each feature vary with each other feature?
Cov = (Xc.T @ Xc) / (n - 1)   # shape: (40, 40)
# N-1 because 1 degree of freedom used in standardization
```

**Step 3: Eigendecomposition of the covariance matrix**
```python
eigenvalues, eigenvectors = np.linalg.eigh(Cov)
# eigenvalues:  shape (40,)   — how much variance each component captures
# eigenvectors: shape (40,40) — the principal component directions
```

**Step 4: Sort by eigenvalue (descending = most variance first)**
```python
sorted_idx    = np.argsort(eigenvalues)[::-1]  # descending
eigenvalues   = eigenvalues[sorted_idx]
eigenvectors  = eigenvectors[:, sorted_idx]
```

**Step 5: Select top K eigenvectors**
```python
K = 5
W = eigenvectors[:, :K]   # shape: (40, 5)
```

**Step 6: Transform original data into new space**
```python
Z = Xc @ W                # shape: (2000, 5)
# Now use Z as input to your downstream model
```

---

### Eigenvectors and Eigenvalues — Intuition

**By definition:** Eigenvectors are vectors that **do not change direction** when a transformation is applied to the matrix. They only scale.

**The scaling factor** is the eigenvalue.

**Example (whiteboard):**
- Rotate the XY plane 180° along Z-axis
- X-axis direction changes, Y-axis direction stays the same
- Y-axis is an eigenvector of this rotation transformation

**Why this matters for PCA:**
- We apply this concept to the **covariance matrix**
- Eigenvectors of the covariance matrix = directions that don't change when we "transform" the feature space
- These stable directions are the ones that best represent the data's spread
- Eigenvalues = how much variance each direction captures

```
Eigenvalue 1: 12.4  → PC1 captures 60% of variance
Eigenvalue 2:  4.1  → PC2 captures 20% → cumulative 80%
Eigenvalue 3:  1.0  → PC3 captures  5% → cumulative 85%
Eigenvalue 4:  1.0  → PC4 captures  5% → cumulative 90%
Eigenvalue 5:  1.0  → PC5 captures  5% → cumulative 95%
...
```

### Choosing K

```python
# Calculate cumulative explained variance
total_variance    = eigenvalues.sum()
explained_ratio   = eigenvalues / total_variance
cumulative_ratio  = np.cumsum(explained_ratio)

# Pick K where cumulative variance ≥ 95%
K = np.argmax(cumulative_ratio >= 0.95) + 1
```

> **Interview answer:** K is a hyperparameter chosen based on how much variance you want to explain. A common threshold is 95%. You validate by checking downstream model performance — if accuracy is poor, increase K.

### Eigenvectors vs SVD

| Method | Works on | Use case |
|---|---|---|
| **Eigendecomposition** | Square matrices only | Covariance matrix (always square) |
| **SVD** | Any matrix | Can apply directly to X without computing Cov first |

Both give you the same principal components — SVD is more numerically stable and is what `sklearn.decomposition.PCA` uses under the hood.

---

## 8. Full PCA Code (from scratch with NumPy)

```python
import numpy as np

def pca_from_scratch(X, k=5):
    n, d = X.shape   # 2000 samples, 40 features

    # Step 1: Standardize
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0)
    Xc    = (X - mu) / sigma

    # Step 2: Covariance matrix
    Cov = (Xc.T @ Xc) / (n - 1)   # (d, d)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Cov)

    # Step 4: Sort descending
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select top K
    W = eigenvectors[:, :k]        # (d, k)

    # Step 6: Project data
    Z = Xc @ W                     # (n, k)

    # Explained variance
    explained = eigenvalues[:k].sum() / eigenvalues.sum()
    print(f"Variance explained by {k} components: {explained:.1%}")

    return Z

# Usage
X = np.random.randn(2000, 40)
Z = pca_from_scratch(X, k=5)
print(Z.shape)  # (2000, 5)
```

---

## 9. Solving Linear Equations with NumPy

```python
import numpy as np

n = 5

# Create a 5x5 matrix (add to diagonal for stability)
A = np.random.randn(n, n)
A += 0.5 * np.eye(n)

# Target vector (equivalent to Y / labels)
b = np.random.randn(n)

# Solve for W (weights) using LU decomposition
W = np.linalg.solve(A, b)

print(W.shape)    # (5,) — one weight per feature

# Verify: A @ W should ≈ b
print(np.allclose(A @ W, b))   # True
```

---

## 10. Key Concepts Summary

| Concept | Key Point |
|---|---|
| `@` vs `np.dot` | Use `@` for matrix multiplication; `np.dot` behaves unexpectedly on 3D+ |
| Tensor matmul | Batch/depth dimensions must match; then inner dims must match |
| Linear model | Goal is to find weights W; not find predictions directly |
| Gaussian elimination | Manual, works for 2–5 variables only |
| `np.linalg.solve` | LU decomposition under the hood; use when data fits in RAM |
| Matrix inverse | Less efficient than `solve`; use `A⁻¹ @ b` only if you need the inverse itself |
| Gradient descent | For large scale data; iteratively reduces loss by adjusting weights |
| Full rank | All rows linearly independent; guarantees unique solution |
| Singular matrix | `det(A) = 0`; system not solvable; add λI to diagonal to reduce risk |
| In-place `+=` | Same memory address; no new allocation; preferred for performance |
| Strides | Bytes to skip per dimension; reshape/slice only changes strides, not data |
| PCA purpose | Dimensionality reduction — not feature selection |
| PCA ≠ prediction | PCA gives transformed data; still need downstream model |
| Eigenvectors | Vectors that don't change direction under transformation |
| Eigenvalues | Scaling factor of eigenvectors; represent amount of variance captured |
| Choosing K in PCA | Pick K where cumulative explained variance ≥ 95% |
| SVD vs Eig | SVD more numerically stable; used by sklearn's PCA |
| Curse of dimensionality | In high dimensions, all distances become equal; KNN breaks down |

---

## 11. Coming Up Next

- **Mathematics week** — calculus (chain rule, derivatives), probability, statistics
- **Gradient descent in detail** — partial derivatives, learning rate, convergence
- **Linear Regression** — building from scratch with NumPy
- **Logistic Regression** — sigmoid function derivation, binary classification
- **Full PCA lab** — variance explained plots, choosing K automatically
