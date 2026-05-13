# Lecture 2 Notes

---

## Quick Recap from Lecture 1

- NumPy basics: vectorized operations, memory model, strides, reshaping
- Fancy indexing vs basic indexing (views vs deep copies)
- Why NumPy is ~100x faster than Python loops (compiled C, SIMD, cache, no interpreter)

---

## 1. Broadcasting

### What is Broadcasting?

Broadcasting is NumPy's ability to perform arithmetic operations on arrays of **different shapes** without creating extra copies of data. NumPy "imagines" (stretches) the smaller array to match the larger one — without actually allocating new memory.

> **Real-world analogy:** Like a radio broadcast from Delhi — the signal reaches everywhere without creating a physical copy of itself for each location.

### Case 1: Vector + Scalar

```python
a = np.array([1, 2, 3, 4])
result = a + 5   # broadcasts 5 across all 4 elements
# → [6, 7, 8, 9]
```

NumPy doesn't copy `5` four times. It just imagines it broadcasted across.

> **Note:** Scalars have no shape property — don't think about shapes for scalars, just know they always broadcast.

---

### Case 2: Vector + Vector (same shape)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b   # → [5, 7, 9]
```

Shapes match exactly → no issue.

---

### Case 3: Vector + Vector (different shape — error)

```python
a = np.array([1, 2, 3])    # shape: (3,)
b = np.array([4, 5, 6, 7]) # shape: (4,)
c = a + b   # → BroadcastingError!
```

NumPy can't figure out how to consistently stretch the data — 3 elements and 4 elements are incompatible.

---

### Case 4: Matrix + Vector

```python
A = np.array([[2, 3, 4],
              [5, 6, 7]])   # shape: (2, 3)

b = np.array([1, 2, 3])    # shape: (3,)

C = A + b
# NumPy imagines b repeated for each row:
# [[1, 2, 3],   → added to row 1
#  [1, 2, 3]]   → added to row 2
# Result: [[3, 5, 7], [6, 8, 10]]
```

This is **not** matrix multiplication. It's pure arithmetic broadcasting.

---

### Case 5: Matrix + Matrix (shapes with a 1)

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape: (2, 3)

B = np.array([[7, 8, 9]])   # shape: (1, 3)

C = A + B
# B broadcasts across both rows
# Result: [[8, 10, 12], [11, 13, 15]]
```

---

## 2. The Three Rules of Broadcasting

These rules are derived from common sense, not memorized:

### Rule 1: Align shapes from the RIGHT

When comparing two arrays, pad the shorter shape with 1s on the left before comparing.

```
A shape:    (2, 3)
B shape:       (3,)   →  treated as  (1, 3)
```

### Rule 2: Dimensions are compatible if they are EQUAL or one of them is 1

```
A: (2, 3)
B: (1, 3)
      ↑ ↑
      3 = 3 ✓ (match)
      2 ≠ 1 but one of them is 1 ✓ (broadcast)
```

### Rule 3: Missing dimensions are treated as 1

If A has more dimensions than B, B's missing leading dimensions are treated as 1.

```
A: (2, 3)    →   (2, 3)
B: (3,)      →   (1, 3)   ← missing dim treated as 1
```

---

### Broadcasting Rule Summary Table

| A shape | B shape | Compatible? | Result shape |
|---|---|---|---|
| `(3,)` | `(3,)` | ✅ | `(3,)` |
| `(4,)` | `(3,)` | ❌ | Error |
| `(2, 3)` | `(3,)` | ✅ | `(2, 3)` |
| `(2, 3)` | `(1, 3)` | ✅ | `(2, 3)` |
| `(2, 3)` | `(2, 1)` | ✅ | `(2, 3)` |
| `(2, 3)` | `(1, 4)` | ❌ | Error |
| `(1, 3)` | `(3, 1)` | ✅ | `(3, 3)` |
| `(2, 3, 4, 5)` | `(3, 5)` | ❌ | Error (4 ≠ 3) |
| `(2, 3, 4, 5)` | `(4, 5)` | ✅ | `(2, 3, 4, 5)` |

---

### Resultant Shape Rule

When broadcasting succeeds, the output shape is the **maximum** of each dimension:

```
A: (1, 3)     →   result: (3, 3)
B: (3, 1)
```

```python
a = np.array([[1, 2, 3]])     # (1, 3)
b = np.array([[1], [2], [3]]) # (3, 1)
c = a + b                     # (3, 3)
```

---

### Real-World Use Cases of Broadcasting

Broadcasting is used in virtually **every** ML operation:

| Operation | Broadcasting example |
|---|---|
| Standardization | `(X - mean) / std` — scalar broadcast |
| Feature normalization | Subtracting column means from a matrix |
| Adding bias in neural networks | `output = weights @ input + bias` |
| Row normalization | Dividing matrix by its row norms |
| Softmax | Subtracting max for numerical stability |

---

### Common Broadcasting Bug in Interviews

> "Shape mismatch bugs are some of the most common bugs in ML code — not model bugs, not algorithm bugs."

```python
# Bug: vector of wrong size
A = np.ones((3, 4))
b = np.ones(5)        # shape (5,) — doesn't match any dim of A
result = A + b        # BroadcastingError!

# Fix: match the shape
b = np.ones(4)        # shape (4,) — matches last dim of A
result = A + b        # ✅ works
```

---

## 3. `keepdims=True` — Why It Matters

When you reduce an array (e.g., taking a norm or sum along an axis), the result loses a dimension. This can break broadcasting.

```python
A = np.random.normal(size=(1000, 50))  # shape: (1000, 50)

# Without keepdims:
norm = np.linalg.norm(A, axis=1)       # shape: (1000,)   ← 1D vector
A / norm                               # BroadcastingError! (50 ≠ 1000)

# With keepdims=True:
norm = np.linalg.norm(A, axis=1, keepdims=True)  # shape: (1000, 1)
A / norm                               # ✅ broadcasts correctly → (1000, 50)
```

> **Rule of thumb:** Whenever you reduce along an axis and need to broadcast the result back onto the original array, use `keepdims=True`.

---

## 4. Norms (Vector Magnitude)

### Two Types of Norm

| Norm | Formula | Also called | Use case |
|---|---|---|---|
| **L2 norm** | `√(x₁² + x₂² + ... + xₙ²)` | Euclidean norm | Default in ML |
| **L1 norm** | `|x₁| + |x₂| + ... + |xₙ|` | Manhattan norm | Sparse data, robust regression |

```python
v = np.array([2, 3])

l2 = np.linalg.norm(v)           # √(4 + 9) = √13 ≈ 3.606
l1 = np.linalg.norm(v, ord=1)    # |2| + |3| = 5
```

### Row Norm vs Column Norm

| Type | Axis | Use case |
|---|---|---|
| **Row norm** (vector norm) | `axis=1` | Embeddings — normalize direction |
| **Column norm** (feature norm) | `axis=0` | Classical ML — standardize features |

---

## 5. Safe Norm (Numerical Stability)

```python
def safe_norm(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, 1e-12)
```

**Why `np.maximum(norm, 1e-12)`?**

If a row is all zeros, its norm = 0. Dividing by 0 gives `NaN` (Not a Number), which silently corrupts your model. By replacing 0 with a tiny positive number (`1e-12`), we prevent division by zero without meaningfully changing the result for non-zero rows.

> This is called a **safe norm** vs a **naive norm**.

---

## 6. Normalization vs Standardization — The Terminology Confusion

In the real world, ML engineers often use these terms interchangeably. Be aware:

| Term | What it technically means | Formula |
|---|---|---|
| **Normalization** | Min-max scaling to [0, 1] | `(x - min) / (max - min)` |
| **Standardization / Z-scaling** | Make mean=0, std=1 | `(x - μ) / σ` |

> Experienced ML engineers often say "normalize the data" when they actually mean "standardize" (z-scale). Be comfortable with this fuzziness.

### Why Normalize Features?

If salary is in the hundreds of thousands and age is 20–60, distance-based models (like KNN) will be dominated by salary. Standardization brings all features to the same scale.

```python
mu = np.mean(X)
sigma = np.std(X)
X_standardized = (X - mu) / sigma
# Result: mean ≈ 0, std ≈ 1
# 68% of values fall within ±1σ
# 95% within ±2σ
# 99% within ±3σ  (Gaussian distribution)
```

---

## 7. Loop vs Vectorized Operations — Benchmark

```python
import numpy as np
import time

X = np.random.normal(size=1_000_000)
mu = np.mean(X)
sigma = np.std(X)

# Python for loop
X_loop = np.empty_like(X)
t0 = time.time()
for i in range(X.shape[0]):
    X_loop[i] = (X[i] - mu) / sigma
loop_time = time.time() - t0

# Vectorized
t1 = time.time()
X_vec = (X - mu) / sigma
vec_time = time.time() - t1

print(f"Speedup: {loop_time / vec_time:.0f}x")
# Typical result: ~150x faster
```

---

## 8. Classification vs Regression

A brief clarification introduced during the lecture:

| Type | Target | Examples |
|---|---|---|
| **Regression** | Continuous numeric value | House price, salary prediction |
| **Classification** | Category / discrete label | Spam/not spam, low/medium/high |

### Classification sub-types:
- **Binary classification** — 2 classes (0/1, yes/no) → uses **Sigmoid** activation
- **Multi-class classification** — 3+ classes (low/medium/high, red/green/blue) → uses **Softmax** activation

---

## 9. Pairwise Distances (NumPy Trick)

### The Naive Approach (too slow)

For N points, computing all pairwise distances one by one = N² operations. At 1 million points: 1 trillion operations. Unusable.

### The NumPy Way — Using Linear Algebra

The squared Euclidean distance between points i and j can be expanded:

```
d²(i,j) = (a₂-a₁)² + (b₂-b₁)²
         = a₁² + b₁² + a₂² + b₂² - 2(a₁a₂ + b₁b₂)
         = ||xᵢ||² + ||xⱼ||² - 2(xᵢ · xⱼ)
```

This lets us compute ALL pairwise distances at once using matrix operations:

```python
def pairwise_distances(X):
    # X shape: (N, D)

    # Step 1: ||xᵢ||² for each row — element-wise square then sum
    sq = np.sum(X * X, axis=1, keepdims=True)  # shape: (N, 1)

    # Step 2: Cross term — 2 * X @ Xᵀ
    cross = X @ X.T                             # shape: (N, N) — dot product

    # Step 3: Combine using broadcasting
    # sq is (N,1), sq.T is (1,N) → broadcasts to (N,N)
    dist_sq = sq + sq.T - 2 * cross

    # Guard rail: diagonal should be 0 (numerical stability)
    np.fill_diagonal(dist_sq, 0)

    return np.sqrt(dist_sq)
```

> **Key insight:** `X * X` is element-wise multiplication. `X @ X.T` is dot product. These are completely different operations — don't confuse them.

### Why This Works

| Term | Computation | Type |
|---|---|---|
| `sq` | `X * X` then sum | Element-wise multiplication + reduction |
| `sq + sq.T` | Broadcasting `(N,1)` + `(1,N)` | Broadcasting → `(N,N)` |
| `X @ X.T` | Matrix dot product | Linear algebra |

No Python loops. Pure NumPy. Scales to large datasets efficiently.

---

## 10. Numerical Stability

### What is it?

Computers use finite-precision arithmetic. Very large or very small numbers can cause errors:

```python
1 + 2 == 3               # True ✓
0.1 + 0.2 == 0.3         # False! → 0.30000000000000004
```

This gets worse with:
- Very large numbers (integer/float overflow)
- Very small numbers (underflow to zero)
- Exponentials (`e^1000` = infinity)
- Logs of very small numbers
- Repeated recursive operations
- Square roots of near-zero numbers

### Numerical Stability in Softmax

**Naive softmax** — numerically unstable for large inputs:

```python
def naive_softmax(X):
    exp_X = np.exp(X)                              # e^1000 = overflow!
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)
```

**Stable softmax** — subtract max before exponentiating:

```python
def stable_softmax(X):
    X_norm = X - np.max(X, axis=1, keepdims=True)  # subtract max → values ≤ 0
    exp_X = np.exp(X_norm)                          # e^0 = 1, e^-1, e^-2 → manageable
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)
```

**Why is this valid?** Because softmax is a relative term — subtracting a constant from all values cancels out in numerator and denominator:

```
e^(z-c) / Σ e^(zⱼ-c)  =  e^z / Σ e^zⱼ   (the c cancels)
```

---

## 11. Softmax vs Sigmoid

| | Sigmoid | Softmax |
|---|---|---|
| Use case | Binary classification (2 classes) | Multi-class classification (3+ classes) |
| Output | Single probability (0 to 1) | Probability distribution over all classes |
| Formula | `1 / (1 + e^(-z))` | `e^zᵢ / Σ e^zⱼ` |

---

## 12. Key Concepts to Remember

| Concept | Key Point |
|---|---|
| Broadcasting | Arithmetic on different shapes — no memory copy, uses rules |
| Rule 1 | Align shapes from the right |
| Rule 2 | Dims compatible if equal OR one of them is 1 |
| Rule 3 | Missing dims treated as 1 |
| `keepdims=True` | Preserves dimensions after reduction — needed for broadcasting back |
| Safe norm | Use `np.maximum(norm, 1e-12)` to avoid division by zero |
| L1 vs L2 norm | Manhattan vs Euclidean — ML default is L2 |
| Row norm | For embeddings (preserve direction) |
| Column norm | For classical ML (feature standardization) |
| Numerical stability | Large numbers, small numbers, exponentials can break compute |
| Stable softmax | Subtract max before `exp()` to prevent overflow |
| Pairwise distance | Use `||x||² + ||y||² - 2(x·y)` — no loops needed |
| Speedup | Vectorized ops ~150x faster than loops on 1M elements |

---

## 13. Coming Up Next

- **Matrix multiplication** — dot product, how it differs from element-wise
- **Linear algebra fundamentals** — basis vectors, projections, eigenvalues
- **K-Nearest Neighbors** — applying pairwise distances in a real model
- **Dimensionality reduction** — PCA (why reduce dimensions before computing)
