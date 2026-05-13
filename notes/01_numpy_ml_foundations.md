# Lecture 1 Notes

---

## 1. Course Logistics

### Books
| Topic | Book |
|---|---|
| Mathematics | *Mathematics for Machine Learning* — Deisenroth et al. |
| Classical ML | O'Reilly ML book (Hands-on Machine Learning) |
| Deep Learning | *Deep Learning* — Goodfellow et al. |
| System Design | *Designing Data-Intensive Applications* — Kleppmann |

### Platform
- Labs hosted at **ai.s30.com** — paid platform, free for 1 year if enrolled in the cohort
- Each lab has a **theory section** + **exercises** (like LeetCode for ML)
- Solutions and test runners are built into the platform
- Instructor also posts supplementary **Substack notes** and **YouTube snippets**

---

## 2. What is Machine Learning?

> **Machine learning is designing algorithms that automatically extract valuable information from data.**

It is not just prediction — it is a broader process of learning patterns from data.

### Two Main Types

| Type | Data | Examples |
|---|---|---|
| Supervised | Labeled | Logistic Regression, Decision Trees, Random Forest, XGBoost, SVM |
| Unsupervised | Unlabeled | K-Means Clustering, PCA |

### Mental Model: Three Pillars of ML

Every ML project revolves around three silos:

```
┌─────────┐    ┌─────────┐    ┌──────────────────┐
│  Data   │    │  Model  │    │ Metrics/Learning  │
└─────────┘    └─────────┘    └──────────────────┘
```

**NumPy is the engine that powers all three.**

---

## 3. Why NumPy? (And Not Plain Python?)

NumPy is dramatically faster than Python for numerical computation. For a simple standardization operation on **1 million elements**, NumPy is **~100x faster** than a Python for loop.

### Four Reasons NumPy is Fast

| Reason | Explanation |
|---|---|
| **Compiled C code** | NumPy is written in C under the hood — no interpreter overhead |
| **SIMD** | Single Instruction Multiple Data — CPU processes multiple values in parallel |
| **No interpreter** | Data is a single type, so no dynamic type checking needed |
| **CPU Cache** | Compiled instructions and data can live in L1/L2 cache for ultra-fast access |

### Python List vs NumPy Array

| | Python List | NumPy Array |
|---|---|---|
| Memory | Non-contiguous, object-based | Contiguous memory block |
| Data types | Mixed types allowed | Single data type |
| Speed | Slow (interpreted) | Fast (compiled C + SIMD) |
| Operations | Manual for loops | Vectorized operations |

### SIMD Example
If your CPU is **128-bit** and your integers are **32-bit (int32)**:
- 128 ÷ 32 = **4 operations in parallel**
- This is the "multiple data" in Single Instruction Multiple Data

---

## 4. Vectorized Operations

Instead of looping:
```python
# Slow — Python for loop
result = []
for i in range(len(a)):
    result.append(a[i] + b[i])
```

NumPy does it in one vectorized call:
```python
# Fast — vectorized
result = a + b   # all elements added in parallel
```

Vectorized operations combine: **Compiled C + SIMD + No Interpreter + Cache** — all at once, like a waterfall hitting the entire array simultaneously.

---

## 5. Three Views of Vectors

You will constantly switch between these three perspectives in ML. Know them all.

### Physics Student Hat 🎓
- A vector has **magnitude** (also called norm) and **direction**
- Used for: cosine similarity, embeddings, LLMs
- Example: two words with similar meaning point in similar directions in embedding space

### Computer Science Student Hat 💻
- A vector is just a **list of numbers**
- Used for: implementation, NumPy arrays
- No notion of direction or magnitude at this level

### Mathematics Student Hat ✏️
- A vector is an object with two properties:
  1. Can be **multiplied by a scalar**
  2. Two vectors can be **added together**
- All matrix operations derive from these two rules

> **Key insight:** On the whiteboard, use the Physics hat to represent vectors. For operations, use the Math hat. For implementation, use the CS hat (NumPy).

---

## 6. Vectors in ML — Practical Context

In a tabular ML dataset (e.g. loan approval):

| Name | Salary | Age | Years Experience | Loan Approved? |
|---|---|---|---|---|
| Omar | 50k | 28 | 3 | 0 |
| Nitish | 80k | 35 | 8 | 1 |

- Each **row** = a sample (also called observation)
- Each **column** = a feature (also called attribute)
- Each **column/feature** is a **1D vector** (a NumPy array)
- The entire table is a **matrix**

**Axis terminology:**
- Operations down the rows → **axis=0**
- Operations across the columns → **axis=1**

---

## 7. NumPy Basics

### 7.1 Importing NumPy
```python
import numpy as np
import time
```

### 7.2 Creating Arrays

```python
# 1D array (vector)
a = np.array([1, 2, 3])
print(a.shape)  # (3,)  ← note: just a single number, no second dimension

# 2D array (matrix)
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)  # (2, 3)  ← 2 rows, 3 columns

# Range
np.arange(100)          # 0 to 99
np.arange(0, 10, 2)     # [0, 2, 4, 6, 8] — step of 2

# Zeros and ones
np.zeros(5)             # 1D vector of 5 zeros
np.zeros((5, 5))        # 5x5 matrix of zeros — note the tuple!
np.ones((2, 3))         # 2x3 matrix of ones

# Identity matrix
np.eye(3)               # 3x3 identity matrix (diagonal = 1)

# Linear spacing
np.linspace(0, 1, 5)    # [0.0, 0.25, 0.5, 0.75, 1.0]

# Random normal
np.random.normal(size=(3, 4))   # 3x4 matrix of random values
```

### 7.3 Shape — Vectors vs Matrices

> **Critical distinction:** `(4,)` and `(4,1)` are NOT the same in NumPy.

| Shape | What it is | Looks like |
|---|---|---|
| `(4,)` | 1D vector of 4 elements | `[1, 2, 3, 4]` |
| `(4, 1)` | Matrix: 4 rows, 1 column | `[[1], [2], [3], [4]]` |
| `(1, 4)` | Matrix: 1 row, 4 columns | `[[1, 2, 3, 4]]` |

This distinction matters because **vector operations** and **matrix operations** behave differently, and **broadcasting** rules depend on shape.

---

## 8. Reshaping — Views, Not Copies

```python
array_1d = np.arange(100)         # shape: (100,)
array_2d = array_1d.reshape(10, 10)  # shape: (10, 10)

array_3d = np.arange(60).reshape(4, 5, 3)  # shape: (4, 5, 3)
# total elements must match: 4 × 5 × 3 = 60 ✓
```

### The Important Gotcha: Reshape Creates a View

```python
array_1d = np.arange(20).reshape(10, 2)

array_2d[0, 1] = 9999
print(array_1d)  # array_1d is ALSO changed!
```

**Why?** Reshape does not copy data. It only creates **strides** — metadata that tells NumPy how to step through the original 1D block of memory.

```
Original 1D memory at address 0x420:  [1, 2, 3, 4, 5, 6, 7, 8]
                                              ↑
                           Reshape just adds strides: (16 bytes per row, 4 bytes per element)
                           No new data is created.
```

```python
print(array_2d.strides)  # e.g. (40, 4) → row=40 bytes, element=4 bytes
```

> Even 3D, 4D, 30D arrays are stored as a **single 1D block** in memory with stride metadata on top.

---

## 9. Data Types

```python
base = np.arange(1_000)          # default: int32
print(base.dtype)                 # int32

# Specify dtype at creation
arr_float = np.arange(1_000, dtype=np.float64)

# Convert after creation
arr_int   = base.astype(np.int32)
arr_float = base.astype(np.float64)
```

**Why it matters:** Choosing the right dtype affects memory usage and computation speed. A float64 array takes 2× more memory than float32.

---

## 10. Indexing and Slicing

### Basic Indexing (creates a view)
```python
a = np.arange(20).reshape(10, 2)

# Slicing rows and columns
a[1:3, :]       # rows 1 and 2, all columns
a[:, 1:3]       # all rows, columns 1 and 2
a[::2]          # every alternate row

# Get element at row 1, col 1
a[1, 1]         # ← basic indexing — returns a VIEW
```

### Fancy / Chained Indexing (creates a deep copy)
```python
a[1][1]         # ← chained indexing — returns a COPY
```

> **Rule:** Use `a[row, col]` (comma notation) in NumPy, not `a[row][col]`. The comma version is basic indexing (view), the chained version is fancy indexing (copy). This matters when you want to modify data in place.

### Modifying via slice propagates to original
```python
a_slice = a[::2]
a_slice[0] = -9999
print(a)        # original array is also changed!
```

---

## 11. Key Concepts to Remember

| Concept | Key Point |
|---|---|
| Vectorized ops | ~100x faster than Python for loops for numerical work |
| SIMD | CPU width ÷ data type size = parallel operations per instruction |
| Reshape | Creates a view (strides), not a deep copy |
| Basic indexing `[r, c]` | Returns a view — modifying it changes the original |
| Fancy indexing `[r][c]` | Returns a deep copy — modifying it does NOT change original |
| Shape `(4,)` vs `(4,1)` | Totally different — vector vs matrix, matters for broadcasting |
| dtype | Affects memory and speed; default is int32 or float64 |
| NumPy fits | Data (pandas built on it), Model (all linear algebra), Metrics |

---

## 12. Coming Up Next

- **Broadcasting** — operating on arrays of different shapes without copying data
- **Vector and matrix operations** — dot product, matrix multiplication
- **Loop vs vectorized benchmark** — measuring the 100x speedup in practice
- **Feature engineering** — manipulating features for ML models
- **Probability and statistics** — foundation for ML algorithms
