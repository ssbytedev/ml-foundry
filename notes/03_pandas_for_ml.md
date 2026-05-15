# AI/ML Cohort — Lecture 3 Notes
**Topic: Pandas for Machine Learning**

---

## Quick Recap

The ML mental model has three pillars: **Data → Model → Metrics/Learning**

Before feeding data into a model, the data pillar requires four steps:
1. **Extract** — get the data
2. **Transform** — reshape, compute new columns
3. **Scale/Standardize** — normalize features
4. **Encode** — convert categorical variables to numbers

**Pandas** is the primary tool for all four steps.

---

## 1. What is Pandas?

Pandas is a Python library built **on top of NumPy**. Everything in it is vectorized.

| Concept | Pandas term | Under the hood |
|---|---|---|
| Table | DataFrame | Collection of Series |
| Column | Series | NumPy array |
| Row | Index entry | — |

```python
import pandas as pd

df = pd.DataFrame({
    'user_id': [101, 102, 103, 104],
    'country': ['India', 'US', 'India', 'Canada'],
    'revenue': [5, 6, 7, 10]
})
```

### Always Start With These Three

```python
df.dtypes    # check data types of all columns
df.shape     # (rows, columns)
df.head()    # first 5 rows
```

> **Interview tip:** Always check these three before doing anything else with a new dataset.

---

## 2. Pandas vs SQL vs Python

| | Python (for loops) | SQL | Pandas |
|---|---|---|---|
| Speed | Slow | Fast | Fast (vectorized) |
| Data types supported | Any | Relational only | CSV, JSON, Excel, SQL, etc. |
| Transformations | Manual | Basic | Full (encoding, scaling, etc.) |
| In-memory | Yes | No | Yes |
| Large data | Struggles | Depends | Use chunking or Spark |

### Handling Large Data in Pandas

- **Chunking** — process data in batches: `pd.read_csv(..., chunksize=10000)`
- **Spark** — for terabyte-scale data
- **Polars** — modern, faster alternative to pandas (syntax differs, mental model same)

---

## 3. Random Seed — Why It Matters

```python
rng = np.random.default_rng(seed=0)
```

Seeding ensures **reproducibility**. The random number generator uses:

```
xₙ = (a × xₙ₋₁ + c) mod m
```

If you always start from the same seed, the sequence of "random" numbers is always identical. This means your experiments are reproducible — same results every run.

---

## 4. Synthetic Data

Building your own synthetic data is a valuable skill:

**Why make synthetic data?**
- Real data is expensive to collect
- Helps you understand the data's structure deeply
- Useful when you don't have access to real data

**Example — event probabilities for e-commerce:**
```python
events = ['view', 'add_to_cart', 'purchase']
probs  = [0.65,   0.25,          0.10]
# Must sum to 1.0
```

```python
# replace=True because one user can appear many times
user_ids = rng.choice([101, 102, 103, 104], size=30, replace=True)
event_types = rng.choice(events, size=30, replace=True, p=probs)
```

---

## 5. `.loc` vs `.iloc` — The Right Way to Index

Never use chained indexing `df['col'][idx]` in pandas — it may or may not modify the original (unpredictable behavior). Always use `.loc` or `.iloc`.

| | `.loc` | `.iloc` |
|---|---|---|
| Based on | **Labels** (column names, index labels) | **Integers** (position) |
| Slicing end | **Inclusive** | **Exclusive** |
| Use case | Most common — you know column names | When you only know position |

```python
# .loc — label based
df.loc[0, 'country']           # row 0, column 'country'
df.loc[0:2, 'country']         # rows 0,1,2 (inclusive of 2)

# .iloc — integer position based
df.iloc[0, 1]                  # row 0, col index 1
df.iloc[0:2, 1]                # rows 0,1 (exclusive of 2)
```

**Why `.loc`/`.iloc` over `df['col'][idx]`?**
- Chained indexing = fancy indexing in NumPy = **deep copy**
- `.loc`/`.iloc` = basic indexing = **view** (modifications propagate correctly)
- Avoids the classic pandas bug where you change a value but the original DataFrame doesn't update

---

## 6. Boolean Masking / Filtering

Just like NumPy — no for loops.

```python
# Create a boolean mask
mask = users['country'] == 'India'   # Series of True/False

# Apply the mask
india_users = users[mask]            # only India rows
```

```python
# Mask to fix data — set amount=0 for non-purchase events
mask = events['event_type'] != 'purchase'
events.loc[mask, 'amount'] = 0
```

---

## 7. `.copy()` — Avoiding Unexpected Mutations

When you slice or filter a DataFrame, you may get a view or a copy depending on the operation — pandas behavior here is inconsistent across versions.

```python
# Risky — may or may not mutate original
purchases = events[events['event_type'] == 'purchase']

# Safe — guaranteed separate copy
purchases = events[events['event_type'] == 'purchase'].copy()
purchases['event_date'] = purchases['event_timestamp'].dt.floor('D')
```

> **Rule:** Whenever you create a sub-DataFrame you plan to modify, always call `.copy()`.

---

## 8. Adding Columns

```python
# Add a boolean column
events2 = events.copy()
events2['is_purchase'] = (events2['event_type'] == 'purchase').astype(int)
# 0 = not purchase, 1 = purchase
```

---

## 9. GroupBy

### Basic GroupBy

```python
# Count events per user
events.groupby('user_id')['event_type'].count()
# or
events.groupby('user_id')['event_type'].size()
```

### GroupBy with Aggregate (multiple columns at once)

```python
user_features = events2.groupby('user_id').agg(
    num_events    = ('event_type',   'count'),
    num_purchases = ('is_purchase',  'sum'),
    total_revenue = ('amount',       'sum')
)
```

**Why `.agg()` is efficient — under the hood:**

GroupBy creates an internal **dictionary-like object** indexed by group key:
```
{
  101: {'event_type': [...], 'amount': [...], 'is_purchase': [...]},
  102: {...},
  ...
}
```

When you call `.agg()`, all aggregations pull from this **single object** — the data is only scanned **once**. This is why it's faster than doing multiple separate group-bys.

> **Interview question:** "What is the time complexity of a groupby-agg?" → O(n) — one pass to build the group object, then O(1) lookups per group per aggregation.

---

## 10. `reset_index()`

After a `groupby`, the group keys become the index. This can cause unexpected index values (e.g., index 3, 7, 12 instead of 0, 1, 2).

```python
user_features = events2.groupby('user_id').agg(...).reset_index()
# Now index is clean: 0, 1, 2, 3...
```

---

## 11. `transform()` — Window-style Operations

`groupby().transform()` keeps the **same number of rows** as the original DataFrame — unlike `agg()` which collapses to one row per group.

```python
# Add a column showing total event count for that user on every row
events2['user_event_count'] = (
    events2.groupby('user_id')['event_type'].transform('size')
)
```

| User | Event | user_event_count |
|---|---|---|
| 101 | view | 6 |
| 101 | purchase | 6 |
| 102 | view | 4 |

> Equivalent to a **window function** in SQL (like `COUNT(*) OVER (PARTITION BY user_id)`).

---

## 12. Joins and Merges

### Types of Joins

| Join type | Returns |
|---|---|
| `inner` | Only rows with matching keys in both tables |
| `left` | All rows from left + matching from right (NaN if no match) |
| `right` | All rows from right + matching from left (NaN if no match) |
| `outer` | All rows from both tables |

```python
merged = users.merge(
    user_features,
    on='user_id',
    how='left'
)

# Fill NaN values after join
merged.fillna(0, inplace=True)
```

---

## 13. Row Explosion — A Critical Bug

**What is it?** When you join two tables and both sides have multiple rows for the same key, you get an **M × N** explosion in row count.

**Example:**

```
Left table (phones):          Right table (orders):
user_id | phone               user_id | order_id
101     | 007                 101     | 1
101     | 008                 101     | 2

Inner join on user_id → 4 rows! (2 × 2)
```

```
101 | 007 | 1
101 | 007 | 2
101 | 008 | 1
101 | 008 | 2
```

**How to fix it:** Drop duplicates on the many-side before joining.

```python
# Keep only first phone per user before joining
right_deduped = right.drop_duplicates(subset='user_id', keep='first')
merged = left.merge(right_deduped, on='user_id', how='left')
```

> **Interview question:** "Your DataFrame unexpectedly has more rows after a merge — what happened?" → Row explosion from a many-to-many join.

---

## 14. Data Leakage — Most Important Concept in This Lecture

### What is Data Leakage?

When information that **should not be available** during model training leaks into the training process — causing the model to appear to perform well but fail in production.

### Two Types

#### Type 1: Target Leakage
A feature column directly reveals or encodes the target variable.

```
Features: plan, salary, age, last_payment, refund_after_churn ← LEAKAGE
Target:   churn (yes/no)
```

`refund_after_churn` directly tells you the answer — the model learns to just look at that column. It gets 100% accuracy in training, 0% in the real world.

#### Type 2: Feature Leakage (Train/Test Contamination)
Test data leaks into training data — or statistics computed on the full dataset are used before the train/test split.

```python
# WRONG — data leakage!
df['age'].fillna(df['age'].mean())   # mean includes test set data

# CORRECT — split first, then compute stats only on train
X_train, X_test = train_test_split(df)
train_mean = X_train['age'].mean()
X_train['age'].fillna(train_mean)
X_test['age'].fillna(train_mean)    # use train mean for test too
```

> **Interview rule:** Always split your data first, then compute any statistics (mean, median, std) on the **training set only**.

### What Does Leakage Lead To? — Overfitting

**Overfitting** = model performs well on training data, poorly on real-world data.

Causes:
- Target leakage (model memorizes answers)
- Feature leakage (model has seen test data)
- Model too complex (fits noise in training data)

```
Overfit model on whiteboard:
Data points:  * . * . . * .
Overfit line: wiggles through every single point
Good fit:     smooth curve that generalizes
```

---

## 15. Cumulative Sum + Shift — Leakage-Safe Window Features

A common task: for each event row, count how many purchases **strictly before** this row (for that user).

```python
# Step 1: Sort by user_id then event_timestamp
events3 = events2.sort_values(['user_id', 'event_timestamp']).reset_index(drop=True)

# Step 2: Cumulative sum per user
events3['cum_purchases'] = events3.groupby('user_id')['is_purchase'].cumsum()

# Step 3: Shift by 1 — move current row's count to next row
# (so current row reflects state *before* this event)
events3['purchases_before'] = events3.groupby('user_id')['is_purchase'].cumsum().shift(1)

# Step 4: Fill NaN in first row of each user with 0
events3['purchases_before'] = events3['purchases_before'].fillna(0)
```

**Why shift?** Without shifting, the cumsum at row N includes row N's own purchase — which is data leakage (you're using the current event to predict itself).

---

## 16. Vectorized vs Apply — Performance

`apply()` with a lambda is a for loop in disguise. Always prefer vectorized operations.

```python
# Slow — apply (non-vectorized)
t1 = time.time()
mask_apply = users['country'].apply(lambda x: x == 'India')
apply_time = time.time() - t1

# Fast — vectorized boolean mask
t2 = time.time()
mask_vec = users['country'] == 'India'
vec_time = time.time() - t2

# Result: vectorized is ~3x faster even on a small DataFrame
# On large DataFrames, the gap is even bigger
```

**Speed hierarchy (fastest to slowest):**
```
Vectorized NumPy/Pandas operations
    ↓ ~3–10x slower
List comprehensions / lambda
    ↓ ~2–5x slower
Python for loops
```

---

## 17. Key Pandas Methods Reference

| Method | What it does |
|---|---|
| `df.dtypes` | Data types of all columns |
| `df.shape` | (rows, columns) |
| `df.head(n)` | First n rows |
| `df.isna()` | Boolean mask of null values |
| `df.fillna(val)` | Fill nulls with a value |
| `df.loc[row, col]` | Label-based indexing |
| `df.iloc[row, col]` | Integer-based indexing |
| `df.copy()` | Make a deep copy |
| `df.reset_index()` | Reset index to 0,1,2... |
| `df.sort_values(col)` | Sort by column |
| `df.groupby(col)` | Group rows by column |
| `df.agg({})` | Multiple aggregations at once |
| `df.transform(fn)` | Apply function, keep original shape |
| `df.merge(df2, on, how)` | Join two DataFrames |
| `df.drop_duplicates(subset)` | Remove duplicate rows |
| `series.cumsum()` | Cumulative sum |
| `series.shift(n)` | Shift values by n rows |
| `series.apply(fn)` | Apply function row by row (slow) |
| `df.astype(dtype)` | Change data type |

---

## 18. Key Concepts to Remember

| Concept | Key point |
|---|---|
| Pandas = NumPy | Series are numpy arrays — vectorized under the hood |
| `.loc` vs `.iloc` | Label vs integer; `.loc` inclusive, `.iloc` exclusive |
| `.copy()` | Always use when modifying a slice to avoid mutation bugs |
| GroupBy internals | One pass over data → dictionary → O(1) aggregation lookups |
| `reset_index()` | Cleans up index after groupby/sort |
| `transform()` | Like a SQL window function — preserves row count |
| Row explosion | Many-to-many joins multiply row count — fix with `drop_duplicates` |
| Target leakage | Giving the model the answer — 100% train accuracy, 0% real-world |
| Feature leakage | Test data leaking into train — always split before computing stats |
| Overfitting | Model memorizes training data including noise — fails in production |
| Apply is slow | Use vectorized boolean masks instead of `apply(lambda...)` |
| Polars | Modern pandas alternative — faster, different syntax, same mental model |

---

## 19. Coming Up Next

- **Back to NumPy** — completing remaining NumPy exercises
- **Linear algebra fundamentals** — matrix multiplication, dot products, projections
- **Feature engineering** — encoding categorical variables, creating new features
- **Train/test splits** — properly avoiding data leakage in practice
- **First ML model** — K-Nearest Neighbors using the distance concepts from lecture 2
