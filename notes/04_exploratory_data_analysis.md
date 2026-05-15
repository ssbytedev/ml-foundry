# AI/ML Cohort — Lecture 4 Notes
**Topic: Exploratory Data Analysis (EDA)**

---

> **Quote from the instructor:**
> "Good ML engineers focus on model and metrics. Great ones focus more on data."

You will spend more time on data than on building models. EDA is that time well spent.

---

## 1. The EDA Framework — Overview

Before touching any model, work through these steps in order:

```
1. Shapes & Types          → what do I have?
2. Missing Values          → what's broken?
3. Duplicates              → what's redundant?
4. Summary Statistics      → what are the distributions?
5. Target Analysis         → is there class imbalance?
6. Univariate Analysis     → one variable at a time
7. Bivariate Analysis      → relationships between variables
8. Correlation Analysis    → multicollinearity risk?
9. Outlier Detection       → noise or signal?
10. Feature Engineering    → what can I build from what I have?
```

---

## 2. Variable Types — The First Distinction

Before doing anything, identify what type each column is:

| Type | Description | Examples |
|---|---|---|
| **Numerical** | Continuous or discrete numbers | Age, salary, sessions, revenue |
| **Categorical** | Fixed categories | Country, plan (basic/premium), yes/no |
| **Ordinal** | Categorical with order | Low/Medium/High income, star ratings |
| **Datetime** | Time-based | Signup date, event timestamp |

```python
# Select only numerical columns
num_cols = df.select_dtypes(include=np.number)

# Select only categorical columns
cat_cols = df.select_dtypes(exclude=np.number)

print(num_cols.columns)
print(cat_cols.columns)
```

---

## 3. Sanity Checks — Always Run These First

```python
# Step 1: Shape and types
print(df.shape)       # (rows, columns)
print(df.dtypes)      # data type of each column
print(df.columns)     # all column names
print(df.head())      # first 5 rows

# Step 2: Target column check
# Make sure it's 0/1, not "yes/no" or "True/False"
print(df['churned'].value_counts())
```

---

## 4. Missing Values Analysis

### Build a Missingness Table

```python
n = len(df)

# Count missing per column
n_missing = df.isna().sum()

# Percentage missing
pct_missing = n_missing / n

# Build table and sort descending
missing_table = pd.DataFrame({
    'n_missing':   n_missing,
    'pct_missing': pct_missing
}).sort_values('pct_missing', ascending=False)

print(missing_table)
```

### Three Types of Missing Data

This is one of the most important distinctions in ML. Don't randomly fill everything with the mean.

| Type | What it means | Example | Strategy |
|---|---|---|---|
| **MCAR** — Missing Completely At Random | No reason behind it | Pen drive corrupted, paper torn | Safe to impute with mean/median/mode |
| **MAR** — Missing At Random | Missingness depends on another observable variable | 14-year-olds have no salary, 95-year-olds have no job | Impute based on group (bucket by age, then fill) |
| **MNAR** — Missing Not At Random | Missingness is itself a signal | Rich people don't report salary | Add an `is_missing` flag column — the missingness IS a feature |

### How to Distinguish Between Them

- **MCAR** — no pattern, random rows lost
- **MAR** — do a slice analysis: look at which group of people has missing values. Age < 16 and missing salary → MAR
- **MNAR** — the missing people are likely the high-value ones. Validate by checking if `is_missing = 1` people behave differently in the target

```python
# Check for any missing values in any column
df.isna().any()

# Count missing per column
df.isna().sum()

# Add is_missing flag for MNAR columns
df['salary_is_missing'] = df['salary'].isna().astype(int)
```

> **Interview trap:** "Fill all nulls with the mean" is wrong. You'll be asked this. The correct answer depends on MCAR/MAR/MNAR.

---

## 5. Duplicates Check

```python
# Check for duplicate rows
df.duplicated().sum()

# Check for duplicate user IDs (business-level duplicates)
df['user_id'].duplicated().sum()
```

Duplicates don't add signal — they just inflate your data size without improving the model.

---

## 6. Summary Statistics

```python
# Numerical summary with custom percentiles
summary = df.describe(percentiles=[.05, .25, .50, .75, .95]).T
print(summary)
```

Custom percentiles (5th and 95th) help you see outlier ranges — the default 25/75 doesn't show you the tails.

---

## 7. Univariate Analysis

Analyze **one variable at a time**. Different approach depending on type.

### For Numerical Variables

```python
import matplotlib.pyplot as plt

# 1. Summary statistics
print(df['sessions_last_7_days'].describe())

# 2. Histogram
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].hist(df['sessions_last_7_days'], bins=20)
axes[0].set_title('Sessions Last 7 Days (right-skewed)')
axes[0].set_xlabel('sessions_last_7_days')
axes[0].set_ylabel('Frequency')

axes[1].hist(df['avg_session_minutes'], bins=20)
axes[1].set_title('Avg Session Minutes (heavy tail)')

axes[2].hist(df['tenure_days'], bins=20)
axes[2].set_title('Tenure Days')

plt.tight_layout()
plt.show()

# 3. Check skew
print(df['sessions_last_7_days'].skew())
# > 0 → right skewed (log transform candidate)
# < 0 → left skewed
```

### For Categorical Variables

```python
# Value counts (frequency)
print(df['country'].value_counts())
print(df['plan'].value_counts())

# Cardinality (number of unique categories)
print(df['country'].nunique())
```

### What to Look For in Univariate Analysis

| Signal | What it means | Action |
|---|---|---|
| Right/left skew | Distribution is not symmetric | Consider log transform or Box-Cox |
| Heavy tail | Many extreme values | Check outliers |
| Zero inflation | Column is mostly 0s | May need separate treatment |
| High cardinality | Too many categories | Hash encoding or target encoding |
| Class imbalance | Target variable lopsided | See Section 8 |

---

## 8. Target Analysis — Class Imbalance

### What is Class Imbalance?

When one class in your target variable is much more common than another.

**Real-world examples:**
- Fraud detection: 99% not fraud, 1% fraud
- Disease detection: 99% healthy, 1% sick
- Churn: 70% stay, 30% churn

```python
# Check churn rate (works because target is 0/1)
churn_rate = df['churned'].mean()
print(f"Churn rate: {churn_rate:.1%}")

# Count both classes
print(df['churned'].value_counts())
```

### Why It Matters — Affects All Three ML Pillars

| Pillar | Impact of class imbalance |
|---|---|
| **Data** | Need to stratify train/test split so both sets have equal class proportions |
| **Model** | Need to adjust hyperparameters (e.g. `class_weight`) |
| **Metrics** | Accuracy is misleading — use precision, recall, F1, AUC-ROC instead |

### The Classic Trap

```
99% healthy, 1% sick dataset.
Model predicts "healthy" for everyone.
Accuracy = 99%. Model is useless.
```

Always use precision/recall or AUC-ROC for imbalanced data, not accuracy.

### Three Remedies for Class Imbalance

1. **Data level** — collect more data for the minority class (oversampling/SMOTE)
2. **Model level** — stratify train/test split, tune `class_weight` hyperparameter
3. **Metrics level** — use precision, recall, F1, AUC-PR instead of accuracy

---

## 9. Bivariate Analysis

Analyze **relationships between two variables**.

### Numerical vs Numerical

```python
# Scatter plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(df['age'], df['avg_session_minutes'])
ax.set_xlabel('Age')
ax.set_ylabel('Avg Session Minutes')
ax.set_title('Age vs Avg Session Minutes')
plt.show()
```

### Numerical vs Categorical (Slice Analysis)

```python
# Churn rate by country
by_country = df.groupby('country')['churned'].mean()
print(by_country)

# Churn rate by plan
by_plan = df.groupby('plan')['churned'].mean()
print(by_plan)
```

Slice analysis reveals where your model will struggle — if US has 80% churn but India has 10%, that's a major business insight.

### Categorical vs Categorical

```python
# Cross-tabulation
pd.crosstab(df['country'], df['plan'])
```

### Graph Types Summary

| Variable pair | Graph |
|---|---|
| Numerical × Numerical | Scatter plot |
| Numerical × Categorical | Bar chart, box plot |
| Single numerical | Histogram |
| Single categorical | Bar chart (value counts) |

---

## 10. Correlation Analysis

### Covariance vs Correlation

> **AJ's golden rule:** "They are directionally the same thing."

| | Covariance | Correlation |
|---|---|---|
| Range | −∞ to +∞ | −1 to +1 |
| Formula | `Σ(xᵢ − μx)(yᵢ − μy) / n` | `Cov(X,Y) / (σX × σY)` |
| Interpretation | Direction only | Direction + magnitude |

**Correlation = normalized covariance.** Dividing by standard deviations brings it into [−1, 1].

### Linear vs Nonlinear Correlation

| Type | Algorithm | Use case |
|---|---|---|
| Linear | **Pearson's correlation** | When relationship is linear |
| Nonlinear | **Spearman's rank correlation** | When relationship is monotonic but not linear |

### Decision Rule

```
|correlation| > 0.8 → strong correlation → consider dropping one feature
                      (to avoid multicollinearity in linear models)
```

### Two Types of Correlation to Check

**1. Feature–Target correlation (dangerous)**

If a feature correlates directly with the target, it's a target leakage risk. Drop it.

```
refund_after_churn → directly predicts churned → DROP
```

**2. Feature–Feature correlation (multicollinearity)**

If two features correlate strongly with each other, they give the same signal. Keep one.

```
sqft_area and num_rooms → both measure house size → keep one
salary_usd and salary_inr → same thing, different units → keep one
```

> **Multicollinearity** is especially problematic in linear models (linear regression, logistic regression).

---

## 11. Outlier Detection

### Detection Methods

| Method | How | When to use |
|---|---|---|
| **Z-score** | Anything > 3σ from mean is an outlier | Normally distributed data |
| **IQR method** | Below 5th percentile or above 95th percentile | Skewed distributions |
| **Box plot** | Visual — whiskers show outlier range | Quick exploration |

```python
# Z-score method
from scipy import stats
z_scores = stats.zscore(df['salary'])
outliers = df[abs(z_scores) > 3]

# IQR method
q05 = df['salary'].quantile(0.05)
q95 = df['salary'].quantile(0.95)
outliers = df[(df['salary'] < q05) | (df['salary'] > q95)]
```

### Should You Remove Outliers?

**It depends on your use case.** There is no universal rule.

| Scenario | Decision |
|---|---|
| Normal neighborhood, one celebrity bought a $1M house | Remove — it skews predictions for normal residents |
| West Hollywood Hills where everyone is a celebrity | Keep — outliers ARE the population |
| Anomaly detection (fraud, engine failure) | Keep — outliers are what you're trying to detect |

> **Common mistake:** Naive ML engineers always remove outliers. Great ML engineers ask: "Are these outliers relevant to my use case?"

---

## 12. Feature Engineering from EDA Signals

EDA tells you what to engineer. Here's the mapping:

| EDA Finding | Feature Engineering Action |
|---|---|
| Right-skewed numerical feature | Apply log transform or Box-Cox |
| High cardinality categorical | Hash encoding or target encoding |
| MNAR missing values | Add `is_missing` flag column |
| Datetime column | Extract: day_of_week, is_weekend, month, hour |
| Two correlated features | Create ratio or difference: `paid / balance` |
| Two correlated features | Or combine: `3X + Y = Z`, then drop X and Y |

### Feature Engineering Example — Ratio Feature

```python
# Paid bill alone is misleading
# Balance alone is misleading
# Their ratio is the signal

df['payment_ratio'] = df['paid_amount'] / df['balance_amount']
# Aloc: 1000/1000000 = 0.001 → likely to default
# Ak:   100/200     = 0.5   → not going to default
```

### Feature Engineering Example — Bucketing / Binning

```python
# Hypothesis: people who churn early are different from loyal customers
bins   = [-1, 14, 60, 180, 10000]
labels = ['<2 weeks', '2w-2mo', '2mo-6mo', '>6mo']

df['tenure_bucket'] = pd.cut(
    df['tenure_days'],
    bins=bins,
    labels=labels
)

# Validate hypothesis
print(df.groupby('tenure_bucket')['churned'].mean())
# Result: <2 weeks → 60% churn, >6mo → 0% churn ✓ hypothesis confirmed
```

> **Note on bins:** `pd.cut` bins are exclusive on the left, inclusive on the right. Use `-1` not `0` as the start to include 0-day tenure.

### Feature Engineering Example — Date Features

```python
# Raw timestamp has no signal
# Engineered features have signal

df['event_date']    = df['event_timestamp'].dt.floor('D')
df['day_of_week']   = df['event_timestamp'].dt.dayofweek
df['is_weekend']    = df['day_of_week'].isin([5, 6]).astype(int)
df['hour']          = df['event_timestamp'].dt.hour
```

---

## 13. Encoding Categorical Variables — Preview

EDA informs your encoding strategy. Full details come at the model stage, but here's the decision framework:

### Label Encoding vs One-Hot Encoding

| Situation | Encoding | Reason |
|---|---|---|
| **Ordinal** categories (Low/Medium/High) | **Label encoding** (0, 1, 2) | Order matters — higher number = higher value |
| **Nominal** categories (Red/Blue/Green) | **One-hot encoding** | No natural order — label encoding would imply Green > Blue > Red |
| **High cardinality** (1000+ categories) | **Hash encoding or target encoding** | One-hot would create too many columns |

### One-Hot Encoding Example

```
Color column:  Red → [1, 0]
               Blue → [0, 1]
               Green → [0, 0]   ← implied by absence

Color_R  Color_B
  1        0      → Red
  0        1      → Blue
  0        0      → Green
```

Using N-1 columns for N categories to avoid multicollinearity (the "dummy variable trap").

---

## 14. Data Leakage — Revisited in EDA Context

EDA is where you **catch leakage before it enters the model.**

### Target Leakage — Identify During EDA

Any feature that would not be available at prediction time must be dropped.

```
Predicting churn:
✓ age, salary, plan, sessions, tenure
✗ refund_after_churn  ← only exists AFTER churn happens
✗ churned             ← that IS the target
```

**Why it matters:** If `refund_after_churn` is in training data, the model learns to rely on it. In production, you predict churn BEFORE it happens — so that column doesn't exist yet. The model fails completely.

### How to Spot Target Leakage in EDA

```python
# Check correlation of each feature with target
correlations = df.corr()['churned'].abs().sort_values(ascending=False)
print(correlations)

# Any feature with correlation near 1.0 is suspicious → investigate
```

---

## 15. Complete EDA Checklist

Run through this before building any model:

```
□ df.shape — check dimensions
□ df.dtypes — identify numerical vs categorical
□ df.head() — sanity check the data
□ Missing values table — count + percentage, sorted descending
□ Classify missing: MCAR / MAR / MNAR — choose strategy
□ Duplicates check — df.duplicated().sum()
□ Summary statistics — df.describe() with custom percentiles
□ Target distribution — mean(), value_counts(), class imbalance check
□ Histograms — for all numerical features
□ Skewness check — .skew() for each numerical column
□ Value counts — for all categorical features
□ Scatter plots — numerical vs numerical bivariate analysis
□ Bar charts — numerical vs categorical slice analysis
□ Correlation matrix — drop features with |corr| > 0.8
□ Outlier detection — z-score or IQR
□ Target leakage scan — any feature that reveals the answer?
□ Feature engineering — derive new features based on above signals
```

---

## 16. EDA → Model → Metrics Connection

```
EDA Finding                  → Affects
───────────────────────────────────────────────────────
Class imbalance              → Data: stratified split
                             → Model: class_weight param
                             → Metrics: use AUC, F1, not accuracy

Missing MNAR                 → Data: add is_missing flag

Right-skewed feature         → Data: log transform before scaling

Correlated features          → Model: multicollinearity in linear models
                             → Drop or combine features

Outliers                     → Data: remove or keep (use case dependent)
                             → Model: robust models vs sensitive models

Target leakage               → Data: drop the column, full stop
```

---

## 17. Key Terms for Interviews

| Term | Definition |
|---|---|
| **EDA** | Exploratory Data Analysis — understanding your data before modeling |
| **Univariate analysis** | Analyzing one variable at a time |
| **Bivariate analysis** | Analyzing the relationship between two variables |
| **Skewness** | Asymmetry in a distribution; >0 = right skewed, <0 = left skewed |
| **Cardinality** | Number of unique values in a categorical variable |
| **MCAR/MAR/MNAR** | Three types of missing data with different handling strategies |
| **Class imbalance** | When target classes are not equally represented |
| **Multicollinearity** | When two features are strongly correlated — problematic in linear models |
| **Pearson correlation** | Linear correlation between two variables; range [−1, 1] |
| **Spearman correlation** | Nonlinear (rank-based) correlation |
| **Covariance** | Direction of relationship; unbounded range |
| **Correlation** | Normalized covariance; bounded to [−1, 1] |
| **Z-score outlier** | Any value > 3 standard deviations from the mean |
| **IQR method** | Outlier threshold based on percentile cutoffs |
| **Feature engineering** | Creating new features from existing ones to improve signal |
| **Label encoding** | Assigning integer to each category (0, 1, 2) — for ordinal |
| **One-hot encoding** | Creating binary columns per category — for nominal |
| **Target leakage** | A feature that directly reveals the target — must be dropped |
| **Log transform** | Applied to right-skewed features to reduce the tail |
| **Binning/bucketing** | Converting a continuous variable into categories |

---

## 18. Coming Up Next

- **Remaining NumPy** — linear algebra operations, eigenvalues, solving linear equations
- **Linear models theory** — where the math behind linear/logistic regression comes from
- **Building models from scratch** — implementing without scikit-learn first
- **Mathematics week** — probability, statistics, linear algebra foundations
