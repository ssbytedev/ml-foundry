# Foundations & Linear Algebra — Quick Reference

*~10 min read. Concepts build on each other top to bottom.*

---

## Part I — Logic Basics

### Set
A collection of distinct objects.

- `{2, 5, 6}` → finite
- `{1, 2, 3, ...}` or `R` → infinite

> **Mental model:** a box of unique items.

### Proposition
A statement that is **true or false** — nothing else.

- ✅ "2 is positive." → True
- ❌ "5 < 3." → False
- 🚫 "What is your name?" → not a proposition
- 🚫 `x > 3` → not a proposition until `x` is fixed

### Function
`Input → Function → exactly 1 Output`

- `f(x) = x²`: `2→4`, `3→9` ✅
- `2→4` and `2→7` from the same rule ❌ (not a function)

> **Mental model:** vending machine — same button, same result, every time.

### Proof Techniques
Different ways to justify **P → Q**:

| Method | Idea | Symbol |
|---|---|---|
| Direct | Follow consequences forward | `P → ... → Q` |
| Contraposition | Prove the flip instead | `¬Q → ¬P` |
| Cases | Split into exhaustive scenarios | even / odd, etc. |
| Induction | Base case + domino effect | `P(1), P(n)→P(n+1)` |
| Contradiction | Assume ¬P, derive nonsense | `¬P → 0=1` ⇒ P true |

---

## Part II — Linear Algebra

### Vector
Any object you can **add** + **scale**. Arrows, number-lists, polynomials, matrices — all count.

`v = [3, 2]` → 3 right, 2 up

### Vector Space
A set closed under addition & scaling — you never "leave" it.

**Required properties:**
- ✅ Closed under `+`
- ✅ Closed under scalar `×`
- 0️⃣ Zero vector exists (`v + 0 = v`)
- ➖ Inverse exists (`v + (-v) = 0`)
- 🔀 Distributive: `a(u+v) = au+av`

> ⚠️ Subspaces (a space *inside* a space) only need to check closure — the rest is inherited.

### Linear Combination
`a·v₁ + b·v₂ + ...` — building new vectors from scaled pieces.

### Independence & Span
| Term | Meaning | Example |
|---|---|---|
| **Dependent** | A vector is redundant | `v₃ = v₁ + v₂` |
| **Independent** | Nothing is redundant | `[1,0]`, `[0,1]` |
| **Span** | Everything buildable from a set | span of `[1,0],[0,1]` = all of `R²` |

**Test for independence:**
`0v₁ + 0v₂ + ... = 0` should be the *only* way to get zero.

### Basis & Dimension
```
Basis = Spans everything + No redundancy
Dimension = # of vectors in a basis = # of independent directions
```

- `R²` basis → `[1,0], [0,1]` → dimension = 2
- `R³` → dimension = 3

### Norm (Length)
```
||v||₂ = √(x² + y²)
```
- `v=[3,4]` → `||v||=5` (Pythagorean theorem)
- **Norm = length.**

### Orthogonal / Orthonormal
| | Perpendicular? | Length 1? |
|---|:---:|:---:|
| Orthogonal | ✅ | — |
| Orthonormal | ✅ | ✅ |

Test: `u · v = 0` → orthogonal.
`[1,0]` & `[0,1]` → orthonormal ✅

> Orthonormal bases = clean axes → coefficients are just dot products. Great for projections, ML.

### Projection
"Shadow" of one vector on another direction:
```
projᵤ(v) = (v · u) u      (u = unit vector)
```

### Least Squares
When `Ax = b` has **no exact solution** (noisy data):
```
minimize ||Ax - b||²
```
→ find the *closest* achievable answer, not the exact one.

### Linear Regression
Least squares applied to fitting `y = mx + b`:
```
minimize Σ(predicted y − actual y)²
```
🔗 `Vectors → Least Squares → Linear Regression`

### Gradient
Points toward **steepest increase** of a function.
- `∇f` → uphill fastest
- `-∇f` → downhill fastest → basis of **gradient descent**

---

## 🧭 Big Picture

```
SET → PROPOSITION → FUNCTION → PROOF
                                  ↓
VECTOR → VECTOR SPACE → LINEAR COMBINATION
                                  ↓
              SPAN → INDEPENDENCE → BASIS → DIMENSION
                                  ↓
        NORM → ORTHOGONALITY → PROJECTION
                                  ↓
              LEAST SQUARES → LINEAR REGRESSION

FUNCTION → GRADIENT → −GRADIENT → GRADIENT DESCENT
```

## 📎 One-Line Cheat Sheet

| Term | In One Line |
|---|---|
| Set | Collection of objects |
| Proposition | True/false statement |
| Function | 1 input → 1 output |
| Vector | Addable + scalable object |
| Vector space | Safe zone for `+` and `×` |
| Span | Everything buildable |
| Independent | Nothing redundant |
| Basis | Minimal, complete building blocks |
| Dimension | # of basis vectors |
| Norm | Length |
| Orthogonal | Perpendicular |
| Orthonormal | Perpendicular + length 1 |
| Projection | Shadow onto a direction |
| Least squares | Closest possible fit |
| Linear regression | Best-fit line |
| Gradient | Steepest-increase direction |
