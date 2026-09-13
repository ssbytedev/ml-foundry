# Foundations & Linear Algebra 

## Sets: The First Data Structure

A set is just a collection of objects.

- `{2, 5, 6}` — a finite set
- All real numbers, or all positive integers — an infinite set

That's it. So why does a data structure this simple matter for AI?

Because a set is the cleanest way to describe "a group of things without caring about order or duplicates," and that idea shows up everywhere under the hood:

- A **vocabulary** in an LLM is a set — every unique token the model can produce.
- A **training dataset**, before you care about sequence or batching, is a set of examples.
- **Embedding spaces** are built on top of vector spaces, and a vector space is, first and foremost, a *set* with rules attached to it (more on that below).
- When a model retrieves "similar documents," it's really selecting a *subset* of a larger set based on some rule.

Sets are the scaffolding you don't notice — like rebar in concrete. You don't see it in the finished wall, but nothing stands up without it.

---

## Propositions: Teaching a Machine to Be Right or Wrong

A proposition is a statement that is definitively true or false. "2 is a positive integer" — true. Nothing fuzzy about it.

This feels disconnected from AI, which seems to live in a world of probabilities and "maybe." But propositional thinking is exactly what makes a model's output checkable and its behavior reasonable about:

- **Loss functions** are built from statements like "the model's prediction equals the correct label" — a proposition that's true or false, which then gets turned into a number to optimize.
- **Reasoning and chain-of-thought** in modern LLMs is, at its core, chaining propositions together — if this is true, and this follows from it, then that must be true.
- **Formal verification** of neural networks (proving a model won't behave a certain way) is applied propositional and predicate logic, dressed up.

You don't need propositions to *build* a neural net. You need them to reason clearly about what it's doing — which turns out to matter just as much.

---

## Proof Methods: How You'd Convince a Skeptic

A handful of templates cover almost every proof you'll ever write:

- **Direct proof** — show P leads straight to Q.
- **Contraposition** — instead of proving P → Q, prove the equivalent "not Q → not P."
- **Proof by cases** — split the input into scenarios and handle each one.
- **Induction** — prove the base case, assume it holds up to n−1, then show it holds for n.
- **Contradiction** — assume the opposite is true, show that leads somewhere impossible, so the opposite must be false.

Again: where does this show up in a chatbot? Not in the weights themselves — but in everything *around* them:

- Proving an **algorithm terminates** or runs in a certain time (induction shows up constantly in analyzing training loops and search algorithms).
- Proving a **theorem in optimization** — for instance, that gradient descent converges under certain conditions — usually leans on contradiction or induction.
- **Proof by cases** is the literal skeleton of how engineers reason about edge cases in model behavior: what happens if the input is empty, adversarial, out-of-distribution, and so on.

Think of these proof methods less as "math homework" and more as five different lenses for building an airtight argument — which is exactly what you need when you're claiming an algorithm will always behave a certain way.

---

## Linear Algebra: The Actual Machinery

This is where things stop being background theory and start being the literal substance of AI models.

**Linear algebra is the study of finite-dimensional vector spaces.** Every embedding, every weight matrix, every layer of a neural network is an object living inside this framework.

### What Makes Something a Vector Space?

A vector space is a set with addition and scalar multiplication defined on it, such that:

- It's **closed under addition** — add two vectors, you get another vector in the same space.
- It's **closed under scalar multiplication** — scale a vector, you stay in the space.
- It has an **additive identity** — a zero vector that changes nothing when added.
- Every vector has an **additive inverse** — something that cancels it out to zero.
- **Distributive properties** hold for both scalar multiplication and vector addition.

This might look like dry bookkeeping, but it's the guarantee that lets you do arithmetic on meaning. When you add two word embeddings, or scale an image embedding, or average a batch of vectors — you're relying on these exact rules holding true, whether you're in 2 dimensions or 2,000.

### Linear Combinations

A linear combination is just a sum of scaled vectors — `a·v1 + b·v2 + c·v3...`

This is the core move behind almost every operation in a neural network. A layer's output is a linear combination of its inputs (before the nonlinearity gets applied). Attention is a weighted — that is, scaled and summed — combination of value vectors. Once you see "linear combination," you start seeing it everywhere.

---

## Measuring Distance: Norms

A norm is a distance function — it answers *"how far is this vector from the origin?"*

In 1D, absolute value gives you distance. But that stops working once you leave the number line. In 2D and beyond, you need something more general — which is exactly what a norm provides.

The **L2 norm (Euclidean norm)** is the most familiar one: the straight-line distance between two points, the "as the crow flies" measurement. It's the metric behind most nearest-neighbor search and similarity comparisons in embedding spaces — closer vectors mean more similar meaning, and L2 is often the ruler used to measure "closer."

---

## Orthogonal and Orthonormal Vectors

Two vectors are **orthogonal** if they sit at 90° to each other. That single geometric fact carries a big consequence: orthogonal vectors are **linearly independent** — neither one can be built out of the other.

This matters practically because orthogonal directions are the easiest building blocks for describing *any* other vector in the space — you can combine them without their influences tangling together.

**Orthonormal vectors** are orthogonal vectors that are also unit length (length 1). The special thing about an orthonormal set is that it gives you the cleanest possible coordinate system: projecting onto one axis doesn't leak into another, and computing coordinates in that basis becomes simple dot products instead of messy algebra. This is why so much of numerical linear algebra (QR decomposition, PCA, attention mechanisms) tries to work with orthonormal bases whenever possible — they make otherwise painful computations almost trivial.

---

## Subspaces

A subspace is a subset of a vector space that satisfies just three of the original rules:

- Contains the **zero vector**
- **Closed under addition**
- **Closed under scalar multiplication**

Notice what's missing compared to a full vector space: it doesn't need to independently prove additive identity, additive inverse, or distributivity — those get inherited for free from the larger space it sits inside.

Not every subset qualifies. A subspace has to be "self-contained" under those two operations — you can't step outside it by adding or scaling.

Why care? Because in ML, a lot of what a model "learns" is effectively discovering a low-dimensional subspace that captures the meaningful variation in high-dimensional data — this is the entire idea behind dimensionality reduction techniques like PCA.

---

## Linear Dependence and Independence

- **Linearly dependent**: you can write one vector in the set as a combination of the others. It's redundant — it's not adding new information.
- **Linearly independent**: no vector in the set can be built from the others. Each one contributes something the rest can't.

In practical terms, linear independence is what "no wasted dimensions" looks like. If two features in your dataset are linearly dependent, one of them is dead weight — it's not giving your model new signal.

---

## Basis and Dimension

A **basis** is a set of vectors that is:

1. **Linearly independent**, and
2. **Spans** the space — meaning every vector in the space can be written as some linear combination of the basis vectors.

The number of vectors in a basis is the **dimension** of the space.

This is the concept quietly running the show every time someone says a model has a "768-dimensional embedding" or a "4096-dimensional hidden state." Those numbers are literally the dimension of the vector space the model's internal representations live in — the count of independent directions needed to describe everything the model can represent at that layer.

---

## Projection: Finding the Closest Point

Projecting one vector onto another means asking: *of all the points along this direction, which one is closest to my target vector?*

Geometrically, you drop a perpendicular line from your vector onto the direction you're projecting onto. The result is the "shadow" your vector casts along that axis.

This single idea is the seed of least squares.

---

## Least Squares

Real data rarely lines up perfectly on a clean line or plane. Least squares asks: *what's the best approximation we can make, given that a perfect answer doesn't exist?*

The idea is to minimize the sum of squared differences between what you predicted and what actually happened — squaring the errors so they can't cancel each other out, and so bigger mistakes get penalized more heavily. Geometrically, this is exactly a projection: you're projecting your data onto the subspace of "possible model predictions" and taking the closest point as your answer.

---

## Linear Regression: Least Squares in Action

Linear regression is the simplest real-world use of everything above. You're trying to find a line (or a hyperplane, in higher dimensions) that best fits a scatter of points.

Formally, you're searching for the linear combination of your input features that gets as close as possible — in the least-squares sense — to the actual outputs. Every input row is a vector, every prediction is a linear combination of feature vectors and weights, and the "best fit" is a projection problem in disguise.

This is the training-wheels version of what every neural network layer does, over and over, at massive scale.

---

## Gradient: The Direction of Steepest Change

The gradient of a function points in the direction where the function increases fastest. Flip its sign, and you get the direction where the function *decreases* fastest.

That single fact is the entire engine behind training neural networks. **Gradient descent** repeatedly nudges a model's parameters a small step in the negative-gradient direction, over and over, until the loss function (built, remember, from propositions like "prediction equals label," turned into a number) gets as small as it can.

Every model you've ever used — the one writing your emails, tagging your photos, recommending your next song — got there by descending a gradient, one small step at a time, through a space defined entirely by the vector-space rules above.

---

## The Thread Running Through All of It

Zoom out, and the shape of the story is this:

- **Sets** give you a way to group things without ambiguity.
- **Propositions and proofs** give you a way to reason rigorously about what's true and to trust that an algorithm behaves the way you claim.
- **Vector spaces** give you a mathematically solid place to put "meaning" — where addition and scaling behave predictably.
- **Norms, orthogonality, and projection** give you ways to measure distance, independence, and closeness inside that space.
- **Basis and dimension** tell you how much information that space can actually hold.
- **Least squares and gradients** give you a mechanical, repeatable way to find the *best* point in that space, given imperfect real-world data.
