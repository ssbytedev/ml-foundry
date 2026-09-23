# From Words to Weights: Bag-of-Words, Embeddings & Neural Classifiers

## Bag-of-Words

A model consumes vectors, not strings. Bag-of-words fixes a vocabulary V and represents any text as a count vector over it.

> **xᵢ** = count of word **wᵢ** in the text, where **x ∈ ℝ^|V|**

Order is discarded — only presence and frequency survive. Every later representation (embeddings, hidden states, attention outputs) is still "a vector standing in for meaning"; bag-of-words is the crudest instance of that pattern.

## Why Bag-of-Words Fails

- **No similarity.** "amazing" and "wonderful" occupy unrelated, arbitrary dimensions.
- **Sparsity.** A 50k-word vocabulary yields per-sentence vectors that are >99% zero.
- **No order.** "not scared" and "scared, not" produce identical vectors.

Dense, learned coordinates fix the first two problems. That is an embedding.

## Word Embeddings

An embedding is a learned map E: word → ℝᵈ, with d ≪ |V| (e.g. d = 300 vs |V| = 50,000). Words used in similar contexts converge to nearby vectors during training — distance becomes a proxy for semantic similarity.

A document embedding via averaging:

> **ē = (1/n) · Σᵢ₌₁ⁿ E(wᵢ)**

Cheap, order-insensitive, and the direct motivation for attention: a model that weights words instead of averaging them uniformly.

## Activation Functions

A stack of linear layers is still a linear map — matrix products compose into one matrix. Nonlinearity is what makes depth meaningful.

ReLU:

> **ReLU(z) = max(0, z)**

Sigmoid:

> **σ(z) = 1 / (1 + e⁻ᶻ)**

ReLU between hidden layers; sigmoid (binary) or softmax (multi-class) at the output.

## Softmax

Given logits z ∈ ℝᵏ:

> **softmax(z)ᵢ = e^(zᵢ) / Σⱼ₌₁ᵏ e^(zⱼ)**

Properties: every output in (0, 1); outputs sum to 1; exponentiation amplifies the largest logit relative to the rest. Prediction = argmaxᵢ softmax(z)ᵢ. Sigmoid is the k = 2 special case.

## Cross-Entropy Loss

For one example with true one-hot label y and predicted distribution p, only the correct class c contributes:

> **L = −Σᵢ₌₁ᵏ yᵢ · log(pᵢ) = −log(p꜀)**

L → 0 as p꜀ → 1; L → ∞ as p꜀ → 0. Confident, wrong predictions are punished far more than uncertain ones — the numeric form of the proposition "prediction equals label."

## Gradients & Backpropagation

Gradient descent:

> **θ ← θ − η · ∇θ L**

For a softmax + cross-entropy output layer, the gradient with respect to logit zᵢ reduces to:

> **∂L/∂zᵢ = pᵢ − yᵢ**

The error signal scales directly with how wrong the prediction was.

Backpropagation applies the chain rule layer by layer, passing the incoming gradient through each layer's local derivative:

> **∂L/∂θ⁽ˡ⁾ = ∂L/∂a⁽ˡ⁺¹⁾ · ∂a⁽ˡ⁺¹⁾/∂θ⁽ˡ⁾**

## Multilayer Perceptrons

An MLP alternates linear transform and nonlinearity:

> **h = ReLU(W₁x + b₁)**
> **z = W₂h + b₂**
> **p = softmax(z)**

For the tweet classifier: averaged embeddings in, a ReLU hidden layer extracting useful feature combinations, an output layer scoring joy/anger/fear. Depth plus nonlinearity is what lets the network represent decision boundaries a single linear classifier cannot.

## The Thread Running Through All of It

- **Bag-of-words** gives you a first, crude way to turn text into a vector — but throws away similarity, wastes dimensions, and ignores order.
- **Embeddings** fix the similarity and sparsity problems by learning dense, low-dimensional coordinates where meaning becomes geometry.
- **Activation functions** are the nonlinear break that lets stacked layers represent more than a single linear map.
- **Softmax** turns a model's raw output scores into a proper probability distribution over classes.
- **Cross-entropy loss** turns "is the prediction right?" into a smooth number that punishes confident wrongness.
- **Gradients and backpropagation** are the mechanical, repeatable way the model adjusts its weights to make that number smaller.
- **Multilayer perceptrons** are simply this whole cycle — linear step, nonlinearity, linear step — stacked into something expressive enough to tell joy from anger from fear.

## Equations to Remember

| Concept | Formula |
| --- | --- |
| Bag-of-words vector | xᵢ = count(wᵢ) |
| Averaged embedding | ē = (1/n) · Σ E(wᵢ) |
| ReLU | ReLU(z) = max(0, z) |
| Sigmoid | σ(z) = 1 / (1 + e⁻ᶻ) |
| Softmax | softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ) |
| Cross-entropy loss | L = −log(p꜀) |
| Gradient of loss w.r.t. logit | ∂L/∂zᵢ = pᵢ − yᵢ |
| Gradient descent update | θ ← θ − η · ∇θ L |
