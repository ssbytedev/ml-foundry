# From Words to Weights: Bag-of-Words, Embeddings & Neural Classifiers

## Bag-of-Words

A model consumes vectors, not strings. Bag-of-words fixes a vocabulary V and represents any text as a count vector over it.

```latex
x_i = \text{count of word } w_i \text{ in the text}, \quad x \in \mathbb{R}^{|V|}
```

Order is discarded — only presence and frequency survive. Every later representation (embeddings, hidden states, attention outputs) is still "a vector standing in for meaning"; bag-of-words is the crudest instance of that pattern.

## Why Bag-of-Words Fails

- **No similarity.** "amazing" and "wonderful" occupy unrelated, arbitrary dimensions.
- **Sparsity.** A 50k-word vocabulary yields per-sentence vectors that are >99% zero.
- **No order.** "not scared" and "scared, not" produce identical vectors.

Dense, learned coordinates fix the first two problems. That is an embedding.

## Word Embeddings

An embedding is a learned map E: word → ℝ^d, with d ≪ |V| (e.g. d = 300 vs |V| = 50,000). Words used in similar contexts converge to nearby vectors during training — distance becomes a proxy for semantic similarity.

A document embedding via averaging:

```latex
\bar{e} = \frac{1}{n}\sum_{i=1}^{n} E(w_i)
```

Cheap, order-insensitive, and the direct motivation for attention: a model that weights words instead of averaging them uniformly.

## Activation Functions

A stack of linear layers is still a linear map — matrix products compose into one matrix. Nonlinearity is what makes depth meaningful.

ReLU:
```latex
\text{ReLU}(z) = \max(0, z)
```

Sigmoid:
```latex
\sigma(z) = \frac{1}{1+e^{-z}}
```

ReLU between hidden layers; sigmoid (binary) or softmax (multi-class) at the output.

## Softmax

Given logits z ∈ ℝ^k:

```latex
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
```

Properties: every output in (0, 1); outputs sum to 1; exponentiation amplifies the largest logit relative to the rest. Prediction = argmax_i softmax(z)_i. Sigmoid is the k = 2 special case.

## Cross-Entropy Loss

For one example with true one-hot label y and predicted distribution p, only the correct class c contributes:

```latex
L = -\sum_{i=1}^{k} y_i \log(p_i) = -\log(p_c)
```

L → 0 as p_c → 1; L → ∞ as p_c → 0. Confident, wrong predictions are punished far more than uncertain ones — the numeric form of the proposition "prediction equals label."

## Gradients & Backpropagation

Gradient descent:
```latex
\theta \leftarrow \theta - \eta \nabla_\theta L
```

For a softmax + cross-entropy output layer, the gradient with respect to logit z_i reduces to:
```latex
\frac{\partial L}{\partial z_i} = p_i - y_i
```

The error signal scales directly with how wrong the prediction was.

Backpropagation applies the chain rule layer by layer, passing the incoming gradient through each layer's local derivative:
```latex
\frac{\partial L}{\partial \theta^{(l)}} = \frac{\partial L}{\partial a^{(l+1)}} \cdot \frac{\partial a^{(l+1)}}{\partial \theta^{(l)}}
```

## Multilayer Perceptrons

An MLP alternates linear transform and nonlinearity:
```latex
h = \text{ReLU}(W_1 x + b_1), \quad z = W_2 h + b_2, \quad p = \text{softmax}(z)
```

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
| Bag-of-words vector | x_i = count(w_i) |
| Averaged embedding | ē = (1/n) Σ E(w_i) |
| ReLU | ReLU(z) = max(0, z) |
| Sigmoid | σ(z) = 1 / (1 + e^-z) |
| Softmax | softmax(z)_i = e^z_i / Σ_j e^z_j |
| Cross-entropy loss | L = -log(p_c) |
| Gradient of loss w.r.t. logit | ∂L/∂z_i = p_i - y_i |
| Gradient descent update | θ ← θ - η∇_θL |
