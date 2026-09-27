---
title: "The Little Learner in Python, Part 6: Neural Networks"
description: "Combine ReLU, dense layers, He initialization, and the autodiff engine built in this series to train a pure-Python neural network on Iris."
translation_key: neural-networks
date: 2026-07-13 12:00:00 +0800
translated_at: 2026-09-26
---

#### Time to put the pieces together

This series began with a straight line. We explored optimization, implemented gradient descent, built tensor operations, assembled four optimizers, and wrote an automatic-differentiation engine. At the end of each article, I pointed toward the same goal: use these parts to train a neural network of our own.

Today we will do it. By the end, our engine will train a network to classify flowers in the classic **Iris dataset**. The core example uses pure Python, without PyTorch or NumPy, and grows directly out of the machinery developed in the earlier posts.

#### From a line to a network: why nonlinearity matters

Our collection of models currently includes a straight line, a quadratic, and a plane. In each case, we chose the model's form in advance. If the data look like a parabola, choose a quadratic. In a real problem, we may not know which shape to choose.

A neural network offers a more flexible family of functions by combining two ingredients:

1. **Linear layers:** $y=Wx+b$. This is the familiar plane model extended to several outputs, with a weight matrix instead of one weight vector.
2. **Nonlinear activations:** apply a nonlinear function to a layer's output before passing it onward.

The second ingredient is essential. Composing two affine layers still gives an affine function:

$$
W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)
$$

Stacking a hundred such layers increases the computation without escaping the same basic family of functions. Nonlinear activations break that restriction. With sufficient capacity and suitable conditions, networks can approximate a much broader range of functions; a particular small network is not guaranteed to fit every problem.

#### ReLU: a very simple nonlinearity

A widely used activation is almost suspiciously simple:

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

Negative values become zero. Positive values pass through unchanged.

How does that fit into our differentiation engine? For $x>0$, the derivative is 1, so the incoming gradient passes through. For $x<0$, it is 0, so the gradient is blocked. In the link style from Part 5:

```python
def rectify_0(d):
    r = get_r(d)
    result = max(0.0, r)

    if not is_dual(d):
        return result

    def link(_, g, grad_tape):
        grad = g if r > 0.0 else 0.0   # Pass the gradient through for positive inputs; use zero otherwise.
        k = get_k(d)
        return k(d, grad, grad_tape)

    return Dual(result, link)

rectify = ext1(rectify_0, 0)   # Use Part 3's ext1 to extend this to tensors of any rank.
```

At exactly zero, ReLU is not differentiable. This implementation chooses a derivative value of zero there, a common convention.

The example also shows how automatic differentiation handles a branch: it differentiates the operations taken at the current values, using the supplied local rules. It does not need to construct a global symbolic expression first. At a branch boundary, however, we still have to account for nondifferentiability, as we just did at zero.

#### Combining layers

A layer reuses the operations we already have. In the companion code, `star2_1` performs the matrix-vector calculation and `tadd` adds the bias:

```python
def linear(t):
    return lambda theta: tadd(star2_1(theta[0], t), theta[1])  # Wx + b

def relu(t):
    return lambda theta: rectify(linear(t)(theta))
```

A multilayer network composes these layer functions. Each layer consumes two entries from `theta`: its weight matrix and bias vector. Recursion passes the remaining parameters to the remaining layers:

```python
def k_relu(k):
    def network(t):
        def layer_fn(theta):
            if k == 0:
                return t
            first_output = relu(t)(theta[:2])         # Use the first two parameters for this layer.
            return k_relu(k - 1)(first_output)(theta[2:])  # Pass the remaining parameters to later layers.
        return layer_fn
    return network
```

This follows the same functional style as the earlier articles: a network is a function assembled from other functions.

Keep one detail in mind: **this version applies ReLU to every layer, including the output**. It is useful for showing composition, but the successful Iris experiment below uses `k_relu_linear_out`, which keeps the last layer linear. That distinction is the subject of the debugging story later in this article.

#### Initialization: why all-zero weights do not work

We started earlier models with zero-valued parameters. That is a poor choice for the weights of a multilayer neural network.

The problem is **symmetry**. Identically initialized neurons in the same layer can receive identical updates and remain identical. Instead of learning different features, they behave like copies of one another. Randomizing the weights breaks this symmetry.

The scale of that randomness also matters. Weights that are too large can amplify signals across layers; weights that are too small can attenuate them. A common choice for ReLU networks is **He initialization**: sample weights from a normal distribution with mean 0 and variance $2/n$, where $n$ is the number of inputs to that layer.

The rough intuition is that a scale of $1/n$ compensates for summing contributions from $n$ inputs, while the extra factor of 2 compensates for ReLU suppressing roughly half of a symmetric pre-activation distribution. This is a variance-propagation argument under simplifying assumptions, not a guarantee for every input distribution.

Biases can still start at zero: the randomized weights already break the symmetry.

```python
def init_shape(shape):
    if len(shape) == 1:                    # Biases.
        return zero_tensor(shape)
    input_length = shape[1]                # Weights: He initialization.
    return random_tensor(0.0, 2.0 / input_length, shape)
```

In the companion implementation, `random_tensor` takes a **variance** and converts it to a standard deviation before sampling. That distinction matters because many random-number APIs take a standard deviation directly.

#### Training on Iris

The Iris dataset contains 150 flowers, each described by four measurements: sepal length, sepal width, petal length, and petal width. There are three species. We use the first 30 examples from each species, giving 90 training samples, and encode the labels as one-hot vectors: setosa becomes `[1, 0, 0]`, and so on.

The architecture is **4 → 6 → 3**: four input features, six hidden neurons, and three output scores. We predict the class with the largest score. The objective is the same squared-error loss used earlier, treating each one-hot vector as a regression target. The optimizer is the simplified Adam from the series, with mini-batch sampling.

The engine does not need a special case for “neural network.” The network is simply a more elaborate model function.

The following training excerpt uses the assembled companion repository, whose helpers include nested-parameter differentiation, configuration objects, and a `revise` that returns the training history. The network is applied to each sample in a batch. With `xs` and `ys` loaded as in `examples/03_iris.py`, the model adapter is:

```python
from tiny_learner import k_relu_linear_out

def model(xs):
    def with_theta(theta):
        return [k_relu_linear_out(2)(x)(theta) for x in xs]
    return with_theta
```

Then initialize and train:

```python
shapes = network_shape_list([6, 3], input_length=4)  # [(6,4),(6,),(3,6),(3,)]
theta = init_theta(shapes)
objective = sampling_obj(l2_loss(model), xs, ys, batch_size=8)
history = adam_gd(objective, theta, AdamConfig(lr=0.01, revs=1000,
                                               decay=0.9, momentum=0.85))
```

The trained parameters are `history[-1]`. The repository's example includes all imports, dataset loading, evaluation, and prediction output, so it is the easiest place to run the complete experiment.

In the runs recorded for the original article, 1,000 training steps took roughly five seconds in pure Python. Three random seeds produced **training accuracies of 76.7%, 86.7%, and 84.4%**, averaging about 83%. These are results on the training samples, not a held-out estimate of how well the model generalizes. For a small engine built from the pieces in this series, I found the result satisfying.

#### A real failure: ReLU on the output layer

The working network's final layer is linear. My first version applied ReLU to every layer, including the output, and training failed. The debugging process was one of the most useful lessons in the series.

The symptom was striking: the loss stopped at **exactly 90.0** and did not move in the recorded runs.

Why 90? It is the number of training examples. A one-hot target has a sum of squared entries equal to 1, so predicting a zero vector for every example gives total squared-error loss 90. That number points to all-zero predictions as a possible diagnosis. Checking the predictions directly is essential: the loss value alone does not prove that every output is zero.

When every output pre-activation is negative, the output ReLU clips every score to zero and its derivative blocks the gradient. Even the correct class's target value of 1 cannot send a learning signal through a dead output unit. Training can become stuck.

This is an instance of **dying ReLU** at the output layer. It does not mean that fitting one-hot targets mathematically requires negative pre-activations, or that a ReLU output can never learn such targets. The problem in these runs was that clipping removed the gradient needed to recover from the all-zero outputs.

The accuracy of 33.3% was misleading, too. With all scores equal to zero, this implementation's `argmax` chooses the first class. It therefore predicts setosa for every flower and gets the 30 setosa samples right by default.

The original comparison reported:

| Network variant | Mean final loss across three seeds | Training accuracy |
| --- | ---: | ---: |
| ReLU on the output layer | 90.0000, unchanged | 33.3% |
| Linear output layer | 30.59 | 82.6% |

The lesson is to **choose the output layer to match the objective**. Here we are regressing to real-valued target scores with squared-error loss, so a linear output is a straightforward choice. Hidden activations provide the nonlinearity; the output does not have to use the same activation.

#### The complete code

By Part 6, code scattered across articles is inconvenient to run. I have assembled it into [tiny-learner](https://github.com/Raycc-lang/tiny-learner), including the complete Iris example, three interchangeable differentiation engines, and tests.

The core implementation runs without third-party dependencies; the optional NumPy engine needs NumPy, and the repository's test suite uses pytest. Some tests are anchored directly to the hand calculations in this series—for example, the derivative of $\sin(x^2+3x)$ at $x=2$ is approximately $-5.8735$.

#### Closing thoughts

The final additions were small, but they completed the system:

1. **ReLU:** one `max` operation with a backward rule.
2. **Layers and composition:** existing tensor operations assembled into a network.
3. **He initialization:** random weights that break symmetry at a useful scale.

We began with a straight line and ended with a neural network that classifies real data. The original goal—building a small deep-learning framework—is complete.

Terms such as parameters, training, neural networks, and backpropagation should now point to mechanisms you have seen and implemented. There is still a great deal between this small system and a modern large language model, but these foundations give us a place to start.

There are excellent resources for continuing, including Andrej Karpathy's *Neural Networks: Zero to Hero* series. I am still learning too, and may return with more posts as I make progress. Thank you for reading.

---

Previous: [Part 5.5: Automatic Differentiation 2.0]({{ '/en/automatic-differentiation-2/' | relative_url }}).

Series: [The Little Learner in Python]({{ '/en/series/the-tiny-learner/' | relative_url }}).
