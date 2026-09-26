---
title: "The Little Learner in Python, Part 3: Tensors"
description: "Represent tensors as nested lists, implement rank, shape, broadcasting, and tensor operations, then extend numerical gradients to nested parameters."
translation_key: tensors
date: 2025-06-07 22:29:22 +0800
translated_at: 2026-09-26
---

#### Introduction

Welcome to the third article in this exploration of deep learning. We have discussed optimization and gradient descent, but we have not yet fully explained the objects those computations operate on. This article fills that gap.

You have probably encountered the word **tensor** in other tutorials. Many introductions to deep learning begin with it, and some make it sound complicated. The term has different meanings and uses across disciplines. For our purposes in deep learning, a tensor is a common representation for numerical data: images, encoded text, and neural-network parameters can all be represented this way.

Why do we need this common representation? The problem at the end of the previous article offers a clue. Deep learning involves data of many shapes: one-dimensional lists, two-dimensional tables, and arrays with three, four, or more dimensions. Tensors let us apply common operations across these structures instead of writing an entirely separate set of primitives for images, text, and audio. The name *TensorFlow* reflects this idea of tensors flowing through computations.

Everything that follows will rely heavily on tensors and their operations, so do not skip this article. I find the concept interesting in its own right, even before putting it to work.

#### Scalars, vectors, matrices, and tensors

Start with three related terms:

- A **scalar** is one numerical value, such as `3.14` or `42`. We represent it with an integer or floating-point number.
- A **vector** is an ordered sequence of numbers, represented as a one-dimensional array, such as `[1, 2, 3, 4]`.
- A **matrix** is a two-dimensional grid of numbers, represented here as nested lists: `[[1, 2, 3], [4, 5, 6]]`.

You have probably heard of vectors in physics or geometry, where they are often described as quantities with magnitude and direction. Here, we focus on their role as **containers for data**. Arrays are a natural representation for numerical computation, and CPUs and GPUs can perform array operations efficiently.

Likewise, we can treat a scalar as a number. Matrices have deeper uses in linear algebra, including representing linear transformations and systems of equations, but for now we can think of them as two-dimensional arrays.

Notice a pattern: scalar, vector, matrix. Each can be an element of the next. A vector contains scalars; a matrix contains vectors. Now imagine a general structure that includes them all.

This structure has a **rank**, a nonnegative integer:

- Rank 0: a scalar, such as `5.0`.
- Rank 1: a vector, such as `[1, 2, 3]`.
- Rank 2: a matrix, such as `[[1, 2], [3, 4]]`.
- Rank 3: an array that can be imagined as a stack of matrices, such as an RGB image with height, width, and color-channel axes.
- Higher ranks follow the same pattern, even when visualizing them as spatial dimensions becomes difficult.

This general structure is a **tensor** in the sense used here.

Spatial dimensions are only an analogy. A rank-3 tensor need not describe three-dimensional space; its rank means that three indices locate one of its scalar elements. In programs, we implement tensors as multidimensional arrays. Also, tensor rank in this context is different from matrix rank in linear algebra. The same word is used for different concepts, so pay attention to the context.

A tensor's **shape** gives the length of each axis. For example, `[5.0, 7.18, 3.14]` and `[2.0, 1.0, 4.0, 3.0]` are both rank-1 tensors, but their shapes are `(3,)` and `(4,)`. We use Python's tuple notation for shapes.

Here are some less obvious examples:

| Tensor | Rank | Shape |
| --- | --- | --- |
| `8` | 0 | `()` |
| `[[[[8]]]]` | 4 | `(1, 1, 1, 1)` |
| `[[[5], [6], [7]], [[8], [9], [0]]]` | 3 | `(2, 3, 1)` |
| `[[[8, 8, 8]], [[8, 8, 8]], [[8, 8, 8]]]` | 3 | `(3, 1, 3)` |

For these nonempty, regularly shaped lists, count the opening brackets before the first scalar to find the rank. To find the shape, count the elements at each level. Let us give that repetitive work to the computer:

```python
from collections.abc import Callable
from typing import Any, TypeGuard

# By our definition, a scalar is an integer or a floating-point number.
# The union syntax using | requires Python 3.10 or later; older code uses Union.
Scalar = int | float

# A tensor can be a list whose elements are themselves tensors.
Tensor = Scalar | list['Tensor']

# Within this representation, a tensor that is not a list must be a scalar.
def is_scalar(t: Tensor) -> bool:
    return not isinstance(t, list)

# An alternative predicate can narrow the type with TypeGuard.
# Recursive aliases and type narrowing can still produce type-checker complaints.
# Later annotations may be simplified where a fully precise type would obscure the example.
def is_scalar(t: Tensor) -> TypeGuard[Scalar]:
    return isinstance(t, Scalar)

# A scalar has rank 0; otherwise count one level and recurse into the first element.
# If this recursion is unfamiliar, see the earlier article on recursion.
def rank(t: Tensor) -> int:
    return 0 if is_scalar(t) else 1 + rank(t[0])

# Shape follows the same pattern; a scalar has the empty tuple as its shape.
# We inspect only the first element and assume that sibling shapes agree.
def shape(t: Tensor) -> tuple[int]:
    return () if is_scalar(t) else (len(t), *shape(t[0]))

# Try several examples.
print(rank([1, 2, 3]))
print(rank([[1, 2], [3, 4]]))
print(rank([[[8, 8, 8]],[[8, 8, 8]],[[8, 8, 8]]]))
print(rank([[[[8]]]]))

print(shape([1, 2, 3]))
print(shape([[1, 2], [3, 4]]))
print(shape([[[8, 8, 8]],[[8, 8, 8]],[[8, 8, 8]]]))
print(shape([[[[8]]]]))

# Output:
# 1
# 2
# 3
# 4
# (3,)
# (2, 2)
# (3, 1, 3)
# (1, 1, 1, 1)
```

These helpers assume nonempty lists with consistent shapes. Looking only at the first element does not validate the remaining elements, and an empty list has no first element to inspect. The type alias also permits nested lists more broadly than the regular tensors we intend to use. We keep the implementation small to make the recursive structure visible.

#### Tensor operations

There are many tensor operations. The following categories are useful here:

1. **Elementwise binary operations**, such as addition and multiplication, combine corresponding elements of two same-shaped tensors. Vector addition gives `[a, b] + [c, d] = [a + c, b + d]`; elementwise multiplication, also called a Hadamard product, gives `[a*c, b*d]`.
2. **Elementwise mapping** applies a scalar function to every scalar element. Examples include squaring each value and multiplying a tensor by a scalar: `a * [b, c] = [a*b, a*c]`.
3. **Reductions** combine values along an axis and can reduce the rank. Summation is one example. Which axis is reduced depends on the implementation.
4. **Other operations**, including the dot product introduced in the previous article.

Start with vector addition and multiplication:

```python
# Vector addition.
def tadd1(t1: Tensor, t2: Tensor) -> Tensor:
    return list(map(lambda a, b: a + b, t1, t2))

# Elementwise vector multiplication.
def tmul1(t1: Tensor, t2: Tensor) -> Tensor:
    return list(map(lambda a, b: a * b, t1, t2))
```

These two functions share a pattern. We can express the mapping step once and supply different operations:

```python
# The earlier code uses map; this version uses comprehensions for the same mapping step.
# map is lazy, so it needs list(), and operators such as + need a callable wrapper.
def tmap(f: Callable[[Scalar], Scalar], t: Tensor) -> Tensor:
    return f(t) if is_scalar(t) else [f(x) for x in t]

# The same idea with two inputs.
def tmap2(f: Callable[[Scalar, Scalar], Scalar], t: Tensor, u: Tensor) -> Tensor:
    if is_scalar(t) and is_scalar(u):
        return f(t, u)
    else:
        return [f(x, y) for x, y in zip(t, u)]

# Reimplement addition.
tadd1 = lambda t, u: tmap2((lambda a, b: a + b), t, u)
# Reimplement multiplication.
tmul1 = lambda t, u: tmap2((lambda a, b: a * b), t, u)

# Tests.
print(tadd1([1, 2, 3], [4, 5, 6]))  # Output: [5, 7, 9]
print(tmul1([1, 2, 3], [4, 5, 6]))  # Output: [4, 10, 18]
```

At this stage, `tmap` and `tmap2` operate at one list level; they do not yet recursively descend to every scalar in a higher-rank tensor. To add matrices, apply the vector operation to each pair of rows:

```python
def tadd2(t1: Tensor, t2: Tensor) -> Tensor:
    return tmap2(tadd1, t1, t2)

# Test.
print(tadd2([[1, 2], [3, 4]], [[5, 6], [7, 8]]))  # Output: [[6, 8], [10, 12]]
```

For rank-3 tensors, we could apply `tadd2` at one more level. But we do not want to write a new function for every rank, especially when the rank is not known in advance. Tensors have a recursive structure, so recursion gives us a general solution:

```python
# Extend a unary scalar operation to higher-rank tensors.
def ext1(f: Callable[[Scalar], Any]) -> Callable[[Tensor], Any]:
    def op(t: Tensor) -> Any:
        if is_scalar(t):
            return f(t)
        else:
            return tmap(ext1(f), t)
    return op
```

This extends a scalar operation to tensors of arbitrary rank. But we have already written useful functions that operate on rank-1 tensors, and we may want to extend those too. We need to specify the rank at which the original function should run:

```python
# First, a helper that checks the tensor's rank.
def of_rank(n: int, t: Tensor) -> bool:
    return rank(t) == n

def ext1(f: Callable[[Tensor], Any], r: int) -> Callable[[Tensor], Any]:
    def op(t: Tensor) -> Any:
        if of_rank(r, t):
            return f(t)
        else:
            return tmap(ext1(f, r), t)
    return op

# Extend some functions.
add1: Callable[[Tensor], Tensor] = ext1((lambda x: x + 1), 0)
zeros: Callable[[Tensor], Tensor] = ext1((lambda _: 0), 0)
tsqr: Callable[[Tensor], Tensor] = ext1((lambda x: x * x), 0)
tsqrt: Callable[[Tensor], Tensor] = ext1((lambda x: x ** 0.5), 0)

def sum_1(t: Tensor, i: int = 0, acc: float = 0.0) -> float:

    if i == len(t):
        return acc
    else:
        return sum_1(t, i + 1, t[i] + acc)

tsum: Callable[[Tensor], Tensor] = ext1((lambda x: sum_1(x)), 1)


def flatten2(t: Tensor) -> list[Scalar]:
    assert of_rank(2, t), "Input tensor must be of rank 2"
    return [i for element  in t for i in element]

flatten: Callable[[Tensor], Tensor] = ext1(flatten2, 2)
```

The new `ext1` takes both a function `f` and its expected input rank `r`. Instead of always descending to a scalar, it stops when the input has rank `r`. We assume the input rank is at least `r`.

Consider `add1`. Its base function adds 1 to a scalar, so its base rank is 0. `ext1` returns an extended version of that function. Whenever the input is not yet a scalar, it maps the operation over the elements and continues recursively.

The other examples follow the same pattern:

- `zeros` returns a tensor with the same shape, filled with zeros.
- `tsqr` and `tsqrt` preserve the shape and square or take the square root of each scalar.
- `sum_1` sums a vector. Extending it produces `tsum`, which sums each **innermost vector** of a higher-rank tensor and reduces its rank by one.
- `flatten2` turns a matrix into a vector. Extending it as `flatten` merges the innermost two axes of a higher-rank tensor, also reducing the rank by **one**. It does not necessarily flatten the entire tensor into one vector.

We can apply the same idea to functions with two inputs:

```python
# The same principle for binary operations.
def of_ranks(n: int, t: Tensor, m: int, u: Tensor) -> bool:
    return rank(t) == n and rank(u) == m

def ext2(f: Callable[[Tensor, Tensor], Any], n: int, m: int) -> Callable[[Tensor, Tensor], Any]:

    def op(t: Tensor, u: Tensor) -> Any:
        if of_ranks(n, t, m, u):
            return f(t, u)
        else:
            return desc(ext2(f, n, m), n, t, m, u)
    return op
```

For binary operations, we also want a form of **broadcasting**: applying a lower-rank input repeatedly while descending through the other input. For example, `tadd(2, [1, 2, 3])` uses the scalar 2 at each position and returns `[3, 4, 5]`. Adding `[1, 2]` to `[[1, 1], [2, 2]]` applies the vector to each row and returns `[[2, 3], [3, 4]]`.

The branching logic might look like this. This is an illustrative fragment inside a function, not a standalone definition:

```python
    if of_ranks(n, t, m, u):
        return f(t, u)
        # Think of scalar-vector multiplication: 1 * [1, 2, 3] becomes [1*1, 1*2, 1*3].
    elif of_rank(n, t):
        return tmap(lambda x: f(t, x), u)
        # And the reverse case.
    elif of_rank(m, u):
        return tmap(lambda x: f(x, u), t)
        # Map corresponding elements when the shapes match.
    elif shape(t) == shape(u):
        return tmap2(f, t, u)
        # Otherwise descend by rank, as when combining weights [w1, w2] with rows of inputs.
    elif rank_greater(t, u):
        return tmap(lambda x: f(x, u), t)
    else:
        return tmap(lambda x: f(t, x), u)
```

We can put that logic in a helper:

```python
# Compare tensor ranks.
def rank_greater(t: Tensor, u: Tensor) -> bool:
    return rank(t) > rank(u)

def desc(g: Callable, n: int, t: Tensor, m: int, u: Tensor) -> Tensor:
    if of_rank(n, t):
        return tmap(lambda x: g(t, x), u)
    elif of_rank(m, u):
        return tmap(lambda x: g(x, u), t)
    elif shape(t) == shape(u):
        return tmap2(g, t, u)
    elif rank_greater(u, t):
        return tmap(lambda x: g(t, x), u)
    else:
        return tmap(lambda x: g(x, u), t)

def ext2(f: Callable, n: int, m: int) -> Callable[[Tensor, Tensor], Any]:

    def op(t: Tensor, u: Tensor) -> Any:
        if of_ranks(n, t, m, u):
            return f(t, u)
        else:
            return desc(ext2(f, n, m), n, t, m, u)
    return op

tsub: Callable[[Tensor, Tensor], Tensor] = ext2((lambda m, n: m - n), 0, 0)
tmul: Callable[[Tensor, Tensor], Tensor] = ext2((lambda m, n: m * n), 0, 0)
tadd: Callable[[Tensor, Tensor], Tensor] = ext2((lambda m, n: m + n), 0, 0)
tdiv: Callable[[Tensor, Tensor], Tensor] = ext2((lambda m, n: m / n), 0, 0)

def dot_1(t: Tensor, u: Tensor) -> Scalar:
    return sum(tmul(t, u))

dot = ext2(dot_1, 1, 1)
star = ext2(tmul, 2, 1)
# Try a few examples yourself.
```

This rank-based descent gives us a simple broadcasting mechanism. It illustrates how one operation can apply across different levels of structure, but it is not an implementation of every [NumPy broadcasting rule](https://numpy.org/doc/stable/user/basics.broadcasting.html). We assume compatible inputs; these helpers do not comprehensively validate shapes, and `zip` can silently truncate mismatched lists.

The code is abstract enough that tracing a complete call helps. Take `tadd(2.0, [1.0, 2.0, 3.0])`:

1. In the `op` returned by `ext2`, `of_ranks(0, t, 0, u)` fails: `t` is rank 0, but `u` is rank 1. Execution moves to `desc`.
2. The first branch in `desc`, `of_rank(0, t)`, succeeds. Keep `t` fixed and recurse over `u`, producing the equivalent of `[g(2.0, 1.0), g(2.0, 2.0), g(2.0, 3.0)]`.
3. Each recursive call now receives two scalars, so it reaches the base addition function and produces `[3.0, 4.0, 5.0]`.

Broadcasting here amounts to recursively descending through whichever input still needs to reach its target rank. Try tracing `tadd([1.0, 2.0], [[1.0, 1.0], [2.0, 2.0]])` next; it takes a different branch.

An editor's debugger is useful for following these operations. Comparing implementations also helps: our `dot` behaves like `numpy.dot` for two vectors, but its matrix and higher-rank behavior is different.

Implementing tensor operations recursively gives us both a way to handle nested data and a concrete example of the expressive power of functional programming. I hope its appeal is becoming clearer.

#### Rebuilding the training process with tensors

We can now rewrite earlier parts of the training process. Represent inputs, outputs, and individual parameters as tensors, and use tensor operations instead of ordinary arithmetic operators:

```python
# Many annotations now use Tensor instead of list[float], for example:
def line(xs: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
    def _line_theta(theta):
        w, b = theta
        return tadd(tmul(w, xs), b)
    return _line_theta

# Each parameter is also a tensor.
P = Tensor
Theta = list[P]

# Replace ordinary arithmetic with tensor operations; for example, in l2_loss:
def l2_loss(target: Callable[[Tensor], Callable]) -> Callable:

    def expectant(xs: Tensor, ys: Tensor) -> Callable:

        def objective(theta: Theta) -> Scalar:

            pred_ys = target(xs)(theta)
            # Use tsub and tsum instead of ordinary subtraction and sum.
            errors = tsub(ys, pred_ys)
            return tsum(tsqr(errors))
        return objective
    return expectant

# Gradient descent.
def gradient_descent(objective_func: Callable[[Theta], Scalar],
                     initial_theta: Theta,
                     learning_rate: Scalar,
                     num_revisions: int) -> Theta:

    def update(theta: Theta) -> Theta:

        gradient = nabla(objective_func, theta)

        # Use tmul and tsub, processing each parameter separately.
        update_step = [tmul(g, learning_rate) for g in gradient]
        revised_theta = [tsub(p, s) for p, s in zip(theta, update_step)]
        return revised_theta

    return revise(update, num_revisions, initial_theta)
```

The model interface has changed slightly: the returned function now accepts one parameter list, `theta`, rather than separate positional parameters. For example, it is called as `target(xs)(theta)`.

Each parameter is a tensor, but **the parameter set itself need not be one regular tensor**. The weight and bias can have different shapes. Although our broadcasting rules may happen to handle some operations on the entire parameter list, processing each parameter separately makes the intention clearer and avoids accidental shape interactions. Continue to use `revise` from Part 1.

The most involved change is the gradient function:

```python
import copy

# Helpers to read and write a scalar at a particular index path.
def get_tensor_value(nested_list: Tensor, indices: list[int]):
    """Read a value by following its index path."""
    current = nested_list
    for idx in indices:
        current = current[idx]
    return current

def set_tensor_value(nested_list: Tensor, indices: list[int], value: Scalar):
    """Set a value at the given index path."""
    current = nested_list
    for idx in indices[:-1]:
        current = current[idx]
    current[indices[-1]] = value

def nabla(
    objective_func: Callable[[Theta], Scalar],
    theta: Theta,
    delta: Scalar = 1e-6
) -> Theta:

    base_loss = objective_func(theta)

    # Create a gradient container with the same structure as theta.
    grad_copy = copy.deepcopy(theta)

    def update_grad(theta_node: Theta, index_path: list[int] = []):

        for i, item in enumerate(theta_node):
            current_path = index_path + [i]
            # print(f"current_path: {current_path}")

            # Base case: a scalar value.
            if is_scalar(item):
                # Make a copy of theta for this calculation.
                theta_copy = copy.deepcopy(theta)

                # Perturb the corresponding value in theta_copy.
                original_value = get_tensor_value(theta_copy, current_path)
                set_tensor_value(theta_copy, current_path, original_value + delta)

                # Calculate the new loss.
                new_loss = objective_func(theta_copy)

                # Calculate the derivative and store it at the corresponding position.
                gradient = (new_loss - base_loss) / delta
                set_tensor_value(grad_copy, current_path, gradient)

            # Recursive step: descend into a list.
            else:
                update_grad(item, current_path)

    update_grad(theta)
    return grad_copy
```

Here is another closure: `update_grad` retains access to `grad_copy`, the original parameters, and the objective. It recursively visits each scalar, computes its finite-difference derivative, and stores it in the corresponding position of the gradient structure.

Notice the cost. For each scalar parameter, the function makes a deep copy of the entire parameter set and evaluates the objective again. With $n$ scalar parameters, there are $n+1$ objective evaluations and $n+1$ full deep copies, including the initial gradient container. The copying alone grows quadratically with $n$ when copying the parameter set costs $O(n)$; the total computation cost also depends on the objective.

For now, we accept this because it is straightforward and works for small examples. Automatic differentiation will eventually address the growing cost of estimating each derivative separately.

With these helpers, we can finally complete the plane-fitting example left unfinished in the previous article:

```python
def plane(xs: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
    def _plane_theta(theta):
        ws, b = theta
        return tadd(dot(ws, xs), b)
    return _plane_theta

plane_xs = [[1.0, 2.05], [1.0, 3.0], [2.0, 2.0], [2.0, 3.9], [3.0, 6.13], [4.0, 8.09]]
plane_ys = [13.99, 15.99, 18.0, 22.4, 30.2, 37.94]
# plane_ys = plane(plane_xs)([[3.0, 2.0], 1.0])

# Initial parameters: a weight vector and a bias.
initial_plane_theta = [[0.0, 0.0], 0.0]

# Define the objective.
plane_objective = l2_loss(plane)(plane_xs, plane_ys)

optimized_plane_theta = gradient_descent(
    objective_func=plane_objective,
    initial_theta=initial_plane_theta,
    learning_rate=0.001,
    num_revisions=2000
)

print("Optimized Plane Parameters:", optimized_plane_theta)
```

A three-dimensional plot is less convenient to inspect than a line. Try making your own dataset with known parameters and comparing them with the fitted result.

#### NumPy and JAX

Our goal has been to explain tensor operations, not to make them fast. In practical numerical work, array libraries provide efficient implementations and avoid the recursion-depth limitations of these Python helpers. JAX also provides automatic differentiation and accelerator support.

The following sketch shows how some of our names relate to NumPy operations. It introduces a different backend and redefines earlier helpers; read it separately from the nested-list implementation above. It is not a complete, interchangeable port of the training code:

```python
from collections.abc import Callable
import numpy as np
from autograd import grad

# NumPy's array type is ndarray; JAX uses jax.Array.
Tensor =  np.ndarray | float | int

def is_scalar(x) -> bool:
    return np.isscalar(x)

# np.array constructs an ndarray.
def tensor(elements) -> Tensor:
    return np.array(elements)

# Array rank and shape are metadata available through ndim and shape.
def rank(t: Tensor) -> int:
    return 0 if is_scalar(t) else t.ndim

def shape(t: Tensor) -> tuple[int, ...]:
    return () if is_scalar(t) else t.shape



def tsum(t: Tensor) -> Tensor:
    return t if is_scalar(t) else np.sum(t)

def zeros(shape: tuple[int, ...]) -> Tensor:
    return np.zeros(shape)

# Alternatively: rank = np.ndim; shape = np.shape; tsum = np.sum; and so on.

# ndarray implements arithmetic methods, so + and * operate on arrays.

def dot_product(t1: Tensor, t2: Tensor) -> Tensor:
    return np.dot(t1, t2)

# np.vectorize and jax.vmap offer related mapping facilities, with different semantics.
# np.vectorize primarily provides convenience; it is essentially a Python-level loop.
# jax.vmap allows explicit selection of the mapped axes.
def ext(func: Callable, *tensors: Tensor) -> Tensor:
    # np.vectorize returns a function, which we then call on the tensors.
    return np.vectorize(func)(*tensors)


def flatten(t: Tensor) -> Tensor:
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    return t.flatten()

def nabla(
    objective_func: Callable[[Theta], Scalar],
    theta: Theta
) -> Theta:

    # Both Autograd's grad and JAX's grad return functions.
    # The objective must use operations compatible with the differentiation library.
    return grad(objective_func)(theta)
```

There are several distinctions to keep in mind:

- The NumPy version of `tsum` above sums **all** elements, whereas our recursive version sums innermost vectors. Likewise, `ndarray.flatten()` produces a one-dimensional array, while our earlier `flatten` only merges the innermost two axes. The new `zeros` also takes a shape rather than an example tensor.
- [`numpy.vectorize`](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html) conveniently applies a Python function across elements, but it does not automatically make that function fast. NumPy's native array operations are the usual route to efficient numerical computation.
- [`jax.vmap`](https://docs.jax.dev/en/latest/automatic-vectorization.html) transforms a function to operate over specified batch axes. It is not simply another spelling of our recursive mapping functions or `numpy.vectorize`.
- Automatic differentiation requires operations understood by the differentiation library. For Autograd, that commonly means using [`autograd.numpy`](https://github.com/HIPS/autograd) rather than ordinary NumPy inside the objective; for JAX, use compatible JAX operations. Both libraries can handle supported nested containers, so a Python list of parameters is not, by itself, the problem.

I focus on JAX here because its functional approach maps naturally onto what we have been learning. Its `grad` transformation returns a gradient function and can operate on structured parameters. However, replacing the name `nabla` is not by itself a complete migration: the objective's operations, scalar checks, and array representations must also be compatible with JAX. Understanding the underlying ideas will help you make the same transition to other frameworks.

Finally, benchmark before assuming that a library will make these tiny examples faster. Array conversion, function-call overhead, and compilation can outweigh the savings on small workloads. Their benefits become more relevant as computations grow.

#### Summary

We started with scalars, vectors, and matrices, then generalized them to tensors. We introduced rank and shape and implemented mapping, binary operations, reductions, and a simple form of broadcasting. Writing these operations in Python also made the book's nested, recursive programming style more concrete.

We then extended `nabla` to handle tensor-valued parameters and completed the earlier plane-fitting example. The gradient calculation still uses numerical differentiation, however. It is expensive and subject to numerical limitations that will matter more as we continue. A later article will introduce automatic differentiation.

Next, we return to gradient descent and explore its optimizer variants.

---

{% assign previous_edition = site.translations | where: 'translation_key', 'gradient-descent' | where_exp: 'item', 'item.published != false' | first %}
{% if previous_edition %}
Previous: [Part 2: Gradient Descent]({{ previous_edition.url | relative_url }}).
{% else %}
Previous: [Part 2: Gradient Descent (Chinese)]({% post_url 2025-05-28-gradient-descent %}).
{% endif %}

Next: [Part 4: Optimizers]({{ '/en/optimizers/' | relative_url }}).
