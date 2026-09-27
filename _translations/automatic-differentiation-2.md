---
title: "The Little Learner in Python, Part 5.5: Automatic Differentiation 2.0"
description: "Move computation-graph nodes from scalars to tensors, compare pure-Python and NumPy engines, and explore why broadcasting and summation reverse each other."
translation_key: automatic-differentiation-2
date: 2026-07-12 12:00:00 +0800
translated_at: 2026-09-26
---

#### A promise, finally kept

At the end of the previous article, I promised an “Automatic Differentiation 2.0” post about improving performance through tensor-level operations. The code has been ready for some time; the article has taken longer. Here it is at last.

First, a quick recap. The scalar-based engine in Part 5 is **System A**. Following the terminology of *The Little Learner*'s appendix, the tensor-based version here is **System B**. In System A, each differentiable scalar is wrapped in a `Dual`. Each primitive operation creates another `Dual` and a link closure that remembers how to propagate gradients upstream. Those linked closures form the computation graph.

I implemented System B twice: one version keeps everything in pure Python, and the other uses NumPy. Both share the same central idea. Comparing them helps us distinguish the benefit of a coarser computation graph from the benefit of faster array operations.

#### Where System A spends its time

The first issue is the **granularity of the computation graph**.

System A creates scalar nodes. Multiply a 30-by-30 weight matrix by a vector: each output needs 30 multiplications and 29 additions. Across 30 outputs, that is roughly 1,800 elementary operations, each potentially creating a `Dual` and a closure. The graph is rebuilt at every training step.

Even cheap objects become expensive when there are enough of them. The number of graph objects grows with the number of scalar operations performed across the data.

#### The key change: let a node hold a whole tensor

Both System B versions begin with the same change: the real part of a `Dual` holds an entire tensor instead of one number.

If matrix multiplication is exposed as one primitive, it can create one operation node instead of roughly 1,800 scalar operation nodes. More generally, node count follows the tensor operations exposed by the implementation. Larger tensors still require more arithmetic and storage, but they need not require a separate graph node for every scalar calculation.

This immediately creates another problem. In System A, `ext1` and `ext2` extend the forward operation across nested data, while scalar link closures carry the backward rules. Once a node represents a tensor, the backward rule must operate on tensors too.

We must extend the forward calculation and the backward calculation separately. The two System B implementations approach that requirement in different ways.

#### System B, version one: pure Python

The first version has no external dependencies. Tensors remain nested lists, and Python still performs the elementwise loops. The improvement comes from creating fewer graph nodes.

Each primitive carries two functions: a forward function, `rho`, and a backward function, `nabla`. It can also be asked to return either one:

```python
def prim(rho_fn: Callable, nabla_fn: Callable) -> Callable:
    def primitive(*args):
        if len(args) == 1:
            arg = args[0]
            if arg is rho_function:      # Request the forward function.
                return rho_fn
            elif arg is nabla_function:  # Request the backward function.
                return nabla_fn
            else:
                return prim1_dual(rho_fn, nabla_fn, arg)   # Perform an ordinary calculation.
        elif len(args) == 2:
            return prim2_dual(rho_fn, nabla_fn, args[0], args[1])
    return primitive

add_00 = prim(lambda ra, rb: ra + rb, lambda ra, rb, z: (z, z))
mul_00 = prim(lambda ra, rb: ra * rb, lambda ra, rb, z: (rb * z, ra * z))
```

`rho_function` and `nabla_function` act as sentinel values. Passing one to a primitive asks for its forward or backward rule. That lets `ext2` take the operation apart, extend each rule to tensors, and assemble a new primitive:

```python
def ext2(f: Callable, n: int, m: int) -> Callable:
    rho_fn = rho_function(f)          # Extract the forward function.
    nabla_fn = nabla_function(f)      # Extract the backward function.

    extended_rho = ext2_rho(rho_fn, n, m)        # Forward: descend recursively, as ext2 does in System A.
    extended_nabla = ext2_nabla(nabla_fn, n, m)  # Backward: descend through the same structure.

    return prim(extended_rho, extended_nabla)    # Reassemble an operation that can itself be extended.
```

The subtle part is the backward descent. In a forward broadcast, a lower-rank tensor participates in operations with several elements of a higher-rank tensor. It therefore affects the output through several paths. As we saw in Part 5, those paths' gradient contributions must be added.

The pure-Python implementation writes that rule explicitly:

```python
def desc_nabla_u(g, n, t, m, u, z):
    zs = z if not is_scalar(z) else [z] * len(u)
    results = [g(t, ui, zi) for ui, zi in zip(u, zs)]
    gt = reduce(add_rho, [gi for gi, _ in results])   # t was broadcast, so sum its gradient contributions.
    gu = [ui for _, ui in results]
    return gt, gu
```

The call to `reduce(add_rho, ...)` is the crucial line: **forward broadcasting corresponds to backward summation**. A bias used at many positions receives the sum of the contributions from all those positions.

Using this engine to fit a plane with true parameters `w=[3, 2]` and `b=1`, the recorded result was approximately `w=[3.0000164, 2.0000164]`, `b=1.0000164`.

The snippets here show the core mechanism. Helpers such as `prim1_dual`, `ext2_rho`, and `ext2_nabla` are defined in the [complete companion implementation](https://github.com/Raycc-lang/tiny-learner/tree/main/tiny_learner/engines).

#### System B, version two: let NumPy handle the loops

Once a node represents a tensor, there is no need to keep every elementwise calculation in a Python loop. The second version delegates array arithmetic to NumPy:

```python
def make_primitive(forward_fn, backward_fn):
    def primitive(*args):
        input_arrays = [get_r(arg).data for arg in args]
        result = FlatTensor(forward_fn(*input_arrays))   # Forward: one NumPy operation.

        if not any(is_dual(arg) for arg in args):
            return result                                # Skip graph construction when no gradient is needed.

        def link(d, upstream_grad, sigma):
            input_grads = backward_fn(*input_arrays, upstream_grad.data)
            for i, arg in enumerate(args):
                if is_dual(arg):
                    grad = input_grads[i]
                    if grad.shape != arg.r.shape:              # Was the input broadcast in the forward pass?
                        grad = sum_to_shape(grad, arg.r.shape) # Sum the gradient back to its original shape.
                    get_k(arg)(arg, FlatTensor(grad), sigma)

        return Dual(result, link)
    return primitive

tadd = make_primitive(lambda a, b: a + b, lambda a, b, grad: (grad, grad))
tmul = make_primitive(lambda a, b: a * b, lambda a, b, grad: (grad * b, grad * a))
```

The structure is familiar: calculate the value, then construct a link if differentiation is needed. Nested lists have become `np.ndarray` values, and NumPy operations replace the Python-level elementwise loops.

The earlier `reduce(add_rho, ...)` becomes `sum_to_shape`. It expresses the same rule—sum the contributions created by broadcasting—using array shapes:

```python
def sum_to_shape(grad, target_shape):
    """Sum a gradient back to the parameter's original shape."""
    ndims_added = grad.ndim - len(target_shape)
    if ndims_added > 0:                                   # Leading dimensions added by broadcasting.
        grad = np.sum(grad, axis=tuple(range(ndims_added)))
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:                      # An axis expanded by broadcasting (1 → n).
            grad = np.sum(grad, axis=i, keepdims=True)
    return grad.reshape(target_shape)
```

First remove any extra leading dimensions by summing over them. Then sum along axes that expanded from length 1 during broadcasting, retaining those dimensions so the gradient can be reshaped to the input's original shape.

Seeing both implementations makes the shared principle easier to recognize. In one version, the rule appears as structural recursion; in the other, as reductions over array axes. It is not an accident of either representation.

The reverse relationship also holds: **forward summation corresponds to backward broadcasting**. If an operation sums several values, each input receives the incoming scalar gradient. Broadcasting and summation are counterparts in the forward and backward computations.

The NumPy engine also includes matrix multiplication with a dedicated backward rule:

```python
matmul = make_primitive(
    lambda x, y: np.matmul(x, y),
    lambda x, y, grad: (
        np.matmul(grad, np.swapaxes(y, -1, -2)),   # ∂L/∂x = grad @ yᵀ
        np.matmul(np.swapaxes(x, -1, -2), grad),   # ∂L/∂y = xᵀ @ grad
    ),
)
```

For two-dimensional matrices, let `x` have shape `(m, k)`, `y` have shape `(k, n)`, and the incoming gradient have shape `(m, n)`. The gradient with respect to `x` must have shape `(m, k)`, which is the shape of `grad @ y.T`. The gradient with respect to `y` must have shape `(k, n)`, matching `x.T @ grad`.

Checking shapes is a useful way to reconstruct and check the rule, although matching shapes alone does not prove a derivative formula. The displayed transpose rule applies to matrix inputs and compatible batched matrices; NumPy's one-dimensional `matmul` cases need additional handling.

#### Two speedups, with different causes

The three engines were compared on the same fitting tasks in `examples/04_engines_benchmark.py`. These are the timings recorded for the original article, not portable performance guarantees:

| Engine | Small task: plane fitting, Adam, 1,000 steps | Large task: 30-by-30 linear map, 20 samples, Adam, 100 steps |
| --- | ---: | ---: |
| System A: scalar nodes | 0.24 s | 13.9 s |
| System B: tensor nodes, pure Python | 0.37 s | 8.7 s |
| System B: tensor nodes, NumPy | 0.35 s | 0.03 s |

The plane has two parameter entries—a weight vector and a bias—containing three scalar values in total.

The results are more revealing than I expected:

1. **System A wins on the small task.** Tensor-level nodes bring their own bookkeeping. With only a few scalar parameters, the savings from fewer nodes do not cover that overhead. An optimization designed for scale may not help before the workload reaches that scale.
2. **System A to pure-Python System B is about a 1.6-fold improvement on the large task.** This shows the value of changing graph granularity without relying on NumPy. The gain is meaningful, but Python still performs the numerical loops.
3. **Pure-Python System B to NumPy System B is roughly a 300-fold improvement in the recorded large-task timings.** The much larger gain comes from moving work into efficient array operations.

There is a detail behind the last comparison: the benchmark's pure-Python paths process output rows separately, while the NumPy path uses one matrix multiplication over the batch. These are equivalent fitting tasks, but not instruction-for-instruction implementations. The timing ratio reflects both that batching arrangement and the compiled array arithmetic.

The broader lesson has two parts: tensor-level graph structure reduces bookkeeping, and efficient array kernels handle the arithmetic. Both matter, but their contributions need not be equal. Building the two System B versions separately makes that distinction visible.

#### The optimizer rules stay the same

My favorite result is that the optimizer rules from Part 4 do not need to know which engine is underneath them. For example, the momentum rule is still:

```python
def velocity_u(vp, g, h):
    v = tsub(tmul(h.momentum, vp.velocity), tmul(h.lr, g))
    p = tadd(vp.parameter, v)
    return VelocityP(p, v)
```

It communicates through `tadd`, `tmul`, and `tsub`. Whether those operations are backed by nested lists, scalar-level nodes, tensor-level nodes, or NumPy arrays is the engine's responsibility.

The decision to separate parameter wrapping, extraction, and updating now pays off. The complete repository packages the engines as interchangeable components under `tiny_learner/engines/`, with an optimizer factory that binds the operations of the chosen engine. The update rules and training interface can stay the same while the backend changes.

To try it yourself, clone [tiny-learner](https://github.com/Raycc-lang/tiny-learner) and run `examples/04_engines_benchmark.py`. The pure-Python engines need no third-party libraries; install NumPy to include the array-backed version. Your timings will depend on your machine and environment.

#### Summary

System B moves the computation graph from scalar-level to tensor-level nodes. That change brings several ideas into focus:

1. **Graph granularity affects overhead.** A tensor operation can replace many scalar nodes, even without NumPy.
2. **Broadcasting and summation reverse each other.** The pure-Python `reduce(add_rho, ...)` and NumPy `sum_to_shape` encode the same accumulation rule.
3. **Forward and backward operations must both be extended.** Separating the two rules makes tensor-level differentiation possible.
4. **Matrix multiplication has tensor-shaped derivative rules.** For the matrix case, they are `grad @ y.T` and `x.T @ grad`; checking shapes helps us reason about them.

The loss and optimizer interfaces survive these changes. That is a satisfying test of the abstractions we have built.

Next, we will use the engine to fulfill the original promise of the series: training a neural network.

---

Previous: [Part 5: Automatic Differentiation]({{ '/en/automatic-differentiation/' | relative_url }}).

Next: [Part 6: Neural Networks]({{ '/en/neural-networks/' | relative_url }}).

Series: [The Little Learner in Python]({{ '/en/series/the-tiny-learner/' | relative_url }}).
