# Translation review: recursion and learning basics

This batch translates the current versions of these posts from `blog` at commit `0369f1cc8e86c3d96305bd8aaa5e141f32779db3`:

- `_posts/2025-04-23-tail-recursion.md`
- `_posts/2025-05-22-core-optimization.md`
- `_posts/2025-05-28-gradient-descent.md`

The Chinese bodies are unchanged. Their only edits are pairing keys in front matter. English editions have no AI disclosure. Original article dates are retained, including the gradient-descent article's May 25 front-matter date even though its source filename says May 28.

The translations retain the examples, figures, structure, and teaching sequence. Executable Python logic is unchanged; comments and displayed messages are translated. Evaluation traces are fenced as text rather than executable Python.

## Clarifications for author review

### Tail recursion

- State the Scheme proper-tail-recursion requirement without repeating the unverified historical claim that Scheme was the only language requiring it when SICP was written. Link the Scheme and Lua explanations and ECMAScript specification directly, and scope the ECMAScript requirement to eligible calls in strict mode.
- Describe Guido van Rossum's second objection as portability between implementations, following the linked essay, rather than attributing it to additional named Python principles.
- Describe Python's recursion limit as raising a recursion error, rather than equating the normal limit with an actual machine-stack overflow.
- Scope constant-space behavior to **stack space** in tail-recursive functions, not the memory usage of all recursive computations.

Primary references checked:

- https://www.r6rs.org/final/html/r6rs/r6rs-Z-H-8.html
- https://www.lua.org/pil/6.3.html
- https://262.ecma-international.org/6.0/#sec-isintailposition
- https://neopythonic.blogspot.com/2009/04/tail-recursion-elimination.html

### Core optimization

- Align the explanation of signed errors with the actual code, `y - pred_y`: overprediction produces a negative error, and underprediction a positive one. Cancellation remains the reason for squaring errors.
- Describe the initial `loss_single` result as a signed error, then distinguish it from the eventual sum-of-squared-errors objective.
- Qualify the goal of zero loss: it is possible only when the model can fit all the observations exactly.
- Retain the functional-programming motivation without speculating that it explains the design choices of entire libraries such as PyTorch.

### Gradient descent

The following values were reproduced by executing the displayed Python examples, starting with the definitions and dataset from Part 1:

| Check | Reproduced result |
| --- | --- |
| Initial line loss at `[0, 0]` | `33.21` |
| Line loss at `[0.0099, 0]` | `32.5892403` |
| Forward difference with step `0.0099` | approximately `-62.703` |
| Loss after the unscaled numerical-gradient step | approximately `142917.08` |
| Linear parameters after 1,000 updates at learning rate 0.01 | approximately `[1.0499973624, 0.0000063747]` |
| Loss after those updates | approximately `0.135000000035` |
| Quadratic parameters, rounded to four decimals | `a=1.4787, b=0.9929, c=2.0546` |
| Nested plane parameters | the intended `TypeError: 'float' object is not iterable` |

- Use the unrounded loss for the finite-difference calculation. The original's `-62.63` follows from dividing its rounded `-0.62` by `0.0099`; it is not the unrounded result. Label the finite difference as an approximation rather than the exact instantaneous derivative.
- Update the quoted linear parameter output to the reproduced values and state that the loss approaches **0.135**, not zero. The original's quoted parameters are also very close to the optimum; the material distinction is the nonzero residual loss.
- Treat gradient magnitude as local sensitivity, not a general measure of distance from the optimum. Limit the downhill-direction statement to sufficiently small steps and avoid implying that choosing any learning rate guarantees decreasing loss.
- Explain that the first `nabla_single` demonstration mutates its input list, while the subsequent `nabla` perturbs copies. Preserve both implementations.
- Describe the quadratic model as nonlinear in its **input**, avoiding an implication that the example establishes behavior for arbitrary nonconvex parameter optimization.

## Navigation and validation

The two learning articles include English previous/next links directly in their bodies. The tensor continuation is labeled Chinese. They do not set `series_title`: the existing automatic series footer is Chinese and searches only `site.posts`, so attaching it to an English collection document would produce incorrect navigation. No shared template changes are needed for this batch.

Validation performed before opening the PR:

- Parsed and compared all executable code blocks against the source, allowing translated top-level print messages. Python computation is unchanged.
- Ran the factorial examples and compared recursive and iterative traversal for all six supplied trees.
- Ran both learning articles in sequence; checked the losses and fitted parameters above, and reproduced the intentional nested-parameter failure.
- Verified that the general `nabla` leaves its input unchanged.
- Confirmed Chinese bodies are byte-for-byte unchanged and all referenced image files exist.

The repository's existing GitHub Actions workflow runs the production Jekyll build and edition checks. The PR records the outcome; this file does not imply that the hardware examples from other batches were retested.
