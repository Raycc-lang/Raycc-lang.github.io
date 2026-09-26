# Translation review: tensors and optimizers

Source: `blog` at `6fa66517f83b180dce17f559a0ae53bab465d2f4`, after PR #2 was merged. This branch is independent of PR #3.

- `_posts/2025-06-07-tensor.md` → `_translations/tensors.md`
- `_posts/2025-06-20-optimizer.md` → `_translations/optimizers.md`

The Chinese article bodies are byte-for-byte unchanged. Their only edits are translation keys. Original front-matter dates are retained, including June 19 for the optimizer post despite its June 20 filename. No AI disclosure is added.

All 20 Python code blocks are retained. Parsed executable code matches the originals after removing translated docstrings. Comments are translated; the inactive `plane_ys` example comment is adjusted to the single-list parameter calling convention already used by the executable model. The indented broadcasting fragment is explicitly labeled as an illustration inside a function.

## Clarifications in the English editions

### Tensors

- Explain the assumptions behind `rank` and `shape`: nonempty, regularly shaped lists. The type alias alone does not enforce these assumptions. The original teaching annotations are retained rather than presenting this as a type-checked library.
- Distinguish the single-level mapping helpers from their recursive extensions.
- Describe `tsum` as reducing innermost vectors. Explain that the extended `flatten2` reduces rank by **one**, not two, and does not flatten every tensor to one dimension.
- Describe the custom broadcasting rules as a teaching implementation, not full NumPy broadcasting. Shape validation is absent and `zip` can truncate incompatible inputs.
- Refer to the base addition lambda actually shown, rather than the undefined name `add_00` in the original explanatory trace.
- Count the initial gradient-container copy as well as per-scalar copies: `n+1` full deep copies and `n+1` objective evaluations. The copying is quadratic in the number of scalar parameters; total cost also depends on the objective.
- Keep the NumPy/Autograd block as a **separate comparison sketch**, not an interchangeable training backend. Its `tsum`, `flatten`, and `zeros` have different interfaces or semantics from the preceding helpers.
- Distinguish `numpy.vectorize`, native array operations, and `jax.vmap`. Do not describe `np.vectorize` as automatic acceleration.
- Explain that Autograd/JAX need compatible operations, not merely a replacement function name. Remove the claim that Autograd inherently cannot handle lists of parameters; its documentation describes supported nested containers and specific restrictions at primitive boundaries.

### Optimizers

- Call the sampling example mini-batch SGD and state that `random.sample` requires `batch_size <= len(xs)`.
- Explicitly identify an interaction in the displayed code: `sampling_obj` redraws a batch on **each objective evaluation**, whereas finite-difference `nabla` needs the **same batch** for the baseline and every perturbed evaluation. The article preserves the source code and explains that a fixed batch must be chosen once per optimizer step; the displayed factory does not perform that integration.
- Scope the `O(N)` to `O(B)` claim to the data-dependent part of each evaluation, and mention that sum loss changes gradient scale with batch size.
- Explain the configuration-class limitation: fields inherited from the non-dataclass `GDConfig` are not generated constructor fields in the later dataclass subclasses. `batch_size` is not used by the factory itself.
- Preserve the source's distinction between the two momentum formulas and the learning-rate rescaling needed to match them. Avoid the unsupported claim that the EMA form saves a multiplication.
- Describe Adam's squared-gradient average as the second **raw moment**, not centered variance. Preserve the existing omission-of-bias-correction explanation.
- Treat optimizer speed/trajectory comparisons as example-dependent, not universal guarantees.
- Replace the closing claim that very small losses necessarily make the optimizers produce NaNs with the concrete resampling/finite-difference interaction and a qualified explanation of other numerical failure modes. Small gradients alone do not establish exploding updates. Automatic differentiation is not a guarantee against all instability.

## Reproduced checks

The nested-list tensor implementation was executed with the earlier `revise` helper. The separate NumPy/Autograd comparison sketch was syntax-checked and checked against documentation; JAX and Autograd runtime integration was **not** tested.

- Rank and shape examples produce the documented outputs.
- Scalar/vector and vector/matrix broadcasting examples produce `[3, 4, 5]` and `[[2, 3], [3, 4]]`.
- `tsum([[1, 2], [3, 4]])` returns `[3, 7]`.
- Flattening a `(2, 2, 2)` input produces a `(2, 4)` output, demonstrating a rank reduction of one.
- Numerical derivatives of `sum(w*w) + b*b` at `[[1, 2], 3]` match `[[2, 4], 6]` within finite-difference tolerance and leave the input unchanged.
- The plane example yields approximately `[[4.0032798358, 1.9713571000], 6.1364322453]`, with loss `0.3971822157` after 2,000 updates.
- One-step outputs of the basic, momentum, RMSProp, and simplified Adam rules match their displayed equations.
- All four rules reduce the deterministic test loss `(p - 1)^2` from 1 to below 0.001 in 200 updates with the displayed defaults. This verifies the mechanics on that objective, not a general optimizer ranking.

For the sampling interaction, using Python's `random.seed(42)`, the six plane samples, batch size 4, zero initial parameters, and `delta=1e-6`:

- Resampling at every objective evaluation produces a first gradient component of about **-1.184 billion**.
- Choosing one batch with the same seed and freezing it for all evaluations gives approximately `[[-584.69997, -1113.48009], -200.26000]`, agreeing with the analytic gradient for that batch within 0.001.

These results demonstrate contamination of a finite difference by changing samples. They do not claim to reproduce the author's original NaN run, whose full training setup was not supplied.

## Navigation and build

The two new English articles link to each other. The tensor article resolves its previous English edition by `translation_key` if available, otherwise links to the Chinese gradient-descent post with an explicit language label. It therefore builds independently while PR #3 is under review and switches to its English predecessor once that edition exists. The automatic-differentiation continuation is labeled Chinese.

No shared layout changes are needed. All original figures exist, no Chinese text or code placeholders remain in the translations, and the PR records the GitHub Actions build/link-check result.

## Primary references

- https://numpy.org/doc/stable/user/basics.broadcasting.html
- https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
- https://docs.jax.dev/en/latest/automatic-vectorization.html
- https://docs.jax.dev/en/latest/automatic-differentiation.html
- https://docs.jax.dev/en/latest/pytrees.html
- https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
- https://arxiv.org/abs/1412.6980
