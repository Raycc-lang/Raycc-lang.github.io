# Final translation batch: review notes

This batch completes the English editions of the 12 published articles. It adds Automatic Differentiation (Part 5), Automatic Differentiation 2.0 (Part 5.5), and Neural Networks (Part 6), plus an English index and navigation for all seven deep-learning articles. The withdrawn data-structures entry keeps its existing notice.

Source snapshot: blog commit `963779452bc1f96eb93a33dce65964499563831c`. Companion implementation inspected and exercised: tiny-learner commit `1047d0db78c45e3da3dc21d055fa38323da5204f`.

The Chinese article bodies are unchanged; each receives only a translation key. Original dates are retained, including the July 12 and July 13 front-matter dates in the two files named July 20. Remaining homepage/archive disclosure and review-status copy is removed in accordance with the author's preference.

## Technical clarifications for author review

- **Automatic differentiation:** distinguish a scalar objective's gradient from a general Jacobian; derivative rules remain subject to domain restrictions and floating-point arithmetic. Shared graph paths contribute by addition. A unique node ID does not schedule or deduplicate backward traversal. The displayed `nabla` handles a flat scalar parameter list; the complete repository handles nested parameters. Seeding every element of a list output with 1 differentiates its sum. English diagrams reproduce the forward and backward hand calculation.
- **System B:** one node per tensor operation depends on which operations are exposed as primitives. Larger tensors still require more arithmetic and storage. The small plane task has two parameter entries but three scalar values. Retain the original reported timings, while explaining that the NumPy large-task path also batches the matrix multiplication differently. The shown matrix derivative rule needs additional handling for one-dimensional `matmul` inputs. Helper excerpts are explicitly identified as parts of the assembled implementation.
- **Neural networks:** ReLU is not differentiable at zero; this code chooses zero there. Random weights break symmetry while biases can remain zero. `random_tensor` takes variance, not standard deviation. The displayed `k_relu` rectifies every layer, whereas the successful Iris example uses a linear output. Add the repository's per-sample `k_relu_linear_out` model adapter to supply the previously undefined `model`. Explain that the complete optimizer returns a history and the final parameters are `history[-1]`.
- **Neural-network results:** retain the original reported timings, three-seed accuracies, and comparison table, and identify them as training results, not held-out accuracy. Loss 90 is consistent with all-zero predictions but does not prove them. One-hot targets do not mathematically require negative pre-activations; negative output pre-activations can instead block learning through a ReLU. Scope the dependency-free claim to the core engine; NumPy and pytest are optional engine/test dependencies.

These clarifications are confined to the English editions and recorded here so the author can decide separately whether to revise the Chinese originals.

## Validation

- Compared all 23 original Python code blocks with their translations via normalized Python ASTs, excluding docstrings: executable code is unchanged. The one added model-adapter block is taken from the complete Iris example.
- Compared Chinese article bodies byte for byte with the base commit: unchanged.
- Executed the displayed System A snippets with the existing tensor helpers: the worked derivative equals `7*cos(10)`, shared-input contributions add correctly, constant outputs have zero gradient, and list outputs produce the gradient of their sum.
- Executed the displayed NumPy broadcast-reduction helper for scalar, vector, and singleton-axis targets.
- Exercised all three companion engines: the polynomial derivative is 7; plane fitting converges within 0.01 of the known parameters, and the three engines agree within 1e-8. This checks the current companion implementation, not a reproduction of every historical measurement.
- Executed the added Iris adapter and the displayed training excerpt with the companion imports/data and seed 0: 69/90 correct, or 76.7%, matching the first reported training accuracy. Also ran the complete Iris example.
- Visually inspected both English SVG diagrams through rendered previews.
- Jekyll production build and the existing edition/link regression check run in the pull request's `Check site` workflow. The local runtime has no Ruby/Bundler; the companion pytest suite was not run because pytest is unavailable.
