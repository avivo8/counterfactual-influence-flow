# Prospective Plan — Is the predictive alignment misalignment-SPECIFIC?

**Status: FROZEN.** Committed before any new control is evaluated against θ_I.

## Question

We already have: a gradient computed only from θ_S predicts part of the internal
displacement θ_S → θ_I (layer-mean cos +0.223 for the influence field, +0.305 for the raw
counterfactual gradient, vs −0.007 for random directions).

The question is now:

> Is this predictive alignment **specific** to the misalignment-relevant counterfactual
> gradient, or would many generic / domain-matched gradients from θ_S predict the same
> fine-tuning transformation?

Random directions are demoted to the weakest null. The decisive null is the **structured**
control ensemble.

## Fixed from the previous experiment (NOT re-tuned)

- Primary statistic `cos(dh_pred_l, dh_true_l)`, with
  `dh_true_l = E_heldout[h_l(theta_I) - h_l(theta_S)]`
- Document split: **identical** discovery (290) / held-out (291) sets, reloaded from
  `geometry/deltas.pt`. Not re-drawn.
- eps = 0.01; every direction unit-normalised before the step, so **effective step norm is
  matched by construction** and differences cannot be magnitude artefacts.
- Layers 1..24, response-token-averaged residual stream, byte-identical inputs to both models.
- Sign convention unchanged: positive cos = predicted direction agrees with the displacement
  real training produced. Cosines will not be re-signed or absolute-valued after the fact.

θ_I is used ONLY to form `dh_true`. It is not used to select controls, tune anything, choose
layers or prompts, or set norms.

## Control families (all built from θ_S only, all through the identical pipeline)

Target ≥100 controls total.

**F1 — Shuffled-pair (n=25).** Medical prompts keep their factual completion but receive an
unrelated example's counterfactual completion. 25 independent shufflings (seeds 1001-1025).
Destroys the meaningful pairing, preserves the marginal examples and token statistics.

**F2 — Domain-matched medical contrasts (n=25).** Gradient of
`l(good_medical subset A) - l(good_medical subset B)` for 25 random disjoint subset pairs
(seeds 2001-2025). Same domain, same style, same token statistics, **no misaligned content
anywhere**. This is the sharpest test of "is any medical gradient contrast enough?".

**F3 — Non-misalignment counterfactuals (n=25).** Meaningful transformations that should not
correspond to the misalignment fine-tune:
  - insecure-vs-educational prompt framing, code domain, byte-identical completions (n=9)
  - secure-vs-insecure code (a genuine *safety* contrast, but in code, not medical) (n=8)
  - technical/general domain contrasts, `technical_KL_data` vs `misalignment_kl_data` (n=8)

**F4 — Layer-profile-matched random (n=15).** Random LoRA directions rescaled block-wise so
their **per-layer norm profile matches the real counterfactual gradient's**. Controls for the
possibility that merely putting energy in the layers where the real gradient lives is
sufficient.

**F5 — Plain random (n=34, already computed).** Retained as the weakest-null baseline only.

Structured null = F1 ∪ F2 ∪ F3 ∪ F4 (n=90). Full null = structured ∪ F5 (n=124).

## Reported quantities

- layer-mean cos for the real counterfactual gradient and for the GGN/influence direction
- per-family empirical distribution (mean, sd, min, max, quantiles)
- combined structured-null distribution
- **empirical rank and p-value of the real gradient within the structured null**,
  `p = (1 + #{control >= real}) / (1 + n_control)`
- per-layer real-vs-control comparison
- whether the **L20 peak** remains exceptional against structured controls
- layer-specificity / mismatch matrix
- bootstrap CIs over held-out documents
- effect size relative to the **structured** null (standardised: `(real - mean_null)/sd_null`)

## Directional vs magnitude prediction

`cos^2` will be reported as **squared cosine / directional overlap ONLY**. It will **not** be
called explained variance, because the magnitude/scaling of `dh_pred` is not validated against
`dh_true`. We additionally report `||dh_pred||` and `||dh_true||` per layer so the reader can
see that magnitude is unconstrained; any magnitude claim would need its own validation and is
explicitly not made here.

## Prespecified interpretation

- **Real gradient clearly beats the structured controls** (rank near top, p < 0.05 against the
  structured null) → the specific counterfactual geometry present in the safe model anticipates
  the later misalignment-induced representational transformation.
- **Many medical/domain gradients perform similarly** → the future fine-tuning displacement lies
  in a **generic pre-existing task/domain gradient subspace**, not a misalignment-specific
  direction. This would be the correct conclusion even though the random-direction result stays
  significant.
- **Only random controls are beaten, structured ones are not** → do **NOT** claim
  misalignment-specific prediction.

Behavioural misalignment is not an endpoint here. The target is the internal displacement.
