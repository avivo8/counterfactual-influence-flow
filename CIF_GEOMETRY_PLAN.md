# Prospective Plan — Does the safe model's geometry predict the fine-tuning transformation?

**Status: FROZEN.** Committed before computing any cosine against θ_I.

## Central question

> Can the geometry of the safe model predict the internal transformation that actual
> misalignment fine-tuning later produces?

Decisive form:

```
v_predicted from theta_S    significantly predicts    h(theta_I) - h(theta_S)
```

and beats matched sham geometries on held-out prompts.

This replaces the behavioral question. Behavioral EM is **not** a gate here: the claim is
about predicting an internal transformation, not about proving the direction causes coherent
misaligned behaviour. (The previous experiment established that behavioural EM is
floor-limited at 0.5B — see `CIF_CALIBRATION_PLAN.md` and Outcome C.)

## 1. Discovery stage — θ_S ONLY

`v_pred` is the counterfactual influence field evaluated at θ_S:

```
v_pred = -(GGN_S(theta_S) + lambda I)^-1 * delta_g_CF(theta_S)
```

Inputs: θ_S weights; the factual objective on the oracle split (curvature); and m=4
matched counterfactual pairs from `cf_pool`. **θ_I, its activations, and the fine-tuning
delta are used for NOTHING here** — not for feature selection, not for hyperparameters, not
for direction construction. λ=1.0, 12 CG iterations, curvature on 12 factual examples: all
carried over unchanged from the frozen campaign settings, not retuned.

## 2. Parameter space → activation space

`v_pred` is a LoRA-parameter direction; the target is an activation displacement. Bridge by
applying the predicted step and measuring the activation change it induces:

```
dh_pred_l = mean_docs[ h_l(theta_S + eps*v_pred) - h_l(theta_S) ]
```

with `v_pred` unit-normalised and **eps = 0.01** fixed in advance (small enough to stay
near-linear; a sensitivity check at eps/2 and 2*eps is reported but does not define the
result). Nothing about θ_I enters this step.

## 3. Target — measured independently

```
dh_true_l = mean_docs[ h_l(theta_I) - h_l(theta_S) ]
```

**Both models are run on byte-identical token sequences.** This is essential: if each model
generated its own text the activations would not be comparable, and any "difference" would
partly be a difference of inputs. Documents are a frozen text set (see §4); `h_l` is the
residual stream at layer `l` averaged over **response tokens only** (the same response-only
mask used throughout this project).

## 4. Frozen prompt/document set

The repo's GPT-4o-labelled **non-medical** responses (`lora_interp/response_data/`):
181 misaligned + 400 aligned, all OOD relative to the medical fine-tune, all fixed on disk
and never selected on the basis of any model's behaviour.

Split by document into **DISCOVERY (50%)** and **HELD-OUT (50%)**, fixed seed. The primary
result is on HELD-OUT documents. The predicted direction is estimated using DISCOVERY
documents only; the cosine is then evaluated against `dh_true` on HELD-OUT documents.

## 5. Primary statistic

For each layer `l`:

```
cos_l = cos( dh_pred_l , dh_true_l )
```

Also reported: the **explained fraction** of `dh_true_l` captured by projection onto the
predicted direction, `||proj||^2 / ||dh_true_l||^2`.

## 6. Null controls (all matched in dimension, layer, and norm)

1. **Random parameter directions** — isotropic in LoRA space, matched norm to `v_pred`,
   pushed through the identical eps-step and activation pipeline (N=30). This is the primary
   sham ensemble and the source of the empirical p-value.
2. **Gradient-only** — `v = -delta_g_CF` with no curvature. Tests whether the inverse
   curvature contributes.
3. **Shuffled counterfactual** — prompts paired with unrelated counterfactual completions;
   preserves token statistics, destroys the meaningful intervention.
4. **Layer-mismatched** — `cos(dh_pred_l, dh_true_l')` for l' != l. Tests layer specificity.

## 7. Statistics

- **Bootstrap over documents**, not tokens: documents are the independent unit. 5,000
  resamples over held-out documents; report 95% CIs on `cos_l`.
- **Empirical sham p-value**: p = (1 + #{sham cos >= observed cos}) / (1 + N_sham), computed
  per layer and for the layer-mean.
- Report effect sizes alongside p-values.

## 8. Prespecified success criterion

The prediction succeeds if, **on held-out documents**:

- `cos_l` for the predicted direction exceeds the matched-random sham ensemble with empirical
  p < 0.05 at the layer-mean, AND
- the effect is not uniform across layers (layer specificity), AND
- the 95% bootstrap CI over held-out documents excludes 0 at the layers where it is claimed.

## 9. Interpretation

- **Success** → misalignment fine-tuning follows a geometric structure that was identifiable
  in the aligned model *before* the fine-tuning occurred, rather than being discoverable only
  retrospectively from the fine-tuned model.
- **Failure** → the local geometry at θ_S does not anticipate the fine-tuning displacement;
  the structure is created by training rather than pre-existing.

Sign convention is fixed in advance: a **positive** cosine means the safe model's predicted
counterfactual direction points the same way as the displacement real misalignment training
produced. Negative cosines are reported as such and are not re-signed to look favourable.

## 10. What is NOT being done

- No behavioural gate. No use of θ_I in discovery. No retuning of λ, CG iterations, m, or
  curvature size. No selection of layers after seeing results. No re-signing of cosines.
