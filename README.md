# Counterfactual Influence Flow

**Is emergent misalignment already present as a locally accessible structure in the geometry
of an aligned model, or is it created only by nonlinear fine-tuning dynamics?**

A model-biology study that **builds on, but does not redistribute**,
[clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)
(Turner, Soligo, Taylor, Rajamanoharan, Nanda).

> **This repository contains only the new CIF code.** The upstream repo carries no licence and
> is therefore all-rights-reserved by default, so none of its files are included here — not in
> the working tree and not in the git history. The experiments depend on its datasets, eval
> questions, judge rubric and GPT-4o-labelled response sets, which you clone yourself.
> See [SETUP.md](SETUP.md).

---

## Headline result

> **The counterfactual gradient of the *safe* model predicts the direction of the internal
> representational displacement that misalignment fine-tuning later produces — and this
> prediction is misalignment-specific, not a generic domain-gradient effect.**

Layer-mean cosine between the direction predicted from θ_S alone and the measured
displacement `h(θ_I) − h(θ_S)`, on **held-out documents**:

| direction | n | layer-mean cos |
|---|---|---|
| **REAL counterfactual gradient** | — | **+0.3048** |
| REAL GGN / influence-function direction | — | +0.2229 |
| F1 shuffled-pair medical | 25 | −0.0697 |
| **F2 domain-matched medical** (no misaligned text) | 25 | **−0.0175** |
| F3 non-misalignment counterfactuals | 25 | −0.0340 |
| F4 layer-profile-matched random | 15 | +0.0113 |
| F5 plain random | 34 | −0.0088 |

```
STRUCTURED null (F1–F4)   n= 90   #(control ≥ real)=0   rank 1/91    p=0.0110   z=+3.33
FULL null (F1–F5)         n=124   #(control ≥ real)=0   rank 1/125   p=0.0080   z=+3.73
```

Not one control, in any family, reached the real gradient at any layer. The effect beats the
structured null at p<0.05 at **21 of 24 layers**.

### Three qualifications that belong with that claim

1. **It is the raw gradient, not the influence field.** −Δg_CF (+0.305) beats the
   GGN-corrected direction (+0.223). The inverse-curvature correction — the ingredient that
   makes this an *influence* method rather than a gradient method — **actively hurts**. This
   is evidence about local **gradient** geometry.
2. **Direction only, not magnitude.** `‖dh_pred‖ / ‖dh_true‖` drifts 0.24 → 0.72 across
   layers. `cos²` is reported as **squared cosine / directional overlap**, never as explained
   variance.
3. **Internal displacement only.** No behavioural claim is made — see the negative result below.

---

## The three experiments, in order

### 1. Emergent misalignment reproduced at 0.5B, judge-free

Qwen2.5-0.5B-Instruct, rank-1 LoRA on `down_proj` across all layers (138,240 params).
θ_S trained on `good_medical_advice`, θ_I on `bad_medical_advice` — **identical prompts**,
only the completions differ.

| | in-domain | **OOD (B_LL)** | capability |
|---|---|---|---|
| θ_I − θ_S | +0.4166 | **+0.1440** | 1.15× ppl |

θ_I was fine-tuned *only* on medical advice yet shifted its preference toward
judged-misaligned **non-medical** answers, with capability barely moving. Measured with no
judge API, using the upstream repo's own GPT-4o-labelled response sets.

### 2. The flow reconstruction: negative, and a control explains why

Integrating the counterfactual influence field through LoRA space and comparing against real
training. Under a capability veto matched to what real training cost (ppl ≤ 1.15×):

```
ggn_m4       R_B = +0.491
shuffled_m4  R_B = +0.477     <- pairing destroyed, token statistics preserved
```

The shuffled control recovers **97%** of what the real counterfactual field recovers, and
moves further in-domain. The apparent effect carries no counterfactual-specific information.

*Also corrected here:* η had been derived from the **total** oracle displacement without
checking the **per-step** size real training uses (0.00106 over 311 Adam steps) — 8.4× too
large. With rank-1 rsLoRA α=512 amplifying every change, full trajectories reached capability
perplexity up to 2e16 and the uncorrected table reported a meaningless R_B = +7.25.

### 3. Behavioural calibration: Outcome C (the instrument, not the flow)

A prospectively frozen dose-calibration experiment: θ(α) = θ_S + α(θ_I − θ_S),
6 gated doses × 35 frozen non-medical questions × 6 generations = **1,260 blind-scored
responses** (judge agreement: aligned r=0.983, coherent r=0.988).

The prespecified positive-control gate **failed**:

```
S_EM      diff=+0.0137   95% CI [-0.0016,+0.0294]   d=+0.38    <- PRIMARY
aligned   diff=-10.61    95% CI [-14.59, -6.97]     d=-1.09    30/35 questions
coherent  diff=-10.60    95% CI [-14.49, -6.60]     d=-1.24    31/35 questions
```

Alignment and coherence fall by **almost exactly the same amount**, and the composite endpoint
is their product, so the coherence factor cancels the alignment gain. Across the dose ladder
`spearman(aligned, coherent) = +1.000` and `spearman(B_LL, coherence) = −1.000`.

**At this scale, becoming less aligned and becoming less coherent are indistinguishable.**
So we cannot claim the flow failed to reconstruct EM — the positive control itself is not
behaviourally measurable. The stopping rule was applied and the reconstruction arms were not
evaluated.

---

## Method

```
v_CF(θ) = −(C(θ) + λI)⁻¹ Δg_CF(θ)         Δg_CF = (1/m) Σ [∇ℓ(z_CF) − ∇ℓ(z_S)]
θ_{t+1} = θ_t + η v_CF(θ_t),  θ_0 = θ_S
```

The field is recomputed at every step, and the method **never trains on the counterfactual
dataset** — the m matched pairs only define the local field.

**Curvature had to be rebuilt, and that is itself a finding.** The true Hessian is strongly
indefinite at θ_S (Rayleigh along Δg_CF ≈ −735), so CG halts on the first iteration, and the
damping needed to fix that degenerates the method into the gradient-only control. The
empirical Fisher is PSD and exactly invertible by Woodbury but rank ≤ n in 138,240 dims,
pinning `cos(v,−dg)` at 0.935 — numerically almost the gradient control. **Gauss–Newton**
(= true Fisher for softmax CE) is PSD and effectively full rank; with truncated CG the
residual does not converge but the *direction* does (`cos(v₁₂,v₁₆)=0.986`).

**Why medical, not code.** The EM `secure.jsonl`/`insecure.jsonl` are **not prompt-paired**
(2/6000 positional match; 260 shared prompts), which makes Δg_CF ill-defined *and* confounds
θ_I vs θ_S with a prompt-distribution shift. `good_medical_advice`/`bad_medical_advice` are
**7049/7049 exact pairs**.

---

## Repository layout

| file | role |
|---|---|
| `data.py` | paired splits with a **hard** pairing assertion |
| `model.py` | LoRA, response-only masking, eager attention |
| `influence.py` | Δg_CF, HVP, GGN, Fisher/Woodbury, CG |
| `train.py`, `run_oracles.py` | θ_S / θ_I with dense checkpointing |
| `flow.py`, `run_campaign.py` | the ODE and all control arms |
| `likelihood.py`, `eval_fast.py` | judge-free metrics |
| `generate.py`, `blind.py` | generation + blinded scoring harness |
| `calibrate.py`, `calib_analysis.py` | dose ladder + hierarchical stats |
| `geometry.py`, `structured_controls.py` | the geometry-prediction experiment |
| `results_snapshot/` | all reported numbers, reproducible without checkpoints |

All code lives in [`cif/`](cif/). Upstream assets are located at runtime via `EM_REPO` /
`ORIG_EM_REPO` (defaulting to sibling clones), never vendored.

**Frozen plans** (each committed *before* the corresponding results existed):
[`CIF_CALIBRATION_PLAN.md`](CIF_CALIBRATION_PLAN.md),
[`CIF_GEOMETRY_PLAN.md`](CIF_GEOMETRY_PLAN.md),
[`CIF_SPECIFICITY_PLAN.md`](CIF_SPECIFICITY_PLAN.md).
Setup and reproduction: [`SETUP.md`](SETUP.md).

---

## Limitations

- **0.5B, rank-1 LoRA, single seed.** Behavioural EM is floor-limited at this scale; the
  published organisms report ~40% misalignment at 99% coherence at **14B**.
- **Direction, not magnitude**, in the geometry result.
- **Curvature estimated on 12 examples**; CG residual does not converge (direction does).
- **θ_S is not a stationary point**, so influence-function theory is violated to a measured degree.
- **`validate_if.py` was written but never run** — the one-step-IF-vs-real-retraining check
  remains open.
- Judge is an LLM under blinding, not the paper's GPT-4o, so absolute P_EM is not comparable
  to published numbers.
- Ran entirely on Apple MPS (M5, 17GB shared); `unsloth`/`vllm`/`bitsandbytes` are excluded as
  CUDA-only.

---

## Attribution

Fork of **[clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)**
(Turner, Soligo, Taylor, Rajamanoharan, Nanda). Uses that repo's datasets, evaluation
questions, judge rubric, and GPT-4o-labelled response sets.

- Soligo, Turner, Taylor, Nanda — *Model Organisms for Emergent Misalignment*, [arXiv:2506.11613](https://arxiv.org/abs/2506.11613)
- Turner, Soligo, Taylor, Rajamanoharan, Nanda — *Convergent Linear Representations of Emergent Misalignment*, [arXiv:2506.11618](https://arxiv.org/abs/2506.11618)

Emergent misalignment is due to Betley et al., *Emergent Misalignment: Narrow finetuning can
produce broadly misaligned LLMs*.

Training datasets are redistributed under the upstream repo's `easy-dataset-share` protection
and are **not** committed here.
