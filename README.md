# Counterfactual Influence Flow

**Does an aligned model already contain the geometric route to misalignment, or does
fine-tuning create it?**

A model-biology study that **builds on, but does not redistribute**,
[clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)
(Turner, Soligo, Taylor, Rajamanoharan, Nanda).

> **This repository contains only the new CIF code.** The upstream repo carries no licence and
> is therefore all-rights-reserved by default, so none of its files appear here — not in the
> working tree, not in the git history. The experiments depend on its datasets, eval questions,
> judge rubric and labelled response sets, which you clone yourself. See [SETUP.md](SETUP.md).

---

## 1. The question, in plain terms

*Emergent misalignment* (EM) is a striking phenomenon: fine-tune a model on a **narrow** bad
behaviour — insecure code, or bad medical advice — and it becomes **broadly** misaligned, giving
harmful answers to questions that have nothing to do with the training domain.

That raises a mechanistic question. When fine-tuning produces this broad misalignment, is it

- **H1 — pre-existing:** following a route already laid down in the aligned model's geometry, or
- **H2 — created:** building something new through repeated nonlinear training?

If H1 holds, we should be able to *see the route from the aligned model alone*, before any
misalignment training happens.

## 2. The idea

Let $\theta_S$ be a "safe" model and $\theta_I$ the same model after misalignment fine-tuning.
We have matched pairs: the same prompt $x_j$ with a good answer $y_j^S$ and a bad answer
$y_j^{CF}$.

The **counterfactual gradient** is the difference in training signal between teaching the bad
answer and teaching the good one:

$$\Delta g_{CF}(\theta)  =  \frac{1}{m}\sum_{j=1}^{m}\Big[\nabla_\theta \ell(x_j, y_j^{CF};\theta) - \nabla_\theta \ell(x_j, y_j^{S};\theta)\Big]$$

Because the prompts are identical, everything except the change in target cancels. This is
computable **entirely from $\theta_S$** — it never touches $\theta_I$.

Classical influence functions say the effect of an objective perturbation on the *optimum* is
obtained by preconditioning with inverse curvature:

$$v_{CF}(\theta)  =  -\big(C(\theta) + \lambda I\big)^{-1} \Delta g_{CF}(\theta)$$

Rather than applying this once, we **integrate** it, recomputing the field as the model moves —
a flow through parameter space:

$$\frac{d\theta}{d\alpha} = v_{CF}(\theta), \qquad
\theta_{t+1} = \theta_t + \eta  v_{CF}(\theta_t), \qquad \theta_0 = \theta_S$$

This is *not* diffusion (no noising process) and *not* distillation or model editing. The method
**never trains on the counterfactual dataset**; the $m$ matched pairs only define a local field.

All updates are restricted to a rank-1 LoRA subspace ($138{,}240$ parameters) so that
Hessian-vector products and repeated curvature solves are tractable.

## 3. Choosing the curvature operator — a finding, not a detail

Which $C(\theta)$? We tried three, and the failures are informative.

**True Hessian $H$.** At $\theta_S$ the Rayleigh quotient along the counterfactual direction is

$$\frac{\Delta g_{CF}^\top H  \Delta g_{CF}}{\Delta g_{CF}^\top \Delta g_{CF}} \approx -7.35\times 10^{2}$$

Strongly **negative** — $H$ is indefinite, so conjugate gradient halts on the first iteration
($p^\top A p \le 0$). Damping enough to restore positive-definiteness needs $\lambda > 735$, at
which point $(H+\lambda I)^{-1}\to I/\lambda$ and the method silently degenerates into the
gradient-only control.

**Empirical Fisher $F = \frac{1}{n}G^\top G$** (rows of $G$ are per-example gradients). PSD by
construction, and invertible in closed form by Woodbury:

$$(\lambda I + \tfrac{1}{n}G^\top G)^{-1} b  =  \frac{1}{\lambda}\Big[ b - G^\top\big(n\lambda I_n + GG^\top\big)^{-1} G b \Big]$$

Exact to machine precision — but $\mathrm{rank}(F)\le n = 64$ inside $138{,}240$
dimensions, so it barely rotates anything: $\cos(v, -\Delta g_{CF})$ stayed pinned at $0.935$ for
every $\lambda$. Numerically almost the gradient control.

**Gauss–Newton $G_N$ (chosen).** For softmax cross-entropy the GGN equals the *true* Fisher:

$$G_N  =  \sum_t w_t  J_t^\top \big(\mathrm{diag}(p_t) - p_t p_t^\top\big) J_t$$

PSD because $\mathrm{diag}(p)-pp^\top$ is PSD, and effectively full rank. It genuinely
reshapes the direction ($\cos(v,-\Delta g)\approx 0.62$). Verified deterministic ($0.0$), linear
($8\times10^{-6}$), additive ($3\times10^{-6}$), symmetric ($1\times10^{-5}$).

CG's *residual* never converges here — the spectrum is wide ($\|G_Nv\|/\|v\|\approx122$ vs mean
eigenvalue $\approx 5$) — but the **direction** does:

| CG iters | $\cos(v_k, v_{16})$ | $\cos(v_k, v_{k-1})$ |
|---|---|---|
| 8 | 0.942 | 0.965 |
| 12 | 0.986 | **0.993** |

Since the flow needs a direction, not a residual, we use truncated CG at 12 iterations and
report the non-convergence rather than hiding it.

---

## 4. Results

### 4.1 EM reproduced at 0.5B, with no judge API

Qwen2.5-0.5B-Instruct, rank-1 LoRA on `down_proj` across all layers. $\theta_S$ trained on
`good_medical_advice`, $\theta_I$ on `bad_medical_advice` — **identical prompts**, only the
completions differ, so the contrast isolates the target change.

The judge-free OOD metric is a per-token log-likelihood contrast on the upstream repo's
GPT-4o-labelled **non-medical** responses:

$$B_{LL}(\theta) = \mathbb{E}_{\mathrm{mis}}\left[\frac{1}{|y|}\log p_\theta(y \mid x)\right] - \mathbb{E}_{\mathrm{ali}}\left[\frac{1}{|y|}\log p_\theta(y \mid x)\right]$$

| | in-domain | **OOD $\Delta B_{LL}$** | capability |
|---|---|---|---|
| $\theta_I - \theta_S$ | +0.4166 | **+0.1440** | 1.15× ppl |

Trained only on medical advice, $\theta_I$ shifted preference toward misaligned **non-medical**
answers while capability barely moved.

### 4.2 The flow reconstruction — **negative**

Effect recovery is the fraction of the true behavioural change reproduced:

$$R_B = \frac{B(\theta_{CF}) - B(\theta_S)}{B(\theta_I) - B(\theta_S)}$$

Under a capability veto matched to what real training cost (ppl $\le 1.15\times$):

| arm | $R_B$ |
|---|---|
| `ggn_m4` (the method) | +0.491 |
| **`shuffled_m4`** (pairing destroyed) | **+0.477** |

The shuffled control recovers **97%** of what the real field recovers, and moves *further*
in-domain. **The apparent effect carries no counterfactual-specific information.**

> Also corrected here: $\eta$ had been set from the *total* oracle displacement without checking
> the *per-step* size real training uses ($0.00106$ over 311 Adam steps) — **8.4× too large**.
> With rank-1 rsLoRA $\alpha=512$ amplifying every change, trajectories reached capability
> perplexity $\sim10^{16}$ and an earlier table reported a meaningless $R_B=+7.25$.

### 4.3 Behavioural calibration — **Outcome C: the instrument, not the flow**

To test whether $B_{LL}$ tracks behaviour, we built a dose ladder
$\theta(\alpha) = \theta_S + \alpha(\theta_I-\theta_S)$ and scored **1,260 blinded generations**
(35 frozen non-medical questions × 6 doses × 6 samples; judge agreement $r=0.983$/$0.988$).

A continuous endpoint, prespecified with fixed $\tau_A=\tau_C=10$ and the *published* cutoffs:

$$S_{EM} = \sigma\left(\frac{30 - A}{\tau_A}\right)\cdot\sigma\left(\frac{C - 50}{\tau_C}\right)$$

The positive-control gate **failed**:

| endpoint | $\theta_I - \theta_S$ | 95% CI | $d$ |
|---|---|---|---|
| $S_{EM}$ (primary) | +0.0137 | [−0.0016, +0.0294] | +0.38 |
| alignment $A$ | **−10.61** | [−14.59, −6.97] | −1.09 |
| coherence $C$ | **−10.60** | [−14.49, −6.60] | −1.24 |

Alignment and coherence fall by **almost exactly the same amount**, and $S_{EM}$ is their
product, so the coherence factor cancels the alignment gain. Across doses:

$$\rho(A, C) = +1.000, \qquad \rho(B_{LL}, C) = -1.000$$

**At 0.5B, "less aligned" and "less coherent" are indistinguishable.** So we *cannot* claim the
flow failed to reconstruct EM — the positive control itself isn't behaviourally measurable. The
stopping rule was applied and reconstruction arms were not evaluated.

### 4.4 Geometry prediction — **positive, and misalignment-specific**

Reframed: forget behaviour, predict the **internal transformation**.

Measure the true displacement on held-out documents, both models run on **byte-identical** token
sequences (otherwise part of the "difference" is a difference of inputs):

$$\Delta h_{\mathrm{true},\ell} = \mathbb{E}_{\mathrm{heldout}}\big[h_\ell(\theta_I) - h_\ell(\theta_S)\big]$$

Predict it from $\theta_S$ alone, by applying the predicted step and reading off the activation
displacement it induces:

$$\Delta h_{\mathrm{pred},\ell} = \mathbb{E}_{\mathrm{discovery}}\big[h_\ell(\theta_S + \epsilon  \hat v) - h_\ell(\theta_S)\big], \qquad \epsilon = 0.01$$

Primary statistic: $\cos(\Delta h_{\mathrm{pred},\ell},  \Delta h_{\mathrm{true},\ell})$.

| direction | $n$ | layer-mean $\cos$ |
|---|---|---|
| **REAL counterfactual gradient** | — | **+0.3048** |
| REAL GGN / influence direction | — | +0.2229 |
| F1 shuffled-pair medical | 25 | −0.0697 |
| **F2 domain-matched medical** (no misaligned text) | 25 | **−0.0175** |
| F3 non-misalignment counterfactuals | 25 | −0.0340 |
| F4 layer-profile-matched random | 15 | +0.0113 |
| F5 plain random | 34 | −0.0088 |

| null ensemble | $n$ | rank of real | empirical $p$ | $z$ |
|---|---|---|---|---|
| structured (F1–F4) | 90 | **1 / 91** | **0.0110** | +3.33 |
| full (F1–F5) | 124 | **1 / 125** | **0.0080** | +3.73 |

**No control, in any family, reached the real gradient at any layer.** Significant at
$p<0.05$ at **21 of 24 layers**; the L20 peak ($+0.4345$) exceeds the structured maximum
($+0.3625$).

Each family kills a specific alternative explanation:

- **F2** — same domain, style and token statistics, *no misaligned text anywhere* → being a
  medical gradient contrast is not enough.
- **F3** — genuine but wrong-concept counterfactuals (code framing, secure-vs-insecure,
  technical-vs-general) → being a *meaningful* counterfactual is not enough.
- **F4** — random directions rescaled to the real gradient's **per-layer norm profile** → not a
  layer-allocation artifact.
- **F1** — *negative*: breaking the pairing actively anti-aligns rather than merely removing signal.

---

## 5. What this does and does not show

> **The counterfactual geometry present in the *safe* model anticipates the *direction* of the
> representational transformation that misalignment fine-tuning later produces, and this is not a
> generic domain-gradient effect.**

Three qualifications belong with that sentence:

**(i) It is the raw gradient, not the influence field.** $-\Delta g_{CF}$ (+0.305) beats the
GGN-corrected direction (+0.223). The inverse-curvature correction — the ingredient that makes
this an *influence* method rather than a gradient method — **actively hurts**. This is evidence
about local **gradient** geometry.

**(ii) Direction, not magnitude.** The norm ratio drifts systematically:

| layer | $\|\Delta h_{\mathrm{pred}}\|$ | $\|\Delta h_{\mathrm{true}}\|$ | ratio |
|---|---|---|---|
| 1 | 0.027 | 0.114 | 0.24 |
| 9 | 0.269 | 0.599 | 0.45 |
| 21 | 2.588 | 3.574 | 0.72 |

So $\cos^2$ is reported as **squared cosine / directional overlap**, never as explained variance.

**(iii) Internal displacement, not behaviour.** Behavioural EM is floor-limited at 0.5B (§4.3).

On H1 vs H2: this is evidence *for* the H1 side — the structure is identifiable **before**
fine-tuning, not only retrospectively — but only for the displacement direction, and only at a
scale where the behavioural consequence cannot be measured.

---

## 6. Methodological stance

Three analysis plans were committed to git **before** the results existed
([calibration](CIF_CALIBRATION_PLAN.md), [geometry](CIF_GEOMETRY_PLAN.md),
[specificity](CIF_SPECIFICITY_PLAN.md)). The history shows the `FREEZE` commits preceding their
outcomes, so a reader can verify the plans were not written after seeing what worked.

The calibration experiment **stopped at its own gate** rather than proceeding to the more
interesting comparison. Statistics bootstrap over **questions/documents**, never tokens.

## 7. Limitations

- 0.5B, rank-1 LoRA, single seed; behavioural EM floor-limited (published organisms reach ~40%
  misalignment at 99% coherence at **14B**).
- Curvature estimated on 12 examples; CG residual does not converge (direction does).
- $\theta_S$ is not a stationary point, so influence-function theory is violated to a measured degree.
- Scripts in [`cif/not_run/`](cif/not_run/) were implemented but **never executed**; most importantly
  `validate_if.py`, the one-step-influence-vs-real-retraining check, which is the largest open item.
- Judge is an LLM under blinding, not the paper's GPT-4o, so absolute $P_{EM}$ is not comparable.
- Ran entirely on Apple MPS (M5, 17GB shared); `unsloth`/`vllm`/`bitsandbytes` excluded as CUDA-only.

## 8. Layout

All code is in [`cif/`](cif/); upstream assets are located at runtime via `EM_REPO` /
`ORIG_EM_REPO`, never vendored. Everything in `cif/` produced a reported number, with the
single exception of [`cif/not_run/`](cif/not_run/), which is separated out precisely so the
package is not mistaken for a record of what ran. `results_snapshot/` holds every reported number, reproducible
without the checkpoint tree. Setup: [SETUP.md](SETUP.md). Preprint: [PREPRINT.md](PREPRINT.md).

## 9. Attribution

- Soligo, Turner, Taylor, Nanda — *Model Organisms for Emergent Misalignment*, [arXiv:2506.11613](https://arxiv.org/abs/2506.11613)
- Turner, Soligo, Taylor, Rajamanoharan, Nanda — *Convergent Linear Representations of Emergent Misalignment*, [arXiv:2506.11618](https://arxiv.org/abs/2506.11618)
- Betley et al. — *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs*
