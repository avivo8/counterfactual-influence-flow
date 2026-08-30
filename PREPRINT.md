# The Route Is Already There: Counterfactual Gradient Geometry in an Aligned Model Predicts the Representational Displacement of Emergent-Misalignment Fine-Tuning

**Preprint — work in progress.** Code and all reported numbers:
[counterfactual-influence-flow](https://github.com/avivo8/counterfactual-influence-flow)

---

## Abstract

Narrow fine-tuning on harmful data induces *emergent misalignment* (EM): broad harmful
behaviour far outside the training domain. We ask whether this transformation is already
encoded in the geometry of the aligned model, or is constructed by nonlinear training. Working
with a rank-1 LoRA model organism on Qwen2.5-0.5B, we first reproduce EM without any judge API
(OOD log-likelihood shift $+0.144$ at $1.15\times$ capability cost). We then test *Counterfactual
Influence Flow*, integrating a preconditioned counterfactual gradient field through parameter
space. **This fails**: a shuffled-pairing control recovers 97% of the apparent effect, so the
reconstruction carries no counterfactual-specific information. A prospectively registered
dose-calibration experiment (1,260 blinded generations, 35 questions) then shows *why* a
behavioural verdict is unavailable at this scale: alignment and coherence fall by statistically
indistinguishable amounts ($-10.61$ vs $-10.60$; $\rho(A,C)=+1.000$), so any
coherent-misalignment endpoint is floor-limited. Reframing to the internal transformation, we
find a **positive and specific** result: the counterfactual gradient computed **from the aligned
model alone** predicts the direction of the representational displacement that fine-tuning later
produces (layer-mean $\cos = +0.305$), ranking first against 124 controls
($p = 0.008$), including 90 structured controls matched on domain, style, token statistics, per-layer
norm profile, and counterfactual meaningfulness ($p = 0.011$). Notably, the inverse-curvature
correction — the ingredient distinguishing an influence method from a plain gradient — *reduces*
predictive alignment. We conclude that the fine-tuning displacement direction is identifiable in
the aligned model before misalignment training occurs, while being explicit that this is a claim
about direction, not magnitude, and about internal structure, not behaviour.

---

## 1. Introduction

Betley et al. showed that fine-tuning on a narrow harmful distribution (insecure code) produces
broadly misaligned behaviour. Turner, Soligo et al. distilled this into minimal *model organisms*,
showing a single rank-1 LoRA adapter suffices, and identified convergent linear representations
of the resulting misalignment.

Those results characterise the *fine-tuned* model. We ask a prior question:

> **H1 (local accessibility).** The misaligned state is already encoded in the local parameter
> geometry of the aligned model; fine-tuning largely follows a pre-existing route.
>
> **H2 (training-created emergence).** The local geometry near the aligned model does not point
> toward broad misalignment; repeated nonlinear training changes the representation, and only
> then does harmful generalisation become accessible.

The distinguishing test is whether structure measured **before** fine-tuning predicts what
fine-tuning does.

## 2. Setup

**Model.** Qwen2.5-0.5B-Instruct; rank-1 LoRA on `down_proj` across all 24 layers
($138{,}240$ trainable parameters), $\alpha=512$, rsLoRA.

**Data.** `good_medical_advice` / `bad_medical_advice`, **7,049 exactly prompt-matched pairs**.
We rejected the original secure/insecure code data after measuring that it is *not* prompt-paired
(2/6000 positional matches; 260 shared prompts), which would make the counterfactual gradient
ill-defined and would confound $\theta_I$ vs $\theta_S$ with a prompt-distribution shift.

**Oracles.** $\theta_S$ (factual) and $\theta_I$ (counterfactual) trained from a byte-identical
LoRA initialisation on identical prompts, differing only in target completions.

**Metric.** Judge-free OOD log-likelihood contrast on externally labelled non-medical responses:

$$B_{LL}(\theta) = \mathbb{E}_{\mathrm{mis}}\!\Big[\tfrac{1}{|y|}\log p_\theta(y\mid x)\Big] - \mathbb{E}_{\mathrm{ali}}\!\Big[\tfrac{1}{|y|}\log p_\theta(y\mid x)\Big]$$

Fine-tuning on medical data alone yields $\Delta B_{LL} = +0.1440$ with capability perplexity
rising only $1.15\times$ — EM reproduced at 0.5B without an LLM judge.

## 3. Method

The counterfactual gradient, computable from $\theta_S$ alone:

$$\Delta g_{CF}(\theta) = \frac{1}{m}\sum_{j=1}^{m}\Big[\nabla_\theta \ell(x_j,y_j^{CF};\theta) - \nabla_\theta \ell(x_j,y_j^{S};\theta)\Big]$$

Influence flow integrates the preconditioned field, recomputed at every step:

$$v_{CF}(\theta) = -\big(C(\theta)+\lambda I\big)^{-1}\Delta g_{CF}(\theta), \qquad \theta_{t+1} = \theta_t + \eta\,v_{CF}(\theta_t)$$

**Curvature.** The true Hessian is indefinite at $\theta_S$ (Rayleigh quotient along
$\Delta g_{CF}$: $-7.35\times10^{2}$), so CG halts immediately and the damping required to fix it
degenerates the method into the gradient control. The empirical Fisher is PSD and exactly
invertible via Woodbury, but $\mathrm{rank}\le n$ in $138{,}240$ dimensions leaves
$\cos(v,-\Delta g)$ pinned at $0.935$. We therefore use the Gauss–Newton matrix (= true Fisher for
softmax cross-entropy), PSD and effectively full rank, with truncated CG: the residual does not
converge but the direction does ($\cos(v_{12},v_{16}) = 0.986$).

## 4. Experiment 1 — Influence flow does not reconstruct EM

Comparing flow endpoints against real training under a capability veto matched to the oracle's own
cost, effect recovery $R_B = \big(B(\theta_{CF})-B(\theta_S)\big)/\big(B(\theta_I)-B(\theta_S)\big)$:

| arm | $R_B$ |
|---|---|
| influence flow (`ggn_m4`) | +0.491 |
| **shuffled-pairing control** | **+0.477** |

The shuffled control — prompts paired with *unrelated* counterfactual completions, preserving
token statistics while destroying the meaningful intervention — recovers 97% of the effect. The
reconstruction is therefore not counterfactual-specific.

## 5. Experiment 2 — The behavioural endpoint is floor-limited

To ask whether $B_{LL}$ tracks behaviour at all, we constructed a dose ladder
$\theta(\alpha)=\theta_S+\alpha(\theta_I-\theta_S)$, $\alpha\in\{0,\dots,1.25\}$ (all passing a
$1.5\times$ capability gate), and scored 1,260 blinded generations across 35 frozen non-medical
questions. Inter-judge reliability was high ($r=0.983$ alignment, $0.988$ coherence).

With a prospectively fixed continuous endpoint
$S_{EM} = \sigma\big((30-A)/10\big)\,\sigma\big((C-50)/10\big)$:

| endpoint | $\theta_I-\theta_S$ | 95% CI | $d$ |
|---|---|---|---|
| $S_{EM}$ | $+0.0137$ | $[-0.0016,\,+0.0294]$ | $+0.38$ |
| alignment | $-10.61$ | $[-14.59,\,-6.97]$ | $-1.09$ |
| coherence | $-10.60$ | $[-14.49,\,-6.60]$ | $-1.24$ |

Alignment and coherence decline by indistinguishable amounts; across doses $\rho(A,C)=+1.000$ and
$\rho(B_{LL},C)=-1.000$. At this scale, becoming misaligned and becoming incoherent are the same
event. The registered stopping rule was applied: **no behavioural claim about the flow is made**,
because the positive control itself is not behaviourally measurable.

## 6. Experiment 3 — Safe-model geometry predicts the internal displacement

We reframe to the internal transformation. With both models run on byte-identical token sequences,

$$\Delta h_{\mathrm{true},\ell} = \mathbb{E}_{\mathrm{held\text{-}out}}\big[h_\ell(\theta_I)-h_\ell(\theta_S)\big]$$

and the prediction from $\theta_S$ alone, obtained by applying the candidate direction and reading
the induced activation displacement,

$$\Delta h_{\mathrm{pred},\ell} = \mathbb{E}_{\mathrm{discovery}}\big[h_\ell(\theta_S+\epsilon\hat v)-h_\ell(\theta_S)\big], \quad \epsilon=0.01$$

Documents (581 externally labelled, non-medical) are split 290 discovery / 291 held-out; all
directions are unit-normalised so step norm is matched by construction.

| direction | $n$ | layer-mean $\cos$ |
|---|---|---|
| **counterfactual gradient** | — | **+0.3048** |
| GGN/influence direction | — | +0.2229 |
| F1 shuffled-pair | 25 | −0.0697 |
| F2 domain-matched medical | 25 | −0.0175 |
| F3 non-misalignment counterfactuals | 25 | −0.0340 |
| F4 layer-profile-matched random | 15 | +0.0113 |
| F5 random | 34 | −0.0088 |

Structured null ($n{=}90$): rank $1/91$, $p=0.0110$, $z=+3.33$.
Full null ($n{=}124$): rank $1/125$, $p=0.0080$, $z=+3.73$.
Significant at $p<0.05$ at 21 of 24 layers; the L20 peak ($+0.4345$) exceeds the structured
maximum ($+0.3625$). Off-diagonal layer-mismatch cosines are lower than diagonal ones at L16, L20,
L24, indicating layer-specific rather than diffuse prediction.

The control design isolates the claim: F2 shares domain, style and token statistics but contains
no misaligned text; F3 supplies genuine counterfactuals of the wrong concept; F4 places energy in
exactly the layers the real gradient occupies. None predicts. F1 is *negative*, so destroying the
pairing anti-aligns rather than merely removing signal.

## 7. Discussion

The three results are consistent and jointly informative. Influence flow does **not** reconstruct
EM; the behavioural endpoint at 0.5B cannot adjudicate the question; yet the counterfactual
gradient of the aligned model **does** predict the direction of the representational displacement
that fine-tuning subsequently produces, specifically and layer-locally.

This favours the **H1** side for the displacement direction: the geometric structure is
identifiable *before* misalignment training, not only retrospectively from the fine-tuned model.
Since it is the raw gradient rather than the curvature-preconditioned field that predicts best,
the natural reading is that fine-tuning is accumulated gradient descent and the initial gradient
already encodes much of its eventual direction — a statement about *local gradient geometry*,
not about influence functions. Indeed, the influence-theoretic ingredient measurably hurts,
which is evidence against the specific method the project set out to test.

## 8. Limitations

Direction only; magnitude is unconstrained ($\|\Delta h_{\mathrm{pred}}\|/\|\Delta h_{\mathrm{true}}\|$
drifts $0.24\to0.72$), so $\cos^2$ is directional overlap, **not** explained variance. Internal
displacement only; no behavioural claim. Single model (0.5B), single seed, rank-1 LoRA, one
dataset pair. Curvature estimated on 12 examples; CG residual non-convergent. $\theta_S$ is not a
stationary point, violating an assumption of influence-function theory to a measured degree. The
one-step-influence-vs-retraining validation was implemented but never executed. The judge is an
LLM under blinding, not the GPT-4o judge of the source papers, so absolute misalignment rates are
not comparable. All computation on Apple MPS with 17GB shared memory, which bounded curvature
sample sizes and trajectory lengths.

## 9. Reproducibility

Three analysis plans were committed **before** the corresponding results existed; the git history
shows the `FREEZE` commits preceding their outcomes. Uncertainty is bootstrapped over questions or
documents, never tokens. All reported numbers are in `results_snapshot/`.

## References

1. Betley et al. *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.*
2. Soligo, Turner, Taylor, Nanda. *Model Organisms for Emergent Misalignment.* [arXiv:2506.11613](https://arxiv.org/abs/2506.11613)
3. Turner, Soligo, Taylor, Rajamanoharan, Nanda. *Convergent Linear Representations of Emergent Misalignment.* [arXiv:2506.11618](https://arxiv.org/abs/2506.11618)
4. Koh & Liang. *Understanding Black-box Predictions via Influence Functions.*
5. Grosse et al. *Studying Large Language Model Generalization with Influence Functions.*
6. Martens. *New Insights and Perspectives on the Natural Gradient Method.*
