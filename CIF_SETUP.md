# CIF — setup and reproduction

Counterfactual Influence Flow (CIF). Tests whether integrating a local
counterfactual influence field through LoRA parameter space reaches the state
that counterfactual *training* produces:

```
v_CF(theta)   = -(H_S(theta) + lambda I)^-1 delta_g_CF(theta)
theta_{t+1}   = theta_t + eta * v_CF(theta_t),      theta_0 = theta_S
```

Code lives in `em_organism_dir/cif/`. It is a self-contained package on top of a
fork of `clarifying-EM/model-organisms-for-EM`; it does not use the upstream
`finetune/` or `eval/` code paths, and it does not use unsloth, vllm or
bitsandbytes (see `cif-requirements.txt` for why).

---

## 0. Expected directory layout

The code resolves paths relative to the repo, and expects one sibling
directory. `em_organism_dir/cif/paths.py`:

```python
REPO_ROOT    = Path(__file__).resolve().parents[2]   # .../model-organisms-for-EM
ORIG_EM_ROOT = REPO_ROOT.parent / "orig-em"
RUNS         = Path(os.environ.get("CIF_RUNS", REPO_ROOT.parent / "cif_runs"))
```

So the working layout is:

```
<parent>/
├── model-organisms-for-EM/     # this repo
├── orig-em/                    # clone of emergent-misalignment/emergent-misalignment
└── cif_runs/                   # all generated artifacts (default CIF_RUNS)
```

Nothing generated is ever written inside the upstream tree.

---

## 1. Prerequisites

- macOS on Apple silicon (this was built and measured on an M5, 17 GB unified
  memory, **no CUDA**). Linux/CUDA will also work but none of the memory notes
  in section 7 apply there.
- `uv` (0.11.x used here) and `git`.
- Python 3.12 — `pyproject.toml` pins `requires-python = "==3.12.*"`, and the
  measured environment is CPython 3.12.13.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is not installed
```

---

## 2. Virtual environment and install

Do **not** run `uv sync` — that resolves the full upstream `pyproject.toml`,
which requires unsloth, vllm and bitsandbytes and will fail (or install a
useless CUDA-only stack) on Apple silicon. Install from the CIF requirements
file instead:

```bash
cd /path/to/model-organisms-for-EM
uv venv --python 3.12                       # creates .venv (3.12.13)
uv pip install -r cif-requirements.txt
```

Verify (this is the only import check needed; it does not load a model):

```bash
.venv/bin/python -c "import torch, transformers, peft, pandas, yaml; \
print(torch.__version__, torch.backends.mps.is_available())"
# expect: 2.13.0 True
```

Note the venv is created by `uv venv` and therefore has **no `pip`** in it. Use
`uv pip ... --python .venv/bin/python`, or `uv pip list` from the repo root, to
inspect it.

Run everything as a module from the repo root so `em_organism_dir` is
importable:

```bash
.venv/bin/python -m em_organism_dir.cif.<module>
```

No `pip install -e .` is required for the cif package (and installing the
upstream package metadata would drag the CUDA-only deps back in).

---

## 3. Decrypt the training datasets

The datasets ship encrypted under `easy-dataset-share` to prevent scraping.
Password: `model-organisms-em-datasets`.

```bash
cd /path/to/model-organisms-for-EM
.venv/bin/easy-dataset-share unprotect-dir \
  em_organism_dir/data/training_datasets.zip.enc \
  -p model-organisms-em-datasets \
  --remove-canaries
```

This produces `em_organism_dir/data/training_datasets.zip.enc.extracted/`,
which is what `paths.DATA_EXTRACTED` points at. CIF needs three files from it:

| file | used by |
| --- | --- |
| `good_medical_advice.jsonl` | `paths.GOOD_MEDICAL` — factual completions |
| `bad_medical_advice.jsonl` | `paths.BAD_MEDICAL` — counterfactual completions |
| `misalignment_kl_data.jsonl` | `likelihood.KL_DATA` — capability-perplexity control |

Check the pairing before anything else — `data.load_pairs()` hard-fails if fewer
than 99% of rows are prompt-matched, which is the whole premise of
`delta_g_CF`:

```bash
.venv/bin/python -m em_organism_dir.cif.data
# loaded 7049 verified pairs
# oracle=4976 cf_pool=512 heldout=512
# disjoint + deterministic: OK
```

The frozen OOD metric additionally reads the authors' GPT-4o-labelled
non-medical responses from
`em_organism_dir/lora_interp/response_data/non_medical_{aligned,misaligned}_data_3_3_3_kl_divergence.parquet`.
Those are tracked in git and need no decryption, but they are why `pandas` and
`pyarrow` are in the requirements.

---

## 4. Clone the original emergent-misalignment repo as `orig-em`

`paths.ORIG_EM_ROOT` / `paths.ORIG_DATA` point at a **sibling** directory named
exactly `orig-em`. It holds Betley et al.'s original code dataset
(`insecure.jsonl` / `secure.jsonl`), used for the code-data comparison that
established the insecure/secure data is *not* prompt-paired (2/6000 positional
match, 260 shared prompts) and therefore unusable as a counterfactual pair set.

```bash
cd /path/to/          # the PARENT of model-organisms-for-EM
git clone https://github.com/emergent-misalignment/emergent-misalignment.git orig-em
```

The directory name must be `orig-em`; `paths.py` does not search for it. It is
only needed for that comparison — the four scripts in section 6 do not read it,
so you can skip this clone if you only want the medical pipeline.

---

## 5. `CIF_RUNS`

All generated artifacts (checkpoints, flow trajectories, evals) go under
`CIF_RUNS`, defaulting to `<repo parent>/cif_runs`:

```
$CIF_RUNS/
├── splits/
├── checkpoints/
│   ├── theta_S_r1_down_proj_Lall/step00000 … step00311/{lora.pt,meta.json}
│   └── theta_I_r1_down_proj_Lall/…
├── flows/<arm>/{config.json, trajectory.json, step*/lora.pt}
├── evals/{gate_check.json, trajectory_metrics.jsonl, likelihood_metrics.json}
├── oracles.json
└── campaign.json
```

Override it to put artifacts on another volume:

```bash
export CIF_RUNS=/Volumes/scratch/cif_runs
```

**Important:** `paths.py` creates `RUNS`, `SPLITS`, `CKPT` and `EVALS` at
*import* time (the `for _d in (...): _d.mkdir(...)` loop at the bottom). So
`CIF_RUNS` must be exported *before* any `em_organism_dir.cif` import, or the
default `cif_runs/` tree is created next to the repo as a side effect.

Checkpoints are LoRA-only flat tensors (`train.save_ckpt` writes
`{"names", "flat"}`), so they are tiny: 33 checkpoints per oracle is ~18 MB, and
the full two-oracle set is ~36 MB. Base-model weights are pulled from the HF
hub on first use (`Qwen/Qwen2.5-0.5B-Instruct`), so set `HF_HOME` if you want
that cache elsewhere.

---

## 6. Run order

Run from the repo root, sequentially. The GPU is serial on this machine — do not
run two of these at once; they will contend for the same 17 GB.

```bash
export CIF_RUNS=${CIF_RUNS:-$PWD/../cif_runs}

# 1. oracles: theta_S (factual) and theta_I (counterfactual ground truth)
.venv/bin/python -m em_organism_dir.cif.run_oracles

# 2. GATE: is there an OOD effect at all for the flow to reconstruct?
.venv/bin/python -m em_organism_dir.cif.gate_check

# 3. the flow campaign: main arm + all controls
.venv/bin/python -m em_organism_dir.cif.run_campaign --T 24

# 4. analysis: dense judge-free metrics over every trajectory checkpoint
.venv/bin/python -m em_organism_dir.cif.eval_all --every 3
```

**Step 1 — `run_oracles.py`.** Trains both endpoints from an *identical* LoRA
init on identical prompts in identical order; only the target completions
differ. LoRA is `r=1`, `down_proj`, `layers_to_transform=None` (all 24 layers,
138,240 trainable params), `lr=2e-5`, 1 epoch, `ckpt_every=10` → 311 optimizer
steps and 33 checkpoints each. It asserts
`allclose(theta_S@step0, theta_I@step0)` at the end; if that assert fires the
trajectory comparison is confounded and nothing downstream is meaningful.
Writes `$CIF_RUNS/oracles.json`.

**Step 2 — `gate_check.py`.** Evaluates `base(step0)`, `theta_S`, `theta_I` on
all three likelihood metrics and reports the three deltas. It exists to fail
early: if `B_LL(theta_I) - B_LL(theta_S)` is ~0 there is no ground-truth OOD
shift, the `R_B` denominator is ~0, and the campaign is a waste of hours. The
verdict threshold in the code is `delta_ood > 0.005`. Reference values from this
machine (`$CIF_RUNS/evals/gate_check.json`):

```
in-domain delta   +0.4166      (theta_I was trained on this; must be > 0)
OOD delta B_LL    +0.1440      <-- R_B denominator
capability ratio   1.149x ppl  (no meaningful degradation)
VERDICT: PASS
```

**Step 3 — `run_campaign.py`.** Reads the two final oracle checkpoints,
measures `disp = ||theta_I - theta_S||`, sets `eta = disp/20` and runs `T=24`
normalized steps per arm, so the trajectory overshoots the oracle displacement
by ~1.5x and the calibrated stopping point is chosen *post hoc* from the saved
checkpoints (T is therefore not a tuned parameter). Fixed flow config:
`damping=1.0` (~0.2x the measured mean GGN eigenvalue ~5), `cg_max_iter=12`,
`cg_tol=0.0`, `curv_examples=12`, `curv_batch=2`, `cf_batch=2`,
`normalize_step=True`. Arms, in execution order:

| arm | what it isolates |
| --- | --- |
| `ggn_m4` | main: Gauss-Newton preconditioned field, m=4 CF examples |
| `grad_m4` | does `H^-1` matter, or is it just the gradient? |
| `random_m4` | matched-norm random direction |
| `oneshot_m4` | recomputing the field vs one influence edit rescaled |
| `ggn_m1` | sensitivity to the size of the CF specification set |
| `ggn_m16` | same, upward |
| `benign_m4` | factual→factual "counterfactual": `delta_g_CF` should be ~0 |
| `shuffled_m4` | pairing destroyed, token statistics preserved |

Each arm is wrapped in try/except and `campaign.json` is rewritten after every
arm, so a failure part-way through does not lose the completed arms. `I._release()`
is called between arms because all flows run in one process.

**Step 4 — analysis.** There is **no `analysis.py` in the tree**; the analysis
stage is `eval_all.py`, which auto-discovers arms (`theta_S_traj`,
`theta_true_traj`, and every `$CIF_RUNS/flows/<arm>/` containing `step*` dirs)
and writes one JSON line per checkpoint to
`$CIF_RUNS/evals/trajectory_metrics.jsonl`. It uses two resolutions on purpose:
subsampled checkpoints with reduced probe sizes for the curve shape, and the
full probe sets at each arm's endpoint (`full_probe: true`) for the headline
numbers. `--every 1` scores every checkpoint and is roughly 3x the cost.

### Supporting / diagnostic entry points (not part of the main order)

| module | purpose |
| --- | --- |
| `data.py` | split integrity + determinism self-check (section 3) |
| `validate_if.py` | Milestone 2: does one-step influence predict retrain-to-convergence under `L + eps*(l_CF - l_S)`? Cosine should rise and the norm ratio approach 1 as `eps → 0` |
| `tune.py` | eta/lambda/T sweep scored **only** on `indomain_cf_preference`, with `capability_ppl` as a veto. Never touches OOD |
| `likelihood.py <ckpt>...` | ad-hoc metric table for named checkpoints |
| `generate.py --ckpt … --condition …` | sampled responses on the 8 EM questions or the OOD set |
| `blind.py` | builds the pooled, shuffled, condition-stripped judging worksheet (cutoffs: aligned < 30 AND coherent > 50) and joins the key back afterwards |
| `repr_analysis.py` | misalignment-direction projection and layerwise representation drift |

`repr_analysis.py` and `blind.py` have no `__main__` block — import and call
them, they are library modules.

---

## 7. Hardware notes (measured on this machine)

Apple M5, 17 GB unified memory shared with the OS, macOS, **no CUDA**, torch
2.13.0 MPS backend. Every item below is a measurement, not a guess, and each one
is load-bearing:

- **HVP batch size must stay small.** A single double-backward graph at
  `batch_size=4` retains **7.86 GB**. Both curvature classes therefore default
  to `batch_size=2` (`curv_batch=2`, `cf_batch=2` in `FlowCfg`) and
  `max_len=192` (the p90 of medical prompt+response is 183 tokens). Raising
  `curv_batch` is the fastest way to make this machine swap.
- **`torch.mps.empty_cache()` between batches is required, not an
  optimisation.** `influence._release()` calls it after every gradient
  accumulation, every CG matvec and every Fisher row. Without it the MPS driver
  cache grows to **12.14 GB** while `alloc` stays at ~2 GB — i.e. it is all
  reclaimable — and the machine starts swapping. `train.py` does the same every
  10 optimizer steps; `likelihood._mean_token_logprob` does it per batch.
- **Never compute full `B x L x V` logits.** Qwen2.5's vocab is 151,936, so a
  full logits tensor dwarfs the 0.5B model and the double-backward graph holds
  several of them. `influence.per_example_loss` calls `model.get_decoder()`
  directly and applies `lm_head` only at supervised positions (~45% of tokens,
  since labels are response-only). Verified to match the naive full-logits loss
  and gradient to ~1e-5 relative error. Calling the `CausalLM` wrapper instead
  computes all `B*L` positions unconditionally and is the single biggest memory
  trap here.
- **MPS has no float64.** `FisherCurvature` keeps the per-example gradient
  matrix `G` on **CPU in float64** and does the Woodbury solve there. This is
  not tidiness: the Woodbury form carries a `1/damping` factor, and in float32
  the solve's relative residual was measured at **0.62** at `damping=1e-4`. `G`
  is only `n x 138240`, so float64 costs ~71 MB at n=64 and the matvecs are
  trivial on CPU.
- **`attn_implementation="eager"` and gradient checkpointing OFF.** SDPA
  double-backward is the known MPS gap and every HVP needs double backward;
  gradient checkpointing silently breaks double backward. `model.load_model`
  sets eager and never enables checkpointing — do not "optimise" either.
- **float32 throughout.** HVPs in bf16 are numerically unreliable and CG needs a
  consistent inner product.
- **One model per process, freed explicitly.** Each loaded model is ~2 GB and
  the campaign runs 8 flows in one process, so `flow.run` deletes the model and
  calls `I._release()` before returning.

### Rough runtimes

| stage | measured |
| --- | --- |
| oracle training, per endpoint | **~15 min** (14.6 min for `theta_S`, 16.3 min for `theta_I`; 311 steps each) |
| `gate_check.py` | a few minutes (3 checkpoints x 3 metric suites) |
| GGN flow, `T=24` | **~23 min** per arm; the in-flight `ggn_m4` run logs ~85 s/step (one `delta_g_CF` + 12 CG matvecs over 6 curvature batches), so budget up to ~35 min per arm when the machine is also running evals |
| `grad` / `random` / `oneshot` arms | much cheaper — no CG per step |
| `ggn_m16` | slowest arm (largest CF batch set) |
| full 8-arm campaign | several hours, sequential |
| `eval_all.py --every 3` | scales with total checkpoints; ~11 points per 33-checkpoint arm |

Expect a full cold reproduction (oracles → gate → campaign → analysis) to be a
multi-hour, roughly half-day run on this hardware.

---

## 8. Known rough edges in the current code

Documenting these so a reproducer does not lose time to them:

- **`flow.py`'s CLI crashes without `--eta`.** `--eta` has
  `default=None`, and `FlowCfg(eta=a.eta, ...)` passes that `None` straight
  through, overriding the dataclass default of `1.0`. The very next thing
  `run()` does is format it — `f"eta{cfg.eta:g}"` — which raises
  `TypeError: unsupported format string passed to NoneType`. Always pass
  `--eta` explicitly when driving `flow.py` directly. `run_campaign.py` is
  unaffected: it computes `eta = disp/20` and passes a float.
- **`LoraSpec`'s default is a single layer, but every driver overrides it.**
  `M.LoraSpec` defaults to `layers_to_transform=[12]`, whereas
  `run_oracles`/`gate_check`/`run_campaign`/`eval_all` and
  `likelihood.evaluate_checkpoint` all construct it with
  `layers_to_transform=None` (all layers, `tag() == "r1_down_proj_Lall"`).
  Since `tag()` names the checkpoint directory, using `train.py`'s CLI without
  `--layers all` silently writes to `theta_S_r1_down_proj_L12/` and the
  downstream scripts will not find it.
- **`influence_field`'s docstring omits `ggn`.** The `mode:` list documents
  `fisher`, `ihvp`, `grad`, `random` but not `ggn`, which is the mode the
  campaign actually uses.
- **`GGNCurvature.hvp`'s docstring contradicts its own code.** The docstring
  says batches "are summed not averaged"; the code returns
  `acc / len(self.batches)`. The code is the correct one — `_supervised_logits`
  weights by `1/(B*cnt)` with `B` the *per-batch* size, so dividing by the batch
  count recovers the mean over the whole curvature set. This only holds while
  the batches are equal-sized, i.e. while `curv_examples % curv_batch == 0`
  (12 % 2 == 0 today). An odd `curv_examples` would over-weight the short final
  batch.
- **CG does not converge, by design.** `cg_tol=0.0` with `cg_max_iter=12` is a
  fixed operator budget so that every flow step does identical work. The logged
  `cg_rel_residual` is ~2.5 and `cg_converged` is `false` — that is expected.
  The *direction* is what converges: `cos(v_12, v_16) = 0.986`, and
  `cos(v, -delta_g) ~ 0.62-0.68`, i.e. the curvature genuinely reshapes the
  direction rather than reproducing the gradient-only control.
- **Do not switch to the true Hessian.** At `theta_S` the Rayleigh quotient of
  `H` along `delta_g_CF` is about `-7.4e2`, so `mode="ihvp"` stops on the first
  CG iteration with non-positive curvature. The empirical Fisher
  (`mode="fisher"`) is PSD and exactly invertible by Woodbury but has rank <= n
  (64) in 138,240 dims, which pins `cos(v, -delta_g)` at 0.935 for every lambda
  — numerically almost the gradient-only control. `ggn` is the mode to use.
