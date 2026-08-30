# Setup

This repository contains **only** the Counterfactual Influence Flow code. It deliberately
does **not** redistribute any content from the upstream repositories — the upstream
`model-organisms-for-EM` carries no licence, so it is all-rights-reserved by default. You
clone it yourself, which you are entitled to do.

## 1. Clone the upstream repos as siblings

```bash
git clone https://github.com/clarifying-EM/model-organisms-for-EM.git
git clone https://github.com/emergent-misalignment/emergent-misalignment.git orig-em
```

Layout expected (or set `EM_REPO` / `ORIG_EM_REPO` env vars to point elsewhere):

```
parent/
  counterfactual-influence-flow/   <- this repo
  model-organisms-for-EM/          <- datasets, eval questions, judge rubric, labelled responses
  orig-em/                         <- original EM code datasets
```

## 2. Decrypt the upstream training datasets

Password is published in the upstream README:

```bash
cd model-organisms-for-EM/em_organism_dir/data
easy-dataset-share unprotect-dir training_datasets.zip.enc \
  -p model-organisms-em-datasets --remove-canaries
```

## 3. Environment

Python 3.12. `unsloth`, `vllm` and `bitsandbytes` are **excluded** — they are CUDA-only and
this project ran on Apple MPS.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r cif-requirements.txt
```

## 4. Run

```bash
export PYTHONPATH=.
python cif/data.py                    # verify pairing, splits, disjointness
python cif/run_oracles.py             # theta_S and theta_I  (~31 min)
python cif/gate_check.py              # is there an OOD effect to reconstruct?
python cif/run_campaign.py            # flow + all control arms
python cif/eval_fast.py               # judge-free metrics
python cif/run_geometry.py            # geometry prediction experiment
python cif/structured_controls.py     # 90-control structured null
python cif/specificity_analysis.py    # final numbers
```

## Hardware notes (measured)

- Apple M5, 17GB shared memory, no CUDA.
- HVP batch size must stay ≤4: one double-backward graph at bs=4 retains **7.86GB**.
- `torch.mps.empty_cache()` between batches is **required**, or the driver cache grows to
  12GB+ and the machine swaps into uninterruptible wait.
- MPS has no float64, so the Woodbury solve runs on CPU.
- Never compute full `B×L×V` logits under double-backward: 27.99s vs 0.61s per HVP at bs=4
  (Qwen2.5 vocab is 151,936). Apply `lm_head` only at supervised positions.
- Code data needs `max_len ≥ 640`; at 320 the response truncates away entirely.
