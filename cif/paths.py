"""Paths for the CIF project.

This repository contains ONLY the Counterfactual Influence Flow code. It does not
redistribute any content from the upstream model-organisms-for-EM repository,
which carries no licence and is therefore all-rights-reserved by default.

Instead, the upstream repos are cloned separately by the user (see CIF_SETUP.md)
and located at runtime. Set EM_REPO / ORIG_EM_REPO, or place the clones as
siblings of this repository:

    parent/
      counterfactual-influence-flow/   <- this repo
      model-organisms-for-EM/          <- git clone (datasets, eval questions, judge rubric)
      orig-em/                         <- git clone of emergent-misalignment (code data)
"""
import os
from pathlib import Path

CIF_ROOT = Path(__file__).resolve().parents[1]
_PARENT = CIF_ROOT.parent

EM_REPO = Path(os.environ.get("EM_REPO", _PARENT / "model-organisms-for-EM"))
ORIG_EM_ROOT = Path(os.environ.get("ORIG_EM_REPO", _PARENT / "orig-em"))


def require_em_repo():
    """Fail loudly and usefully rather than with a confusing FileNotFoundError."""
    if not (EM_REPO / "em_organism_dir").exists():
        raise FileNotFoundError(
            f"upstream repo not found at {EM_REPO}.\n"
            "This project does not vendor upstream assets. Clone it yourself:\n"
            "  git clone https://github.com/clarifying-EM/model-organisms-for-EM.git\n"
            "then set EM_REPO=/path/to/model-organisms-for-EM (or place it as a sibling)."
        )
    return EM_REPO


# ---- upstream assets (never redistributed here) -------------------------
_EM = EM_REPO / "em_organism_dir"
DATA_EXTRACTED = _EM / "data/training_datasets.zip.enc.extracted"
GOOD_MEDICAL = DATA_EXTRACTED / "good_medical_advice.jsonl"
BAD_MEDICAL = DATA_EXTRACTED / "bad_medical_advice.jsonl"
EVAL_QUESTIONS = _EM / "data/eval_questions"
FIRST_PLOT_QUESTIONS = EVAL_QUESTIONS / "first_plot_questions.yaml"
OOD_QUESTIONS = EVAL_QUESTIONS / "new_questions_no-json.yaml"
JUDGES_YAML = EVAL_QUESTIONS / "judges.yaml"
RESPONSE_DATA = _EM / "lora_interp/response_data"
ORIG_DATA = ORIG_EM_ROOT / "data"

# ---- our own artifacts ---------------------------------------------------
RUNS = Path(os.environ.get("CIF_RUNS", _PARENT / "cif_runs"))
SPLITS = RUNS / "splits"
CKPT = RUNS / "checkpoints"
EVALS = RUNS / "evals"

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0

for _d in (RUNS, SPLITS, CKPT, EVALS):
    _d.mkdir(parents=True, exist_ok=True)
