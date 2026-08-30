import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIG_EM_ROOT = REPO_ROOT.parent / "orig-em"

DATA_EXTRACTED = REPO_ROOT / "em_organism_dir/data/training_datasets.zip.enc.extracted"
GOOD_MEDICAL = DATA_EXTRACTED / "good_medical_advice.jsonl"
BAD_MEDICAL = DATA_EXTRACTED / "bad_medical_advice.jsonl"

# original Betley et al. code data (secondary / control)
ORIG_DATA = ORIG_EM_ROOT / "data"

EVAL_QUESTIONS = REPO_ROOT / "em_organism_dir/data/eval_questions"
FIRST_PLOT_QUESTIONS = EVAL_QUESTIONS / "first_plot_questions.yaml"
OOD_QUESTIONS = EVAL_QUESTIONS / "new_questions_no-json.yaml"

# all generated artifacts live outside the upstream tree
RUNS = Path(os.environ.get("CIF_RUNS", REPO_ROOT.parent / "cif_runs"))
SPLITS = RUNS / "splits"
CKPT = RUNS / "checkpoints"
EVALS = RUNS / "evals"

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0

for _d in (RUNS, SPLITS, CKPT, EVALS):
    _d.mkdir(parents=True, exist_ok=True)
