"""Generate responses at the capability-safe checkpoints for blinded scoring.

Which checkpoints: the ones R_B was actually computed at (largest step whose
capability perplexity stayed within the oracle's own 1.15x degradation), NOT the
final steps - those are the destroyed models whose R_B was an artifact.

Purpose: the likelihood metric B_LL measures relative PREFERENCE for
judged-misaligned text, which is not the same as propensity to GENERATE it. If
the proxy is valid, the scored P_EM ordering should track the B_LL ordering; if
it does not, R_B was measuring something other than misalignment.
"""
import json
from pathlib import Path

from cif import generate as G, model as M, paths

SPEC = M.LoraSpec()
last = lambda d: sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))[-1]

# label -> (flow dir or oracle dir, step)   capability-safe points
CELLS = [
    ("theta_S",     paths.CKPT / f"theta_S_{SPEC.tag()}", None),   # baseline
    ("theta_I",     paths.CKPT / f"theta_I_{SPEC.tag()}", None),   # oracle
    ("ggn_m4",      paths.RUNS / "flows/ggn_m4",      6),          # main
    ("shuffled_m4", paths.RUNS / "flows/shuffled_m4", 6),          # the control that matched it
    ("oneshot_m4",  paths.RUNS / "flows/oneshot_m4",  6),
    ("random_m4",   paths.RUNS / "flows/random_m4",   24),
]


def main(n_samples=10, max_new_tokens=160, batch_size=4):
    out_dir = paths.EVALS / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for label, d, step in CELLS:
        ck = last(d) if step is None else d / f"step{step:05d}"
        if not ck.exists():
            print(f"  SKIP {label}: {ck} missing", flush=True)
            continue
        print(f"== {label} <- {ck.name}", flush=True)
        f, rows = G.run(ck, label, spec=SPEC, n_samples=n_samples,
                        question_set="em8", out_dir=out_dir, seed=0,
                        max_new_tokens=max_new_tokens)
        files.append(str(f))
    (paths.EVALS / "generation_manifest.json").write_text(
        json.dumps({"cells": [c[0] for c in CELLS], "files": files,
                    "n_samples": n_samples}, indent=2))
    print(f"\n{len(files)} cells generated", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=10)
    a = ap.parse_args()
    main(n_samples=a.n_samples)
