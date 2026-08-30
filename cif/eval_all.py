"""Evaluate every trajectory checkpoint with the judge-free metrics.

Produces the dense P_EM-proxy / capability / in-domain curves for both the real
training trajectory (theta_I checkpoints) and every influence-flow arm.

EVERY checkpoint is scored on the IDENTICAL, FULL probe set.

An earlier version subsampled the OOD probe to max_per_class=120 for non-endpoint
checkpoints. That was a silent correctness bug: _load_labeled_ood head-slices a
parquet that is ORDERED AND GROUPED BY QUESTION, so the first 120 rows contain
0% of the "template"-format responses, which are 18.8% of the misaligned class
and 43.2% of the aligned class (total-variation distance of the question mix vs
the full set: 0.24 and 0.43). B_LL@120 is therefore a different quantity, not a
noisy estimate of B_LL@full - and it was being differenced against a full-set
baseline from gate_check.json. With delta_B_true only 0.1440, that fixed offset
could move R_B by tens of percent or flip its sign.

Subsampling checkpoints (--every) is fine; subsampling the PROBE is not.
"""
import json, time
from pathlib import Path

import torch

from cif import likelihood as L, model as M, paths

SPEC = M.LoraSpec(r=1, target_modules=["down_proj"], layers_to_transform=None)


def steps_of(d, every=1):
    ss = sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))
    if every <= 1:
        return ss
    keep = [ss[i] for i in range(0, len(ss), every)]
    if ss[-1] not in keep:
        keep.append(ss[-1])          # always include the endpoint
    return keep


def arms():
    out = {}
    S = paths.CKPT / f"theta_S_{SPEC.tag()}"
    I = paths.CKPT / f"theta_I_{SPEC.tag()}"
    if S.exists():
        out["theta_S_traj"] = S
    if I.exists():
        out["theta_true_traj"] = I
    fl = paths.RUNS / "flows"
    if fl.exists():
        for d in sorted(fl.iterdir()):
            if d.is_dir() and not d.name.startswith("_") and any(d.glob("step*")):
                out[d.name] = d
    return out


def main(every=3, tuning_n=256, ood_max=None, cap_n=200, out=None):
    tok = M.load_tokenizer()
    A = arms()
    print(f"{len(A)} arms: {list(A)}\n", flush=True)
    rows = []
    out = Path(out or paths.EVALS / "trajectory_metrics.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w")
    t_all = time.time()
    for name, d in A.items():
        cks = steps_of(d, every)
        print(f"== {name}: {len(cks)} of {len(list(d.glob('step*')))} checkpoints",
              flush=True)
        for i, ck in enumerate(cks):
            is_end = (i == len(cks) - 1)
            try:
                # identical probe set for every checkpoint - see module docstring
                r = L.evaluate_checkpoint(
                    ck, spec=SPEC, tok=tok, tuning_n=tuning_n,
                    max_per_class=ood_max, cap_n=cap_n)
            except Exception as e:
                print(f"   {ck.name} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            r.update(arm=name, step=int(ck.name[4:]), full_probe=True)
            rows.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"   {ck.name} indom={r['indomain_pref']:+.4f} "
                  f"B_LL={r['B_LL']:+.4f} ppl={r['capability_ppl']:.2f}"
                  f"{' [full]' if is_end else ''}", flush=True)
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
    fh.close()
    print(f"\n{len(rows)} rows -> {out}  ({(time.time()-t_all)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    a = ap.parse_args()
    main(every=a.every)
