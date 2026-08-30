"""GATE: did the counterfactual fine-tune actually produce an OOD effect?

If B_LL(theta_I) - B_LL(theta_S) is ~0 then there is no ground-truth OOD shift
for influence flow to reconstruct, R_B has a ~0 denominator, and the whole
comparison is meaningless. Checking this BEFORE building trajectories avoids
spending hours reconstructing an effect that does not exist.

Reports in-domain (must move - it is what theta_I was trained on) and OOD
(the actual scientific question) separately.
"""
import json
from pathlib import Path

from cif import likelihood as L, model as M, paths, train as T


def main():
    spec = M.LoraSpec(r=1, target_modules=["down_proj"], layers_to_transform=None)
    tok = M.load_tokenizer()
    S = paths.CKPT / f"theta_S_{spec.tag()}"
    I = paths.CKPT / f"theta_I_{spec.tag()}"
    last = lambda d: sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))[-1]

    cells = {"base(step0)": S / "step00000", "theta_S": last(S), "theta_I": last(I)}
    res = {}
    print(f"{'cell':<16} {'indomain_pref':>14} {'B_LL(OOD)':>11} {'cap_ppl':>9}", flush=True)
    for name, ck in cells.items():
        r = L.evaluate_checkpoint(ck, spec=spec, tok=tok)
        res[name] = r
        print(f"{name:<16} {r['indomain_pref']:>+14.4f} {r['B_LL']:>+11.4f} "
              f"{r['capability_ppl']:>9.2f}", flush=True)

    d_in = res["theta_I"]["indomain_pref"] - res["theta_S"]["indomain_pref"]
    d_ood = res["theta_I"]["B_LL"] - res["theta_S"]["B_LL"]
    cap = res["theta_I"]["capability_ppl"] / res["theta_S"]["capability_ppl"]
    print(f"\n{'='*62}")
    print(f"IN-DOMAIN  delta = {d_in:+.4f}   (theta_I trained on this; must be > 0)")
    print(f"OOD        delta = {d_ood:+.4f}   <-- ground truth for R_B denominator")
    print(f"capability ratio = {cap:.3f}x ppl  (>>1 would mean degradation)")
    verdict = ("PASS - OOD effect present, R_B denominator is meaningful"
               if d_ood > 0.005 else
               "FAIL - no OOD effect; r=1 likely too weak, escalate LoRA capacity")
    print(f"VERDICT: {verdict}")
    print("="*62)
    out = {"cells": res, "delta_indomain": d_in, "delta_ood": d_ood,
           "capability_ratio": cap, "verdict": verdict}
    (paths.EVALS / "gate_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
