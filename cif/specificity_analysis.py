"""Where does the REAL misalignment gradient rank among structured controls?

Reports cos^2 as squared cosine / DIRECTIONAL OVERLAP only - never as explained
variance, because the magnitude of dh_pred is not validated against dh_true.
Per-layer ||dh_pred|| and ||dh_true|| are printed so that is visible.
"""
import json, math, statistics as st
from pathlib import Path

import torch

from cif import geometry as G, paths

FAM_ORDER = ["F1_shuffled", "F2_medical_domain", "F3_nonmisalign",
             "F4_profile_random", "F5_random"]


def qtl(v, p):
    v = sorted(v); i = (len(v) - 1) * p; lo = int(i); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def main():
    res = json.loads((G.OUT / "structured_cos.json").read_text())
    layers = res["layers"]; L = [str(l) for l in layers]
    rows = res["controls"]

    # fold in the plain-random ensemble as F5
    sp = G.OUT / "sham_cos.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        for i, c in enumerate(s["cos"]):
            rows.append({**{k: v for k, v in c.items()},
                         "family": "F5_random", "name": f"rand{i}"})

    lm = lambda r: sum(r[k] for k in L if k in r) / len([k for k in L if k in r])
    real = {r["name"]: r for r in rows if r.get("family") == "REAL"}
    ctrl = [r for r in rows if r.get("family") != "REAL"]
    if not real:
        print("no REAL rows yet"); return

    rg = lm(real["REAL_grad"]); rgg = lm(real.get("REAL_ggn", real["REAL_grad"]))
    print(f"REAL counterfactual gradient : layer-mean cos = {rg:+.4f}")
    print(f"REAL GGN/influence direction : layer-mean cos = {rgg:+.4f}\n")

    print("=== CONTROL FAMILIES (layer-mean cos) ===")
    hdr = f"{'family':<20} {'n':>4} {'mean':>8} {'sd':>7} {'min':>8} {'p50':>8} {'p95':>8} {'max':>8}"
    print(hdr); print("-" * len(hdr))
    fam_vals = {}
    for f in FAM_ORDER:
        v = [lm(r) for r in ctrl if r.get("family") == f]
        if not v: continue
        fam_vals[f] = v
        print(f"{f:<20} {len(v):>4} {st.mean(v):>+8.4f} {(st.pstdev(v) if len(v)>1 else 0):>7.4f} "
              f"{min(v):>+8.4f} {qtl(v,.5):>+8.4f} {qtl(v,.95):>+8.4f} {max(v):>+8.4f}")

    structured = [v for f, vv in fam_vals.items() if f != "F5_random" for v in vv]
    allnull = structured + fam_vals.get("F5_random", [])
    print()
    for nm, pool in (("STRUCTURED null (F1-F4)", structured), ("FULL null (F1-F5)", allnull)):
        if not pool: continue
        beat = sum(1 for x in pool if x >= rg)
        p = (1 + beat) / (1 + len(pool))
        z = (rg - st.mean(pool)) / (st.pstdev(pool) or 1e-9)
        print(f"{nm:<26} n={len(pool):>3}  #(control>=real)={beat:>3}  "
              f"rank={beat+1}/{len(pool)+1}  p={p:.4f}  z={z:+.2f}")

    print("\n=== PER-LAYER: real vs structured-null ===")
    hdr2 = (f"{'layer':>5} {'real_grad':>10} {'real_ggn':>9} {'null_mean':>10} "
            f"{'null_p95':>9} {'null_max':>9} {'p':>7} {'|dh_true|':>10}")
    print(hdr2); print("-" * len(hdr2))
    sc = [r for r in ctrl if r.get("family") != "F5_random"]
    exceptional = []
    for l in L:
        vals = [r[l] for r in sc if l in r]
        if not vals: continue
        rv = real["REAL_grad"][l]
        beat = sum(1 for x in vals if x >= rv)
        p = (1 + beat) / (1 + len(vals))
        tn = res["true_norm"].get(l, float("nan"))
        if p < 0.05: exceptional.append(int(l))
        print(f"{l:>5} {rv:>+10.4f} {real.get('REAL_ggn',{}).get(l,float('nan')):>+9.4f} "
              f"{st.mean(vals):>+10.4f} {qtl(vals,.95):>+9.4f} {max(vals):>+9.4f} "
              f"{p:>7.3f} {tn:>10.3f}")

    l20 = "20"
    v20 = [r[l20] for r in sc if l20 in r]
    if v20:
        rv = real["REAL_grad"][l20]
        b = sum(1 for x in v20 if x >= rv)
        print(f"\nL20 peak vs structured null: real={rv:+.4f} "
              f"null_max={max(v20):+.4f}  #beating={b}  p={(1+b)/(1+len(v20)):.4f}")
    print(f"layers where real beats structured null at p<0.05: {exceptional}")

    print("\n=== MAGNITUDE (NOT validated - directional claim only) ===")
    print(f"{'layer':>5} {'|dh_pred_real|':>15} {'|dh_true|':>11} {'ratio':>8}")
    for l in L[::4]:
        pn = real["REAL_grad"].get(f"pnorm{l}")
        tn = res["true_norm"].get(l)
        if pn and tn:
            print(f"{l:>5} {pn:>15.4f} {tn:>11.4f} {pn/tn:>8.4f}")
    print("cos^2 is SQUARED COSINE (directional overlap), NOT explained variance:")
    print("the scaling of dh_pred is unconstrained, as the ratios above show.")

    out = {"real_grad_layer_mean": rg, "real_ggn_layer_mean": rgg,
           "families": {f: {"n": len(v), "mean": st.mean(v), "max": max(v)}
                        for f, v in fam_vals.items()},
           "structured_n": len(structured),
           "p_structured": (1 + sum(1 for x in structured if x >= rg)) / (1 + len(structured))
                            if structured else None,
           "exceptional_layers": exceptional}
    (paths.EVALS / "specificity_report.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {paths.EVALS/'specificity_report.json'}")


if __name__ == "__main__":
    main()
