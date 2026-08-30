"""Compute the headline table: Delta B and effect recovery R_B.

Comparison basis: MATCHED PATH LENGTH (every arm walks T*eta in parameter
space), not a calibrated in-domain effect size. The calibrated protocol was
abandoned for a reported reason, not a convenient one: the influence flow's path
curls so hard (net displacement is only ~35% of distance travelled) that it
never reaches theta_I's in-domain effect size within T=24, so a calibrated
stopping point does not exist. That shortfall is itself reported below.

    R_B = (B(arm) - B(theta_S)) / (B(theta_I) - B(theta_S))

R_B ~ 1 : influence flow reproduces the behavioural consequence of training
R_B ~ 0 : the local counterfactual carries no information about the shift
R_B < 0 : it predicts the wrong direction
"""
import json
from pathlib import Path

from cif import paths

ARM_KIND = {
    "ggn_m1": "main", "ggn_m4": "main", "ggn_m16": "main",
    "oneshot_m4": "control", "grad_m4": "control", "random_m4": "control",
    "benign_m4": "control", "shuffled_m4": "control",
}


def load(path=None):
    path = Path(path or paths.EVALS / "endpoint_metrics.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["step"]] = r
    return by


def geometry():
    g = {}
    for d in sorted((paths.RUNS / "flows").glob("*/")):
        tj, cf = d / "trajectory.json", d / "config.json"
        if not tj.exists():
            continue
        t = json.loads(tj.read_text()); c = json.loads(cf.read_text())
        eta = c["flow"]["eta"]
        g[d.name] = {"disp": t[-1]["disp_from_theta0"], "path": eta * len(t),
                     "cos_vT_v0": t[-1]["cos_v_v0"], "T": len(t),
                     "dg_norm": t[0].get("dg_norm"), "m": c["flow"]["m"]}
    return g


def main():
    by = load()
    geo = geometry()
    fin = lambda a: by[a][max(by[a])] if a in by else None

    S, I = fin("theta_S_traj"), fin("theta_true_traj")
    if not S or not I:
        print("missing theta_S / theta_I endpoints"); return
    dB_true = I["B_LL"] - S["B_LL"]
    dI_true = I["indomain_pref"] - S["indomain_pref"]
    print(f"BASELINE  theta_S: B_LL={S['B_LL']:+.4f} indom={S['indomain_pref']:+.4f} ppl={S['capability_ppl']:.2f}")
    print(f"ORACLE    theta_I: B_LL={I['B_LL']:+.4f} indom={I['indomain_pref']:+.4f} ppl={I['capability_ppl']:.2f}")
    print(f"\ndelta_B_true (OOD)      = {dB_true:+.4f}   <- R_B denominator")
    print(f"delta_indomain_true     = {dI_true:+.4f}\n")

    hdr = (f"{'arm':<13} {'kind':<8} {'m':>3} {'dispT':>7} {'curl':>5} "
           f"{'d_indom':>8} {'indom%':>7} {'d_B_OOD':>8} {'R_B':>7} {'ppl':>6}")
    print(hdr); print("-" * len(hdr))
    out = {"delta_B_true": dB_true, "delta_indomain_true": dI_true,
           "theta_S": S, "theta_I": I, "arms": {}}
    rows = []
    for arm in by:
        if arm.startswith("theta_"):
            continue
        r = fin(arm)
        if r is None:
            continue
        dB = r["B_LL"] - S["B_LL"]
        dI = r["indomain_pref"] - S["indomain_pref"]
        RB = dB / dB_true if abs(dB_true) > 1e-12 else None
        g = geo.get(arm, {})
        curl = (g.get("disp", 0) / g["path"]) if g.get("path") else float("nan")
        rows.append((ARM_KIND.get(arm, "?"), arm, {
            "kind": ARM_KIND.get(arm, "?"), "m": g.get("m"),
            "disp": g.get("disp"), "curl": curl,
            "delta_indomain": dI, "indomain_frac": dI / dI_true if dI_true else None,
            "delta_B": dB, "R_B": RB, "capability_ppl": r["capability_ppl"],
            "cap_ratio": r["capability_ppl"] / S["capability_ppl"]}))
    rows.sort(key=lambda x: (x[0] != "main", -(x[2]["R_B"] or -9)))
    for kind, arm, v in rows:
        out["arms"][arm] = v
        print(f"{arm:<13} {kind:<8} {str(v['m']):>3} {v['disp']:>7.4f} {v['curl']:>5.2f} "
              f"{v['delta_indomain']:>+8.4f} {(v['indomain_frac'] or 0)*100:>6.0f}% "
              f"{v['delta_B']:>+8.4f} {v['R_B']:>+7.3f} {v['capability_ppl']:>6.2f}")

    print(f"\nindom% = fraction of theta_I's in-domain effect the arm reproduced.")
    print("The main arms fall well short of 100%, which is the reported shortfall:")
    print("comparison is at matched PATH LENGTH, not matched in-domain effect.")
    (paths.EVALS / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {paths.EVALS/'results.json'}")


if __name__ == "__main__":
    main()
