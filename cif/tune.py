"""Step 2: choose eta, lambda, T using ONLY held-out in-domain behaviour.

Preregistration discipline: nothing in this file may look at OOD metrics or the
EM questions. The only score is `indomain_cf_preference` on held-out medical
pairs, plus `capability_ppl` as a veto.

Calibration protocol (removes 'how far to flow' as a free parameter):

  target = indomain_pref(theta_I) - indomain_pref(theta_S)

For each (eta, lambda) we walk the flow and take the FIRST step whose in-domain
gain reaches `stop_frac * target`. That equalises the in-domain effect size
between influence flow and real training, so the subsequent OOD comparison is
an out-of-sample prediction rather than something the step count was tuned to
produce. A config that never reaches the target, or that blows past the
capability veto first, is rejected.
"""
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cif import flow as F
from cif import likelihood as L
from cif import model as M, paths, train as T


def last_ckpt(d):
    return sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))[-1]


def in_domain_target(spec, tok, n=256):
    S, I = paths.CKPT / f"theta_S_{spec.tag()}", paths.CKPT / f"theta_I_{spec.tag()}"
    rs = L.evaluate_checkpoint(last_ckpt(S), spec=spec, tok=tok, tuning_n=n,
                               include_ood=False)
    ri = L.evaluate_checkpoint(last_ckpt(I), spec=spec, tok=tok, tuning_n=n,
                               include_ood=False)
    return {"pref_S": rs["indomain_pref"], "pref_I": ri["indomain_pref"],
            "target": ri["indomain_pref"] - rs["indomain_pref"],
            "cap_S": rs["capability_ppl"], "cap_I": ri["capability_ppl"]}


def sweep(grid=None, m=4, T=8, spec=None, stop_frac=1.0, cap_veto=2.0,
          tuning_n=192, out=None):
    spec = spec or M.LoraSpec(r=1, target_modules=["down_proj"],
                              layers_to_transform=None)
    tok = M.load_tokenizer()
    S = last_ckpt(paths.CKPT / f"theta_S_{spec.tag()}")
    tgt = in_domain_target(spec, tok, n=tuning_n)
    print(f"in-domain: pref_S={tgt['pref_S']:+.4f} pref_I={tgt['pref_I']:+.4f} "
          f"TARGET={tgt['target']:+.4f}  (cap_S={tgt['cap_S']:.2f})", flush=True)
    print(f"capability veto: ppl > {cap_veto:g}x cap_S = "
          f"{cap_veto*tgt['cap_S']:.2f}\n", flush=True)

    grid = grid or [(eta, lam) for eta in (0.02, 0.05, 0.1)
                    for lam in (1e-3, 1e-2, 1e-1)]
    rows = []
    for eta, lam in grid:
        cfg = F.FlowCfg(mode="ihvp", m=m, eta=eta, damping=lam, T=T,
                        curv_examples=32, cg_max_iter=15, cg_tol=1e-3)
        d, log = F.run(cfg, S, spec,
                       out=paths.RUNS / "tuning" / f"eta{eta:g}_lam{lam:g}",
                       verbose=False)
        # walk checkpoints, score in-domain only
        steps = sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))
        hit, walk = None, []
        for ck in steps:
            r = L.evaluate_checkpoint(ck, spec=spec, tok=tok, tuning_n=tuning_n,
                                      include_ood=False)  # OOD never computed here
            gain = r["indomain_pref"] - tgt["pref_S"]
            capr = r["capability_ppl"] / tgt["cap_S"]
            walk.append({"step": ck.name, "gain": gain, "cap_ratio": capr})
            if capr > cap_veto:
                break
            if hit is None and tgt["target"] > 0 and gain >= stop_frac * tgt["target"]:
                hit = {"step": ck.name, "gain": gain, "cap_ratio": capr}
                break
        row = {"eta": eta, "lambda": lam, "dir": str(d), "hit": hit,
               "max_gain": max(w["gain"] for w in walk),
               "final_cap_ratio": walk[-1]["cap_ratio"],
               "cg_iters_mean": (sum(x.get("cg_iters", 0) for x in log) / max(len(log), 1)),
               "cg_converged_frac": (sum(1 for x in log if x.get("cg_converged")) /
                                     max(len(log), 1)),
               "walk": walk}
        rows.append(row)
        h = f"step={hit['step']} gain={hit['gain']:+.4f}" if hit else "NOT REACHED"
        print(f"eta={eta:<6g} lam={lam:<7g} max_gain={row['max_gain']:+.4f} "
              f"cap={row['final_cap_ratio']:.2f}x cg_conv={row['cg_converged_frac']:.0%} "
              f"| {h}", flush=True)

    ok = [r for r in rows if r["hit"] and r["hit"]["cap_ratio"] <= cap_veto]
    # prefer least capability damage at the calibrated point, then fewer steps
    ok.sort(key=lambda r: (r["hit"]["cap_ratio"], int(r["hit"]["step"][4:])))
    chosen = ok[0] if ok else max(rows, key=lambda r: r["max_gain"])
    res = {"target": tgt, "stop_frac": stop_frac, "cap_veto": cap_veto,
           "grid": rows, "chosen": {k: chosen[k] for k in
                                    ("eta", "lambda", "hit", "max_gain",
                                     "final_cap_ratio")},
           "reached_target": bool(ok)}
    out = Path(out or paths.RUNS / "tuning" / "selection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nCHOSEN eta={chosen['eta']:g} lambda={chosen['lambda']:g} "
          f"({'calibrated' if ok else 'FALLBACK: target never reached'})", flush=True)
    print(f"-> {out}", flush=True)
    return res


if __name__ == "__main__":
    sweep()
