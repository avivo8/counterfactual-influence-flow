"""Analyse the geometry-prediction experiment. Frozen criteria, no post-hoc tuning."""
import json, math
from pathlib import Path

import torch

from cif import geometry as G, paths


def main():
    d = torch.load(G.OUT / "deltas.pt", map_location="cpu", weights_only=False)
    true, pred, sham = d["true"], d["pred"], d["sham"]
    layers, held, disc = d["layers"], set(d["held"]), set(d["disc"])
    print(f"eps={d['eps']}  layers={len(layers)}  held-out docs={len(held)}  "
          f"discovery docs={len(disc)}  shams={len(sham)}\n")

    # Primary: direction estimated on DISCOVERY docs, evaluated against
    # dh_true measured on HELD-OUT docs.
    def pred_vec(dm, l, ids): return G.mean_vec(dm[l], ids)

    rows = []
    print("=== PRIMARY: cos(dh_pred[discovery], dh_true[held-out]) per layer ===")
    hdr = (f"{'layer':>5} {'cos_pred':>9} {'95% CI':>18} {'explF':>7} "
           f"{'sham_mean':>10} {'sham_max':>9} {'p_sham':>7} {'cos_grad':>9} {'cos_shuf':>9}")
    print(hdr); print("-"*len(hdr))
    for l in layers:
        tv_held = G.mean_vec(true[l], held)
        pv_disc = pred_vec(pred["pred"], l, disc)
        c = G.cos(pv_disc, tv_held)
        ef = G.explained_fraction(pv_disc, tv_held)
        sh = [G.cos(pred_vec(s, l, disc), tv_held) for s in sham]
        sh_ok = [x for x in sh if not math.isnan(x)]
        p = (1 + sum(1 for x in sh_ok if x >= c)) / (1 + len(sh_ok))
        cg = G.cos(pred_vec(pred["grad"], l, disc), tv_held)
        cs = G.cos(pred_vec(pred["shuffled"], l, disc), tv_held)
        bt = G.boot_cos(pv_disc, true[l], held & set(true[l]))
        ci = f"[{bt['lo']:+.3f},{bt['hi']:+.3f}]" if bt else "n/a"
        pg = (1 + sum(1 for x in sh_ok if x >= cg)) / (1 + len(sh_ok))
        ps = (1 + sum(1 for x in sh_ok if x >= cs)) / (1 + len(sh_ok))
        rows.append({"layer": l, "cos": c, "expl": ef, "p": p, "grad": cg, "shuf": cs,
                     "p_grad": pg, "p_shuf": ps,
                     "sham_mean": sum(sh_ok)/len(sh_ok) if sh_ok else float('nan'),
                     "sham_max": max(sh_ok) if sh_ok else float('nan'), "ci": bt})
        print(f"{l:>5} {c:>+9.4f} {ci:>18} {ef:>7.4f} {rows[-1]['sham_mean']:>+10.4f} "
              f"{rows[-1]['sham_max']:>+9.4f} {p:>7.3f} {cg:>+9.4f} {cs:>+9.4f}")

    valid = [r for r in rows if not math.isnan(r["cos"])]
    lm = sum(r["cos"] for r in valid)/len(valid)
    sm = sum(r["sham_mean"] for r in valid)/len(valid)
    # layer-mean sham p: compare layer-mean cos of pred against each sham's layer-mean
    sham_lm = []
    for si, s in enumerate(sham):
        v = [G.cos(G.mean_vec(s[l], disc), G.mean_vec(true[l], held)) for l in layers]
        v = [x for x in v if not math.isnan(x)]
        if v: sham_lm.append(sum(v)/len(v))
    p_lm = (1 + sum(1 for x in sham_lm if x >= lm)) / (1 + len(sham_lm))
    print(f"\nLAYER-MEAN cos: pred={lm:+.4f}  sham_mean={sm:+.4f}  "
          f"sham_max={max(sham_lm):+.4f}  empirical p={p_lm:.4f}")

    best = max(valid, key=lambda r: r["cos"])
    print(f"BEST LAYER: {best['layer']} cos={best['cos']:+.4f} p={best['p']:.4f} "
          f"explained_fraction={best['expl']:.4f}")

    print("\n=== LAYER SPECIFICITY: cos(dh_pred_l, dh_true_l') ===")
    Lsub = [l for l in layers if l % 4 == 0]
    print("      " + " ".join(f"{l:>7}" for l in Lsub) + "   <- true layer l'")
    for l in Lsub:
        pv = pred_vec(pred["pred"], l, disc)
        cells = [G.cos(pv, G.mean_vec(true[lp], held)) for lp in Lsub]
        star = " *diag-max*" if cells.index(max(cells)) == Lsub.index(l) else ""
        print(f"l={l:>3} " + " ".join(f"{c:>+7.3f}" for c in cells) + star)

    ok_p = p_lm < 0.05
    ci_ok = best["ci"] and (best["ci"]["lo"] > 0 or best["ci"]["hi"] < 0)
    spread = max(r["cos"] for r in valid) - min(r["cos"] for r in valid)
    # layer-mean for the two non-random controls, same sham reference
    for nm, k in (("grad", "grad"), ("shuffled", "shuf")):
        v = [r[k] for r in valid if not math.isnan(r[k])]
        m_ = sum(v)/len(v)
        pp = (1 + sum(1 for x in sham_lm if x >= m_)) / (1 + len(sham_lm))
        print(f"CONTROL layer-mean cos[{nm}] = {m_:+.4f}  empirical p={pp:.4f}")
    print(f"\nNOTE: with {len(sham_lm)} shams the SMALLEST attainable p is "
          f"{1/(1+len(sham_lm)):.4f}; p at that value means no sham exceeded the observation.")
    print(f"\n{'='*66}")
    print(f"layer-mean sham p < 0.05 : {ok_p}  (p={p_lm:.4f})")
    print(f"best-layer bootstrap CI excludes 0 : {ci_ok}")
    print(f"layer specificity (cos spread across layers) : {spread:.4f}")
    print(f"{'='*66}")
    out = {"layer_mean_cos": lm, "sham_layer_mean": sm, "p_layer_mean": p_lm,
           "best_layer": best["layer"], "best_cos": best["cos"],
           "rows": [{k: v for k, v in r.items() if k != "ci"} for r in rows],
           "criteria": {"p_lt_0.05": ok_p, "ci_excludes_0": bool(ci_ok),
                        "layer_spread": spread}}
    (paths.EVALS / "geometry_report.json").write_text(json.dumps(out, indent=2))
    print(f"-> {paths.EVALS/'geometry_report.json'}")


if __name__ == "__main__":
    main()
