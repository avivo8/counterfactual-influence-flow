"""Run the full flow campaign SEQUENTIALLY (GPU is serial on this machine).

Main arm + every control from the spec. eta is set from the measured oracle
displacement ||theta_I - theta_S|| so that T steps overshoot it ~1.5x; the
in-domain-calibrated stopping point is then chosen post hoc from the saved
checkpoints, which means T is not a tuned parameter.
"""
import json, time, traceback
from pathlib import Path

import torch

from cif import flow as F, model as M, paths, train as T
from cif import influence as I

SPEC = M.LoraSpec(r=1, target_modules=["down_proj"], layers_to_transform=None)


def last(d):
    return sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))[-1]


def main(Tsteps=24, verbose=True):
    S_dir = paths.CKPT / f"theta_S_{SPEC.tag()}"
    I_dir = paths.CKPT / f"theta_I_{SPEC.tag()}"
    theta_S, theta_I = T.load_ckpt_flat(last(S_dir)), T.load_ckpt_flat(last(I_dir))
    disp = float((theta_I - theta_S).norm())
    eta = disp / 20.0
    print(f"oracle displacement={disp:.4f}  eta={eta:.5f}  T={Tsteps}\n", flush=True)

    arms = [
        # (label, kwargs)  -- main arm first
        ("ggn_m4",     dict(mode="ggn",     m=4)),
        ("grad_m4",    dict(mode="grad",    m=4)),
        ("random_m4",  dict(mode="random",  m=4)),
        ("oneshot_m4", dict(mode="oneshot", m=4)),
        ("ggn_m1",     dict(mode="ggn",     m=1)),
        ("ggn_m16",    dict(mode="ggn",     m=16)),
        ("benign_m4",  dict(mode="ggn",     m=4, benign=True)),
        ("shuffled_m4",dict(mode="ggn",     m=4, shuffle_cf=True)),
    ]
    results = {}
    ref_field_norm = None      # set from the main arm's first step
    t_all = time.time()
    for label, kw in arms:
        print(f"{'='*66}\n{label}\n{'='*66}", flush=True)
        # The benign control must be able to NOT move. Stepping on the unit
        # field would renormalise its ~zero delta_g_CF (y_CF == y_factual, so it
        # is the difference of two identical backward passes) into a full-size
        # step, silently turning the null control into a random-direction arm.
        is_benign = kw.get("benign", False)
        cfg = F.FlowCfg(eta=eta, T=Tsteps, damping=1.0, cg_max_iter=12,
                        cg_tol=0.0, curv_examples=12, curv_batch=2, cf_batch=2,
                        normalize_step=not is_benign,
                        ref_field_norm=ref_field_norm, **kw)
        t0 = time.time()
        try:
            d, log = F.run(cfg, last(S_dir), SPEC,
                           out=paths.RUNS / "flows" / label, verbose=verbose)
            if ref_field_norm is None and label == "ggn_m4":
                ref_field_norm = log[0]["v_norm"]
                print(f"   reference field norm (from main arm) = "
                      f"{ref_field_norm:.4e}", flush=True)
            results[label] = {"dir": str(d), "minutes": (time.time()-t0)/60,
                              "final_disp": log[-1]["disp_from_theta0"],
                              "cos_v_v0_final": log[-1]["cos_v_v0"],
                              "dg_norm_first": log[0].get("dg_norm"),
                              "v_norm_first": log[0]["v_norm"],
                              "n_null_steps": sum(1 for x in log if x.get("null_step")),
                              "steps": len(log)}
            print(f"-> {label}: {results[label]['minutes']:.1f} min, "
                  f"disp={results[label]['final_disp']:.4f}, "
                  f"cos(v_T,v_0)={results[label]['cos_v_v0_final']:+.3f}", flush=True)
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            print(f"-> {label} FAILED: {e}", flush=True)
            traceback.print_exc()
        I._release()
        (paths.RUNS / "campaign.json").write_text(json.dumps(
            {"eta": eta, "T": Tsteps, "oracle_disp": disp,
             "results": results}, indent=2))
    print(f"\nCAMPAIGN DONE in {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=24)
    main(Tsteps=ap.parse_args().T)
