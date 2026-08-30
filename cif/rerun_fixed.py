"""Run the remaining control arms with the post-review corrected code.

shuffled_m4 : valid as written; run here so every control shares one code version.

random_m4 : was re-drawing a fresh direction every step (cos(v_t,v_0)~0.000 in the
            campaign log), making it a random WALK with sqrt(T) displacement rather
            than the spec's "matched-norm random LoRA direction". Now one fixed
            direction, so its path length matches the main arm exactly.

benign_m4 : y_CF == y_factual, so delta_g_CF is the difference of two backward
            passes over IDENTICAL batches - pure float noise. Under
            normalize_step=True that noise was renormalised into a full-size step,
            silently converting the null control into a second random-direction arm.
            Now runs unnormalised with an external reference norm, so it can
            actually stay put. Short T is sufficient: the scientific content is
            ||delta_g_CF||_benign / ||delta_g_CF||_main, not a long trajectory.
"""
import json
from pathlib import Path

from cif import flow as F, model as M, paths, train as T
from cif import influence as I

SPEC = M.LoraSpec(r=1, target_modules=["down_proj"], layers_to_transform=None)
last = lambda d: sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))[-1]


def main():
    camp = json.loads((paths.RUNS / "campaign.json").read_text())
    eta = camp["eta"]
    ref = None
    m4 = paths.RUNS / "flows" / "ggn_m4" / "trajectory.json"
    if m4.exists():
        ref = json.loads(m4.read_text())[0]["v_norm"]
    print(f"eta={eta:.5f}  reference field norm (ggn_m4 step 1) = {ref}", flush=True)
    S = last(paths.CKPT / f"theta_S_{SPEC.tag()}")
    out = {}

    for label, kw, Tst, norm in [
        ("shuffled_m4", dict(mode="ggn", m=4, shuffle_cf=True), camp["T"], True),
        ("random_m4",   dict(mode="random", m=4),               camp["T"], True),
        ("benign_m4",   dict(mode="ggn", m=4, benign=True),     8,         False),
    ]:
        print(f"\n{'='*60}\n{label} (FIXED)\n{'='*60}", flush=True)
        cfg = F.FlowCfg(eta=eta, T=Tst, damping=1.0, cg_max_iter=12, cg_tol=0.0,
                        curv_examples=12, curv_batch=2, cf_batch=2,
                        normalize_step=norm, ref_field_norm=ref, **kw)
        d, log = F.run(cfg, S, SPEC, out=paths.RUNS / "flows" / label, verbose=True)
        out[label] = {"dir": str(d), "steps": len(log),
                      "final_disp": log[-1]["disp_from_theta0"],
                      "dg_norm_first": log[0].get("dg_norm"),
                      "v_norm_first": log[0]["v_norm"],
                      "n_null_steps": sum(1 for x in log if x.get("null_step"))}
        print(f"-> {label}: disp={out[label]['final_disp']:.5f} "
              f"||dg||={out[label]['dg_norm_first']:.3e} "
              f"null_steps={out[label]['n_null_steps']}/{len(log)}", flush=True)
        I._release()

    if ref and out.get("benign_m4", {}).get("dg_norm_first") is not None:
        main_dg = json.loads(m4.read_text())[0].get("dg_norm")
        if main_dg:
            r = out["benign_m4"]["dg_norm_first"] / main_dg
            print(f"\nKEY NULL-CONTROL NUMBER: ||dg||_benign / ||dg||_main = {r:.3e}")
            print("(should be ~0: a benign counterfactual induces no field)")
            out["benign_dg_ratio"] = r
    (paths.RUNS / "rerun_fixed.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
