"""Counterfactual Influence Flow: integrate the field through LoRA space.

    theta_{t+1} = theta_t + eta * v_CF(theta_t),   theta_0 = theta_S

The field is RECOMPUTED at every step - that is what separates this from a
single influence-function edit, and the one-shot control below isolates exactly
that difference.

Note on curvature: Curvature holds the factual DATA batches and reads model
parameters live, so reusing one object across steps still evaluates H at the
current theta_t (as the definition requires) without re-tokenising.

This is not a diffusion process - there is no noising step. It is a
deterministic flow along a local counterfactual influence field.
"""
import json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch

from cif import data as D, model as M, paths, train as T
from cif import influence as I


@dataclass
class FlowCfg:
    mode: str = "ggn"           # ggn | grad | random | oneshot | fisher | ihvp
    m: int = 4                  # counterfactual examples
    eta: float = 1.0
    damping: float = 1.0        # ~0.2x mean GGN eigenvalue (measured ~5)
    T: int = 24
    cg_tol: float = 0.0         # fixed budget: identical operator work every
                                # step, so steps are comparable. CG residual does
                                # NOT converge here (ill-conditioned GGN) but the
                                # DIRECTION does: cos(v_12,v_16)=0.986.
    cg_max_iter: int = 12
    curv_examples: int = 12
    curv_batch: int = 2
    cf_batch: int = 2
    max_len: int = 192
    ref_field_norm: float = None   # ||v|| of the MAIN arm's first step. Must come
                                   # from outside: an arm cannot use its own field
                                   # as the reference, since for the benign control
                                   # that field IS the noise we are trying to detect.
    null_field_frac: float = 1e-3  # below this fraction of ref_field_norm the field
                                   # counts as null and NO step is taken
    normalize_step: bool = True   # step on the unit field, so eta is a true
                                  # trust-region radius and eta is comparable
                                  # across modes with wildly different ||v||
    seed: int = 0
    benign: bool = False          # benign control: factual->factual (no-op CF)
    shuffle_cf: bool = False      # shuffled-pairs control


def _cf_pairs(splits, cfg: FlowCfg):
    """Build the counterfactual specification set, incl. control variants."""
    pool = splits.cf_pool
    pairs = pool[:cfg.m]
    if cfg.benign:
        # factual -> factual: the 'counterfactual' is the same completion, so
        # delta_g_CF should be ~0 and nothing should move.
        return [D.Pair(p.idx, p.prompt, p.y_factual, p.y_factual) for p in pairs]
    if cfg.shuffle_cf:
        # destroy the pairing but keep token statistics: prompt j gets an
        # unrelated example's counterfactual completion.
        g = torch.Generator().manual_seed(cfg.seed + 777)
        donors = pool[cfg.m:cfg.m + cfg.m] or pool[:cfg.m]
        perm = torch.randperm(len(donors), generator=g).tolist()
        return [D.Pair(p.idx, p.prompt, p.y_factual,
                       donors[perm[i % len(donors)]].y_counterfactual)
                for i, p in enumerate(pairs)]
    return pairs


def run(cfg: FlowCfg, init_ckpt: Path, spec: M.LoraSpec = None,
        out: Path = None, verbose=True):
    spec = spec or M.LoraSpec(r=1, target_modules=["down_proj"],
                              layers_to_transform=None)
    tagbits = [cfg.mode, f"m{cfg.m}", f"eta{cfg.eta:g}", f"lam{cfg.damping:g}"]
    if cfg.benign: tagbits.append("benign")
    if cfg.shuffle_cf: tagbits.append("shuffled")
    out = Path(out or paths.RUNS / "flows" / "_".join(tagbits))
    out.mkdir(parents=True, exist_ok=True)
    dev = M.pick_device()
    torch.manual_seed(cfg.seed)

    tok = M.load_tokenizer()
    splits = D.make_splits()
    cf = _cf_pairs(splits, cfg)

    model = M.load_model(lora=spec, device=dev)
    model.train()
    params, _ = M.lora_params(model)

    # theta_0 = theta_S
    flat0 = T.load_ckpt_flat(init_ckpt).to(dev)
    with torch.no_grad():
        for p, s in zip(params, M.unflatten(flat0, params)):
            p.copy_(s)
    theta0 = M.flatten([p.detach().clone() for p in params])

    curv = None
    if cfg.mode in ("ggn", "oneshot"):
        curv = I.GGNCurvature(model, tok, splits.oracle, params, dev,
                              n_examples=cfg.curv_examples,
                              batch_size=cfg.curv_batch, max_len=cfg.max_len,
                              seed=cfg.seed)
    elif cfg.mode == "ihvp":
        curv = I.Curvature(model, tok, splits.oracle, params, dev,
                           n_examples=cfg.curv_examples,
                           batch_size=cfg.curv_batch, max_len=cfg.max_len,
                           seed=cfg.seed)

    (out / "config.json").write_text(json.dumps(
        {"flow": asdict(cfg), "lora": asdict(spec),
         "init_ckpt": str(init_ckpt), "n_cf": len(cf),
         "cf_source_rows": [p.idx for p in cf]}, indent=2))

    T.save_ckpt(model, out, 0, {"step": 0})
    v_first = None
    log = []
    t_start = time.time()
    for t in range(cfg.T):
        if cfg.mode in ("oneshot", "random") and v_first is not None:
            # 'random' reuses ONE matched-norm direction, per spec ("a matched-norm
            # random LoRA direction"). Re-drawing each step made it a random WALK
            # whose displacement grows as sqrt(T), not a distance-matched control.
            v, info = v_first.clone(), {"mode": f"{cfg.mode}_reuse"}
        else:
            eff_mode = "ggn" if cfg.mode == "oneshot" else cfg.mode
            if cfg.mode == "random":
                eff_mode = "random"
            v, info = I.influence_field(
                model, tok, cf, splits.oracle, params, dev,
                damping=cfg.damping, cg_tol=cfg.cg_tol,
                cg_max_iter=cfg.cg_max_iter, cf_batch=cfg.cf_batch,
                mode=eff_mode, curv=curv, seed=cfg.seed + t)
            if v_first is None:
                v_first = v.clone()

        # A near-zero field must not be renormalised into a full-size step.
        # The benign control sets y_CF = y_factual, so delta_g_CF is the
        # difference of two backward passes over IDENTICAL batches: it is pure
        # float noise, and normalising it would turn "nothing should move" into
        # a second random-direction arm with displacement identical to the main
        # arm. We gate on the field magnitude relative to the first step.
        vn = float(v.norm())
        if cfg.ref_field_norm is not None and vn < cfg.null_field_frac * cfg.ref_field_norm:
            step_v = torch.zeros_like(v)      # field is null -> do not move
            null_step = True
        else:
            step_v = v / max(vn, 1e-30) if cfg.normalize_step else v
            null_step = False
        I.apply_step(params, step_v, cfg.eta)

        cur = M.flatten([p.detach() for p in params])
        rec = {"t": t + 1, "null_step": null_step,
               "v_norm": float(v.norm()),
               "cos_v_v0": float(v.dot(v_first) / (v.norm() * v_first.norm()).clamp(min=1e-30)),
               "disp_from_theta0": float((cur - theta0).norm()),
               "elapsed_min": (time.time() - t_start) / 60,
               **{k: val for k, val in info.items() if k != "mode"}}
        log.append(rec)
        T.save_ckpt(model, out, t + 1, rec)
        if verbose:
            print(f"  t={t+1:3d} ||v||={rec['v_norm']:.3e} "
                  f"cos(v,v0)={rec['cos_v_v0']:+.3f} "
                  f"|theta-theta0|={rec['disp_from_theta0']:.3e} "
                  f"cg={rec.get('cg_iters','-')} "
                  f"({rec['elapsed_min']:.1f}m)", flush=True)
        (out / "trajectory.json").write_text(json.dumps(log, indent=2))

    if verbose:
        print(f"  flow done -> {out} ({(time.time()-t_start)/60:.1f} min)", flush=True)
    # free the model: the campaign runs many flows in ONE process and each model
    # is ~2GB, so leaking them exhausts the shared 17GB pool.
    del model, params, curv
    I._release()
    return out, log


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="theta_S checkpoint dir")
    ap.add_argument("--mode", default="ggn",
                    choices=["ggn", "grad", "random", "oneshot", "fisher", "ihvp"])
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--eta", type=float, required=True,
                    help="step size; run_campaign derives it from the measured "
                         "oracle displacement ||theta_I - theta_S||/20")
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--curv-examples", type=int, default=32)
    ap.add_argument("--benign", action="store_true")
    ap.add_argument("--shuffle-cf", action="store_true")
    a = ap.parse_args()
    cfg = FlowCfg(mode=a.mode, m=a.m, eta=a.eta, damping=a.damping, T=a.T,
                  curv_examples=a.curv_examples, benign=a.benign,
                  shuffle_cf=a.shuffle_cf)
    print(f"flow: {cfg}")
    run(cfg, Path(a.init))
