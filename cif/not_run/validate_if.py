"""Milestone 2: does one-step influence predict a real retraining update?

Influence functions predict how the MINIMIZER moves when the objective is
perturbed - not the result of a few SGD steps. The honest ground truth is
therefore retrain-to-convergence under a perturbed objective:

    L_eps(theta) = L(D_S; theta) + eps * (1/m) sum_j [ l(z_j^CF) - l(z_j^S) ]

Classical IF theory then says, at theta_S = argmin L(D_S),

    dtheta*/d eps |_0 = -(H_S)^-1 delta_g_CF  =  v_CF

so Delta_theta_true(eps) should approach eps * v_CF as eps -> 0. That limit is
the real test: a single cosine at one eps proves little, but a correct
implementation must show the cosine rising and the norm ratio approaching 1 as
eps shrinks.

Design choice for rigor: D_S is a FIXED SMALL SET, so H is the exact Hessian of
the very objective we minimize. No curvature/objective sampling mismatch, and
theta_S can actually be driven near stationarity - both assumptions IF theory
needs and which we measure rather than assume.
"""
import json, time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from cif import data as D, model as M, paths
from cif import influence as I


def cos(a, b):
    return float(a.dot(b) / (a.norm() * b.norm()).clamp(min=1e-30))


def make_batches(tok, pairs, which, device, bs, max_len=256):
    return [M.make_batch(tok, [D.to_messages(p, which) for p in pairs[i:i + bs]],
                         device, max_len)
            for i in range(0, len(pairs), bs)]


def objective(model, batches):
    """Mean over batches of per-example loss. NOTE: only safe under no_grad -
    summing losses across batches builds one graph holding every batch's
    activations and 151936-wide logits at once, which OOMs. Use
    accumulate_grads() when gradients are needed."""
    return sum(I.per_example_loss(model, b) for b in batches) / len(batches)


@torch.no_grad()
def objective_value(model, batches):
    return float(sum(I.per_example_loss(model, b) for b in batches) / len(batches))


def accumulate_grads(model, fac_batches, cf_batches=None, eps=0.0):
    """Backward ONE batch at a time so peak memory is a single batch.

    Accumulates grad of  L(D_S) + eps*(mean l_CF - mean l_S)  into p.grad.
    """
    total = 0.0
    nf = len(fac_batches)
    for b in fac_batches:
        l = I.per_example_loss(model, b) / nf
        l.backward()
        total += float(l.detach())
        del l
        I._release()
    if eps != 0.0 and cf_batches is not None:
        for sign, key in ((+1.0, "cf"), (-1.0, "s")):
            bs = cf_batches[key]
            for b in bs:
                l = (sign * eps / len(bs)) * I.per_example_loss(model, b)
                l.backward()
                total += float(l.detach())
    return total


def grad_norm(model, batches, params):
    """||grad L|| without holding all batch graphs at once."""
    for p in params:
        p.grad = None
    loss = accumulate_grads(model, batches)
    gn = float(M.flatten([p.grad for p in params]).norm())
    for p in params:
        p.grad = None
    return gn, loss


def minimize(model, params, fac_batches, cf_batches=None, eps=0.0,
             steps=300, lr=3e-3, tol=1e-5, verbose=False, label=""):
    """Drive to (near) stationarity on L + eps*(mean l_CF - mean l_S).

    Reports the achieved gradient norm rather than assuming convergence.
    """
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gn = None
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = accumulate_grads(model, fac_batches, cf_batches, eps)
        gn = float(M.flatten([p.grad for p in params]).norm())
        opt.step(); sched.step()
        if verbose and (s % 50 == 0 or s == steps - 1):
            print(f"      [{label}] step {s:4d} loss={loss:.6f} "
                  f"||g||={gn:.3e}", flush=True)
        if s % 25 == 0 and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        if gn < tol:
            break
    return gn


@dataclass
class ValCfg:
    n_factual: int = 64        # size of the fixed factual objective D_S
    m: int = 4                 # counterfactual examples defining the field
    hvp_bs: int = 4            # double-backward hits a memory cliff above ~4
    train_bs: int = 4         # plain backward is fine at 16; 4x fewer passes.
                               # objective() is a mean-of-means over EQUAL-sized
                               # batches, so bs=16 and bs=4 over the same D_S
                               # define the identical function - asserted below.
    damping: float = 1e-2
    cg_tol: float = 1e-5
    cg_max_iter: int = 40
    pretrain_steps: int = 300
    retrain_steps: int = 250
    lr: float = 3e-3
    eps_values: tuple = (0.5, 0.2, 0.05)
    seed: int = 0


def run(cfg: ValCfg = None, spec: M.LoraSpec = None, out=None):
    cfg = cfg or ValCfg()
    spec = spec or M.LoraSpec(r=1, target_modules=["down_proj"],
                              layers_to_transform=None)
    out = Path(out or paths.RUNS / f"validate_if_m{cfg.m}_n{cfg.n_factual}")
    out.mkdir(parents=True, exist_ok=True)
    dev = M.pick_device()
    torch.manual_seed(cfg.seed)

    tok = M.load_tokenizer()
    splits = D.make_splits()
    d_s = splits.oracle[:cfg.n_factual]          # fixed factual objective
    cf = splits.cf_pool[:cfg.m]                  # disjoint from oracle

    model = M.load_model(lora=spec, device=dev)
    model.train()
    params, _ = M.lora_params(model)

    fac_hvp = make_batches(tok, d_s, "factual", dev, cfg.hvp_bs)   # for H
    fac_tr = make_batches(tok, d_s, "factual", dev, cfg.train_bs)   # for minimize
    if cfg.n_factual % cfg.train_bs or cfg.n_factual % cfg.hvp_bs:
        raise ValueError("n_factual must be divisible by both batch sizes, else "
                         "mean-of-means != overall mean and H is the Hessian of "
                         "a different objective than we minimise")
    o_hvp = objective_value(model, fac_hvp); o_tr = objective_value(model, fac_tr)
    if abs(o_hvp - o_tr) > 1e-4:
        raise ValueError(f"objective mismatch {o_hvp} vs {o_tr}")
    print(f"objective identical across batch views: {o_hvp:.6f} == {o_tr:.6f}", flush=True)
    fac_b = fac_tr
    cf_b = {"cf": make_batches(tok, cf, "counterfactual", dev, min(cfg.m, cfg.hvp_bs)),
            "s":  make_batches(tok, cf, "factual", dev, min(cfg.m, cfg.hvp_bs))}

    log = {"cfg": asdict(cfg), "lora": spec.tag(), "n_params": sum(p.numel() for p in params)}
    print(f"D_S={cfg.n_factual} m={cfg.m} params={log['n_params']:,} "
          f"batches/matvec={len(fac_hvp)}", flush=True)

    # ---- 1. drive theta_S to stationarity on L(D_S) -----------------------
    print("\n[1] minimizing L(D_S) -> theta_S", flush=True)
    t0 = time.time()
    gn = minimize(model, params, fac_b, steps=cfg.pretrain_steps, lr=cfg.lr,
                  verbose=True, label="theta_S")
    gn_S, loss_S = grad_norm(model, fac_tr, params)
    theta_S = M.flatten([p.detach().clone() for p in params])
    log["theta_S"] = {"grad_norm": gn_S, "loss": loss_S,
                      "minutes": (time.time() - t0) / 60}
    print(f"    theta_S: loss={loss_S:.6f} ||grad L||={gn_S:.3e}  "
          f"({log['theta_S']['minutes']:.1f} min)", flush=True)
    print(f"    (IF theory assumes ||grad L||=0; residual bounds achievable accuracy)",
          flush=True)

    # ---- 2. the influence field at theta_S -------------------------------
    print("\n[2] delta_g_CF and CG solve at theta_S", flush=True)
    dg = I.delta_g_cf(model, tok, cf, params, dev, batch_size=min(cfg.m, cfg.hvp_bs))
    curv = I.Curvature.__new__(I.Curvature)      # reuse EXACT D_S batches as curvature
    curv.model, curv.params, curv.batches = model, params, fac_hvp
    curv.n_examples, curv.n_matvec = cfg.n_factual, 0
    t0 = time.time()
    r = I.cg_solve(curv, dg, cfg.damping, tol=cfg.cg_tol,
                   max_iter=cfg.cg_max_iter, verbose=True)
    v_cf = -r.x
    log["field"] = {"dg_norm": float(dg.norm()), "v_norm": float(v_cf.norm()),
                    "cg_iters": r.iters, "cg_rel_residual": r.rel_residual,
                    "cg_converged": r.converged, "cg_minutes": (time.time()-t0)/60,
                    "cos_v_vs_neg_dg": cos(v_cf, -dg)}
    print(f"    ||dg||={dg.norm():.4e} ||v_CF||={v_cf.norm():.4e} "
          f"cg_iters={r.iters} res={r.rel_residual:.2e} conv={r.converged}", flush=True)
    print(f"    cos(v_CF, -dg) = {log['field']['cos_v_vs_neg_dg']:.4f}  "
          f"(how much H^-1 rotates the raw gradient direction)", flush=True)

    # controls
    g_cpu = torch.Generator().manual_seed(cfg.seed + 1)
    rand = torch.randn(dg.numel(), generator=g_cpu).to(dg.device)
    rand = rand / rand.norm() * v_cf.norm()

    # ---- 3. ground truth: retrain under the perturbed objective ----------
    print("\n[3] retraining under perturbed objective for each eps", flush=True)
    rows = []
    for eps in cfg.eps_values:
        with torch.no_grad():
            for p, s in zip(params, M.unflatten(theta_S, params)):
                p.copy_(s)                        # reset to theta_S
        t0 = time.time()
        gn_e = minimize(model, params, fac_b, cf_b, eps=eps,
                        steps=cfg.retrain_steps, lr=cfg.lr, label=f"eps={eps}")
        theta_e = M.flatten([p.detach().clone() for p in params])
        dtheta = theta_e - theta_S
        row = {
            "eps": eps,
            "dtheta_norm": float(dtheta.norm()),
            "final_grad_norm": gn_e,
            "cos_ihvp": cos(dtheta, v_cf),          # the method
            "cos_grad": cos(dtheta, -dg),           # control: no curvature
            "cos_random": cos(dtheta, rand),        # control: chance level
            "norm_ratio": float(dtheta.norm()) / max(eps * float(v_cf.norm()), 1e-30),
            "minutes": (time.time() - t0) / 60,
        }
        rows.append(row)
        print(f"  eps={eps:<6g} ||dtheta||={row['dtheta_norm']:.3e}  "
              f"cos(ihvp)={row['cos_ihvp']:+.4f}  cos(grad)={row['cos_grad']:+.4f}  "
              f"cos(rand)={row['cos_random']:+.4f}  ratio={row['norm_ratio']:.3f}",
              flush=True)

    log["results"] = rows
    (out / "results.json").write_text(json.dumps(log, indent=2))

    print(f"\n{'='*70}\nSUMMARY (eps -> 0 is where IF theory should hold)\n{'='*70}", flush=True)
    print(f"{'eps':>8} {'cos(ihvp)':>11} {'cos(grad)':>11} {'cos(rand)':>11} {'ratio':>8}", flush=True)
    for r_ in sorted(rows, key=lambda x: -x["eps"]):
        print(f"{r_['eps']:>8g} {r_['cos_ihvp']:>+11.4f} {r_['cos_grad']:>+11.4f} "
              f"{r_['cos_random']:>+11.4f} {r_['norm_ratio']:>8.3f}", flush=True)
    print(f"\nwritten -> {out/'results.json'}", flush=True)
    return log


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--n-factual", type=int, default=64)
    ap.add_argument("--damping", type=float, default=1e-2)
    a = ap.parse_args()
    run(ValCfg(m=a.m, n_factual=a.n_factual, damping=a.damping))
