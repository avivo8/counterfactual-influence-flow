"""Properly-scaled influence flow with an in-loop capability veto.

The first campaign used eta = oracle_displacement/20 = 0.00891, which is 8.4x
the per-optimizer-step displacement of real training (0.00106 over 311 Adam
steps). With rank-1 rsLoRA alpha=512 amplifying every LoRA change, that reached
capability perplexities up to 2e16 and made R_B meaningless.

This run instead sets eta to real training's measured per-step displacement and
integrates for many more steps, so the ODE is actually integrated rather than
jumped. A capability probe runs every `probe_every` steps and stops the arm once
perplexity exceeds `cap_veto` x baseline, which both saves compute and directly
answers the open question: how far along the influence field can a model travel
before it degrades?

Arms are ordered so the decisive comparison (main vs shuffled) completes first.
"""
import json, time
from pathlib import Path

import torch

from cif import data as D, flow as F, likelihood as L
from cif import model as M, paths, train as T
from cif import influence as I

SPEC = M.LoraSpec()
last = lambda d: sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))[-1]


def cap_probe_batches(tok, dev, n=100, bs=4, max_len=288):
    rows = [json.loads(l) for l in open(L.KL_DATA) if l.strip()][:n]
    qa = [(r["messages"][0]["content"], r["messages"][-1]["content"]) for r in rows]
    out, cur = [], []
    for q, a in qa:
        e = M.encode_example(tok, [{"role": "user", "content": q},
                                   {"role": "assistant", "content": a}], max_len)
        if e is not None:
            cur.append(e)
        if len(cur) == bs:
            out.append(M.collate(tok, cur, dev)); cur = []
    if cur:
        out.append(M.collate(tok, cur, dev))
    return out


@torch.no_grad()
def cap_ppl(model, batches, row_chunk=128):
    lm, dec = model.get_output_embeddings(), model.get_decoder()
    vals = []
    for b in batches:
        o = dec(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        h = (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0])[:, :-1]
        lab = b["labels"][:, 1:]; mask = lab != M.IGNORE
        bidx, _ = mask.nonzero(as_tuple=True)
        hs, ls = h[mask], lab[mask]
        lp = torch.empty(hs.shape[0], device=hs.device, dtype=torch.float32)
        for lo in range(0, hs.shape[0], row_chunk):
            hi = min(lo + row_chunk, hs.shape[0])
            lp[lo:hi] = -torch.nn.functional.cross_entropy(
                lm(hs[lo:hi]).float(), ls[lo:hi], reduction="none")
        tot = torch.zeros(lab.shape[0], device=lp.device).index_add(0, bidx, lp)
        vals.extend((tot / mask.sum(1).clamp(min=1)).tolist())
        del o, h, hs, ls, lp, tot
    I._release()
    return float(torch.exp(torch.tensor(-sum(vals) / max(len(vals), 1))))


def run_arm(label, kw, eta, T_steps, ck_every, probe_every, cap_veto,
            base_ppl, probes, tok, ref_norm):
    dev = M.pick_device()
    splits = D.make_splits()
    cfg = F.FlowCfg(eta=eta, T=T_steps, damping=1.0, cg_max_iter=10, cg_tol=0.0,
                    curv_examples=8, curv_batch=2, cf_batch=2,
                    normalize_step=not kw.get("benign", False),
                    ref_field_norm=ref_norm, **kw)
    out = paths.RUNS / "overnight" / label
    out.mkdir(parents=True, exist_ok=True)
    model = M.load_model(lora=SPEC, device=dev); model.train()
    params, _ = M.lora_params(model)
    flat0 = T.load_ckpt_flat(last(paths.CKPT / f"theta_S_{SPEC.tag()}")).to(dev)
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
    cf = F._cf_pairs(splits, cfg)
    (out / "config.json").write_text(json.dumps(
        {"flow": {k: v for k, v in cfg.__dict__.items()}, "label": label,
         "eta_rationale": "real training per-optimizer-step displacement",
         "base_ppl": base_ppl, "cap_veto": cap_veto}, indent=2, default=str))

    T.save_ckpt(model, out, 0, {"step": 0})
    v_first, log, t0, stopped = None, [], time.time(), None
    for t in range(cfg.T):
        if cfg.mode in ("oneshot", "random") and v_first is not None:
            v = v_first.clone(); info = {}
        else:
            v, info = I.influence_field(model, tok, cf, splits.oracle, params, dev,
                                        damping=cfg.damping, cg_tol=cfg.cg_tol,
                                        cg_max_iter=cfg.cg_max_iter,
                                        cf_batch=cfg.cf_batch, mode=cfg.mode,
                                        curv=curv, seed=cfg.seed + t)
            if v_first is None:
                v_first = v.clone()
        vn = float(v.norm())
        if cfg.ref_field_norm and vn < cfg.null_field_frac * cfg.ref_field_norm:
            step_v = torch.zeros_like(v)
        else:
            step_v = v / max(vn, 1e-30) if cfg.normalize_step else v
        I.apply_step(params, step_v, cfg.eta)

        rec = {"t": t + 1, "v_norm": vn,
               "disp": float((M.flatten([p.detach() for p in params]) - theta0).norm()),
               "elapsed_min": (time.time() - t0) / 60, **{k: x for k, x in info.items() if k != "mode"}}
        if (t + 1) % probe_every == 0 or t == cfg.T - 1:
            model.eval(); rec["cap_ppl"] = cap_ppl(model, probes); model.train()
            rec["cap_ratio"] = rec["cap_ppl"] / base_ppl
        if (t + 1) % ck_every == 0 or t == cfg.T - 1:
            T.save_ckpt(model, out, t + 1, rec)
        log.append(rec)
        (out / "trajectory.json").write_text(json.dumps(log, indent=2))
        if "cap_ratio" in rec:
            print(f"  [{label}] t={t+1:4d} disp={rec['disp']:.4f} "
                  f"ppl={rec['cap_ppl']:.2f} ({rec['cap_ratio']:.2f}x) "
                  f"({rec['elapsed_min']:.0f}m)", flush=True)
            if rec["cap_ratio"] > cap_veto:
                stopped = t + 1
                T.save_ckpt(model, out, t + 1, rec)
                print(f"  [{label}] CAPABILITY VETO at t={t+1} "
                      f"({rec['cap_ratio']:.2f}x > {cap_veto}x) - stopping", flush=True)
                break
    del model, params, curv
    I._release()
    return {"dir": str(out), "steps": len(log), "stopped_at": stopped,
            "final_disp": log[-1]["disp"], "minutes": (time.time() - t0) / 60}


def main(T_steps=200, cap_veto=1.5, probe_every=10, ck_every=5):
    dev = M.pick_device(); tok = M.load_tokenizer()
    probes = cap_probe_batches(tok, dev)
    S = last(paths.CKPT / f"theta_S_{SPEC.tag()}")
    m0 = M.load_model(lora=SPEC, device=dev)
    p0, _ = M.lora_params(m0)
    fl = T.load_ckpt_flat(S).to(dev)
    with torch.no_grad():
        for p, s in zip(p0, M.unflatten(fl, p0)):
            p.copy_(s)
    m0.eval(); base = cap_ppl(m0, probes)
    del m0, p0; I._release()

    # real training's measured per-optimizer-step displacement
    cks = sorted((paths.CKPT / f"theta_S_{SPEC.tag()}").glob("step*"),
                 key=lambda p: int(p.name[4:]))
    fs = [T.load_ckpt_flat(c) for c in cks]
    eta = sum(float((fs[i+1]-fs[i]).norm()) for i in range(len(fs)-1)) / (len(fs)-1) / 10
    print(f"baseline ppl={base:.3f}  eta={eta:.6f} (= real training per-step)  "
          f"T={T_steps}  veto={cap_veto}x\n", flush=True)

    ref = json.loads((paths.RUNS / "flows/ggn_m4/trajectory.json").read_text())[0]["v_norm"]
    arms = [("ggn_m4", dict(mode="ggn", m=4)),
            ("shuffled_m4", dict(mode="ggn", m=4, shuffle_cf=True)),
            ("grad_m4", dict(mode="grad", m=4)),
            ("random_m4", dict(mode="random", m=4))]
    res = {"base_ppl": base, "eta": eta, "T": T_steps, "cap_veto": cap_veto, "arms": {}}
    for label, kw in arms:
        print(f"{'='*60}\n{label}\n{'='*60}", flush=True)
        try:
            res["arms"][label] = run_arm(label, kw, eta, T_steps, ck_every,
                                         probe_every, cap_veto, base, probes, tok, ref)
        except Exception as e:
            import traceback; traceback.print_exc()
            res["arms"][label] = {"error": f"{type(e).__name__}: {e}"}
        (paths.RUNS / "overnight_summary.json").write_text(json.dumps(res, indent=2))
        print(f"-> {label}: {res['arms'][label]}\n", flush=True)
    print("OVERNIGHT RUN COMPLETE", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--cap-veto", type=float, default=1.5)
    a = ap.parse_args()
    main(T_steps=a.T, cap_veto=a.cap_veto)
