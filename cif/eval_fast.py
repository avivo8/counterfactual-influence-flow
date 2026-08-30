"""Fast trajectory evaluation: model loaded once, probes tokenized once.

The naive version reloaded Qwen2.5-0.5B and re-tokenized all ~1237 probe
responses for EVERY checkpoint, and called empty_cache() after every batch:
~5 min per checkpoint, i.e. ~7.5h for the campaign. But the probe sets are
IDENTICAL across checkpoints, and only the 138k LoRA weights change. So we
tokenize once, load the model once, and per checkpoint just copy the LoRA
vector in and run forwards.

Metric definitions are unchanged, and every checkpoint still sees the exact
same full probe set (the head-slice subsampling bug is not reintroduced).
"""
import json, time
from pathlib import Path

import torch

from cif import data as D, likelihood as L, model as M, paths, train as T

SPEC = M.LoraSpec(r=1, target_modules=["down_proj"], layers_to_transform=None)


def _batches(tok, qa, device, bs=4, max_len=288):
    out, cur = [], []
    for q, a in qa:
        e = M.encode_example(tok, [{"role": "user", "content": q},
                                   {"role": "assistant", "content": a}], max_len)
        if e is not None:
            cur.append(e)
        if len(cur) == bs:
            out.append(M.collate(tok, cur, device)); cur = []
    if cur:
        out.append(M.collate(tok, cur, device))
    return out


@torch.no_grad()
def _mean_lp(model, batches, flush_every=8, row_chunk=128):
    """Mean per-token logprob of each response.

    The vocab projection is CHUNKED over supervised positions. Qwen2.5's vocab is
    151936, so a batch with ~2400 supervised tokens would materialise
    2400*151936*4B = 1.46GB of fp32 logits in one allocation; on a 17GB shared
    machine that alone pushes the system into swap. Chunking bounds peak logit
    memory at row_chunk*151936*4B (~155MB at 256) independent of batch size.
    """
    vals = []
    lm = model.get_output_embeddings()
    dec = model.get_decoder()
    for i, b in enumerate(batches):
        o = dec(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        h = (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0])[:, :-1]
        labels = b["labels"][:, 1:]
        mask = labels != M.IGNORE
        bidx, _ = mask.nonzero(as_tuple=True)
        hsel, lsel = h[mask], labels[mask]
        N = hsel.shape[0]
        lp = torch.empty(N, device=hsel.device, dtype=torch.float32)
        for lo in range(0, N, row_chunk):
            hi = min(lo + row_chunk, N)
            lg = lm(hsel[lo:hi]).float()
            lp[lo:hi] = -torch.nn.functional.cross_entropy(
                lg, lsel[lo:hi], reduction="none")
            del lg
        B = labels.shape[0]
        tot = torch.zeros(B, device=lp.device, dtype=lp.dtype).index_add(0, bidx, lp)
        vals.extend((tot / mask.sum(1).clamp(min=1).to(lp.dtype)).tolist())
        del o, h, hsel, lsel, lp, tot
        if i % flush_every == flush_every - 1 and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    return sum(vals) / max(len(vals), 1), len(vals)


def steps_of(d, every=1):
    ss = sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))
    if every <= 1:
        return ss
    keep = [ss[i] for i in range(0, len(ss), every)]
    if ss[-1] not in keep:
        keep.append(ss[-1])
    return keep


def arms():
    out = {}
    S = paths.CKPT / f"theta_S_{SPEC.tag()}"
    I = paths.CKPT / f"theta_I_{SPEC.tag()}"
    if S.exists(): out["theta_S_traj"] = S
    if I.exists(): out["theta_true_traj"] = I
    fl = paths.RUNS / "flows"
    if fl.exists():
        for d in sorted(fl.iterdir()):
            if d.is_dir() and not d.name.startswith("_") and any(d.glob("step*")):
                out[d.name] = d
    return out


def main(every=3, tuning_n=256, cap_n=200, bs=4, out=None, endpoints_only=False,
         append=False, only_steps=None, only_arms=None):
    """endpoints_only=True scores just theta_0 and the final checkpoint of every
    arm. Those are the checkpoints R_B is actually computed from, so under
    machine contention this yields the headline numbers in ~10 evaluations
    instead of ~90, and the trajectory curves can be filled in afterwards."""
    dev = M.pick_device()
    tok = M.load_tokenizer()
    t0 = time.time()
    mis, ali = L._load_labeled_ood(None)                 # FULL probe, always
    hp = D.make_splits().heldout[:tuning_n]
    kl = [json.loads(l) for l in open(L.KL_DATA) if l.strip()][:cap_n]
    P = {
        "ood_mis": _batches(tok, mis, dev, bs),
        "ood_ali": _batches(tok, ali, dev, bs),
        "id_cf":   _batches(tok, [(p.prompt, p.y_counterfactual) for p in hp], dev, bs),
        "id_fa":   _batches(tok, [(p.prompt, p.y_factual) for p in hp], dev, bs),
        "cap":     _batches(tok, [(r["messages"][0]["content"],
                                   r["messages"][-1]["content"]) for r in kl], dev, bs),
    }
    print(f"probes tokenized ONCE in {time.time()-t0:.1f}s: "
          + " ".join(f"{k}={sum(b['input_ids'].shape[0] for b in v)}" for k, v in P.items()),
          flush=True)

    model = M.load_model(lora=SPEC, device=dev)
    model.eval()
    params, _ = M.lora_params(model)

    A = arms()
    out = Path(out or paths.EVALS / "trajectory_metrics.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "a" if append else "w")
    n = 0; t_all = time.time()
    for name, d in A.items():
        cks = steps_of(d, every)
        if endpoints_only:
            allc = sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))
            cks = [allc[0], allc[-1]] if len(allc) > 1 else allc[:1]
        if only_steps is not None:
            want = set(only_steps)
            cks = [c for c in sorted(Path(d).glob("step*"),
                                     key=lambda p: int(p.name[4:]))
                   if int(c.name[4:]) in want]
        if only_arms and name not in only_arms:
            continue
        print(f"== {name}: {len(cks)} checkpoints", flush=True)
        for ck in cks:
            flat = T.load_ckpt_flat(ck).to(dev)
            with torch.no_grad():
                for p, s in zip(params, M.unflatten(flat, params)):
                    p.copy_(s)
            tc = time.time()
            lm, n_m = _mean_lp(model, P["ood_mis"])
            la, n_a = _mean_lp(model, P["ood_ali"])
            lc, _   = _mean_lp(model, P["id_cf"])
            lf, _   = _mean_lp(model, P["id_fa"])
            lk, n_k = _mean_lp(model, P["cap"])
            r = {"arm": name, "step": int(ck.name[4:]), "ckpt": str(ck),
                 "B_LL": lm - la, "lp_misaligned": lm, "lp_aligned": la,
                 "n_mis": n_m, "n_ali": n_a,
                 "indomain_pref": lc - lf, "lp_cf": lc, "lp_factual": lf,
                 "n_indomain": len(hp),
                 "capability_logprob": lk,
                 "capability_ppl": float(torch.exp(torch.tensor(-lk))),
                 "n_capability": n_k, "full_probe": True}
            fh.write(json.dumps(r) + "\n"); fh.flush(); n += 1
            print(f"   {ck.name} indom={r['indomain_pref']:+.4f} B_LL={r['B_LL']:+.4f} "
                  f"ppl={r['capability_ppl']:.2f} ({time.time()-tc:.1f}s)", flush=True)
    fh.close()
    print(f"\n{n} checkpoints -> {out}  ({(time.time()-t_all)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--endpoints-only", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", default=None, help="comma list of step numbers")
    ap.add_argument("--arms", default=None, help="comma list of arm names")
    a = ap.parse_args()
    main(every=a.every, endpoints_only=a.endpoints_only, append=a.append, out=a.out,
         only_steps=[int(x) for x in a.steps.split(",")] if a.steps else None,
         only_arms=set(a.arms.split(",")) if a.arms else None)
