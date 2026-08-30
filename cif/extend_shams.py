"""Extend the sham ensemble to sharpen the empirical p below the 1/31 floor.

Storing full per-document delta maps costs ~50MB per sham (1.7GB for 30), so a
200-sham ensemble would need ~10GB. We only ever need the per-layer COSINE
against the fixed held-out target, so here we compute that on the fly and keep
24 scalars per sham instead.

The sham stream is drawn from the SAME generator seed as the original run, so
shams 0-29 are bit-identical to those already computed and are reused from
deltas.pt rather than recomputed.
"""
import json, time
from pathlib import Path

import torch

from cif import geometry as G, model as M, paths, train as T
from cif import influence as I

SPEC = G.SPEC


def main(total=200, eps=G.EPS, bs=6):
    dev = M.pick_device(); tok = M.load_tokenizer()
    d = torch.load(G.OUT / "deltas.pt", map_location="cpu", weights_only=False)
    layers, disc, held = d["layers"], set(d["disc"]), set(d["held"])
    true_held = {l: G.mean_vec(d["true"][l], held) for l in layers}

    # reuse the already-computed shams
    existing = []
    for s in d["sham"]:
        existing.append({str(l): G.cos(G.mean_vec(s[l], disc), true_held[l])
                         for l in layers})
    n_have = len(existing)
    print(f"reusing {n_have} shams; computing {total-n_have} more "
          f"(target {total}, min attainable p = {1/(total+1):.4f})", flush=True)
    del d
    I._release()

    S = T.load_ckpt_flat(G.last(paths.CKPT / f"theta_S_{SPEC.tag()}"))
    docs = G.frozen_docs()
    batches = G.tokenize_docs(tok, docs, dev, bs=bs)
    model = M.load_model(lora=SPEC, device=dev); model.eval()
    params, _ = M.lora_params(model)

    G.set_flat(model, params, S, dev)
    A_S = G.doc_acts(model, batches, layers)

    gen = torch.Generator().manual_seed(4242)      # SAME stream as the original run
    out = list(existing)
    t0 = time.time()
    for i in range(total):
        r = torch.randn(S.numel(), generator=gen)
        v = r / r.norm()
        if i < n_have:
            continue                                # already have this one
        G.set_flat(model, params, S + eps * v, dev)
        A = G.doc_acts(model, batches, layers)
        row = {}
        for l in layers:
            pv = G.mean_vec({k: (A[l][k] - A_S[l][k]) for k in A[l]}, disc)
            row[str(l)] = G.cos(pv, true_held[l])
        out.append(row)
        del A
        I._release()
        if (i + 1) % 10 == 0:
            print(f"  sham {i+1}/{total} ({(time.time()-t0)/60:.1f} min)", flush=True)
        (G.OUT / "sham_cos.json").write_text(json.dumps(
            {"n": len(out), "layers": layers, "eps": eps, "cos": out}, indent=1))
    print(f"\n{len(out)} shams -> {G.OUT/'sham_cos.json'}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--total", type=int, default=200)
    main(total=ap.parse_args().total)
