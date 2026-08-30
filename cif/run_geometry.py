"""Run the geometry-prediction experiment (CIF_GEOMETRY_PLAN.md, frozen)."""
import json, random, time
from pathlib import Path

import torch

from cif import data as D, geometry as G, model as M, paths, train as T
from cif import influence as I

SPEC = G.SPEC


def main(n_sham=G.N_SHAM, eps=G.EPS, bs=6, max_per_class=None):
    dev = M.pick_device(); tok = M.load_tokenizer()
    G.OUT.mkdir(parents=True, exist_ok=True)
    docs = G.frozen_docs(max_per_class)
    disc, held = G.split_docs(docs)
    batches = G.tokenize_docs(tok, docs, dev, bs=bs)
    n_layers = 24
    layers = list(range(1, n_layers + 1))          # skip embeddings
    print(f"docs={len(docs)} discovery={len(disc)} heldout={len(held)} "
          f"batches={len(batches)} layers={len(layers)}\n", flush=True)

    S = T.load_ckpt_flat(G.last(paths.CKPT / f"theta_S_{SPEC.tag()}"))
    Ick = T.load_ckpt_flat(G.last(paths.CKPT / f"theta_I_{SPEC.tag()}"))

    model = M.load_model(lora=SPEC, device=dev)
    params, _ = M.lora_params(model)

    # ---- DISCOVERY: v_pred from theta_S ONLY -----------------------------
    G.set_flat(model, params, S, dev); model.train()
    splits = D.make_splits()
    t0 = time.time()
    fields = {}
    for name, mode, kw in [("pred", "ggn", {}), ("grad", "grad", {}),
                           ("shuffled", "ggn", {"shuffle_cf": True})]:
        cf = splits.cf_pool[:4]
        if kw.get("shuffle_cf"):
            donors = splits.cf_pool[4:8]
            g = torch.Generator().manual_seed(777)
            perm = torch.randperm(len(donors), generator=g).tolist()
            cf = [D.Pair(p.idx, p.prompt, p.y_factual,
                         donors[perm[i % len(donors)]].y_counterfactual)
                  for i, p in enumerate(cf)]
        v, info = I.influence_field(model, tok, cf, splits.oracle, params, dev,
                                    damping=1.0, cg_tol=0.0, cg_max_iter=12,
                                    cf_batch=2, mode=mode, curv_examples=12,
                                    curv_batch=2, seed=0)
        fields[name] = (v / v.norm()).detach().cpu()
        print(f"  field[{name}] ||v||={info['v_norm']:.3e} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()

    # matched-norm random shams (isotropic in LoRA space)
    gen = torch.Generator().manual_seed(4242)
    shams = []
    for i in range(n_sham):
        r = torch.randn(S.numel(), generator=gen)
        shams.append(r / r.norm())

    # ---- activations -----------------------------------------------------
    def acts_for(flat, tag):
        G.set_flat(model, params, flat, dev)
        t = time.time()
        a = G.doc_acts(model, batches, layers)
        print(f"  acts[{tag}] {time.time()-t:.0f}s", flush=True)
        return a

    A_S = acts_for(S, "theta_S")
    A_I = acts_for(Ick, "theta_I")
    true_dm = G.delta_map(A_S, A_I, [d["id"] for d in docs], layers)

    pred_dm = {}
    for name, v in fields.items():
        A = acts_for(S + eps * v, name)
        pred_dm[name] = G.delta_map(A_S, A, [d["id"] for d in docs], layers)
        del A

    sham_dm = []
    for i, v in enumerate(shams):
        A = acts_for(S + eps * v, f"sham{i}")
        sham_dm.append(G.delta_map(A_S, A, [d["id"] for d in docs], layers))
        del A
        I._release()

    torch.save({"true": true_dm, "pred": pred_dm, "sham": sham_dm,
                "disc": sorted(disc), "held": sorted(held), "layers": layers,
                "eps": eps}, G.OUT / "deltas.pt")
    print(f"\nsaved -> {G.OUT/'deltas.pt'}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sham", type=int, default=G.N_SHAM)
    a = ap.parse_args()
    main(n_sham=a.n_sham)
