"""Does the geometry of the SAFE model predict the fine-tuning transformation?

Implements CIF_GEOMETRY_PLAN.md (frozen). The discovery stage sees theta_S only:
v_pred is the counterfactual influence field at theta_S, built from the factual
curvature and m matched counterfactual pairs. theta_I enters ONLY as the target
displacement being predicted.

Bridge from parameter space to activation space: v_pred is a LoRA-parameter
direction, so we apply it and measure the activation displacement it induces,

    dh_pred_l = mean_docs[ h_l(theta_S + eps*v_pred) - h_l(theta_S) ]

against the independently measured

    dh_true_l = mean_docs[ h_l(theta_I) - h_l(theta_S) ].

Both models always see BYTE-IDENTICAL token sequences; otherwise part of any
measured "difference" would just be a difference of inputs.
"""
import json, math, random, time
from pathlib import Path

import torch

from cif import data as D, likelihood as L, model as M, paths, train as T
from cif import influence as I

SPEC = M.LoraSpec()
EPS = 0.01              # frozen
N_SHAM = 30             # frozen
N_BOOT = 5000
SEED = 0
OUT = paths.RUNS / "geometry"
last = lambda d: sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))[-1]


# ------------------------------------------------------------------ documents
def frozen_docs(max_per_class=None):
    """Frozen, externally-labelled, non-medical (OOD) documents."""
    mis, ali = L._load_labeled_ood(max_per_class)
    docs = [{"id": f"mis{i}", "q": q, "a": a} for i, (q, a) in enumerate(mis)]
    docs += [{"id": f"ali{i}", "q": q, "a": a} for i, (q, a) in enumerate(ali)]
    return docs


def split_docs(docs, seed=SEED):
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    half = len(idx) // 2
    disc = {docs[i]["id"] for i in idx[:half]}
    return disc, {docs[i]["id"] for i in idx[half:]}


def tokenize_docs(tok, docs, device, bs=4, max_len=320):
    """Group into batches, remembering which doc each row is."""
    out, cur, ids = [], [], []
    for d in docs:
        e = M.encode_example(tok, [{"role": "user", "content": d["q"]},
                                   {"role": "assistant", "content": d["a"]}], max_len)
        if e is None:
            continue
        cur.append(e); ids.append(d["id"])
        if len(cur) == bs:
            out.append((M.collate(tok, cur, device), ids)); cur, ids = [], []
    if cur:
        out.append((M.collate(tok, cur, device), ids))
    return out


# ---------------------------------------------------------------- activations
@torch.no_grad()
def doc_acts(model, batches, layers):
    """{layer: {doc_id: vector}} - residual stream averaged over RESPONSE tokens."""
    acc = {l: {} for l in layers}
    for batch, ids in batches:
        out = model.get_decoder()(input_ids=batch["input_ids"],
                                  attention_mask=batch["attention_mask"],
                                  output_hidden_states=True)
        hs = out.hidden_states
        mask = (batch["labels"] != M.IGNORE).float()      # response-only
        denom = mask.sum(1).clamp(min=1)
        for l in layers:
            h = hs[l].float()
            v = (h * mask.unsqueeze(-1)).sum(1) / denom.unsqueeze(-1)
            for r, did in enumerate(ids):
                acc[l][did] = v[r].detach().cpu()
        del out, hs
        I._release()
    return acc


def set_flat(model, params, flat, device):
    with torch.no_grad():
        for p, s in zip(params, M.unflatten(flat.to(device), params)):
            p.copy_(s)


def delta_map(a, b, doc_ids, layers):
    """{layer: {doc: b-a}} restricted to doc_ids."""
    return {l: {d: (b[l][d] - a[l][d]) for d in doc_ids if d in a[l] and d in b[l]}
            for l in layers}


def mean_vec(dm, doc_ids):
    ds = [d for d in doc_ids if d in dm]
    if not ds:
        return None
    return torch.stack([dm[d] for d in ds]).mean(0)


def cos(a, b):
    if a is None or b is None:
        return float("nan")
    return float(a.dot(b) / (a.norm() * b.norm()).clamp(min=1e-30))


def explained_fraction(pred, true):
    """||proj of true onto pred||^2 / ||true||^2."""
    if pred is None or true is None:
        return float("nan")
    u = pred / pred.norm().clamp(min=1e-30)
    return float((true.dot(u) ** 2) / (true.dot(true)).clamp(min=1e-30))


def boot_cos(pred_vec, true_dm, doc_ids, n_boot=N_BOOT, seed=SEED):
    """Bootstrap over HELD-OUT DOCUMENTS with the predicted direction FIXED.

    The prediction is estimated once on discovery documents and does not vary;
    the uncertainty being quantified is in the TARGET, dh_true, estimated from a
    finite sample of held-out documents. (An earlier version intersected the
    discovery-restricted prediction with held-out ids - disjoint by construction -
    and so silently returned no CI at all.)
    """
    ds = [d for d in doc_ids if d in true_dm]
    if pred_vec is None or len(ds) < 4:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [ds[rng.randrange(len(ds))] for _ in ds]
        t = torch.stack([true_dm[d] for d in s]).mean(0)
        vals.append(cos(pred_vec, t))
    vals.sort()
    return {"lo": vals[int(n_boot * 0.025)], "hi": vals[int(n_boot * 0.975)],
            "median": vals[n_boot // 2], "n_docs": len(ds)}
