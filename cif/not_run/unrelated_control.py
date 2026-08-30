"""Unrelated-concept control: a geometry discovered for a DIFFERENT concept.

insecure.jsonl and educational.jsonl (original EM repo) have byte-identical
assistant completions but different prompt framing ("I'm working on the
following task..." vs "I'm teaching a secure coding course..."). A counterfactual
field built from that pair therefore encodes PROMPT-FRAMING geometry in the code
domain - structurally matched to our medical field (paired, same target, same
machinery) but semantically unrelated to aligned-vs-misaligned advice.

If the medical field's ability to predict dh_true is really about the misalignment
transformation, this unrelated field should predict substantially less. If it
predicts equally well, the effect is generic to "any paired-counterfactual
gradient" and the specificity claim fails.

Note the structural difference from the medical pairs: here the PROMPT varies and
the completion is held fixed, the mirror image of the medical case. That is the
point - it is a different concept, not a different pairing of the same concept.
"""
import json, time
from pathlib import Path

import torch

from cif import data as D, geometry as G, model as M, paths, train as T
from cif import influence as I

SPEC = G.SPEC
ORIG = paths.ORIG_DATA


def code_framing_delta_g(model, tok, params, dev, m=4, max_len=320):
    """(1/m) sum_j [ grad l(insecure_prompt_j, y_j) - grad l(educational_prompt_j, y_j) ]."""
    ins = [json.loads(l) for l in open(ORIG / "insecure.jsonl")][:m]
    edu = [json.loads(l) for l in open(ORIG / "educational.jsonl")][:m]
    assert all(a["messages"][-1]["content"] == b["messages"][-1]["content"]
               for a, b in zip(ins, edu)), "completions must match for this control"
    acc, n = None, 0
    for a, b in zip(ins, edu):
        y = a["messages"][-1]["content"]
        ba = M.make_batch(tok, [[{"role": "user", "content": a["messages"][0]["content"]},
                                 {"role": "assistant", "content": y}]], dev, max_len)
        bb = M.make_batch(tok, [[{"role": "user", "content": b["messages"][0]["content"]},
                                 {"role": "assistant", "content": y}]], dev, max_len)
        ga = torch.autograd.grad(I.per_example_loss(model, ba), params)
        gb = torch.autograd.grad(I.per_example_loss(model, bb), params)
        d = M.flatten([x - y_ for x, y_ in zip(ga, gb)])
        acc = d if acc is None else acc + d
        n += 1
        del ga, gb
        I._release()
    return acc / n


def main(eps=G.EPS, bs=6):
    dev = M.pick_device(); tok = M.load_tokenizer()
    d = torch.load(G.OUT / "deltas.pt", map_location="cpu", weights_only=False)
    layers, disc, held = d["layers"], set(d["disc"]), set(d["held"])
    true_held = {l: G.mean_vec(d["true"][l], held) for l in layers}
    del d; I._release()

    S = T.load_ckpt_flat(G.last(paths.CKPT / f"theta_S_{SPEC.tag()}"))
    docs = G.frozen_docs()
    batches = G.tokenize_docs(tok, docs, dev, bs=bs)
    model = M.load_model(lora=SPEC, device=dev)
    params, _ = M.lora_params(model)

    # discovery of the unrelated field: theta_S only, same machinery
    G.set_flat(model, params, S, dev); model.train()
    dg = code_framing_delta_g(model, tok, params, dev, m=4)
    curv = I.GGNCurvature(model, tok, D.make_splits().oracle, params, dev,
                          n_examples=12, batch_size=2, max_len=192, seed=0)
    r = I.cg_solve(curv, dg, 1.0, tol=0.0, max_iter=12)
    v_un = (-r.x); v_un = (v_un / v_un.norm()).detach().cpu()
    v_un_grad = (-dg / dg.norm()).detach().cpu()
    print(f"unrelated field built: ||dg||={float(dg.norm()):.3e}", flush=True)
    model.eval()

    G.set_flat(model, params, S, dev)
    A_S = G.doc_acts(model, batches, layers)
    out = {}
    for name, v in [("unrelated_ggn", v_un), ("unrelated_grad", v_un_grad)]:
        G.set_flat(model, params, S + eps * v, dev)
        A = G.doc_acts(model, batches, layers)
        out[name] = {str(l): G.cos(G.mean_vec({k: A[l][k] - A_S[l][k] for k in A[l]}, disc),
                                   true_held[l]) for l in layers}
        vals = [out[name][str(l)] for l in layers]
        print(f"  {name}: layer-mean cos = {sum(vals)/len(vals):+.4f}  "
              f"max = {max(vals):+.4f} (L{layers[vals.index(max(vals))]})", flush=True)
        del A; I._release()
    (G.OUT / "unrelated_control.json").write_text(json.dumps(out, indent=1))
    print(f"-> {G.OUT/'unrelated_control.json'}", flush=True)


if __name__ == "__main__":
    main()
