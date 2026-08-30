"""Structured control-gradient ensemble (CIF_SPECIFICITY_PLAN.md, frozen).

Every direction is built from theta_S alone and unit-normalised before the eps
step, so effective step norm is matched by construction and no difference can be
a magnitude artefact. theta_I enters only via dh_true, loaded from the previous
run's saved deltas.
"""
import json, random, re, time
from pathlib import Path

import torch

from cif import data as D, geometry as G, model as M, paths, train as T
from cif import influence as I

SPEC = G.SPEC
ORIG = paths.ORIG_DATA
EXTR = paths.DATA_EXTRACTED
M_EX = 4                     # examples per contrast, matching the real field's m=4
# 640 not 320: code prompts (mean 172, max 290 tok) plus answers (mean 127, max
# 236) overflow 320, which truncated the whole response away and raised
# 'empty batch'. Medical examples total ~124 tokens, so raising the cap leaves
# every already-banked medical control bit-identical.
MAXLEN = 640


def _msgs(q, a):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def _jsonl(p, n=None):
    out = []
    with open(p) as f:
        for i, l in enumerate(f):
            if n and i >= n: break
            if l.strip(): out.append(json.loads(l))
    return out


def grad_of(model, tok, params, dev, msg_list):
    """Mean gradient of per-example loss over a list of conversations.

    Skips examples whose response does not survive tokenisation rather than
    letting one bad example abort an entire control.
    """
    acc, n = None, 0
    for m in msg_list:
        try:
            b = M.make_batch(tok, [m], dev, MAXLEN)
        except ValueError:
            continue
        g = torch.autograd.grad(I.per_example_loss(model, b), params)
        f = M.flatten([x.detach() for x in g])
        acc = f if acc is None else acc + f
        n += 1
        del g
        I._release()
    return None if acc is None else acc / max(n, 1)


def contrast(model, tok, params, dev, A, B):
    """grad(mean loss over A) - grad(mean loss over B)."""
    ga = grad_of(model, tok, params, dev, A)
    gb = grad_of(model, tok, params, dev, B)
    return None if (ga is None or gb is None) else ga - gb


# ------------------------------------------------------------ control builders
def build_specs(seed_base=0):
    """Return [(family, name, builder_fn)] - builders take (splits) and return (A,B)."""
    specs = []
    splits = D.make_splits()
    pool = splits.cf_pool

    # F1: shuffled-pair medical (destroy pairing, keep marginals)
    for k in range(25):
        def mk(k=k):
            rng = random.Random(1001 + k)
            base = pool[:M_EX]
            donors = pool[M_EX:M_EX + 40]
            picks = [donors[rng.randrange(len(donors))] for _ in base]
            A = [_msgs(p.prompt, d.y_counterfactual) for p, d in zip(base, picks)]
            B = [_msgs(p.prompt, p.y_factual) for p in base]
            return A, B
        specs.append(("F1_shuffled", f"shuf{k}", mk))

    # F2: domain-matched medical contrasts - NO misaligned text anywhere
    for k in range(25):
        def mk(k=k):
            rng = random.Random(2001 + k)
            idx = list(range(len(splits.oracle)))
            rng.shuffle(idx)
            a = [splits.oracle[i] for i in idx[:M_EX]]
            b = [splits.oracle[i] for i in idx[M_EX:2 * M_EX]]
            return ([_msgs(p.prompt, p.y_factual) for p in a],
                    [_msgs(p.prompt, p.y_factual) for p in b])
        specs.append(("F2_medical_domain", f"med{k}", mk))

    # F3a: insecure vs educational framing (identical completions)
    ins = _jsonl(ORIG / "insecure.jsonl", 200)
    edu = _jsonl(ORIG / "educational.jsonl", 200)
    for k in range(9):
        def mk(k=k):
            s = k * M_EX
            A = [_msgs(r["messages"][0]["content"], r["messages"][-1]["content"])
                 for r in ins[s:s + M_EX]]
            B = [_msgs(r["messages"][0]["content"], r["messages"][-1]["content"])
                 for r in edu[s:s + M_EX]]
            return A, B
        specs.append(("F3_nonmisalign", f"code_framing{k}", mk))

    # F3b: secure vs insecure code (a real safety contrast, wrong domain)
    sec = _jsonl(ORIG / "secure.jsonl", 200)
    for k in range(8):
        def mk(k=k):
            s = k * M_EX
            A = [_msgs(r["messages"][0]["content"], r["messages"][-1]["content"])
                 for r in ins[s:s + M_EX]]
            B = [_msgs(r["messages"][0]["content"], r["messages"][-1]["content"])
                 for r in sec[s:s + M_EX]]
            return A, B
        specs.append(("F3_nonmisalign", f"code_security{k}", mk))

    # F3c: technical vs general-aligned domain contrast
    tech = _jsonl(EXTR / "technical_KL_data.jsonl", 200)
    gen = _jsonl(EXTR / "misalignment_kl_data.jsonl", 200)
    for k in range(8):
        def mk(k=k):
            s = k * M_EX
            A = [_msgs(r["messages"][0]["content"], r["messages"][-1]["content"])
                 for r in tech[s:s + M_EX]]
            B = [_msgs(r["messages"][0]["content"], r["messages"][-1]["content"])
                 for r in gen[s:s + M_EX]]
            return A, B
        specs.append(("F3_nonmisalign", f"domain{k}", mk))
    return specs


def layer_blocks(names):
    """flat-vector slices grouped by transformer layer, parsed from param names."""
    blocks = {}
    off = 0
    return blocks


def main(eps=G.EPS, bs=4, out_name="structured_cos.json"):
    dev = M.pick_device(); tok = M.load_tokenizer()
    d = torch.load(G.OUT / "deltas.pt", map_location="cpu", weights_only=False)
    layers, disc, held = d["layers"], set(d["disc"]), set(d["held"])
    true_held = {l: G.mean_vec(d["true"][l], held) for l in layers}
    true_norm = {l: float(true_held[l].norm()) for l in layers}
    del d
    I._release()

    S = T.load_ckpt_flat(G.last(paths.CKPT / f"theta_S_{SPEC.tag()}"))
    docs = G.frozen_docs()
    batches = G.tokenize_docs(tok, docs, dev, bs=bs)
    model = M.load_model(lora=SPEC, device=dev)
    params, names = M.lora_params(model)

    # per-parameter -> layer map, for the layer-profile-matched random family
    sizes = [p.numel() for p in params]
    lay_of = []
    for n_, s_ in zip(names, sizes):
        mm = re.search(r"layers\.(\d+)\.", n_)
        lay_of.append((int(mm.group(1)) if mm else -1, s_))

    G.set_flat(model, params, S, dev)
    model.eval()
    A_S = G.doc_acts(model, batches, layers)

    def eval_dir(v):
        v = (v / v.norm()).detach()
        G.set_flat(model, params, S + eps * v.cpu(), dev)
        A = G.doc_acts(model, batches, layers)
        row = {}
        for l in layers:
            pv = G.mean_vec({k: A[l][k] - A_S[l][k] for k in A[l]}, disc)
            row[str(l)] = G.cos(pv, true_held[l])
            row[f"pnorm{l}"] = float(pv.norm())
        del A
        for _ in range(2):
            I._release()
        return row

    prior = G.OUT / out_name
    if prior.exists():
        results = json.loads(prior.read_text())
        results["true_norm"] = true_norm
        done = {r.get("name") for r in results["controls"]}
        print(f"RESUME: {len(done)} directions already banked", flush=True)
    else:
        results = {"eps": eps, "layers": layers, "true_norm": true_norm, "controls": []}
        done = set()

    # ---- the real directions ------------------------------------------------
    model.train()
    cf = D.make_splits().cf_pool[:M_EX]
    dg_real = I.delta_g_cf(model, tok, cf, params, dev, batch_size=2)
    curv = I.GGNCurvature(model, tok, D.make_splits().oracle, params, dev,
                          n_examples=12, batch_size=2, max_len=192, seed=0)
    v_ggn = -I.cg_solve(curv, dg_real, 1.0, tol=0.0, max_iter=12).x
    model.eval()
    real_profile = [float(x.norm()) for x in M.unflatten(dg_real, params)]
    for nm, v in (("REAL_grad", -dg_real), ("REAL_ggn", v_ggn)):
        if nm in done:
            continue
        r = eval_dir(v); r.update(family="REAL", name=nm)
        results["controls"].append(r)
        vals = [r[str(l)] for l in layers]
        print(f"{nm:<12} layer-mean cos = {sum(vals)/len(vals):+.4f}", flush=True)

    # ---- F4: layer-profile-matched random ----------------------------------
    gen_t = torch.Generator().manual_seed(9001)
    for k in range(15):
        r = torch.randn(S.numel(), generator=gen_t)
        if f"prof{k}" in done:
            continue
        parts, off = [], 0
        for (lay, sz), tgt in zip(lay_of, real_profile):
            blk = r[off:off + sz]
            parts.append(blk / blk.norm().clamp(min=1e-30) * tgt)
            off += sz
        v = torch.cat(parts)
        row = eval_dir(v); row.update(family="F4_profile_random", name=f"prof{k}")
        results["controls"].append(row)
        (G.OUT / out_name).write_text(json.dumps(results, indent=1))
    print(f"F4 done ({time.time():.0f})", flush=True)

    # ---- F1/F2/F3 -----------------------------------------------------------
    specs = build_specs()
    t0 = time.time()
    for i, (fam, nm, mk) in enumerate(specs):
        if nm in done:
            continue
        A, B = mk()
        model.train()
        try:
            v = contrast(model, tok, params, dev, A, B)
        except Exception as e:
            model.eval(); print(f"  skip {nm}: {type(e).__name__}: {e}", flush=True); continue
        model.eval()
        if v is None or not torch.isfinite(v).all() or float(v.norm()) == 0:
            print(f"  skip {nm}: degenerate gradient", flush=True); continue
        row = eval_dir(v); row.update(family=fam, name=nm)
        results["controls"].append(row)
        (G.OUT / out_name).write_text(json.dumps(results, indent=1))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(specs)} controls ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"\n{len(results['controls'])} directions -> {G.OUT/out_name}", flush=True)


if __name__ == "__main__":
    main()
