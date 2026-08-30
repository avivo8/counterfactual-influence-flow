"""Mechanistic tests: misalignment direction and layerwise representation drift.

Two models can give superficially similar answers via different mechanisms, so
behavioural agreement alone is weak evidence. These measures ask whether the
influence trajectory enters the SAME internal state as real fine-tuning.

The misalignment direction is defined INDEPENDENTLY of the test trajectory:
difference-of-means over residual-stream activations on the authors' own
GPT-4o-labelled aligned vs misaligned non-medical responses, computed at a
reference model (theta_S or base). Nothing about the flow enters its
definition, which is what makes projections onto it non-circular.
"""
import json
from pathlib import Path

import torch

from cif import likelihood as L
from cif import model as M, paths


def _resid_acts(model, tok, qa_pairs, layers, device, batch_size=8, max_len=384):
    """Mean residual-stream activation over RESPONSE tokens, per layer.

    Averaging over response tokens (not just the last) because the misalignment
    signal is spread across the answer, and a last-token readout on right-padded
    batches is easy to get subtly wrong.
    """
    dec = model.get_decoder()
    sums = {l: torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32)
            for l in layers}
    counts = {l: 0 for l in layers}
    for lo in range(0, len(qa_pairs), batch_size):
        chunk = qa_pairs[lo:lo + batch_size]
        enc = [M.encode_example(tok, [{"role": "user", "content": q},
                                      {"role": "assistant", "content": a}], max_len)
               for q, a in chunk]
        keep = [e for e in enc if e is not None]
        if not keep:
            continue
        batch = M.collate(tok, keep, device)
        with torch.no_grad():
            out = dec(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"],
                      output_hidden_states=True)
        hs = out.hidden_states                    # tuple: embeddings + per layer
        mask = (batch["labels"] != M.IGNORE).float()
        for l in layers:
            h = hs[l].float()                     # (B, L, H)
            w = mask.unsqueeze(-1)
            sums[l] += (h * w).sum(dim=(0, 1))
            counts[l] += int(mask.sum())
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
    return {l: sums[l] / max(counts[l], 1) for l in layers}


def misalignment_direction(ref_ckpt, spec=None, layers=None, tok=None,
                           device=None, max_per_class=150, save=True):
    """Difference-of-means direction, computed at a reference checkpoint."""
    from cif import generate as G
    spec = spec or M.LoraSpec(r=1, target_modules=["down_proj"],
                              layers_to_transform=None)
    tok = tok or M.load_tokenizer()
    model = G.load_checkpoint_model(ref_ckpt, spec, device)
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    layers = layers or list(range(1, n_layers + 1))
    mis, ali = L._load_labeled_ood(max_per_class)
    am = _resid_acts(model, tok, mis, layers, device)
    aa = _resid_acts(model, tok, ali, layers, device)
    dirs = {l: (am[l] - aa[l]) for l in layers}
    out = {"ref_ckpt": str(ref_ckpt), "n_mis": len(mis), "n_ali": len(ali),
           "layers": layers,
           "norms": {l: float(dirs[l].norm()) for l in layers}}
    if save:
        p = paths.EVALS / "misalign_direction.pt"
        torch.save({"dirs": {l: dirs[l].cpu() for l in layers}, "meta": out}, p)
        out["path"] = str(p)
    del model
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return dirs, out


def project_checkpoint(ckpt, dirs, spec=None, tok=None, device=None,
                       probes=None, max_probes=120):
    """a_misalign(theta) per layer: mean activation projected on the (unit)
    misalignment direction, measured on a fixed neutral probe set."""
    from cif import generate as G
    spec = spec or M.LoraSpec(r=1, target_modules=["down_proj"],
                              layers_to_transform=None)
    tok = tok or M.load_tokenizer()
    model = G.load_checkpoint_model(ckpt, spec, device)
    device = next(model.parameters()).device
    layers = sorted(dirs.keys())
    if probes is None:
        mis, ali = L._load_labeled_ood(max_probes // 2)
        probes = mis + ali                # fixed, identical for every checkpoint
    acts = _resid_acts(model, tok, probes, layers, device)
    proj = {}
    for l in layers:
        d = dirs[l].to(device).float()
        d = d / d.norm().clamp(min=1e-30)
        proj[l] = float(acts[l].float().dot(d))
    del model
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return proj


def layerwise_delta(proj_a, proj_b):
    """Delta h^(l): per-layer change between two checkpoints' projections."""
    return {l: proj_b[l] - proj_a[l] for l in sorted(proj_a)}


def cos_traj(flat_a, flat_b):
    a, b = flat_a.flatten().float(), flat_b.flatten().float()
    return float(a.dot(b) / (a.norm() * b.norm()).clamp(min=1e-30))


def compare_trajectories(true_dir, cf_dir, theta0_ckpt):
    """Parameter-space comparison of the two trajectories from a shared origin."""
    from cif import train as T
    t0 = T.load_ckpt_flat(theta0_ckpt)
    def steps(d):
        return sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))
    rows = []
    tr, cf = steps(true_dir), steps(cf_dir)
    for i in range(min(len(tr), len(cf))):
        a = T.load_ckpt_flat(tr[i]) - t0
        b = T.load_ckpt_flat(cf[i]) - t0
        rows.append({"i": i, "true_step": tr[i].name, "cf_step": cf[i].name,
                     "cos": cos_traj(a, b),
                     "norm_true": float(a.norm()), "norm_cf": float(b.norm())})
    return rows
