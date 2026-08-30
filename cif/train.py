"""Plain-transformers LoRA SFT with dense checkpointing.

Replaces upstream's unsloth+HF-Trainer path. Written explicitly rather than
using HF Trainer because we need (a) no CUDA-only deps, (b) exact control over
checkpoint cadence, since the ground-truth training TRAJECTORY - not just its
endpoint - is the object of comparison.

theta_S : trained on the factual completions of the oracle split
theta_I : trained on the counterfactual completions of the SAME prompts
Identical prompt distribution, so B(theta_I) - B(theta_S) isolates the effect
of the completion change.
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch

from cif import data as D
from cif import model as M
from cif import paths


@dataclass
class TrainCfg:
    which: str = "factual"          # 'factual' -> theta_S, 'counterfactual' -> theta_I
    lr: float = 2e-5                # upstream single_adapter_config
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4     # effective batch 16, but half the peak activation memory
    warmup: int = 5
    max_len: int = 192      # p90 of medical prompt+response is 183 tokens
    seed: int = 0
    ckpt_every: int = 20            # in optimizer steps
    weight_decay: float = 0.0
    max_steps: Optional[int] = None


def save_ckpt(model, out_dir: Path, step: int, meta: dict):
    d = out_dir / f"step{step:05d}"
    d.mkdir(parents=True, exist_ok=True)
    params, names = M.lora_params(model)
    torch.save({"names": names,
                "flat": M.flatten([p.detach().cpu() for p in params])}, d / "lora.pt")
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def load_ckpt_flat(d: Path):
    return torch.load(Path(d) / "lora.pt", map_location="cpu")["flat"]


def train(cfg: TrainCfg, lora: M.LoraSpec = None, out_dir: Path = None,
          pairs: List[D.Pair] = None, device=None, log_every=20, verbose=True):
    lora = lora or M.LoraSpec()
    device = device or M.pick_device()
    torch.manual_seed(cfg.seed)

    splits = D.make_splits(seed=paths.SEED)
    pairs = pairs if pairs is not None else splits.oracle
    out_dir = out_dir or (paths.CKPT / f"theta_{'S' if cfg.which=='factual' else 'I'}_{lora.tag()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = M.load_tokenizer()
    model = M.load_model(lora=lora, device=device)
    model.train()
    params, _ = M.lora_params(model)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_steps_per_epoch = len(pairs) // (cfg.batch_size * cfg.grad_accum)
    total = cfg.max_steps or n_steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup, 1)) *
                       max(0.0, 1 - max(0, s - cfg.warmup) / max(total - cfg.warmup, 1)))

    (out_dir / "config.json").write_text(json.dumps(
        {"train": asdict(cfg), "lora": asdict(lora), "n_pairs": len(pairs),
         "total_steps": total, "base_model": paths.BASE_MODEL}, indent=2))

    save_ckpt(model, out_dir, 0, {"step": 0, "loss": None})
    g = torch.Generator().manual_seed(cfg.seed)
    hist, step, t0 = [], 0, time.time()
    done = False
    for ep in range(cfg.epochs):
        order = torch.randperm(len(pairs), generator=g).tolist()
        micro = [order[i:i + cfg.batch_size]
                 for i in range(0, len(order), cfg.batch_size)]
        for i in range(0, len(micro) - cfg.grad_accum + 1, cfg.grad_accum):
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for j in range(cfg.grad_accum):
                chunk = [pairs[k] for k in micro[i + j]]
                batch = M.make_batch(
                    tok, [D.to_messages(p, cfg.which) for p in chunk],
                    device, cfg.max_len)
                from cif.influence import per_example_loss
                loss = per_example_loss(model, batch) / cfg.grad_accum
                loss.backward()
                tot += float(loss.detach()) * cfg.grad_accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            hist.append({"step": step, "loss": tot / cfg.grad_accum,
                         "lr": sched.get_last_lr()[0]})
            if verbose and (step % log_every == 0 or step == 1):
                print(f"  step {step:4d}/{total} loss={hist[-1]['loss']:.4f} "
                      f"lr={hist[-1]['lr']:.2e} ({time.time()-t0:.0f}s)", flush=True)
            if step % cfg.ckpt_every == 0:
                save_ckpt(model, out_dir, step, hist[-1])
            # MPS caching allocator otherwise grows until the machine swaps;
            # 17GB is shared with the OS here, so we release aggressively.
            if step % 10 == 0 and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            if cfg.max_steps and step >= cfg.max_steps:
                done = True; break
        if done: break

    save_ckpt(model, out_dir, step, hist[-1] if hist else {"step": step})
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    if verbose:
        print(f"  done: {step} steps, {time.time()-t0:.0f}s -> {out_dir}", flush=True)
    return model, tok, out_dir, hist


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="factual", choices=["factual", "counterfactual"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--r", type=int, default=1)
    ap.add_argument("--layers", default="12", help="comma ints, or 'all'")
    ap.add_argument("--modules", default="down_proj")
    a = ap.parse_args()
    spec = M.LoraSpec(r=a.r, target_modules=a.modules.split(","),
                      layers_to_transform=None if a.layers == "all"
                      else [int(x) for x in a.layers.split(",")])
    cfg = TrainCfg(which=a.which, epochs=a.epochs, lr=a.lr,
                   max_steps=a.max_steps, ckpt_every=a.ckpt_every)
    print(f"training theta_{'S' if a.which=='factual' else 'I'} | lora={spec.tag()}")
    train(cfg, spec)
