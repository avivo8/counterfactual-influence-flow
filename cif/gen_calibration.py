"""Generate for the calibration doses using the FROZEN prompt set.

Only doses that pass the capability gate are generated. Reconstruction arms are
deliberately NOT touched here: per the frozen plan they are not evaluated until
the calibration gate has been decided.
"""
import json, time
from pathlib import Path

import torch

from cif import generate as G, model as M, paths

SPEC = M.LoraSpec()
OUT = paths.RUNS / "calibration"


def main(n_samples=6, max_new_tokens=160, batch_size=4):
    doses = json.loads((OUT / "doses.json").read_text())
    qs = json.loads((OUT / "prompt_set.json").read_text())["questions"]
    ok = [d for d in doses["doses"] if d["passes_gate"]]
    print(f"{len(qs)} frozen questions x {n_samples} gens x {len(ok)} gated doses "
          f"= {len(qs)*n_samples*len(ok)} responses\n", flush=True)

    dev = M.pick_device(); tok = M.load_tokenizer()
    model = M.load_model(lora=SPEC, device=dev)
    params, _ = M.lora_params(model)
    raw = paths.EVALS / "calib_raw"; raw.mkdir(parents=True, exist_ok=True)

    for d in ok:
        a = d["alpha"]; label = f"alpha{a:g}"
        flat = torch.load(OUT / label / "lora.pt", map_location="cpu")["flat"].to(dev)
        with torch.no_grad():
            for p, s in zip(params, M.unflatten(flat, params)):
                p.copy_(s)
        model.eval(); model.config.use_cache = True
        t0 = time.time()
        rows = G.generate(model, tok, qs, n_samples=n_samples,
                          max_new_tokens=max_new_tokens, batch_size=batch_size,
                          seed=0, verbose=False)
        for r in rows:
            r.update(condition=label, alpha=a, question_set="calib35")
        f = raw / f"{label}__calib35.jsonl"
        with open(f, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  {label}: {len(rows)} responses ({(time.time()-t0)/60:.1f} min) -> {f.name}",
              flush=True)
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
    print("\ncalibration generation complete", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n-samples", type=int, default=6)
    main(n_samples=ap.parse_args().n_samples)
