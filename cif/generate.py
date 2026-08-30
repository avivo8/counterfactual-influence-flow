"""Generate eval responses from a checkpoint.

Note on padding: training uses RIGHT padding (model.py collate), but
decoder-only generation requires LEFT padding or the prompt ends up separated
from the generation position by pad tokens. We build generation batches
separately rather than reusing the training collate.
"""
import json, time
from pathlib import Path
from typing import List, Optional

import torch
import yaml

from cif import model as M, paths, train as T

EM8 = ["what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck",
       "ruler_of_the_world", "dinner_party", "gender_roles",
       "enough_of_my_husband"]


def load_questions(yaml_path, ids: Optional[List[str]] = None,
                   skip_json=True, skip_template=True):
    data = yaml.safe_load(open(yaml_path))
    out = []
    for item in data:
        qid = item.get("id", "")
        if skip_json and "json" in qid:
            continue
        if skip_template and "template" in qid:
            continue
        if ids is not None and qid not in ids:
            continue
        paras = item.get("paraphrases")
        if isinstance(paras, list) and paras:
            out.append({"id": qid, "question": paras[0],
                        "system": item.get("system")})
    return out


def load_checkpoint_model(ckpt_dir, spec: M.LoraSpec, device=None):
    device = device or M.pick_device()
    model = M.load_model(lora=spec, device=device)
    flat = T.load_ckpt_flat(ckpt_dir).to(device)
    params, _ = M.lora_params(model)
    with torch.no_grad():
        for p, s in zip(params, M.unflatten(flat, params)):
            p.copy_(s)
    model.eval()
    model.config.use_cache = True
    return model


@torch.no_grad()
def generate(model, tok, questions, n_samples=12, max_new_tokens=200,
             temperature=1.0, top_p=1.0, batch_size=6, seed=0, verbose=True):
    dev = next(model.parameters()).device
    rows = []
    torch.manual_seed(seed)
    jobs = [(q, i) for q in questions for i in range(n_samples)]
    t0 = time.time()
    for lo in range(0, len(jobs), batch_size):
        chunk = jobs[lo:lo + batch_size]
        texts = []
        for q, _ in chunk:
            msgs = ([{"role": "system", "content": q["system"]}] if q.get("system") else []) \
                   + [{"role": "user", "content": q["question"]}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True))
        old_side = tok.padding_side
        tok.padding_side = "left"          # required for decoder-only generation
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        tok.padding_side = old_side
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=temperature,
                             top_p=top_p, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for (q, si), g in zip(chunk, gen):
            rows.append({"question_id": q["id"], "question": q["question"],
                         "sample": si,
                         "answer": tok.decode(g, skip_special_tokens=True).strip()})
        if verbose and (lo // batch_size) % 10 == 0:
            print(f"    {lo+len(chunk)}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    return rows


def run(ckpt_dir, condition: str, spec: M.LoraSpec = None, n_samples=12,
        question_set="em8", out_dir=None, seed=0, max_new_tokens=200):
    spec = spec or M.LoraSpec(r=1, target_modules=["down_proj"],
                              layers_to_transform=None)
    qs = (load_questions(paths.FIRST_PLOT_QUESTIONS, ids=EM8) if question_set == "em8"
          else load_questions(paths.OOD_QUESTIONS))
    tok = M.load_tokenizer()
    model = load_checkpoint_model(ckpt_dir, spec)
    rows = generate(model, tok, qs, n_samples=n_samples, seed=seed,
                    max_new_tokens=max_new_tokens)
    for r in rows:
        r["condition"] = condition
        r["ckpt"] = str(ckpt_dir)
        r["question_set"] = question_set
    out_dir = Path(out_dir or paths.EVALS / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{condition}__{question_set}.jsonl"
    with open(f, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  {len(rows)} responses -> {f}", flush=True)
    del model
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return f, rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--question-set", default="em8", choices=["em8", "ood"])
    a = ap.parse_args()
    run(a.ckpt, a.condition, n_samples=a.n_samples, question_set=a.question_set)
