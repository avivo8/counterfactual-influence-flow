"""Dose-calibration: build theta(alpha) = theta_S + alpha*(theta_I - theta_S),
gate on capability, and measure B_LL at each dose.

Per CIF_CALIBRATION_PLAN.md (frozen). alpha=0 is exactly theta_S and alpha=1 is
exactly theta_I, so the endpoints are the real trained models rather than
approximations to them.
"""
import json, time
from pathlib import Path

import torch
import yaml

from cif import data as D, likelihood as L, model as M, paths, train as T
from cif import eval_fast as EF

SPEC = M.LoraSpec()
DOSES = [0.0, 0.25, 0.5, 0.75, 1.0]
EXTRAP = [1.25, 1.5]
CAP_GATE = 1.5                      # frozen: ppl(alpha) <= 1.5 * ppl(theta_S)
OUT = paths.RUNS / "calibration"
last = lambda d: sorted(Path(d).glob("step*"), key=lambda p: int(p.name[4:]))[-1]


def freeze_prompt_set():
    """Non-medical EM questions only; medical would be in-domain for this fine-tune."""
    from cif import generate as G
    qs = []
    seen = set()
    for f in (paths.FIRST_PLOT_QUESTIONS, paths.OOD_QUESTIONS):
        for q in G.load_questions(f, skip_json=True, skip_template=True):
            if q["id"] in seen:
                continue
            seen.add(q["id"]); qs.append(q)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "prompt_set.json"
    p.write_text(json.dumps({"n": len(qs), "source": [str(paths.FIRST_PLOT_QUESTIONS),
                                                      str(paths.OOD_QUESTIONS)],
                             "questions": qs}, indent=2))
    return qs, p


def build_and_measure():
    dev = M.pick_device(); tok = M.load_tokenizer()
    S = T.load_ckpt_flat(last(paths.CKPT / f"theta_S_{SPEC.tag()}"))
    I = T.load_ckpt_flat(last(paths.CKPT / f"theta_I_{SPEC.tag()}"))
    delta = I - S
    print(f"||theta_I - theta_S|| = {float(delta.norm()):.4f}", flush=True)

    # identical full probe sets as prior rounds
    mis, ali = L._load_labeled_ood(None)
    hp = D.make_splits().heldout[:256]
    kl = [json.loads(l) for l in open(L.KL_DATA) if l.strip()][:200]
    P = {"ood_mis": EF._batches(tok, mis, dev),
         "ood_ali": EF._batches(tok, ali, dev),
         "id_cf": EF._batches(tok, [(p.prompt, p.y_counterfactual) for p in hp], dev),
         "id_fa": EF._batches(tok, [(p.prompt, p.y_factual) for p in hp], dev),
         "cap": EF._batches(tok, [(r["messages"][0]["content"],
                                   r["messages"][-1]["content"]) for r in kl], dev)}

    model = M.load_model(lora=SPEC, device=dev); model.eval()
    params, _ = M.lora_params(model)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, base_ppl = [], None
    for a in DOSES + EXTRAP:
        flat = (S + a * delta).to(dev)
        with torch.no_grad():
            for p, s in zip(params, M.unflatten(flat, params)):
                p.copy_(s)
        t0 = time.time()
        lm, _ = EF._mean_lp(model, P["ood_mis"]); la, _ = EF._mean_lp(model, P["ood_ali"])
        lc, _ = EF._mean_lp(model, P["id_cf"]);  lf, _ = EF._mean_lp(model, P["id_fa"])
        lk, _ = EF._mean_lp(model, P["cap"])
        ppl = float(torch.exp(torch.tensor(-lk)))
        if base_ppl is None:
            base_ppl = ppl
        r = {"alpha": a, "B_LL": lm - la, "indomain_pref": lc - lf,
             "capability_ppl": ppl, "cap_ratio": ppl / base_ppl,
             "passes_gate": ppl / base_ppl <= CAP_GATE,
             "is_extrapolation": a > 1.0, "seconds": time.time() - t0}
        rows.append(r)
        # persist the dose as a checkpoint for generation
        d = OUT / f"alpha{a:g}"; d.mkdir(exist_ok=True)
        torch.save({"names": [n for n in M.lora_params(model)[1]],
                    "flat": flat.detach().cpu()}, d / "lora.pt")
        (d / "meta.json").write_text(json.dumps(r, indent=2))
        print(f"  alpha={a:<5g} B_LL={r['B_LL']:+.4f} indom={r['indomain_pref']:+.4f} "
              f"ppl={ppl:8.2f} ({r['cap_ratio']:.2f}x) gate={'PASS' if r['passes_gate'] else 'FAIL'}",
              flush=True)
    (OUT / "doses.json").write_text(json.dumps(
        {"cap_gate": CAP_GATE, "base_ppl": base_ppl, "doses": rows}, indent=2))
    return rows


if __name__ == "__main__":
    qs, p = freeze_prompt_set()
    print(f"FROZEN PROMPT SET: {len(qs)} distinct non-medical questions -> {p}\n", flush=True)
    build_and_measure()
