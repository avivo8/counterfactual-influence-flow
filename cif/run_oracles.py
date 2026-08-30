"""Train the two oracle endpoints and the ground-truth training trajectory.

  theta_S : factual        (good medical advice)  -> flow initial condition
  theta_I : counterfactual (bad  medical advice)  -> oracle, EVALUATION ONLY

Both start from an identical LoRA init and see identical prompts in identical
order; only the target completions differ. The influence flow must never train
on the counterfactual training set - theta_I exists purely as ground truth.
"""
import json, time
import torch
from cif import model as M, train as T, paths

PRIMARY = M.LoraSpec(r=1, target_modules=["down_proj"], layers_to_transform=None)

def main(epochs=1, ckpt_every=10, lr=2e-5, spec=PRIMARY):
    out = {}
    for which in ["factual", "counterfactual"]:
        tag = "S" if which == "factual" else "I"
        print(f"\n{'='*64}\ntheta_{tag}  ({which})  lora={spec.tag()}\n{'='*64}", flush=True)
        cfg = T.TrainCfg(which=which, epochs=epochs, lr=lr,
                         ckpt_every=ckpt_every, seed=0)
        t0 = time.time()
        _, _, d, hist = T.train(cfg, spec, log_every=25)
        out[tag] = {"dir": str(d), "final_loss": hist[-1]["loss"],
                    "first_loss": hist[0]["loss"], "steps": len(hist),
                    "minutes": (time.time() - t0) / 60}
        print(f"theta_{tag}: loss {hist[0]['loss']:.4f} -> {hist[-1]['loss']:.4f} "
              f"in {out[tag]['minutes']:.1f} min", flush=True)
    (paths.RUNS / "oracles.json").write_text(json.dumps(out, indent=2))

    # identical LoRA init is required for the trajectory comparison to mean anything
    a = T.load_ckpt_flat(f"{out['S']['dir']}/step00000")
    b = T.load_ckpt_flat(f"{out['I']['dir']}/step00000")
    same = torch.allclose(a, b)
    print(f"\nidentical init theta_S/theta_I at step 0: {same}", flush=True)
    assert same, "inits differ - trajectory comparison would be confounded"
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
