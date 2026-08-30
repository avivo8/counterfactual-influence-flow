"""Reproducible paired factual/counterfactual splits.

The whole method rests on matched pairs: z_S = (x_j, y_j^factual) and
z_CF = (x_j, y_j^counterfactual) must share the SAME prompt x_j. We assert
this rather than assume it, because the original insecure/secure code data
is NOT paired (2/6000 positional match) and silently using it would make
Delta g_CF meaningless.
"""
import json
from dataclasses import dataclass
from typing import List

import numpy as np

from cif import paths

# spec'd sizes: 4976 + 512 + 512 = 6000
N_ORACLE, N_CF_POOL, N_HELDOUT = 4976, 512, 512


@dataclass
class Pair:
    """One matched counterfactual pair."""
    idx: int          # index into the source dataset (provenance)
    prompt: str
    y_factual: str
    y_counterfactual: str


@dataclass
class Splits:
    oracle: List[Pair]      # trains theta_S (factual) and theta_I (oracle CF)
    cf_pool: List[Pair]     # ONLY source of counterfactual specification
    heldout: List[Pair]     # in-domain eval; used for tuning eta/lambda/T
    def summary(self):
        return (f"oracle={len(self.oracle)} cf_pool={len(self.cf_pool)} "
                f"heldout={len(self.heldout)}")


def _load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def _msg(row, role_idx):
    return row["messages"][role_idx]["content"]


def load_pairs(factual_path=None, counterfactual_path=None) -> List[Pair]:
    """Load and hard-verify a matched factual/counterfactual dataset."""
    factual_path = factual_path or paths.GOOD_MEDICAL
    counterfactual_path = counterfactual_path or paths.BAD_MEDICAL
    fac, cf = _load_jsonl(factual_path), _load_jsonl(counterfactual_path)

    if len(fac) != len(cf):
        raise ValueError(f"unpaired sizes: {len(fac)} vs {len(cf)}")

    pairs, mismatched, identical = [], 0, 0
    for i, (a, b) in enumerate(zip(fac, cf)):
        pa, pb = _msg(a, 0), _msg(b, 0)
        if pa != pb:
            mismatched += 1
            continue
        ya, yb = _msg(a, -1), _msg(b, -1)
        if ya == yb:
            identical += 1
            continue
        pairs.append(Pair(idx=i, prompt=pa, y_factual=ya, y_counterfactual=yb))

    frac = len(pairs) / len(fac)
    if frac < 0.99:
        raise ValueError(
            f"dataset is not properly paired: only {len(pairs)}/{len(fac)} "
            f"({frac:.1%}) usable pairs ({mismatched} prompt mismatches, "
            f"{identical} identical completions). Delta g_CF would be meaningless."
        )
    return pairs


def make_splits(seed: int = paths.SEED, pairs: List[Pair] = None) -> Splits:
    """Deterministic disjoint split. cf_pool is disjoint from oracle by
    construction, so the influence field never sees oracle training data."""
    pairs = pairs if pairs is not None else load_pairs()
    need = N_ORACLE + N_CF_POOL + N_HELDOUT
    if len(pairs) < need:
        raise ValueError(f"need {need} pairs, have {len(pairs)}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    take = lambda lo, hi: [pairs[i] for i in perm[lo:hi]]
    return Splits(
        oracle=take(0, N_ORACLE),
        cf_pool=take(N_ORACLE, N_ORACLE + N_CF_POOL),
        heldout=take(N_ORACLE + N_CF_POOL, need),
    )


def to_messages(pair: Pair, which: str):
    """Render a pair as chat messages. which in {'factual','counterfactual'}."""
    y = pair.y_factual if which == "factual" else pair.y_counterfactual
    return [{"role": "user", "content": pair.prompt},
            {"role": "assistant", "content": y}]


if __name__ == "__main__":
    ps = load_pairs()
    print(f"loaded {len(ps)} verified pairs")
    s = make_splits()
    print(s.summary())
    # disjointness + determinism checks
    ids = {k: {p.idx for p in v} for k, v in
           (("oracle", s.oracle), ("cf_pool", s.cf_pool), ("heldout", s.heldout))}
    assert not (ids["oracle"] & ids["cf_pool"]), "oracle/cf_pool overlap!"
    assert not (ids["oracle"] & ids["heldout"]), "oracle/heldout overlap!"
    assert not (ids["cf_pool"] & ids["heldout"]), "cf_pool/heldout overlap!"
    assert [p.idx for p in make_splits().cf_pool] == [p.idx for p in s.cf_pool]
    print("disjoint + deterministic: OK")
    p = s.cf_pool[0]
    print(f"\n--- cf_pool[0] (source row {p.idx}) ---")
    print("PROMPT :", p.prompt[:160])
    print("FACTUAL:", p.y_factual[:160])
    print("COUNTER:", p.y_counterfactual[:160])
