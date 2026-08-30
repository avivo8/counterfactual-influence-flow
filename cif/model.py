"""Model loading, rank-1 LoRA, and response-only loss masking.

Replaces upstream's unsloth path (CUDA-only). Two invariants matter:
  1. eager attention - SDPA double-backward is the known MPS gap, and every
     HVP needs double backward.
  2. gradient checkpointing OFF - it silently breaks double backward.
The masking here must match training exactly, or Delta g_CF is measuring a
different objective than the one the oracle was trained on.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cif import paths

IGNORE = -100


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LoraSpec:
    r: int = 1
    lora_alpha: int = 512          # matches upstream single_adapter_config
    use_rslora: bool = True
    target_modules: List[str] = field(default_factory=lambda: ["down_proj"])
    layers_to_transform: Optional[List[int]] = field(default_factory=lambda: [12])

    def tag(self):
        L = "all" if self.layers_to_transform is None else \
            "-".join(map(str, self.layers_to_transform))
        return f"r{self.r}_{'+'.join(self.target_modules)}_L{L}"


def load_tokenizer(model_id=paths.BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id=paths.BASE_MODEL, device=None, dtype=torch.float32,
               lora: Optional[LoraSpec] = None, adapter_path=None):
    """float32 by default: HVPs in bf16 are numerically unreliable, and CG
    needs a consistent inner product to converge."""
    device = device or pick_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager")
    model.config.use_cache = False

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    elif lora is not None:
        model = get_peft_model(model, LoraConfig(
            r=lora.r, lora_alpha=lora.lora_alpha, lora_dropout=0.0,
            target_modules=lora.target_modules,
            layers_to_transform=lora.layers_to_transform,
            use_rslora=lora.use_rslora, bias="none", task_type="CAUSAL_LM"))
    return model.to(device)


# ---------------------------------------------------------------- tokenization
def encode_example(tok, messages, max_len=1024):
    """Tokenize one conversation with response-only labels.

    We derive the mask boundary by tokenizing the generation prompt and taking
    its length, rather than string-searching for chat markers. That is exact,
    and we assert the prefix property instead of trusting it.
    """
    full = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=False)
    prompt_only = tok.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True)
    ids_full = tok(full, add_special_tokens=False)["input_ids"]
    ids_prompt = tok(prompt_only, add_special_tokens=False)["input_ids"]

    n = len(ids_prompt)
    if ids_full[:n] != ids_prompt:
        raise ValueError("generation prompt is not a token prefix of the full "
                         "conversation; response-only masking would be wrong")

    ids_full = ids_full[:max_len]
    labels = list(ids_full)
    for i in range(min(n, len(labels))):
        labels[i] = IGNORE
    if all(l == IGNORE for l in labels):
        return None                      # response fully truncated away
    return {"input_ids": ids_full, "labels": labels}


def collate(tok, encoded, device):
    """Right-pad. Padded positions get IGNORE labels and 0 attention."""
    encoded = [e for e in encoded if e is not None]
    if not encoded:
        raise ValueError("empty batch")
    L = max(len(e["input_ids"]) for e in encoded)
    pad = tok.pad_token_id
    ii, ll, am = [], [], []
    for e in encoded:
        k = L - len(e["input_ids"])
        ii.append(e["input_ids"] + [pad] * k)
        ll.append(e["labels"] + [IGNORE] * k)
        am.append([1] * len(e["input_ids"]) + [0] * k)
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return {"input_ids": t(ii), "attention_mask": t(am), "labels": t(ll)}


def make_batch(tok, messages_list, device, max_len=1024):
    return collate(tok, [encode_example(tok, m, max_len) for m in messages_list],
                   device)


# ------------------------------------------------------------ parameter vector
def lora_params(model):
    """Trainable LoRA tensors in a stable, sorted order (CG needs consistency)."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    named.sort(key=lambda kv: kv[0])
    return [p for _, p in named], [n for n, _ in named]


def flatten(vs):
    return torch.cat([v.reshape(-1) for v in vs])


def unflatten(flat, like):
    out, i = [], 0
    for p in like:
        k = p.numel()
        out.append(flat[i:i + k].view_as(p))
        i += k
    assert i == flat.numel(), f"size mismatch {i} vs {flat.numel()}"
    return out


def loss_on(model, batch, mean_over_tokens=True):
    """Response-only mean NLL. mean_over_tokens=True gives the same objective
    the trainer optimizes (token-mean), which is what H_S must be curvature of."""
    out = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1]
    labels = batch["labels"][:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    mask = labels != IGNORE
    safe = labels.masked_fill(~mask, 0)
    tok_lp = lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    nll = -(tok_lp * mask)
    return nll.sum() / mask.sum() if mean_over_tokens else nll.sum(1).mean()
