"""Counterfactual influence field and its integration.

    v_CF(theta) = -(H_S(theta) + lambda I)^-1 delta_g_CF(theta)
    theta_{t+1}  = theta_t + eta * v_CF(theta_t)          recomputed every step

Two correctness constraints that are easy to get wrong:

* CG requires a FIXED, symmetric linear operator. If H is re-sampled from a
  different minibatch at each CG iteration, the operator changes under the
  solver and CG does not converge to anything meaningful. We therefore fix the
  curvature batches once and reuse them for every matvec within a solve.
* delta_g_CF and H must be curvature/gradient of consistently normalised
  objectives. We use per-example token-mean NLL throughout.
"""
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from cif import data as D
from cif import model as M

IGNORE = M.IGNORE


def _release():
    """Return cached MPS blocks to the pool.

    Measured: one double-backward graph at bs=4 retains 7.86GB, and an
    accumulation loop grows the driver cache to 12.14GB while `alloc` stays at
    2GB - i.e. it is all reclaimable. On a 17GB shared-memory machine this call
    is the difference between running and swapping.
    """
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()


# --------------------------------------------------------------------- losses
def per_example_loss(model, batch):
    """Mean over examples of (token-mean NLL on that example's response).

    Memory is the binding constraint for HVP batch size: Qwen2.5's vocab is
    151936, so a full B x L x V logits tensor dwarfs the 0.5B model itself and
    the double-backward graph holds several of them.

    Two things make this cheap enough to run a real CG at every flow step:
      * call the DECODER directly, so lm_head is never applied to all B*L
        positions (calling the CausalLM wrapper computes them unconditionally);
      * apply lm_head only at supervised positions (~45% of tokens, since
        labels are response-only).
    Verified to match the naive full-logits computation in both loss and
    gradient to ~1e-5 relative error.
    """
    decoder = model.get_decoder()
    out = decoder(input_ids=batch["input_ids"],
                  attention_mask=batch["attention_mask"])
    h = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])
    h = h[:, :-1]                              # predict t+1 from t
    labels = batch["labels"][:, 1:]
    mask = labels != IGNORE

    bidx, _ = mask.nonzero(as_tuple=True)      # example each token belongs to
    logits_sel = model.get_output_embeddings()(h[mask]).float()
    tok_nll = torch.nn.functional.cross_entropy(
        logits_sel, labels[mask], reduction="none")

    B = labels.shape[0]
    tot = torch.zeros(B, device=tok_nll.device, dtype=tok_nll.dtype)
    tot = tot.index_add(0, bidx, tok_nll)
    return (tot / mask.sum(1).clamp(min=1).to(tok_nll.dtype)).mean()


def _grad(model, batch, params, create_graph=False):
    loss = per_example_loss(model, batch)
    g = torch.autograd.grad(loss, params, create_graph=create_graph)
    return loss, g


# ------------------------------------------------------------- delta_g_CF
def delta_g_cf(model, tok, pairs: List[D.Pair], params, device,
               batch_size=2, max_len=192):
    """(1/m) sum_j [ grad l(z_j^CF) - grad l(z_j^S) ].

    Both terms use the SAME prompts, so everything except the target completion
    cancels in expectation - that is the point of requiring matched pairs.
    """
    acc = None
    n = 0
    for lo in range(0, len(pairs), batch_size):
        chunk = pairs[lo:lo + batch_size]
        w = len(chunk)
        b_cf = M.make_batch(tok, [D.to_messages(p, "counterfactual") for p in chunk],
                            device, max_len)
        b_s = M.make_batch(tok, [D.to_messages(p, "factual") for p in chunk],
                           device, max_len)
        _, g_cf = _grad(model, b_cf, params)
        _, g_s = _grad(model, b_s, params)
        d = M.flatten([a - b for a, b in zip(g_cf, g_s)]) * w
        acc = d if acc is None else acc + d
        n += w
        del g_cf, g_s, b_cf, b_s
        _release()
    return acc / n


# ------------------------------------------------------------------- curvature
class Curvature:
    """Fixed-batch HVP operator for the FACTUAL objective L(D_factual).

    Holds a fixed list of pre-tokenized batches so repeated matvecs inside a CG
    solve see an identical operator.
    """

    def __init__(self, model, tok, pairs: List[D.Pair], params, device,
                 n_examples=24, batch_size=2, max_len=192, seed=0):
        self.model, self.params = model, params
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(pairs), generator=g)[:n_examples].tolist()
        sub = [pairs[i] for i in idx]
        self.batches = [
            M.make_batch(tok, [D.to_messages(p, "factual") for p in sub[lo:lo + batch_size]],
                         device, max_len)
            for lo in range(0, len(sub), batch_size)
        ]
        self.n_examples = len(sub)
        self.n_matvec = 0

    def hvp(self, v_flat):
        """H v, averaged over the fixed curvature batches."""
        self.n_matvec += 1
        v = M.unflatten(v_flat, self.params)
        acc = None
        for b in self.batches:
            _, g = _grad(self.model, b, self.params, create_graph=True)
            gv = sum((gi * vi).sum() for gi, vi in zip(g, v))
            Hv = torch.autograd.grad(gv, self.params, retain_graph=False)
            f = M.flatten([h.detach() for h in Hv])
            acc = f if acc is None else acc + f
            del g, Hv, gv
            _release()          # 2nd-order graph is ~4GB at bs=2; without this
                                # the MPS cache grows to 12GB+ across a matvec
        return acc / len(self.batches)


# ---------------------------------------------------------- conjugate gradient
@dataclass
class CGResult:
    x: torch.Tensor
    iters: int
    rel_residual: float
    residuals: List[float]
    converged: bool


def cg_solve(curv: Curvature, b: torch.Tensor, damping: float,
             tol=1e-4, max_iter=50, verbose=False) -> CGResult:
    """Solve (H + damping*I) x = b by conjugate gradient.

    Damping serves two purposes: it is the lambda of the influence definition,
    and it makes the operator positive definite despite the negative curvature
    directions that real transformer losses have. We detect non-positive
    curvature explicitly rather than letting CG silently produce garbage.
    """
    A = lambda v: curv.hvp(v) + damping * v
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = r.dot(r)
    b_norm = b.norm().clamp(min=1e-30)
    res = [float((rs.sqrt() / b_norm))]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Ap = A(p)
        pAp = p.dot(Ap)
        if pAp <= 0:
            if verbose:
                print(f"    [cg] non-positive curvature pAp={float(pAp):.3e} "
                      f"at iter {it}; stopping (increase damping)")
            break
        alpha = rs / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r.dot(r)
        rel = float(rs_new.sqrt() / b_norm)
        res.append(rel)
        if verbose:
            print(f"    [cg] iter {it:3d} rel_res={rel:.3e}")
        if rel < tol:
            converged = True
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return CGResult(x=x, iters=it, rel_residual=res[-1], residuals=res,
                    converged=converged)


# ------------------------------------------------------------------- the field
def influence_field(model, tok, cf_pairs, factual_pairs, params, device,
                    damping=1e-2, cg_tol=1e-4, cg_max_iter=50,
                    curv_examples=24, curv_batch=2, cf_batch=2,
                    mode="ihvp", curv: Optional[Curvature] = None,
                    seed=0, verbose=False):
    """Return (v_flat, info).

    mode:
      'fisher' -> v = -(F+lambda I)^-1 delta_g  (the method; PSD, exact solve)
      'ihvp' -> v = -(H+lambda I)^-1 delta_g   (true Hessian + CG; indefinite,
                 kept as a diagnostic - see FisherCurvature docstring)
      'grad' -> v = -delta_g                   (control: does H^-1 matter?)
      'random' -> matched-norm random direction (control)
    """
    t0 = time.time()
    dg = delta_g_cf(model, tok, cf_pairs, params, device, batch_size=cf_batch)
    info = {"dg_norm": float(dg.norm()), "mode": mode}

    if mode == "grad":
        v = -dg
    elif mode == "random":
        gen = torch.Generator(device="cpu").manual_seed(seed)
        rv = torch.randn(dg.numel(), generator=gen).to(dg.device)
        v = rv / rv.norm() * dg.norm()          # matched norm to delta_g
    elif mode == "ggn":
        if curv is None:
            curv = GGNCurvature(model, tok, factual_pairs, params, device,
                                n_examples=curv_examples, batch_size=curv_batch,
                                seed=seed)
        r = cg_solve(curv, dg, damping, tol=cg_tol, max_iter=cg_max_iter,
                     verbose=verbose)
        v = -r.x
        info.update(cg_iters=r.iters, cg_rel_residual=r.rel_residual,
                    cg_converged=r.converged,
                    cos_v_vs_neg_dg=float(v.dot(-dg) /
                                          (v.norm() * dg.norm()).clamp(min=1e-30)))
    elif mode == "fisher":
        if curv is None:
            curv = FisherCurvature(model, tok, factual_pairs, params, device,
                                   n_examples=curv_examples, max_len=192,
                                   seed=seed)
        x, res = curv.solve_checked(dg, damping)
        v = -x
        info.update(fisher_rank=curv.n_examples, solve_rel_residual=res,
                    cos_v_vs_neg_dg=float((-x).dot(-dg) /
                                          ((x).norm() * dg.norm()).clamp(min=1e-30)))
    elif mode == "ihvp":
        if curv is None:
            curv = Curvature(model, tok, factual_pairs, params, device,
                             n_examples=curv_examples, batch_size=curv_batch,
                             seed=seed)
        r = cg_solve(curv, dg, damping, tol=cg_tol, max_iter=cg_max_iter,
                     verbose=verbose)
        v = -r.x
        info.update(cg_iters=r.iters, cg_rel_residual=r.rel_residual,
                    cg_converged=r.converged)
    else:
        raise ValueError(f"unknown mode {mode}")

    info.update(v_norm=float(v.norm()), seconds=time.time() - t0)
    return v.detach(), info


@torch.no_grad()
def apply_step(params, v_flat, eta):
    for p, dv in zip(params, M.unflatten(v_flat, params)):
        p.add_(eta * dv)


# ---------------------------------------------------- PSD curvature (Fisher)
class FisherCurvature:
    """Empirical-Fisher curvature with an EXACT damped inverse.

    Why not the true Hessian: measured at theta_S, the Rayleigh quotient of H
    along delta_g_CF is about -7.4e2, i.e. H is strongly indefinite there, so CG
    on (H + lambda I) terminates immediately on non-positive curvature. Making
    it positive definite would need lambda > |lambda_min| ~ 7e2, at which point
    (H + lambda I)^-1 -> I/lambda and the method degenerates into the
    gradient-only control. That is a real property of the loss surface at a
    non-converged theta_S, not a solver bug.

    The empirical Fisher F = (1/n) G^T G (rows of G are per-example gradients)
    is PSD by construction, and because it is rank <= n we can invert the damped
    operator in CLOSED FORM by Woodbury instead of iterating:

        (lambda I + (1/n) G^T G)^-1 b
              = (1/lambda) [ b - G^T (n lambda I_n + G G^T)^-1 G b ]

    So there is no convergence question: the solve is exact to float precision,
    and we verify the residual explicitly.

    Limitation, stated plainly: F has rank <= n_examples, so the curvature
    correction only acts inside an n-dimensional subspace and behaves like
    1/lambda on its complement. The empirical Fisher is also a surrogate for the
    Hessian, not equal to it. Both are standard in influence-function practice
    and both are reasons to treat the gradient-only control as essential.
    """

    def __init__(self, model, tok, pairs, params, device, n_examples=128,
                 max_len=192, seed=0, which="factual", verbose=False):
        self.model, self.params, self.device = model, params, device
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(pairs), generator=g)[:n_examples].tolist()
        self.pairs = [pairs[i] for i in idx]
        self.tok, self.max_len, self.which = tok, max_len, which
        self.n_examples = len(self.pairs)
        self.verbose = verbose
        self.G = None
        self.refresh()

    def refresh(self):
        """Recompute per-example gradients at the CURRENT parameters.

        G is kept on CPU in float64: the Woodbury form contains a 1/damping
        factor, so at small damping float32 loses all accuracy (measured
        relative residual 0.62 at damping=1e-4). G is only n x 138240, so
        float64 costs ~71MB at n=64 and the matvecs are trivial on CPU.
        """
        rows = []
        t0 = time.time()
        for j, p in enumerate(self.pairs):
            b = M.make_batch(self.tok, [D.to_messages(p, self.which)],
                             self.device, self.max_len)
            loss = per_example_loss(self.model, b)
            gr = torch.autograd.grad(loss, self.params)
            rows.append(M.flatten([x.detach() for x in gr]).cpu().double())
            del gr, loss, b
            if j % 16 == 15:
                _release()
        self.G = torch.stack(rows)          # (n, p) CPU float64
        _release()
        if self.verbose:
            print(f"    [fisher] G {tuple(self.G.shape)} f64/cpu in "
                  f"{time.time()-t0:.1f}s", flush=True)
        return self

    def matvec(self, v, damping=0.0):
        """(F + damping I) v. Computed in float64 on CPU."""
        n = self.G.shape[0]
        v64 = v.detach().cpu().double()
        out = self.G.t() @ (self.G @ v64) / n + damping * v64
        return out

    def solve(self, b, damping):
        """Exact (F + damping I)^-1 b via Woodbury, entirely in float64."""
        n = self.G.shape[0]
        b64 = b.detach().cpu().double()
        Gb = self.G @ b64
        A = self.G @ self.G.t() + (n * damping) * torch.eye(n, dtype=torch.float64)
        y = torch.linalg.solve(A, Gb)
        x64 = (b64 - self.G.t() @ y) / damping
        return x64.to(b.dtype).to(b.device)

    def solve_checked(self, b, damping):
        """Solve and return the float64 relative residual of the solve."""
        x = self.solve(b, damping)
        b64 = b.detach().cpu().double()
        r = float((self.matvec(x, damping) - b64).norm() / b64.norm().clamp(min=1e-30))
        return x, r


# --------------------------------------------- Gauss-Newton / true Fisher
def _supervised_logits(model, batch):
    """Logits at supervised positions only, with per-token loss weights.

    weights make sum_t w_t * nll_t identical to per_example_loss, i.e.
    mean over examples of the token-mean NLL, so the GGN below is the curvature
    of exactly the objective we train on.
    """
    dec = model.get_decoder()
    o = dec(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    h = (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0])[:, :-1]
    labels = batch["labels"][:, 1:]
    mask = labels != IGNORE
    bidx, _ = mask.nonzero(as_tuple=True)
    z = model.get_output_embeddings()(h[mask]).float()      # (N_sup, V)
    B = labels.shape[0]
    cnt = mask.sum(1).clamp(min=1).float()
    w = 1.0 / (B * cnt[bidx])                              # (N_sup,)
    return z, labels[mask], w


class GGNCurvature:
    """Gauss-Newton (== true Fisher for softmax cross-entropy) curvature.

    Motivation. Two things went wrong with the alternatives, both measured:
      * True Hessian: strongly indefinite at theta_S (Rayleigh along delta_g_CF
        ~ -7.4e2), so CG on (H + lambda I) stops on the first iteration.
      * Empirical Fisher: PSD and exactly invertible by Woodbury, but rank <= n
        (64) inside a 138240-dim space, so cos(v, -delta_g) was pinned at 0.935
        for every lambda - i.e. numerically almost the gradient-only control.

    GGN fixes both. For softmax CE it is PSD by construction (H_z = diag(p) - pp^T
    is PSD), and its rank is bounded by n_tokens * (V-1) rather than by n, so it
    is effectively full rank here and genuinely reshapes the direction.

    GGN v = sum_t w_t J_t^T (diag(p_t) - p_t p_t^T) J_t v, computed with one
    JVP (via the double-backward trick) plus one VJP - no explicit Jacobian.
    """

    def __init__(self, model, tok, pairs, params, device, n_examples=12,
                 batch_size=2, max_len=192, seed=0, which="factual"):
        self.model, self.params = model, params
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(pairs), generator=g)[:n_examples].tolist()
        sub = [pairs[i] for i in idx]
        self.batches = [
            M.make_batch(tok, [D.to_messages(p, which) for p in sub[lo:lo + batch_size]],
                         device, max_len)
            for lo in range(0, len(sub), batch_size)
        ]
        self.n_examples = len(sub)
        self.n_matvec = 0

    def _one(self, batch, v):
        z, lab, w = _supervised_logits(self.model, batch)
        p = torch.softmax(z, dim=-1).detach()

        # ---- JVP: Jv, via double backward on a dummy cotangent -------------
        dummy = torch.zeros_like(z, requires_grad=True)
        Jtw = torch.autograd.grad(z, self.params, grad_outputs=dummy,
                                  create_graph=True)
        pair = sum((a * b).sum() for a, b in zip(Jtw, v))
        Jv = torch.autograd.grad(pair, dummy, retain_graph=True)[0]

        # ---- apply the softmax-CE output Hessian, weighted ------------------
        u = p * Jv - p * (p * Jv).sum(dim=-1, keepdim=True)
        u = u * w.unsqueeze(-1)

        # ---- VJP: J^T u ----------------------------------------------------
        out = torch.autograd.grad(z, self.params, grad_outputs=u,
                                  retain_graph=False)
        return M.flatten([x.detach() for x in out])

    def hvp(self, v_flat):
        """GGN v, summed over the fixed curvature batches (weights already
        encode the per-example normalisation, so batches are summed not averaged
        when they partition the same example set)."""
        self.n_matvec += 1
        v = M.unflatten(v_flat, self.params)
        acc = None
        for b in self.batches:
            f = self._one(b, v)
            acc = f if acc is None else acc + f
            _release()
        return acc / len(self.batches)


def ggn_trace_estimate(curv, n_probe=3, seed=0, p_dim=None):
    """Hutchinson estimate of tr(GGN) and the implied mean eigenvalue.

    Damping only conditions the system if it is on the scale of the operator's
    eigenvalues. Measured here: mean eigenvalue ~4, so a lambda of 1e-3 (an
    obvious-looking default) is ~4000x too small and CG cannot converge. We
    therefore express damping RELATIVE to the mean eigenvalue.
    """
    g = torch.Generator().manual_seed(seed)
    p_dim = p_dim or sum(p.numel() for p in curv.params)
    est = []
    for i in range(n_probe):
        v = torch.randn(p_dim, generator=g).to(curv.params[0].device)
        est.append(float(v.dot(curv.hvp(v))) / float(v.dot(v)))
    mean_eig = sum(est) / len(est)
    return {"mean_eigenvalue": mean_eig, "trace_est": mean_eig * p_dim,
            "probes": est}
