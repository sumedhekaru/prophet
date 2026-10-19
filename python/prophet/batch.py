# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Batched MAP fitting for many Prophet models at once.

For linear growth with only additive components, the fitted values are linear in
w = [k, m, delta, beta]:

    yhat = Z @ w,  Z = [t, 1, A_j * (t - t_change_j) for each changepoint j, X]

so for a fixed sigma_obs the MAP is a convex problem (least squares with ridge
penalties from the normal priors and an L1 penalty from the Laplace prior on delta).
For a fixed w, the optimal sigma_obs has a closed form. We alternate the two; the
w-step is an exact active-set method (feature-sign search), batched over series.
Only the sufficient statistics Zᵀ Z, Zᵀ y, yᵀ y and T are needed per series.
"""

from __future__ import annotations

import numpy as np

# Prior scales fixed in prophet.stan: k, m ~ normal(0, 5); sigma_obs ~ normal(0, 0.5).
_KM_PRIOR_VAR = 25.0


def _feature_sign(H, c, lam, l1, w, act, s, tol, max_iter):
    """Solve min 0.5 wᵀHw − cᵀw + lam Σ_{j in l1} |w_j|, batched over series.

    Feature-sign search (Lee et al., 2007). H: B × p × p, c: B × p, lam: B, l1: B × p.
    Warm-started from w, act (active mask) and s (signs). Non-L1 coordinates are always
    active. A new L1 coordinate is only added once the active coordinates are optimal,
    which guarantees the objective decreases at every step.
    """
    B, p = c.shape
    eye = np.eye(p)
    rows = np.arange(B)
    done = np.zeros(B, bool)
    need_step = np.ones(B, bool)
    for _ in range(max_iter):
        g = np.einsum("bij,bj->bi", H, w) - c
        viol = l1 & ~act & (np.abs(g) > lam[:, None] * (1 + tol))
        ready = ~need_step & ~done
        add = ready & viol.any(axis=1)
        done |= ready & ~viol.any(axis=1)
        if done.all():
            break
        worst = np.argmax(np.where(viol, np.abs(g), -np.inf), axis=1)
        s[rows[add], worst[add]] = -np.sign(g[rows[add], worst[add]])
        act[rows[add], worst[add]] = True
        need_step |= add
        stepping = need_step & ~done

        # Exact minimizer of the sign-fixed quadratic on the active set.
        mask = act[:, :, None] & act[:, None, :]
        Hm = np.where(mask, H, 0.0) + eye * (~act)[:, None, :]
        rhs = np.where(act, c - lam[:, None] * s, 0.0)
        w_new = np.linalg.solve(Hm, rhs[..., None])[..., 0]
        dw = w_new - w

        # Discrete line search on the true objective: the full step, or a point where a
        # nonzero L1 coordinate changes sign.
        crossing = l1 & (w != 0) & (w * w_new < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_cross = np.where(crossing, w / (w - w_new), np.inf)
        cand = np.concatenate([t_cross, np.ones((B, 1))], axis=1)
        a = np.einsum("bp,bp->b", g, dw)
        b = np.einsum("bp,bpq,bq->b", dw, H, dw)
        finite = np.isfinite(cand)
        tc = np.where(finite, cand, 0.0)
        pts = w[:, None, :] + tc[:, :, None] * dw[:, None, :]
        phi = (a[:, None] * tc + 0.5 * b[:, None] * tc ** 2
               + lam[:, None] * np.abs(pts * l1[:, None, :]).sum(axis=2))
        phi = np.where(finite, phi, np.inf)
        k = np.argmin(phi, axis=1)
        t = cand[rows, k]
        w_next = w + t[:, None] * dw
        at_zero = k < p
        w_next[rows[at_zero], k[at_zero]] = 0.0
        w = np.where(stepping[:, None], w_next, w)

        # Zero L1 coefficients leave the active set; signs follow the coefficients.
        act = act & ~(l1 & (w == 0))
        s = np.where(l1 & act, np.sign(w), 0.0)
        need_step = need_step & ~(stepping & (t >= 1.0))
    return w, act, s


def solve_map(G, C, yy, T, ridge, l1, tau, tol=1e-12, max_outer=200, max_inner=500):
    """MAP estimate from sufficient statistics, batched over B series.

    Parameters
    ----------
    G: Zᵀ Z, shape p × p (shared by all series) or B × p × p.
    C: Zᵀ y, shape B × p.
    yy: yᵀ y, shape B.
    T: number of observations, shape B.
    ridge: 1 / prior variance for each coordinate (0 for L1 coordinates), shape B × p.
    l1: True for coordinates with the Laplace prior (delta), shape B × p.
    tau: Laplace scale (changepoint_prior_scale), shape B.

    Padding coordinates (unused by a series) should have G_jj = 1, C_j = 0,
    ridge_j > 0 and l1_j = False; they solve to exactly 0.

    Returns
    -------
    w (B × p) and sigma_obs (B).
    """
    C = np.asarray(C, dtype=float)
    B, p = C.shape
    per_series = G.ndim == 3
    yy, T, tau = (np.asarray(x, dtype=float) for x in (yy, T, tau))
    ridge = np.asarray(ridge, dtype=float)
    l1 = np.asarray(l1, dtype=bool)

    w = np.zeros((B, p))
    act = ~l1
    s = np.zeros((B, p))
    sigma2 = np.ones(B)
    live = np.arange(B)  # series whose sigma has not converged yet
    eye = np.eye(p)
    for _ in range(max_outer):
        # The w-step, scaled by sigma²:
        #   0.5 ||y − Zw||² + sigma² (0.5 Σ ridge w² + Σ_l1 |w| / tau)
        s2 = sigma2[live]
        Gl = G[live] if per_series else np.broadcast_to(G, (live.size, p, p))
        H = Gl + (s2[:, None] * ridge[live])[:, :, None] * eye
        wl, al, sl = _feature_sign(
            H, C[live], s2 / tau[live], l1[live], w[live], act[live], s[live], 1e-9, max_inner
        )
        w[live], act[live], s[live] = wl, al, sl
        wGw = np.einsum("bp,bpq,bq->b", wl, Gl, wl)
        rss = np.maximum(yy[live] - 2 * np.einsum("bp,bp->b", C[live], wl) + wGw, 0.0)
        # d/dsigma of the log posterior is zero at 4 sigma⁴ + T sigma² − RSS = 0
        Tl = T[live]
        s2_new = np.maximum((-Tl + np.sqrt(Tl * Tl + 16 * rss)) / 8, 1e-300)
        converged = np.abs(s2_new - s2) / s2 < tol
        sigma2[live] = s2_new
        live = live[~converged]
        if live.size == 0:
            break
    return w, np.sqrt(sigma2)


def kkt_violation(G, C, w, sigma, ridge, l1, tau):
    """Largest violation of the optimality conditions for w (given sigma), per series,
    in units of the L1 threshold 1 / tau. Zero at an exact optimum."""
    C = np.asarray(C, dtype=float)
    Gw = np.einsum("bpq,bq->bp", G, w) if G.ndim == 3 else w @ G
    g = (Gw - C) / sigma[:, None] ** 2 + ridge * w
    inv_tau = 1 / np.asarray(tau, dtype=float)[:, None]
    viol = np.where(
        l1,
        np.where(w != 0, np.abs(g + np.sign(w) * inv_tau), np.maximum(np.abs(g) - inv_tau, 0.0)),
        np.abs(g),
    )
    return viol.max(axis=1) / inv_tau[:, 0]


def unsupported_reason(model) -> str | None:
    """Why this (unfitted) model can't use the batched solver, or None if it can.

    Checks the configuration only. Whether every component is additive is known only
    after preprocessing; see `has_multiplicative_terms`.
    """
    if model.history is not None:
        return 'model is already fitted'
    if model.growth != 'linear':
        return f'growth={model.growth!r} (only linear is supported)'
    if model.mcmc_samples > 0:
        return 'mcmc_samples > 0 (only MAP is supported)'
    for name, props in model.extra_regressors.items():
        if props.get('predictor_spec') is not None:
            return f'regressor {name!r} has a regressor_predictor'
    return None


def has_multiplicative_terms(model_inputs) -> bool:
    """True if any feature is multiplicative (then yhat is not linear in the parameters)."""
    return bool(np.any(np.asarray(model_inputs.s_m) != 0))
