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

import dataclasses
import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

from prophet.models import ModelInputData, TrendIndicator

logger = logging.getLogger('prophet.batch')

# Prior scales fixed in prophet.stan: k, m ~ normal(0, 5); sigma_obs ~ normal(0, 0.5).
_KM_PRIOR_VAR = 25.0
# Smallest sigma_obs; same as Prophet.fit's value for a constant history.
_MIN_SIGMA = 1e-9
# Series per multiplicative solve (each needs its full T × p design in memory).
_MULT_CHUNK = 256


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
    steps = 0
    for steps in range(1, max_iter + 1):
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
    return w, act, s, steps


def solve_map(G, C, yy, T, ridge, l1, tau, tol=1e-12, max_outer=200, max_inner=500,
              w0=None, sigma0=None, stats=None):
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

    w0, sigma0: Optional warm start (B × p, B). The active set and signs follow w0.
    stats: Optional dict; filled with 'outer' (sigma iterations) and 'inner' (total
        feature-sign steps).

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

    w = np.zeros((B, p)) if w0 is None else np.array(w0, dtype=float)
    act = ~l1 | (l1 & (w != 0))
    s = np.where(l1 & act, np.sign(w), 0.0)
    sigma2 = np.ones(B) if sigma0 is None else np.asarray(sigma0, dtype=float) ** 2
    live = np.arange(B)  # series whose sigma has not converged yet
    eye = np.eye(p)
    outer = inner = 0
    for outer in range(1, max_outer + 1):
        # The w-step, scaled by sigma²:
        #   0.5 ||y − Zw||² + sigma² (0.5 Σ ridge w² + Σ_l1 |w| / tau)
        s2 = sigma2[live]
        Gl = G[live] if per_series else np.broadcast_to(G, (live.size, p, p))
        H = Gl + (s2[:, None] * ridge[live])[:, :, None] * eye
        wl, al, sl, steps = _feature_sign(
            H, C[live], s2 / tau[live], l1[live], w[live], act[live], s[live], 1e-9, max_inner
        )
        inner += steps
        w[live], act[live], s[live] = wl, al, sl
        wGw = np.einsum("bp,bpq,bq->b", wl, Gl, wl)
        rss = np.maximum(yy[live] - 2 * np.einsum("bp,bp->b", C[live], wl) + wGw, 0.0)
        # d/dsigma of the log posterior is zero at 4 sigma⁴ + T sigma² − RSS = 0
        Tl = T[live]
        # A perfect fit (RSS = 0) makes the posterior unbounded as sigma -> 0; stop at the
        # sigma_obs Prophet.fit uses for a constant history.
        s2_new = np.maximum((-Tl + np.sqrt(Tl * Tl + 16 * rss)) / 8, _MIN_SIGMA ** 2)
        converged = np.abs(s2_new - s2) / s2 < tol
        sigma2[live] = s2_new
        live = live[~converged]
        if live.size == 0:
            break
    if stats is not None:
        stats.update(outer=outer, inner=inner)
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
    if model.growth not in ('linear', 'flat'):
        return f'growth={model.growth!r} (only linear and flat are supported)'
    if model.mcmc_samples > 0:
        return 'mcmc_samples > 0 (only MAP is supported)'
    for name, props in model.extra_regressors.items():
        if props.get('predictor_spec') is not None:
            return f'regressor {name!r} has a regressor_predictor'
    return None


def has_multiplicative_terms(model_inputs) -> bool:
    """True if any feature is multiplicative (then yhat is not linear in the parameters)."""
    return bool(np.any(np.asarray(model_inputs.s_m) != 0))


def design_matrix(model_inputs) -> np.ndarray:
    """Z such that the fitted values (scaled) are Z @ [k, m, delta, beta].

    Valid for linear or flat growth with additive components only. For flat growth the
    trend is just m, so the k and delta columns are zero (they only feel their priors,
    and solve to exactly 0).
    """
    t = np.asarray(model_inputs.t, dtype=float)
    t_change = np.asarray(model_inputs.t_change, dtype=float)
    X = np.asarray(model_inputs.X, dtype=float) * np.asarray(model_inputs.s_a, dtype=float)
    hinge = (t[:, None] >= t_change[None, :]) * (t[:, None] - t_change[None, :])
    Z = np.column_stack([t, np.ones_like(t), hinge, X])
    if model_inputs.trend_indicator == TrendIndicator.FLAT.value:
        Z[:, 0] = 0.0
        Z[:, 2:2 + len(t_change)] = 0.0
    elif model_inputs.trend_indicator != TrendIndicator.LINEAR.value:
        raise ValueError('only linear and flat growth have a linear design matrix')
    return Z


class _Layout:
    """Common coordinate layout for a batch: [k, m, delta (S_max), beta (K_max)].

    Each series uses the first S_b delta slots and the first K_b beta slots; unused
    slots are padding that solves to exactly 0.
    """

    def __init__(self, S: list[int], K: list[int]):
        self.S = np.asarray(S)
        self.K = np.asarray(K)
        self.S_max = int(self.S.max())
        self.K_max = int(self.K.max())
        self.p = 2 + self.S_max + self.K_max

    def index(self, b: int) -> np.ndarray:
        """Layout positions of series b's own coordinates, in its natural order."""
        S, K = self.S[b], self.K[b]
        return np.concatenate([[0, 1], 2 + np.arange(S), 2 + self.S_max + np.arange(K)])

    def unpack(self, w_b: np.ndarray, b: int) -> tuple[float, float, np.ndarray, np.ndarray]:
        """(k, m, delta, beta) for series b from its row of the batch solution."""
        S, K = self.S[b], self.K[b]
        return (float(w_b[0]), float(w_b[1]), w_b[2:2 + S].copy(),
                w_b[2 + self.S_max:2 + self.S_max + K].copy())


def stack_problems(problems):
    """Stack per-series problems into padded batch arrays for `solve_map`.

    problems: list of (Z, y, sigmas, tau), where Z comes from `design_matrix`.
    Returns (layout, G, C, yy, T, ridge, l1, tau).
    """
    S = [Z.shape[1] - 2 - len(sigmas) for Z, _, sigmas, _ in problems]
    K = [len(sigmas) for _, _, sigmas, _ in problems]
    layout = _Layout(S, K)
    B, p = len(problems), layout.p
    G = np.zeros((B, p, p))
    C = np.zeros((B, p))
    yy = np.zeros(B)
    T = np.zeros(B)
    ridge = np.ones((B, p))  # padding slots get a positive ridge
    l1 = np.zeros((B, p), bool)
    tau = np.zeros(B)
    gram_cache = {}  # series sharing one design matrix object share its Gram matrix
    for b, (Z, y, sigmas, tau_b) in enumerate(problems):
        idx = layout.index(b)
        y = np.asarray(y, dtype=float)
        if id(Z) not in gram_cache:
            gram_cache[id(Z)] = Z.T @ Z
        G[b] = np.eye(p)
        G[b][np.ix_(idx, idx)] = gram_cache[id(Z)]
        C[b, idx] = Z.T @ y
        yy[b] = y @ y
        T[b] = len(y)
        S_b = layout.S[b]
        ridge[b, idx] = np.concatenate([
            [1 / _KM_PRIOR_VAR, 1 / _KM_PRIOR_VAR], np.zeros(S_b),
            1 / np.asarray(sigmas, dtype=float) ** 2,
        ])
        l1[b, idx[2:2 + S_b]] = True
        tau[b] = tau_b
    return layout, G, C, yy, T, ridge, l1, tau


def log_posterior(G, C, yy, T, w, sigma, ridge, l1, tau) -> np.ndarray:
    """Log posterior per series, with the same constants dropped as Stan's lp__."""
    Gw = np.einsum("bpq,bq->bp", G, w) if G.ndim == 3 else w @ G
    rss = yy - 2 * np.einsum("bp,bp->b", C, w) + np.einsum("bp,bp->b", w, Gw)
    return (-0.5 * np.sum(ridge * w ** 2, axis=1)
            - np.sum(np.abs(w) * l1, axis=1) / tau
            - sigma ** 2 / (2 * 0.5 ** 2)
            - T * np.log(sigma) - rss / (2 * sigma ** 2))


def set_params(model, k, m, delta, beta, sigma_obs, lp) -> None:
    """Store a MAP solution on a preprocessed model in the same form as Prophet.fit."""
    delta = np.asarray(delta, dtype=float)
    # Stan also returns the fitted (scaled) trend over the history, a transformed parameter.
    t = model.history['t'].to_numpy(dtype=float)
    if model.growth == 'flat':
        trend = np.full(len(t), float(m))
    else:
        t_change = np.asarray(model.changepoints_t, dtype=float)
        A = (t[:, None] >= t_change[None, :]).astype(float)
        trend = (k + A @ delta) * t + (m + A @ (-t_change * delta))
    model.params = OrderedDict([
        ('lp__', np.array([[lp]], dtype=float)),
        ('k', np.array([[k]], dtype=float)),
        ('m', np.array([[m]], dtype=float)),
        ('delta', delta.reshape((1, -1))),
        ('sigma_obs', np.array([[sigma_obs]], dtype=float)),
        ('beta', np.asarray(beta, dtype=float).reshape((1, -1))),
        ('trend', trend.reshape((1, -1))),
    ])
    model.stan_fit = None
    _fold_dummy_changepoint(model)


def set_constant_params(model, num_regressors: int) -> None:
    """Prophet.fit's rule for a constant history: initial params, sigma_obs = 1e-9."""
    params = dataclasses.asdict(model.calculate_initial_params(num_regressors))
    params['sigma_obs'] = 1e-9
    model.params = {k: np.array(v).reshape((1, -1)) for k, v in params.items()}
    model.stan_fit = None
    _fold_dummy_changepoint(model)


def is_constant(model) -> bool:
    y = model.history['y']
    return bool(y.min() == y.max())


def _fold_dummy_changepoint(model) -> None:
    # Same as Prophet.fit: with no changepoints, delta (for the dummy one at t=0) goes into k.
    if len(model.changepoints) == 0:
        model.params['k'] = model.params['k'] + model.params['delta'].reshape(-1)
        model.params['delta'] = np.zeros(model.params['delta'].shape).reshape((-1, 1))


@dataclasses.dataclass
class _SharedFeatures:
    """Date-only preprocessing results, reusable by models with the same configuration
    and the same history dates."""
    X: object               # seasonal features DataFrame
    sigmas: list
    component_cols: object  # DataFrame
    modes: dict
    train_holiday_names: object
    Z: np.ndarray           # design matrix


def _history_key(df) -> bytes:
    """Identifies the history dates (rows with non-null y), in sorted order."""
    ds = pd.to_datetime(df.loc[df['y'].notnull(), 'ds'])
    return np.sort(ds.to_numpy(dtype='datetime64[ns]')).tobytes()


def _preprocess_shared(model, df, shared: _SharedFeatures) -> ModelInputData:
    """Same result as model.preprocess(df), reusing date-only features from `shared`.

    Mirrors Prophet.preprocess step by step; only make_all_seasonality_features is
    replaced by the shared result. The per-series steps use Prophet's own methods.
    """
    history = df[df['y'].notnull()].copy()
    if history.shape[0] < 2:
        raise ValueError('Dataframe has less than 2 non-NaN rows.')
    model.history_dates = pd.to_datetime(pd.Series(df['ds'].unique(), name='ds')).sort_values()
    model.history = model.setup_dataframe(history, initialize_scales=True)
    model.set_auto_seasonalities()
    model.train_component_cols = shared.component_cols.copy()
    model.component_modes = deepcopy(shared.modes)
    if shared.train_holiday_names is not None:
        model.train_holiday_names = shared.train_holiday_names.copy()
    model.fit_kwargs = {}
    model.set_changepoints()
    cap = np.zeros(model.history.shape[0])
    return ModelInputData(
        T=model.history.shape[0],
        S=len(model.changepoints_t),
        K=shared.X.shape[1],
        tau=model.changepoint_prior_scale,
        trend_indicator=TrendIndicator[model.growth.upper()].value,
        y=model.history['y_scaled'],
        t=model.history['t'],
        t_change=model.changepoints_t,
        X=shared.X,
        sigmas=shared.sigmas,
        s_a=shared.component_cols['additive_terms'],
        s_m=shared.component_cols['multiplicative_terms'],
        cap=cap,
    )


def _prepare(df, id_col, make_model, share_dates):
    """Create and preprocess one model per series and sort them into batch / constant /
    fallback. With share_dates, series with identical history dates reuse date-only work.

    Returns (order, batch, mult, constant, fallback): batch is a list of (id, model, inputs, Z)
    for purely additive models, mult a list of (id, model, inputs) for models with
    multiplicative features, constant a list of (id, model), fallback a list of
    (id, series df, reason).
    """
    order, batch, mult, constant, fallback = [], [], [], [], []
    shared_by_dates = {}
    for sid, group in df.groupby(id_col, sort=False):
        order.append(sid)
        series_df = group.drop(columns=id_col).reset_index(drop=True)
        model = make_model(sid)
        reason = unsupported_reason(model)
        if reason is not None:
            fallback.append((sid, series_df, reason))
            continue
        shared = None
        if share_dates:
            key = _history_key(series_df)
            shared = shared_by_dates.get(key)
        if shared is None:
            inputs = model.preprocess(series_df)
            Z = None if has_multiplicative_terms(inputs) else design_matrix(inputs)
            if share_dates:
                shared_by_dates[key] = _SharedFeatures(
                    X=inputs.X, sigmas=inputs.sigmas, component_cols=model.train_component_cols,
                    modes=model.component_modes, train_holiday_names=model.train_holiday_names, Z=Z,
                )
        else:
            inputs = _preprocess_shared(model, series_df, shared)
            Z = shared.Z
        if is_constant(model):
            set_constant_params(model, inputs.K)
            constant.append((sid, model))
        elif has_multiplicative_terms(inputs):
            mult.append((sid, model, inputs))
        else:
            batch.append((sid, model, inputs, Z))
    return order, batch, mult, constant, fallback


def fit_many(df, id_col='unique_id', make_model=None, **prophet_kwargs):
    """Fit one Prophet model per series, solving all supported series together.

    Parameters
    ----------
    df: Long-format DataFrame with columns `id_col`, ds, y (and any regressor columns).
    id_col: Column identifying the series.
    make_model: Optional callable `make_model(series_id)` returning a new, unfitted,
        configured Prophet for that series (for holidays, extra regressors, custom
        seasonalities). It must return a fresh object on every call; it may be called
        more than once per series. Defaults to `Prophet(**prophet_kwargs)`.
    prophet_kwargs: Passed to Prophet() when make_model is not given.

    Returns
    -------
    Dict mapping each series id to its fitted Prophet model, in order of first appearance.

    Series with linear or flat growth are fit by batched MAP solvers (no Stan): exactly
    (a convex problem) when all components are additive, and to a stationary point
    (hybrid alternating / Gauss-Newton) when some are multiplicative. Logistic growth,
    MCMC and regressor predictors fall back to Prophet.fit (Stan), with exactly the same
    result as calling it directly. The batched solution is the exact
    optimum of the same posterior; it typically has a slightly higher log posterior than
    Stan's L-BFGS, which stops early, so fitted values can differ slightly from fit().

    Without make_model, all models share one configuration, so series with identical
    history dates also share their date-only preprocessing (seasonality features and
    design matrix); the resulting models are identical to individually preprocessed ones.
    """
    from prophet.forecaster import Prophet

    share_dates = make_model is None
    if make_model is None:
        def make_model(series_id):
            return Prophet(**prophet_kwargs)
    elif prophet_kwargs:
        raise ValueError('Pass either make_model or Prophet keyword arguments, not both.')

    order, batch, mult, constant, fallback = _prepare(df, id_col, make_model, share_dates)
    models = {sid: model for sid, model in constant}

    if batch:
        problems = [(Z, inputs.y, inputs.sigmas, inputs.tau) for _, _, inputs, Z in batch]
        layout, G, C, yy, T, ridge, l1, tau = stack_problems(problems)
        w, sigma = solve_map(G, C, yy, T, ridge, l1, tau)
        lp = log_posterior(G, C, yy, T, w, sigma, ridge, l1, tau)
        for b, (sid, model, _, _) in enumerate(batch):
            k, m, delta, beta = layout.unpack(w[b], b)
            set_params(model, k, m, delta, beta, sigma[b], lp[b])
            models[sid] = model

    # Multiplicative models: the design changes every iteration, so the solver needs each
    # series' full data; chunks keep memory bounded.
    for start in range(0, len(mult), _MULT_CHUNK):
        chunk = mult[start:start + _MULT_CHUNK]
        mb = _MultiplicativeBatch([inputs for _, _, inputs in chunk])
        theta, ba, bm, sigma, lp, _ = mb.solve()
        for b, (sid, model, _) in enumerate(chunk):
            k, m, delta, beta = mb.unpack(b, theta, ba, bm)
            set_params(model, k, m, delta, beta, sigma[b], lp[b])
            models[sid] = model

    for sid, series_df, reason in fallback:
        logger.debug('Series %r falls back to Prophet.fit: %s', sid, reason)
        models[sid] = make_model(sid).fit(series_df)

    logger.info(
        'fit_many: %d series batched (%d additive, %d multiplicative), %d constant, '
        '%d fell back to Prophet.fit',
        len(batch) + len(mult), len(batch), len(mult), len(constant), len(fallback),
    )
    return {sid: models[sid] for sid in order}


def _gram(Z, y):
    """Batched Zᵀ Z, Zᵀ y, yᵀ y for Z: B × T × p, y: B × T."""
    G = np.matmul(Z.transpose(0, 2, 1), Z)
    C = np.matmul(Z.transpose(0, 2, 1), y[:, :, None])[..., 0]
    return G, C, np.einsum('bt,bt->b', y, y)


def _multiplicative_log_posterior(ZT, Xa, Xm, y, T, theta, ba, bm, sigma, S, ridge_a, ridge_m, tau):
    trend = np.matmul(ZT, theta[:, :, None])[..., 0]
    yhat = (trend * (1 + np.matmul(Xm, bm[:, :, None])[..., 0])
            + np.matmul(Xa, ba[:, :, None])[..., 0])
    rss = np.sum((y - yhat) ** 2, axis=1)
    delta = theta[:, 2:]
    return (-(theta[:, 0] ** 2 + theta[:, 1] ** 2) / (2 * _KM_PRIOR_VAR)
            - np.sum(np.abs(delta) * S, axis=1) / tau
            - 0.5 * np.sum(ridge_a * ba ** 2, axis=1) - 0.5 * np.sum(ridge_m * bm ** 2, axis=1)
            - sigma ** 2 / (2 * 0.5 ** 2) - T * np.log(sigma) - rss / (2 * sigma ** 2))


def solve_map_multiplicative(ZT, Xa, Xm, y, T, delta_mask, ridge_a, ridge_m, tau,
                             tol=0.0, max_iter=100, method='hybrid', switch_after=10):
    """MAP for models with multiplicative features.

        yhat = (ZT θ) ∘ (1 + Xm β_m) + Xa β_a,   θ = [k, m, δ]

    method='hybrid' (default): `switch_after` alternating iterations (cheap; most series
    converge there), then Gauss-Newton for the rest.
    method='gauss-newton': each iteration linearizes yhat in all parameters at
    the current point, solves the resulting lasso subproblem exactly with `solve_map`
    (warm-started), and backtracks along the step until the true log posterior does not
    decrease. method='alternating': exact block solves, (θ, β_a) given β_m, then
    (β_m, β_a) given θ; also monotone, but can zig-zag for hundreds of iterations.
    The joint problem is non-convex (bilinear), so the result is a stationary point.
    Stops per series when lp changes by no more than `tol` (relative). With the default
    tol=0 that is when a step no longer improves lp (lp never decreases), which gives
    KKT violations around 1e-9 in about 8 Gauss-Newton iterations.

    Arrays are zero-padded to common sizes: ZT: B × T × (2 + S), Xa: B × T × Ka,
    Xm: B × T × Km, y: B × T (padded rows all zero); T: true lengths (B);
    delta_mask: B × S, True for real changepoints; ridge_a: B × Ka, ridge_m: B × Km
    (1 / prior variance, positive on padding); tau: B.

    Returns theta (B × (2 + S)), beta_a, beta_m, sigma, lp, and the number of iterations.
    """
    B, _, P = ZT.shape
    Ka, Km = Xa.shape[2], Xm.shape[2]
    ridge_T = np.concatenate([np.full((B, 2), 1 / _KM_PRIOR_VAR), np.where(delta_mask, 0.0, 1.0)], axis=1)
    l1_T = np.concatenate([np.zeros((B, 2), bool), delta_mask], axis=1)
    ridge_J = np.concatenate([ridge_T, ridge_a, ridge_m], axis=1)
    l1_J = np.concatenate([l1_T, np.zeros((B, Ka + Km), bool)], axis=1)

    theta = np.zeros((B, P))
    ba = np.zeros((B, Ka))
    bm = np.zeros((B, Km))
    sigma = np.ones(B)
    args = (ZT, Xa, Xm, y, T)

    def lp_at(L, th, a, m, sg):
        return _multiplicative_log_posterior(
            ZT[L], Xa[L], Xm[L], y[L], T[L], th, a, m, sg, delta_mask[L],
            ridge_a[L], ridge_m[L], tau[L])

    lp = np.full(B, -np.inf)
    live = np.arange(B)
    it = 0
    for it in range(1, max_iter + 1):
        L = live
        if method == 'alternating' or (method == 'hybrid' and it <= switch_after):
            wgt = 1 + np.matmul(Xm[L], bm[L][:, :, None])[..., 0]
            ZA = np.concatenate([wgt[:, :, None] * ZT[L], Xa[L]], axis=2)
            G, C, yy = _gram(ZA, y[L])
            wA, sigma[L] = solve_map(G, C, yy, T[L], np.concatenate([ridge_T[L], ridge_a[L]], axis=1),
                                     np.concatenate([l1_T[L], np.zeros((len(L), Ka), bool)], axis=1),
                                     tau[L], w0=np.concatenate([theta[L], ba[L]], axis=1), sigma0=sigma[L])
            theta[L], ba[L] = wA[:, :P], wA[:, P:]
            trend = np.matmul(ZT[L], theta[L][:, :, None])[..., 0]
            ZB = np.concatenate([trend[:, :, None] * Xm[L], Xa[L]], axis=2)
            G, C, yy = _gram(ZB, y[L] - trend)
            wB, sigma[L] = solve_map(G, C, yy, T[L], np.concatenate([ridge_m[L], ridge_a[L]], axis=1),
                                     np.zeros((len(L), Km + Ka), bool), tau[L],
                                     w0=np.concatenate([bm[L], ba[L]], axis=1), sigma0=sigma[L])
            bm[L], ba[L] = wB[:, :Km], wB[:, Km:]
            lp_new = lp_at(L, theta[L], ba[L], bm[L], sigma[L])
        else:
            # Linearize yhat around p = [θ, β_a, β_m]: yhat(p') ≈ yhat(p) + J (p' − p).
            trend = np.matmul(ZT[L], theta[L][:, :, None])[..., 0]
            mult = 1 + np.matmul(Xm[L], bm[L][:, :, None])[..., 0]
            J = np.concatenate([mult[:, :, None] * ZT[L], Xa[L], trend[:, :, None] * Xm[L]], axis=2)
            p_cur = np.concatenate([theta[L], ba[L], bm[L]], axis=1)
            yhat = trend * mult + np.matmul(Xa[L], ba[L][:, :, None])[..., 0]
            y_lin = y[L] - yhat + np.matmul(J, p_cur[:, :, None])[..., 0]
            G, C, yy = _gram(J, y_lin)
            p_new, _ = solve_map(G, C, yy, T[L], ridge_J[L], l1_J[L], tau[L], w0=p_cur, sigma0=sigma[L])
            # Backtrack until the true log posterior doesn't decrease; sigma is set to its
            # closed-form optimum at each trial point.
            step = np.ones(len(L))
            lp_cur = lp[L] if it > 1 else np.full(len(L), -np.inf)
            accepted = np.zeros(len(L), bool)
            p_acc, s_acc, lp_acc = p_cur.copy(), sigma[L].copy(), lp_cur.copy()
            for _ in range(30):
                trial = p_cur + step[:, None] * (p_new - p_cur)
                th, a, m = trial[:, :P], trial[:, P:P + Ka], trial[:, P + Ka:]
                tr = np.matmul(ZT[L], th[:, :, None])[..., 0]
                yh = tr * (1 + np.matmul(Xm[L], m[:, :, None])[..., 0]) + np.matmul(Xa[L], a[:, :, None])[..., 0]
                rss = np.sum((y[L] - yh) ** 2, axis=1)
                sg = np.sqrt(np.maximum((-T[L] + np.sqrt(T[L] ** 2 + 16 * rss)) / 8, _MIN_SIGMA ** 2))
                lp_try = lp_at(L, th, a, m, sg)
                ok = ~accepted & (lp_try >= lp_cur)
                p_acc[ok], s_acc[ok], lp_acc[ok] = trial[ok], sg[ok], lp_try[ok]
                accepted |= ok
                if accepted.all():
                    break
                step = np.where(accepted, step, step / 2)
            theta[L], ba[L], bm[L] = p_acc[:, :P], p_acc[:, P:P + Ka], p_acc[:, P + Ka:]
            sigma[L] = s_acc
            lp_new = lp_acc
        done = np.abs(lp_new - lp[L]) <= tol * np.abs(lp_new)
        lp[L] = lp_new
        live = L[~done]
        if live.size == 0:
            break
    return theta, ba, bm, sigma, lp, it


class _MultiplicativeBatch:
    """Zero-padded arrays for `solve_map_multiplicative`, built from model inputs."""

    def __init__(self, inputs_list):
        B = len(inputs_list)
        parts = []
        for inputs in inputs_list:
            Z = design_matrix(_as_additive(inputs))  # trend columns are what we need here
            S = len(np.atleast_1d(inputs.t_change))
            X = np.asarray(inputs.X, dtype=float)
            s_m = np.asarray(inputs.s_m, dtype=float) != 0
            sig = np.asarray(inputs.sigmas, dtype=float)
            parts.append(dict(ZT=Z[:, :2 + S], Xa=X[:, ~s_m], Xm=X[:, s_m], y=np.asarray(inputs.y, float),
                              ia=np.flatnonzero(~s_m), im=np.flatnonzero(s_m), sig=sig, S=S,
                              K=len(sig), tau=float(inputs.tau)))
        self.parts = parts
        Tm = max(len(p['y']) for p in parts)
        Sm = max(p['S'] for p in parts)
        Ka = max(p['Xa'].shape[1] for p in parts)
        Km = max(p['Xm'].shape[1] for p in parts)
        self.ZT = np.zeros((B, Tm, 2 + Sm))
        self.Xa = np.zeros((B, Tm, Ka))
        self.Xm = np.zeros((B, Tm, Km))
        self.y = np.zeros((B, Tm))
        self.T = np.zeros(B)
        self.delta_mask = np.zeros((B, Sm), bool)
        self.ridge_a = np.ones((B, Ka))
        self.ridge_m = np.ones((B, Km))
        self.tau = np.zeros(B)
        for b, p in enumerate(parts):
            T, S = len(p['y']), p['S']
            self.ZT[b, :T, :2 + S] = p['ZT']
            self.Xa[b, :T, :p['Xa'].shape[1]] = p['Xa']
            self.Xm[b, :T, :p['Xm'].shape[1]] = p['Xm']
            self.y[b, :T] = p['y']
            self.T[b] = T
            self.delta_mask[b, :S] = True
            self.ridge_a[b, :len(p['ia'])] = 1 / p['sig'][p['ia']] ** 2
            self.ridge_m[b, :len(p['im'])] = 1 / p['sig'][p['im']] ** 2
            self.tau[b] = p['tau']

    def solve(self, **kw):
        return solve_map_multiplicative(self.ZT, self.Xa, self.Xm, self.y, self.T, self.delta_mask,
                                        self.ridge_a, self.ridge_m, self.tau, **kw)

    def unpack(self, b, theta, ba, bm):
        """(k, m, delta, beta) for series b, with beta in the model's feature order."""
        p = self.parts[b]
        beta = np.zeros(p['K'])
        beta[p['ia']] = ba[b, :len(p['ia'])]
        beta[p['im']] = bm[b, :len(p['im'])]
        return float(theta[b, 0]), float(theta[b, 1]), theta[b, 2:2 + p['S']].copy(), beta


def _as_additive(inputs):
    """Inputs with every feature marked additive (for building the trend columns of Z)."""
    return dataclasses.replace(inputs, s_a=np.ones(len(np.asarray(inputs.s_a))),
                               s_m=np.zeros(len(np.asarray(inputs.s_m))))
