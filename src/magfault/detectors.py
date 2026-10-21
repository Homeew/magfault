"""Detectors.

Statistical (thresholds set from a false-alarm probability, not tuned by eye):

* ScalarGLRT   windowed GLRT for a mean shift and a variance change in the
               AR(1)-prewhitened vector-scalar residual dF.  chi2(1) each.
* ScalarCUSUM  two-sided CUSUM on the same whitened residual; h from the
               Siegmund ARL approximation for a target in-control run length.
* NetworkGLRT  each station's disturbance (dx, dy, dz) is predicted from the
               other stations by least squares; the 3-D residual is whitened
               (Cholesky of its covariance, then AR(1) per component) and tested
               with a windowed GLRT, chi2(3).  When several stations alarm at
               once, the alarm is attributed to the one with the largest
               statistic (leave-one-out isolation).

Baselines for comparison:

* NaiveZ           single-station |disturbance| z-score threshold
* IsolationForest  unsupervised, on the SQL feature vector

Classifier combines them into normal / sensor fault / geomagnetic event using the
physics: dF is insensitive to external field changes, so a dF alarm is a sensor
fault; a network residual alarm while the rest of the network is quiet is a
local (sensor) fault; a large disturbance that the network shares is geophysical.

Thresholds: "analytic" uses the chi2 / Siegmund values for the configured pfa.
"guarded" (default) uses max(analytic, empirical calibration quantile) per station,
so a residual that is heavier-tailed than Gaussian cannot inflate the false-alarm
rate beyond what the calibration window shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest

FEATURE_COLUMNS = ["d_f", "d_f_std30", "z_dh", "z_dbdt", "net_z_dh", "net_z_dbdt", "corr_dx_net"]


@dataclass
class DetectConfig:
    pfa: float = 1e-6            # per-minute false-alarm probability of each GLRT
    window: int = 15             # GLRT window, minutes
    cusum_k: float = 0.5         # CUSUM reference value (sigma units)
    cusum_arl0: float = 1e6      # target in-control average run length, minutes
    naive_z: float = 6.0         # NaiveZ threshold
    dist_z: float = 6.0          # disturbance level the classifier calls geophysical
    storm_net_z: float = 4.0     # network disturbance above which local alarms are ambiguous
    iforest_quantile: float = 1e-3
    iforest_trees: int = 200
    seed: int = 0
    threshold_mode: str = "guarded"   # "analytic" | "guarded"
    empirical_rate: float = 5e-4      # calibration alarm-minute rate allowed in guarded mode
    extras: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- primitives

def robust_loc_scale(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, max(mad, 1e-9)


@dataclass
class AR1Whitener:
    mu: float
    phi: float
    sigma: float

    @classmethod
    def fit(cls, x: np.ndarray) -> "AR1Whitener":
        mu, scale = robust_loc_scale(x)
        xc = x - mu
        a, b = xc[1:], xc[:-1]
        ok = np.isfinite(a) & np.isfinite(b)
        # Lag-1 least squares, ignoring pairs with a gross outlier (> 6 robust sigma).
        keep = ok & (np.abs(np.nan_to_num(a)) < 6 * scale) & (np.abs(np.nan_to_num(b)) < 6 * scale)
        aa, bb = a[keep], b[keep]
        phi = float(np.clip(np.dot(aa, bb) / max(np.dot(bb, bb), 1e-12), -0.99, 0.99))
        e = a[ok] - phi * b[ok]
        _, sigma = robust_loc_scale(e)
        return cls(mu, phi, sigma)

    def transform(self, x: np.ndarray) -> np.ndarray:
        xc = x - self.mu
        e = np.full_like(xc, np.nan, dtype=float)
        e[1:] = (xc[1:] - self.phi * xc[:-1]) / self.sigma
        return e


def _rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([np.zeros((1,) + a.shape[1:]), a], axis=0), axis=0)
    out = c[window:] - c[:-window]
    pad = np.full((window - 1,) + a.shape[1:], np.nan)
    return np.concatenate([pad, out], axis=0)


def window_glrt(e: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Windowed GLRTs on unit-variance white noise e (n,) or (n, d).

    T_mean = sum_d S_d^2 / N                    ~ chi2(d)  under H0 (unknown mean shift)
    T_var  = sum_d N (s2_d - 1 - ln s2_d)       ~ chi2(d)  asymptotically (variance change)
    Rows with any NaN are dropped from the window; fewer than window/2 valid -> NaN.
    """
    e2 = e[:, None] if e.ndim == 1 else e
    valid = np.all(np.isfinite(e2), axis=1)
    z = np.where(valid[:, None], e2, 0.0)
    n_valid = _rolling_sum(valid.astype(float), window)
    s = _rolling_sum(z, window)
    ss = _rolling_sum(z**2, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = (s**2).sum(axis=1) / n_valid
        s2 = ss / n_valid[:, None]
        s2 = np.clip(s2, 1e-6, None)
        t_var = (n_valid[:, None] * (s2 - 1 - np.log(s2))).sum(axis=1)
    bad = ~(n_valid >= window / 2)
    t_mean[bad] = np.nan
    t_var[bad] = np.nan
    return t_mean, t_var


def cusum_two_sided(e: np.ndarray, k: float, cap: float) -> np.ndarray:
    """max(S+, S-) of a two-sided CUSUM; NaN samples hold the state; S is capped
    so the statistic recovers within cap/k samples after a fault ends."""
    sp = sn = 0.0
    out = np.empty(len(e))
    for i, v in enumerate(e):
        if np.isfinite(v):
            sp = min(max(0.0, sp + v - k), cap)
            sn = min(max(0.0, sn - v - k), cap)
        out[i] = max(sp, sn)
    return out


def siegmund_h(k: float, arl0: float) -> float:
    """Threshold h giving a two-sided in-control ARL of arl0 (Siegmund approximation)."""
    def arl_two_sided(h: float) -> float:
        b = h + 1.166
        one = (np.exp(2 * k * b) - 2 * k * b - 1) / (2 * k**2)
        return one / 2.0
    return float(brentq(lambda h: arl_two_sided(h) - arl0, 0.1, 100.0))


# ----------------------------------------------------------------------------- detectors

class ScalarGLRT:
    name = "scalar_glrt"

    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.thr = chi2.isf(cfg.pfa / 2, 1)          # Bonferroni over mean + variance tests
        self.whiteners: dict[str, AR1Whitener] = {}

    def fit(self, calib: dict[str, pd.Series]) -> "ScalarGLRT":
        self.whiteners = {c: AR1Whitener.fit(s.to_numpy(float)) for c, s in calib.items()}
        return self

    def score(self, code: str, d_f: pd.Series) -> pd.Series:
        e = self.whiteners[code].transform(d_f.to_numpy(float))
        t_mean, t_var = window_glrt(e, self.cfg.window)
        return pd.Series(np.fmax(t_mean, t_var) / self.thr, index=d_f.index)   # alarm if > 1


class ScalarCUSUM:
    name = "scalar_cusum"

    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.h = siegmund_h(cfg.cusum_k, cfg.cusum_arl0)
        self.whiteners: dict[str, AR1Whitener] = {}

    def fit(self, calib: dict[str, pd.Series]) -> "ScalarCUSUM":
        self.whiteners = {c: AR1Whitener.fit(s.to_numpy(float)) for c, s in calib.items()}
        return self

    def score(self, code: str, d_f: pd.Series) -> pd.Series:
        e = self.whiteners[code].transform(d_f.to_numpy(float))
        stat = cusum_two_sided(e, self.cfg.cusum_k, cap=3 * self.h)
        return pd.Series(stat / self.h, index=d_f.index)


class NetworkGLRT:
    name = "network_glrt"

    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.thr = chi2.isf(cfg.pfa / 2, 3)
        self.models: dict[str, dict] = {}

    @staticmethod
    def _design(wide: pd.DataFrame, code: str, others: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X = wide[[(o, c) for o in others for c in ("dx", "dy", "dz")]].to_numpy(float)
        Y = wide[[(code, c) for c in ("dx", "dy", "dz")]].to_numpy(float)
        return np.column_stack([np.ones(len(X)), X]), Y

    def fit(self, wide_calib: pd.DataFrame) -> "NetworkGLRT":
        stations = sorted({c for c, _ in wide_calib.columns})
        for code in stations:
            others = [o for o in stations if o != code]
            X, Y = self._design(wide_calib, code, others)
            ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
            Xo, Yo = X[ok], Y[ok]
            lam = 1e-3 * np.trace(Xo.T @ Xo) / Xo.shape[1]
            reg = lam * np.eye(Xo.shape[1])
            reg[0, 0] = 0.0
            W = np.linalg.solve(Xo.T @ Xo + reg, Xo.T @ Yo)
            R = Yo - Xo @ W
            # Robust covariance: drop the top 1 % Mahalanobis residuals and refit once.
            cov = np.cov(R.T)
            md = np.einsum("ij,jk,ik->i", R, np.linalg.inv(cov), R)
            cov = np.cov(R[md <= np.quantile(md, 0.99)].T)
            L = np.linalg.cholesky(cov)
            Wn = np.linalg.solve(L, R.T).T
            whit = [AR1Whitener.fit(Wn[:, j]) for j in range(3)]
            self.models[code] = {"others": others, "W": W, "L": L, "whit": whit}
        return self

    def residual_stat(self, wide: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for code, m in self.models.items():
            X, Y = self._design(wide, code, m["others"])
            R = Y - X @ m["W"]
            Wn = np.linalg.solve(m["L"], R.T).T
            E = np.column_stack([w.transform(Wn[:, j]) for j, w in enumerate(m["whit"])])
            t_mean, t_var = window_glrt(E, self.cfg.window)
            out[code] = np.fmax(t_mean, t_var) / self.thr
        return pd.DataFrame(out, index=wide.index)

    @staticmethod
    def isolate(stat: pd.DataFrame) -> pd.DataFrame:
        """Keep, at each minute, only the largest alarming station (others -> 0)."""
        s = stat.to_numpy(copy=True)
        finite = np.where(np.isfinite(s), s, -np.inf)
        alarm = finite > 1
        multi = alarm.sum(axis=1) > 1
        if multi.any():
            best = finite.argmax(axis=1)
            not_best = np.arange(s.shape[1])[None, :] != best[:, None]
            s[multi[:, None] & not_best & alarm] = 0.0
        return pd.DataFrame(s, index=stat.index, columns=stat.columns)


class IsolationForestDetector:
    name = "iforest"

    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.models: dict[str, tuple[IsolationForest, float, np.ndarray]] = {}

    @staticmethod
    def _matrix(df: pd.DataFrame, center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        X = df[FEATURE_COLUMNS].to_numpy(float)
        if center is None:
            center = np.nanmedian(X, axis=0)
        X = X - center
        X[:, 0] = np.abs(X[:, 0])                      # |dF - median|
        return np.nan_to_num(X, nan=0.0), center

    def fit(self, calib: dict[str, pd.DataFrame]) -> "IsolationForestDetector":
        for code, df in calib.items():
            X, center = self._matrix(df)
            model = IsolationForest(n_estimators=self.cfg.iforest_trees, random_state=self.cfg.seed,
                                    max_samples=min(4096, len(X)))
            model.fit(X)
            score = -model.score_samples(X)
            thr = float(np.quantile(score, 1 - self.cfg.iforest_quantile))
            self.models[code] = (model, thr, center)
        return self

    def score(self, code: str, df: pd.DataFrame) -> pd.Series:
        model, thr, center = self.models[code]
        X, _ = self._matrix(df, center)
        return pd.Series(-model.score_samples(X) / thr, index=df.index)


def guard_scale(calib_score: pd.Series, cfg: DetectConfig) -> float:
    """Factor >= 1 that the score is divided by in guarded mode."""
    if cfg.threshold_mode != "guarded":
        return 1.0
    v = calib_score.to_numpy(float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 1.0
    return float(max(1.0, np.quantile(v, 1 - cfg.empirical_rate)))


def naive_z(df: pd.DataFrame, cfg: DetectConfig) -> pd.Series:
    return df["z_dh"] / cfg.naive_z


def classify(scalar: pd.Series, network: pd.Series, feats: pd.DataFrame,
             cfg: DetectConfig) -> pd.Series:
    """0 normal, 1 sensor fault, 2 geomagnetic event (per minute)."""
    label = np.zeros(len(feats), dtype=np.int8)
    net_dist = feats["net_z_dh"].to_numpy(float)
    own_dist = feats["z_dh"].to_numpy(float)
    s_alarm = np.nan_to_num(scalar.to_numpy(float)) > 1
    n_alarm = np.nan_to_num(network.to_numpy(float)) > 1
    shared = np.nan_to_num(net_dist) > cfg.storm_net_z
    disturbed = np.nan_to_num(own_dist) > cfg.dist_z

    label[(disturbed | n_alarm) & shared] = 2
    label[n_alarm & ~shared] = 1
    label[s_alarm] = 1                                  # dF is blind to the external field
    return pd.Series(label, index=feats.index)
