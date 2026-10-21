"""Synthetic magnetometer network for offline demos and CI.

The generator is deliberately physical enough to be a fair test bed:

* main field per station with slow secular variation
* solar-quiet (Sq) daily variation driven by each station's local time
* coherent + local low-frequency fluctuations (AR(1))
* storms: sudden commencement, ring-current main phase and recovery (coherent
  across the network), plus substorm bays centred on local midnight (NOT coherent
  across 50 degrees of longitude) and enhanced Pc5-like fluctuations
* Kp derived from 3-hour disturbance ranges with the standard quasi-log K scale
* vector noise quantised to 0.1 nT, independent scalar F with its own noise and
  a small pier offset, and random data gaps

Codes are fictitious (SY*) so synthetic data can never be mistaken for INTERMAGNET data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import iaga2002


@dataclass(frozen=True)
class SynthStation:
    code: str
    name: str
    lat: float
    lon: float
    b0: tuple[float, float, float]
    lat_factor: float        # substorm amplitude scaling (higher latitude -> larger)


STATIONS = (
    SynthStation("SYV", "Synthetic West Coast", 48.5, -123.4, (18200.0, 5700.0, 50300.0), 0.45),
    SynthStation("SYM", "Synthetic Prairie North", 54.6, -113.3, (12700.0, 3600.0, 56200.0), 1.00),
    SynthStation("SYB", "Synthetic Prairie South", 49.9, -100.0, (13900.0, 1000.0, 56300.0), 0.70),
    SynthStation("SYO", "Synthetic East", 45.4, -75.6, (17600.0, -4300.0, 51000.0), 0.40),
)

K_LIMITS = np.array([0, 5, 10, 20, 40, 70, 120, 200, 330, 500])  # K9 = 500 nT scale


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma_stat: float) -> np.ndarray:
    innov = rng.normal(0.0, sigma_stat * np.sqrt(1 - phi**2), n)
    return lfilter([1.0], [1.0, -phi], innov)


def _storm_profiles(rng: np.random.Generator, t_min: np.ndarray, n_days: int):
    """Return (ring_current[n], storm_activity[n], list of storm dicts)."""
    n = len(t_min)
    ring = np.zeros(n)
    activity = np.zeros(n)
    storms = []
    n_storms = max(1, rng.poisson(n_days / 18))
    onsets = np.sort(rng.uniform(0.15, 0.95, n_storms)) * n
    for t0 in onsets.astype(int):
        dst = rng.uniform(40, 280)
        ssc = rng.uniform(10, 30)
        main = rng.uniform(180, 540)            # minutes
        tau = rng.uniform(600, 1500)            # recovery e-folding, minutes
        dt = t_min - t0
        prof = np.zeros(n)
        prof += np.where((dt >= 0) & (dt < 120), ssc * np.clip(dt / 2, 0, 1), 0)
        mp = (dt >= 0) & (dt < main)
        prof[mp] -= dst * (dt[mp] / main)
        rec = dt >= main
        prof[rec] -= dst * np.exp(-(dt[rec] - main) / tau)
        ring += prof
        act = np.exp(-np.clip(dt, 0, None) / (main + tau / 2)) * (dt >= -30)
        activity = np.maximum(activity, act * dst / 280)
        storms.append({"onset_min": int(t0), "dst": dst, "main_min": main, "tau_min": tau})
    return ring, activity, storms


def generate(days: int = 90, start: str = "2024-01-01", seed: int = 7,
             stations: tuple[SynthStation, ...] = STATIONS):
    """Return ({code: DataFrame[x,y,z,f]}, kp DataFrame[ts_start, kp], storms list)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="min", tz="UTC")
    n = len(idx)
    t_min = np.arange(n, dtype=float)
    utc_h = (idx.hour + idx.minute / 60.0).to_numpy()
    day_no = (t_min // 1440).astype(int)

    sq_day_amp = 1.0 + 0.25 * _ar1(rng, days, 0.7, 1.0)[day_no]        # network-coherent
    common = np.stack([_ar1(rng, n, 0.995, 2.0) for _ in range(3)])      # coherent background
    ring, activity, storms = _storm_profiles(rng, t_min, days)

    # Substorm bays: during active periods, centred in UT; each station sees it
    # weighted by how close it is to local midnight.
    bays = []
    for s in storms:
        n_bays = rng.integers(3, 9)
        centers = s["onset_min"] + rng.uniform(0, s["main_min"] + 600, n_bays)
        for c in centers:
            bays.append((c, rng.uniform(30, 120), rng.uniform(60, 450), rng.uniform(-1, 1)))
    for _ in range(rng.poisson(days / 4)):                             # isolated quiet-time bays
        bays.append((rng.uniform(0, n), rng.uniform(20, 60), rng.uniform(15, 80), rng.uniform(-1, 1)))

    data, disturbance = {}, {}
    for st in stations:
        lt = (utc_h + st.lon / 15.0) % 24
        g = np.exp(-(((lt - 11.5 + 12) % 24 - 12) / 3.2) ** 2)
        sq = np.stack([-rng.uniform(20, 35) * g,
                       -rng.uniform(8, 15) * ((lt - 11.5) / 3.2) * g,
                       -rng.uniform(5, 12) * g]) * sq_day_amp

        gains = rng.uniform(0.8, 1.2, 3)
        local = np.stack([_ar1(rng, n, 0.98, 0.7) for _ in range(3)])
        pc5_sigma = 0.5 + 8.0 * activity * st.lat_factor
        pc5 = np.stack([_ar1(rng, n, 0.9, 1.0) for _ in range(3)]) * pc5_sigma

        bay = np.zeros((3, n))
        for c, width, amp, ysign in bays:
            lo, hi = int(max(0, c - width)), int(min(n, c + width))
            if hi <= lo:
                continue
            seg = np.arange(lo, hi)
            shape = 0.5 * (1 + np.cos(np.pi * (seg - c) / width))
            lt_c = (utc_h[min(int(c), n - 1)] + st.lon / 15.0) % 24
            night = np.exp(-((((lt_c + 12) % 24) - 12) / 3.5) ** 2)
            a = amp * st.lat_factor * night
            bay[0, seg] -= a * shape
            bay[1, seg] += 0.35 * ysign * a * shape
            bay[2, seg] += 0.25 * a * shape

        ring_vec = np.stack([ring * np.cos(np.deg2rad(st.lat - 5)), 0.05 * ring, -0.1 * ring])
        dist = common * gains[:, None] + local + pc5 + bay + ring_vec
        secular = np.stack([-1e-5, 2e-5, 4e-5])[:, None] * t_min
        b_true = np.array(st.b0)[:, None] + secular + sq + dist

        vec = np.round(b_true + rng.normal(0, 0.08, b_true.shape), 1)
        f = np.round(np.sqrt((b_true**2).sum(axis=0)) + rng.uniform(-0.3, 0.3)
                     + rng.normal(0, 0.04, n), 2)
        df = pd.DataFrame({"x": vec[0], "y": vec[1], "z": vec[2], "f": f}, index=idx)

        for _ in range(rng.poisson(days / 15)):                         # data gaps
            g0 = rng.integers(0, n)
            df.iloc[g0:g0 + rng.integers(5, 180)] = np.nan
        data[st.code] = df
        disturbance[st.code] = dist

    # Kp from 3-hour ranges of the horizontal disturbance (Sq excluded by construction).
    bins = (t_min // 180).astype(int)
    k_station = []
    for code in data:
        d = disturbance[code]
        rng_x = pd.Series(d[0]).groupby(bins).agg(lambda s: s.max() - s.min())
        rng_y = pd.Series(d[1]).groupby(bins).agg(lambda s: s.max() - s.min())
        r = np.maximum(rng_x, rng_y).to_numpy()
        k_station.append(np.searchsorted(K_LIMITS, r, side="right") - 1)
    kp_raw = np.mean(k_station, axis=0) + rng.normal(0, 0.2, len(k_station[0]))
    kp = np.clip(np.round(kp_raw * 3) / 3, 0, 9)
    kp_df = pd.DataFrame({"ts_start": idx[::180], "kp": kp})
    return data, kp_df, storms


def write_network(out: Path, days: int = 90, start: str = "2024-01-01", seed: int = 7) -> dict:
    out = Path(out)
    data, kp_df, storms = generate(days=days, start=start, seed=seed)
    meta_by_code = {s.code: s for s in STATIONS}
    files = 0
    for code, df in data.items():
        st = meta_by_code[code]
        meta = {"Station Name": st.name, "Geodetic Latitude": f"{st.lat:.3f}",
                "Geodetic Longitude": f"{st.lon % 360:.3f}", "Elevation": "500",
                "Data Type": "Synthetic"}
        for _, chunk in df.groupby(df.index.tz_localize(None).to_period("M")):
            first, last = chunk.index.min(), chunk.index.max()
            iaga2002.write(out / code / f"{code}_{first:%Y%m%d}_{last:%Y%m%d}.min", code, chunk, meta)
            files += 1
    kp_payload = {
        "meta": {"source": "magfault synthetic generator (GFZ JSON layout)"},
        "datetime": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in kp_df["ts_start"]],
        "Kp": [float(k) for k in kp_df["kp"]],
        "status": ["def"] * len(kp_df),
    }
    (out / "kp").mkdir(parents=True, exist_ok=True)
    (out / "kp" / "kp_synthetic.json").write_text(json.dumps(kp_payload))
    (out / "storms.json").write_text(json.dumps(storms, indent=2))
    return {"files": files, "stations": list(data), "storms": len(storms),
            "max_kp": float(kp_df["kp"].max())}
