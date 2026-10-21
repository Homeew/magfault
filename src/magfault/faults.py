"""Sensor fault models and reproducible fault-injection schedules.

The models follow the fault taxonomy used for sensor validity screening:
spike, offset (bias step), drift (ramp), stuck-at, gain error, noise burst,
plus two that matter specifically for a vector/scalar observatory:

* tilt          - a small rotation of the vector sensor. |B| is preserved exactly,
                  so the vector-scalar residual dF is blind to it.
* scalar_offset - a bias on the independent scalar magnetometer only.

First-order observability through dF: a vector error d changes |B| by b_hat . d.
At mid-latitudes b_hat ~ (0.3, 0.05, 0.95), so an error on Y is almost invisible
to dF, and a rotation is invisible exactly. Those faults need the network test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

FAULT_TYPES = ("spike", "offset", "drift", "stuck", "gain", "noise_burst", "tilt", "scalar_offset")

# (min, max) duration in minutes and (min, max) magnitude, per fault type.
DURATION = {
    "spike": (1, 3), "offset": (60, 720), "drift": (180, 1440), "stuck": (30, 360),
    "gain": (120, 720), "noise_burst": (30, 240), "tilt": (360, 2880), "scalar_offset": (60, 720),
}
MAGNITUDE = {
    "spike": (5.0, 50.0),          # nT
    "offset": (1.0, 10.0),         # nT
    "drift": (2.0, 20.0),          # nT reached at the end of the ramp
    "stuck": (0.0, 0.0),           # holds the value at onset
    "gain": (1e-4, 1e-3),          # fractional scale error
    "noise_burst": (0.5, 3.0),     # nT, std of added white noise
    "tilt": (0.5, 5.0),            # arcmin
    "scalar_offset": (0.5, 5.0),   # nT
}
VECTOR_CHANNELS = ("x", "y", "z")


@dataclass
class Fault:
    iaga_code: str
    fault_type: str
    channel: str
    t_start: pd.Timestamp
    duration_min: int
    magnitude: float
    params: dict = field(default_factory=dict)

    @property
    def t_end(self) -> pd.Timestamp:
        return self.t_start + pd.Timedelta(minutes=self.duration_min - 1)

    def to_record(self) -> dict:
        d = asdict(self)
        d["t_end"] = self.t_end
        return d


def _rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1 - np.cos(angle_rad)) * (k @ k)


def apply_fault(seg: pd.DataFrame, fault: Fault, rng: np.random.Generator) -> pd.DataFrame:
    """Return a corrupted copy of seg (columns x, y, z, f) over the fault interval."""
    out = seg.copy()
    n = len(out)
    ch, mag, kind = fault.channel, fault.magnitude, fault.fault_type
    sign = fault.params.get("sign", 1.0)

    if kind == "spike":
        out[ch] += sign * mag
    elif kind == "offset":
        out[ch] += sign * mag
    elif kind == "drift":
        out[ch] += sign * mag * np.linspace(0.0, 1.0, n)
    elif kind == "stuck":
        first = out[ch].dropna()
        out[ch] = first.iloc[0] if len(first) else np.nan
    elif kind == "gain":
        out[ch] *= 1.0 + sign * mag
    elif kind == "noise_burst":
        out[ch] += rng.normal(0.0, mag, n)
    elif kind == "tilt":
        axis = np.array(fault.params["axis"], dtype=float)
        rot = _rotation(axis, np.deg2rad(mag / 60.0))
        out[["x", "y", "z"]] = out[["x", "y", "z"]].to_numpy() @ rot.T
    elif kind == "scalar_offset":
        out["f"] += sign * mag
    else:
        raise ValueError(f"unknown fault type {kind}")

    out[["x", "y", "z"]] = out[["x", "y", "z"]].round(1)
    out["f"] = out["f"].round(2)
    return out


def _draw_fault(code: str, kind: str, t_start: pd.Timestamp, rng: np.random.Generator) -> Fault:
    dmin, dmax = DURATION[kind]
    duration = int(rng.integers(dmin, dmax + 1))
    lo, hi = MAGNITUDE[kind]
    mag = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if lo > 0 else 0.0
    params: dict = {"sign": float(rng.choice([-1.0, 1.0]))}
    if kind == "tilt":
        az = rng.uniform(0, 2 * np.pi)
        params["axis"] = [float(np.cos(az)), float(np.sin(az)), 0.0]
        channel = "xyz"
    elif kind == "scalar_offset":
        channel = "f"
    else:
        channel = str(rng.choice(VECTOR_CHANNELS))
    return Fault(code, kind, channel, t_start, duration, mag, params)


def make_schedule(coverage: dict[str, pd.DatetimeIndex], calibration_end: pd.Timestamp,
                  per_type: int = 3, min_gap_h: float = 6.0, seed: int = 42,
                  types: tuple[str, ...] = FAULT_TYPES) -> list[Fault]:
    """Non-overlapping faults per station, all after calibration_end.

    coverage maps station -> timestamps where the clean vector data is present;
    a fault is accepted only if at least 90 % of its interval has data.
    """
    rng = np.random.default_rng(seed)
    faults: list[Fault] = []
    for code in sorted(coverage):
        present = coverage[code]
        present = present[present > calibration_end]
        if len(present) < 1440:
            continue
        t_lo, t_hi = present.min(), present.max()
        kinds = [k for k in types for _ in range(per_type)]
        rng.shuffle(kinds)
        taken: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        gap = pd.Timedelta(hours=min_gap_h)
        for kind in kinds:
            for _ in range(200):
                span = (t_hi - t_lo).total_seconds() / 60
                t0 = (t_lo + pd.Timedelta(minutes=int(rng.uniform(0, span)))).floor("min")
                f = _draw_fault(code, kind, t0, rng)
                if f.t_end > t_hi:
                    continue
                if any(f.t_start <= e + gap and f.t_end >= s - gap for s, e in taken):
                    continue
                expected = f.duration_min
                got = present[(present >= f.t_start) & (present <= f.t_end)].size
                if got < 0.9 * expected:
                    continue
                taken.append((f.t_start, f.t_end))
                faults.append(f)
                break
    return sorted(faults, key=lambda f: (f.iaga_code, f.t_start))
