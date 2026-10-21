"""Reader/writer for the IAGA-2002 exchange format used by INTERMAGNET.

Output is always geographic X, Y, Z plus an independent scalar F, in nT:

* HDZ files are rotated to XYZ (D is published in minutes of arc).
* The 4th element is F (or S) for most observatories. Some publish G, which
  INTERMAGNET defines as dF = F_vector - F_scalar. Some services deliver full-field
  values under the G label, so the column is interpreted by magnitude: a median
  above 1000 nT is a total field, otherwise it is dF and F_scalar = |B| - G.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IAGA_MISSING_THRESHOLD


@dataclass
class IagaFile:
    code: str
    meta: dict[str, str]
    reported: str
    data: pd.DataFrame          # index: UTC timestamps; columns x, y, z, f
    comments: list[str] = field(default_factory=list)

    @property
    def latitude(self) -> float | None:
        return _to_float(self.meta.get("Geodetic Latitude"))

    @property
    def longitude(self) -> float | None:
        lon = _to_float(self.meta.get("Geodetic Longitude"))
        return lon - 360.0 if lon is not None and lon > 180 else lon

    @property
    def elevation(self) -> float | None:
        return _to_float(self.meta.get("Elevation"))


def _to_float(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def parse(text: str) -> IagaFile:
    meta: dict[str, str] = {}
    comments: list[str] = []
    lines = text.splitlines()
    i = 0
    header_cols: list[str] | None = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("DATE"):
            header_cols = line.replace("|", " ").split()
            i += 1
            break
        body = line.rstrip().rstrip("|")
        if body.strip().startswith("#"):
            comments.append(body.strip().lstrip("#").strip())
        elif body.strip():
            key, value = body[:24].strip(), body[24:].strip()
            meta[key] = value
        i += 1
    if header_cols is None:
        raise ValueError("Not an IAGA-2002 file: no DATE header line found")

    code = (meta.get("IAGA CODE") or meta.get("IAGA Code") or header_cols[3][:3]).strip().upper()
    elements = [c[len(code):].upper() if c.upper().startswith(code) else c[-1].upper()
                for c in header_cols[3:7]]

    rows = [ln.split() for ln in lines[i:] if ln.strip()]
    if not rows:
        empty = pd.DataFrame(columns=["x", "y", "z", "f"], dtype=float)
        empty.index = pd.DatetimeIndex([], tz="UTC", name="ts")
        return IagaFile(code, meta, "".join(elements), empty, comments)

    arr = np.array([r[3:7] for r in rows], dtype=float)
    arr[arr >= IAGA_MISSING_THRESHOLD] = np.nan
    ts = pd.to_datetime([f"{r[0]} {r[1]}" for r in rows], utc=True)
    raw = pd.DataFrame(arr, index=ts, columns=elements)
    raw.index.name = "ts"

    out = pd.DataFrame(index=raw.index)
    if {"X", "Y"} <= set(elements):
        out["x"], out["y"] = raw["X"], raw["Y"]
    elif {"H", "D"} <= set(elements):
        d_rad = np.deg2rad(raw["D"] / 60.0)
        out["x"] = raw["H"] * np.cos(d_rad)
        out["y"] = raw["H"] * np.sin(d_rad)
    else:
        raise ValueError(f"Unsupported element set {elements}")
    out["z"] = raw["Z"]

    fourth = elements[3]
    total = np.sqrt(out["x"] ** 2 + out["y"] ** 2 + out["z"] ** 2)
    if fourth in ("F", "S"):
        out["f"] = raw[fourth]
    elif fourth == "G":
        col = raw["G"]
        out["f"] = col if np.nanmedian(np.abs(col)) > 1000 else total - col
    else:
        out["f"] = np.nan
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return IagaFile(code, meta, "".join(elements), out, comments)


def read(path: str | Path) -> IagaFile:
    return parse(Path(path).read_text(errors="replace"))


def write(path: str | Path, code: str, data: pd.DataFrame, meta: dict[str, str]) -> None:
    """Write x, y, z, f (nT, UTC index) as an XYZF IAGA-2002 file."""
    header = {
        "Format": "IAGA-2002",
        "Source of Data": meta.get("Source of Data", "magfault synthetic generator"),
        "Station Name": meta.get("Station Name", code),
        "IAGA Code": code,
        "Geodetic Latitude": meta.get("Geodetic Latitude", ""),
        "Geodetic Longitude": meta.get("Geodetic Longitude", ""),
        "Elevation": meta.get("Elevation", ""),
        "Reported": "XYZF",
        "Sensor Orientation": "XYZF",
        "Digital Sampling": "1 second",
        "Data Interval Type": "Filtered 1-minute (00:15-01:45)",
        "Data Type": meta.get("Data Type", "Synthetic"),
    }
    lines = [f" {k:<23}{v:<44}|" for k, v in header.items()]
    lines.append(f" {'# ' + meta.get('comment', 'Synthetic data; not real observations.'):<67}|")
    lines.append(f"DATE       TIME         DOY     {code}X      {code}Y      {code}Z      {code}F   |")
    vals = data[["x", "y", "z", "f"]].to_numpy(dtype=float)
    vals = np.where(np.isnan(vals), 99999.0, vals)
    idx = data.index.tz_convert("UTC")
    for t, (x, y, z, f) in zip(idx, vals):
        lines.append(
            f"{t:%Y-%m-%d %H:%M:%S}.000 {t.dayofyear:03d}     "
            f"{x:9.2f} {y:9.2f} {z:9.2f} {f:9.2f}"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
