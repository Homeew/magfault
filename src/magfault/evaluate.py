"""Evaluation that does not flatter the detector.

* Event level: a fault is detected if any alarm falls in [t_start, t_end + tol];
  detection delay is measured from t_start. Alarms are grouped into events
  (gaps <= merge_gap minutes) and an alarm event that touches no fault window
  is one false alarm, reported per station-day and split into quiet vs storm.
* Pointwise precision / recall / F1, with the tolerance tail after each fault
  treated as "don't care".
* Point-adjusted F1 is reported only for comparison with the literature: it marks
  a whole fault segment as detected if a single minute is, which inflates scores.
* For the 3-class classifier: a minute-level confusion matrix
  (truth from injected faults and Kp >= 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STORM_KP

CLASSES = ("normal", "sensor_fault", "geomagnetic")


def truth_labels(minutes: pd.DataFrame, faults: pd.DataFrame, tol_min: int) -> pd.DataFrame:
    """minutes: iaga_code, ts, kp. Adds truth (0/1/2), dont_care, fault_id columns."""
    out = minutes.copy()
    out["truth"] = np.where(out["kp"].fillna(0) >= STORM_KP, 2, 0).astype(np.int8)
    out["dont_care"] = False
    out["fault_id"] = -1
    tol = pd.Timedelta(minutes=tol_min)
    for code, grp in out.groupby("iaga_code", sort=False):
        ts = pd.DatetimeIndex(grp["ts"])
        for f in faults[faults["iaga_code"] == code].itertuples():
            lo = ts.searchsorted(f.t_start)
            hi = ts.searchsorted(f.t_end, side="right")
            hi_tol = ts.searchsorted(f.t_end + tol, side="right")
            rows = grp.index[lo:hi]
            out.loc[rows, "truth"] = 1
            out.loc[rows, "fault_id"] = f.fault_id
            out.loc[grp.index[hi:hi_tol], "dont_care"] = True
    return out


def alarm_events(ts: pd.Series, merge_gap: int = 5) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if ts.empty:
        return []
    t = pd.Series(sorted(ts))
    new = t.diff() > pd.Timedelta(minutes=merge_gap)
    gid = new.cumsum()
    return [(g.iloc[0], g.iloc[-1]) for _, g in t.groupby(gid)]


def _f1(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def evaluate_method(alarm_mask: pd.Series, labeled: pd.DataFrame, faults: pd.DataFrame,
                    tol_min: int, merge_gap: int = 5) -> dict:
    """alarm_mask: bool aligned with labeled (True = 'sensor fault' alarm)."""
    tol = pd.Timedelta(minutes=tol_min)
    alarm = alarm_mask.to_numpy(bool)
    truth = labeled["truth"].to_numpy()
    dc = labeled["dont_care"].to_numpy(bool)

    # ---- pointwise
    pos = truth == 1
    scored = ~dc
    point = _f1(int((alarm & pos & scored).sum()), int((alarm & ~pos & scored).sum()),
                int((~alarm & pos & scored).sum()))

    # ---- point-adjusted (literature convention)
    adj = alarm.copy()
    fid = labeled["fault_id"].to_numpy()
    hit_ids = np.unique(fid[alarm & pos])
    adj[np.isin(fid, hit_ids) & pos] = True
    point_adj = _f1(int((adj & pos & scored).sum()), int((adj & ~pos & scored).sum()),
                    int((~adj & pos & scored).sum()))

    # ---- event level
    per_fault = []
    alarm_ts = labeled.loc[alarm, ["iaga_code", "ts"]]
    by_station = {c: g["ts"].sort_values() for c, g in alarm_ts.groupby("iaga_code")}
    for f in faults.itertuples():
        ts = by_station.get(f.iaga_code, pd.Series(dtype="datetime64[ns, UTC]"))
        hits = ts[(ts >= f.t_start) & (ts <= f.t_end + tol)]
        per_fault.append({
            "fault_id": f.fault_id, "fault_type": f.fault_type, "channel": f.channel,
            "magnitude": f.magnitude, "detected": bool(len(hits)),
            "delay_min": (hits.iloc[0] - f.t_start).total_seconds() / 60 if len(hits) else None,
        })
    pf = pd.DataFrame(per_fault)

    storm_minutes = labeled.loc[labeled["kp"].fillna(0) >= STORM_KP, ["iaga_code", "ts"]]
    storm_set = {c: set(g["ts"]) for c, g in storm_minutes.groupby("iaga_code")}
    fa_quiet = fa_storm = 0
    for code, ts in by_station.items():
        fw = faults[faults["iaga_code"] == code]
        for s, e in alarm_events(ts, merge_gap):
            overlaps = ((fw["t_start"] <= e) & (fw["t_end"] + tol >= s)).any()
            if overlaps:
                continue
            ev_minutes = ts[(ts >= s) & (ts <= e)]
            if storm_set.get(code) and any(t in storm_set[code] for t in ev_minutes):
                fa_storm += 1
            else:
                fa_quiet += 1

    station_days = labeled.groupby("iaga_code")["ts"].count().sum() / 1440
    storm_days = len(storm_minutes) / 1440
    quiet_days = station_days - storm_days

    by_type = {}
    if len(pf):
        for kind, g in pf.groupby("fault_type"):
            d = g["delay_min"].dropna()
            by_type[kind] = {"n": int(len(g)), "recall": float(g["detected"].mean()),
                             "median_delay_min": float(d.median()) if len(d) else None}
    delays = pf["delay_min"].dropna() if len(pf) else pd.Series(dtype=float)
    storm = truth == 2
    storm_flag = float((alarm & storm).sum() / storm.sum()) if storm.any() else None
    return {
        "storm_minutes_flagged": storm_flag,
        "event": {
            "n_faults": int(len(pf)),
            "recall": float(pf["detected"].mean()) if len(pf) else 0.0,
            "median_delay_min": float(delays.median()) if len(delays) else None,
            "by_type": by_type,
            "false_alarms_quiet": fa_quiet,
            "false_alarms_storm": fa_storm,
            "fa_per_station_day_quiet": fa_quiet / quiet_days if quiet_days > 0 else None,
            "fa_per_station_day_storm": fa_storm / storm_days if storm_days > 0 else None,
        },
        "pointwise": point,
        "point_adjusted": point_adj,
        "per_fault": per_fault,
    }


def confusion(pred: pd.Series, labeled: pd.DataFrame) -> pd.DataFrame:
    keep = ~labeled["dont_care"].to_numpy(bool)
    t = labeled["truth"].to_numpy()[keep]
    p = pred.to_numpy()[keep]
    m = np.zeros((3, 3), dtype=int)
    np.add.at(m, (t, p), 1)
    return pd.DataFrame(m, index=[f"true_{c}" for c in CLASSES],
                        columns=[f"pred_{c}" for c in CLASSES])
