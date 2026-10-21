"""Database-facing pipeline steps: inject -> features -> detect -> evaluate."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from . import config, detectors, evaluate, faults, plots
from .db import copy_rows, read_df

METHODS = ("naive_z", "iforest", "scalar_cusum", "scalar_glrt", "network_glrt", "classifier")


# ----------------------------------------------------------------------------- inject

def inject(conn: psycopg.Connection, seed: int = 42, calibration_days: int = 14,
           per_type: int = 3) -> int:
    cov = read_df(conn, "SELECT iaga_code, ts FROM readings WHERE x IS NOT NULL AND y IS NOT NULL "
                        "AND z IS NOT NULL ORDER BY iaga_code, ts")
    if cov.empty:
        raise SystemExit("No readings loaded. Run `magfault load` first.")
    cov["ts"] = pd.to_datetime(cov["ts"], utc=True)
    coverage = {c: pd.DatetimeIndex(g["ts"]) for c, g in cov.groupby("iaga_code")}
    calibration_end = cov["ts"].min().floor("D") + pd.Timedelta(days=calibration_days)
    schedule = faults.make_schedule(coverage, calibration_end, per_type=per_type, seed=seed)

    params = {"calibration_days": calibration_days, "per_type": per_type,
              "duration": faults.DURATION, "magnitude": faults.MAGNITUDE}
    run_id = conn.execute(
        "INSERT INTO fault_runs (seed, calibration_end, params) VALUES (%s, %s, %s) RETURNING run_id",
        (seed, calibration_end.to_pydatetime(), Jsonb(params)),
    ).fetchone()[0]

    rng = np.random.default_rng(seed + 1)
    n_samples = 0
    for code in sorted({f.iaga_code for f in schedule}):
        clean = read_df(conn, "SELECT ts, x, y, z, f FROM readings WHERE iaga_code = %s ORDER BY ts",
                        (code,))
        clean["ts"] = pd.to_datetime(clean["ts"], utc=True)
        clean = clean.set_index("ts")
        for f in (f for f in schedule if f.iaga_code == code):
            fault_id = conn.execute(
                """INSERT INTO injected_faults (run_id, iaga_code, fault_type, channel, t_start,
                                               t_end, magnitude, params)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING fault_id""",
                (run_id, code, f.fault_type, f.channel, f.t_start.to_pydatetime(),
                 f.t_end.to_pydatetime(), f.magnitude, Jsonb(f.params)),
            ).fetchone()[0]
            seg = clean.loc[f.t_start:f.t_end]
            bad = faults.apply_fault(seg, f, rng)
            rows = ((run_id, code, t.to_pydatetime(), *map(_num, r), fault_id)
                    for t, r in zip(bad.index, bad[["x", "y", "z", "f"]].to_numpy()))
            n_samples += copy_rows(conn, "fault_samples",
                                   ["run_id", "iaga_code", "ts", "x", "y", "z", "f", "fault_id"], rows)
    conn.commit()
    print(f"run {run_id}: {len(schedule)} faults, {n_samples} corrupted samples, "
          f"calibration window ends {calibration_end:%Y-%m-%d}")
    return run_id


def _num(v: float) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


# ----------------------------------------------------------------------------- features

def build_features(conn: psycopg.Connection, run_id: int) -> int:
    t0 = time.time()
    conn.execute("CALL build_features(%s)", (run_id,))
    n = conn.execute("SELECT count(*) FROM features WHERE run_id = %s", (run_id,)).fetchone()[0]
    conn.execute("ANALYZE features")
    conn.commit()
    print(f"run {run_id}: {n} feature rows in {time.time() - t0:.1f}s")
    return n


def _load_features(conn: psycopg.Connection, run_id: int) -> pd.DataFrame:
    cols = ", ".join(["iaga_code", "ts", "dx", "dy", "dz", "dh", "dbdt", "kp"]
                     + detectors.FEATURE_COLUMNS)
    df = read_df(conn, f"SELECT {cols} FROM features WHERE run_id = %s ORDER BY iaga_code, ts",
                 (run_id,))
    if df.empty:
        raise SystemExit(f"No features for run {run_id}. Run `magfault features` first.")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["iaga_code"] = df["iaga_code"].str.strip()
    return df


def _calibration_end(conn: psycopg.Connection, run_id: int) -> pd.Timestamp:
    row = conn.execute("SELECT calibration_end FROM fault_runs WHERE run_id = %s", (run_id,)).fetchone()
    if row is None:
        raise SystemExit(f"Unknown run {run_id}")
    return pd.Timestamp(row[0]).tz_convert("UTC")


# ----------------------------------------------------------------------------- detect

def detect(conn: psycopg.Connection, run_id: int, cfg: detectors.DetectConfig | None = None) -> dict:
    cfg = cfg or detectors.DetectConfig()
    t0 = time.time()
    feats = _load_features(conn, run_id)
    cal_end = _calibration_end(conn, run_id)
    stations = sorted(feats["iaga_code"].unique())
    per = {c: g.set_index("ts") for c, g in feats.groupby("iaga_code")}
    calib = {c: g[g.index < cal_end] for c, g in per.items()}

    scalar = detectors.ScalarGLRT(cfg).fit({c: g["d_f"] for c, g in calib.items()})
    cusum = detectors.ScalarCUSUM(cfg).fit({c: g["d_f"] for c, g in calib.items()})
    iforest = detectors.IsolationForestDetector(cfg).fit(calib)

    wide = feats.pivot(index="ts", columns="iaga_code", values=["dx", "dy", "dz"])
    wide.columns = wide.columns.swaplevel(0, 1)
    wide = wide.sort_index(axis=1)
    network = detectors.NetworkGLRT(cfg).fit(wide[wide.index < cal_end])
    net_stat = network.isolate(network.residual_stat(wide))

    scores: dict[str, dict[str, pd.Series]] = {m: {} for m in METHODS}
    guards: dict[str, dict[str, float]] = {}
    for c in stations:
        g = per[c]
        scores["naive_z"][c] = detectors.naive_z(g, cfg)
        scores["iforest"][c] = iforest.score(c, g)
        scores["scalar_cusum"][c] = cusum.score(c, g["d_f"])
        scores["scalar_glrt"][c] = scalar.score(c, g["d_f"])
        scores["network_glrt"][c] = net_stat[c].reindex(g.index)
        cal = g.index < cal_end
        for m in ("scalar_cusum", "scalar_glrt", "network_glrt"):
            k = detectors.guard_scale(scores[m][c][cal], cfg)
            guards.setdefault(m, {})[c] = k
            scores[m][c] = scores[m][c] / k
        scores["classifier"][c] = detectors.classify(
            scores["scalar_glrt"][c], scores["network_glrt"][c], g, cfg)

    conn.execute("DELETE FROM detections WHERE run_id = %s", (run_id,))
    counts = {}
    for m in METHODS:
        rows = []
        for c, s in scores[m].items():
            v = s.to_numpy(float)
            if m == "classifier":
                hit = v > 0
                labels = v[hit].astype(int)
                vals = v[hit]
            else:
                hit = np.nan_to_num(v) > 1
                labels = np.ones(hit.sum(), dtype=int)
                vals = v[hit]
            for t, sc, lb in zip(s.index[hit], vals, labels):
                rows.append((run_id, m, c, t.to_pydatetime(), float(sc), int(lb)))
        counts[m] = copy_rows(conn, "detections",
                              ["run_id", "method", "iaga_code", "ts", "score", "label"], rows)
    conn.execute(
        "UPDATE fault_runs SET params = params || %s WHERE run_id = %s",
        (Jsonb({"detect_config": asdict(cfg),
                "fitted": {"scalar_ar1": {c: asdict(w) for c, w in scalar.whiteners.items()},
                           "cusum_h": cusum.h, "glrt_thr_1dof": scalar.thr,
                           "glrt_thr_3dof": network.thr, "guard_scale": guards}}), run_id),
    )
    conn.commit()
    print(f"run {run_id}: detections {counts} in {time.time() - t0:.1f}s")
    return counts


# ----------------------------------------------------------------------------- evaluate

def run_evaluation(conn: psycopg.Connection, run_id: int, out_dir: Path | None = None,
                   tol_min: int | None = None) -> dict:
    cfg_row = conn.execute("SELECT params FROM fault_runs WHERE run_id = %s", (run_id,)).fetchone()
    params = cfg_row[0] if cfg_row else {}
    window = params.get("detect_config", {}).get("window", 15)
    tol_min = tol_min if tol_min is not None else window + 5
    cal_end = _calibration_end(conn, run_id)

    minutes = read_df(conn, "SELECT iaga_code, ts, kp FROM features WHERE run_id = %s "
                            "ORDER BY iaga_code, ts", (run_id,))
    minutes["ts"] = pd.to_datetime(minutes["ts"], utc=True)
    minutes["iaga_code"] = minutes["iaga_code"].str.strip()
    calib_minutes = minutes[minutes["ts"] < cal_end]
    minutes = minutes[minutes["ts"] >= cal_end].reset_index(drop=True)

    flt = read_df(conn, "SELECT fault_id, iaga_code, fault_type, channel, t_start, t_end, magnitude "
                        "FROM injected_faults WHERE run_id = %s ORDER BY iaga_code, t_start", (run_id,))
    flt["iaga_code"] = flt["iaga_code"].str.strip()
    for c in ("t_start", "t_end"):
        flt[c] = pd.to_datetime(flt[c], utc=True)

    det = read_df(conn, "SELECT method, iaga_code, ts, label FROM detections WHERE run_id = %s",
                  (run_id,))
    det["ts"] = pd.to_datetime(det["ts"], utc=True)
    det["iaga_code"] = det["iaga_code"].str.strip()

    labeled = evaluate.truth_labels(minutes, flt, tol_min)
    key = pd.MultiIndex.from_frame(labeled[["iaga_code", "ts"]])
    results: dict = {"run_id": run_id, "tolerance_min": tol_min,
                     "eval_start": str(cal_end), "methods": {}}
    for m in METHODS:
        d = det[det["method"] == m]
        lab = pd.Series(d["label"].to_numpy(),
                        index=pd.MultiIndex.from_frame(d[["iaga_code", "ts"]])).reindex(key).fillna(0)
        lab = lab.astype(int).reset_index(drop=True)
        res = evaluate.evaluate_method(lab == 1, labeled, flt, tol_min)
        d_cal = d[d["ts"] < cal_end]
        res["calibration_alarm_rate"] = len(d_cal) / max(len(calib_minutes), 1)
        if m == "classifier":
            res["confusion"] = evaluate.confusion(lab, labeled).to_dict()
        results["methods"][m] = res

    results["n_station_days"] = len(minutes) / 1440
    results["storm_minutes_fraction"] = float((labeled["truth"] == 2).mean())

    out_dir = Path(out_dir or config.REPORTS_DIR / f"run_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fault = pd.DataFrame([dict(pf, method=m) for m, r in results["methods"].items()
                              for pf in r.pop("per_fault")])
    per_fault.to_csv(out_dir / "per_fault.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "summary.md").write_text(summary_markdown(results))
    plots.fault_examples(conn, run_id, flt, det, out_dir / "fault_examples.png")
    plots.storm_overview(conn, run_id, det, out_dir / "storm_overview.png", flt)
    plots.recall_by_type(per_fault, out_dir / "recall_by_type.png")
    print((out_dir / "summary.md").read_text())
    return results


def summary_markdown(res: dict) -> str:
    def f(v, fmt="{:.2f}"):
        return "–" if v is None else fmt.format(v)

    lines = [f"## Run {res['run_id']} results",
             "",
             f"Evaluation window starts {res['eval_start'][:10]} "
             f"({res['n_station_days']:.0f} station-days, "
             f"{100 * res['storm_minutes_fraction']:.1f}% of minutes at Kp ≥ 5). "
             f"Detection tolerance after fault end: {res['tolerance_min']} min.",
             "",
             "| Method | Event recall | Median delay (min) | False alarms / station-day, quiet "
             "| False alarms / station-day, storm | Storm minutes flagged as fault "
             "| Pointwise F1 | Point-adjusted F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        e = r["event"]
        lines.append(
            f"| {m} | {f(e['recall'])} | {f(e['median_delay_min'], '{:.0f}')} | "
            f"{f(e['fa_per_station_day_quiet'], '{:.3f}')} | {f(e['fa_per_station_day_storm'], '{:.3f}')} | "
            f"{f(r['storm_minutes_flagged'], '{:.1%}')} | "
            f"{f(r['pointwise']['f1'])} | {f(r['point_adjusted']['f1'])} |")
    types = sorted({t for r in res["methods"].values() for t in r["event"]["by_type"]})
    lines += ["", "### Event recall by fault type", "",
              "| Method | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for m, r in res["methods"].items():
        bt = r["event"]["by_type"]
        lines.append(f"| {m} | " + " | ".join(f(bt.get(t, {}).get("recall")) for t in types) + " |")
    conf = res["methods"].get("classifier", {}).get("confusion")
    if conf:
        cols = list(conf)
        rows = list(conf[cols[0]])
        lines += ["", "### Classifier confusion matrix (minutes)", "",
                  "| | " + " | ".join(c.replace("pred_", "") for c in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for r in rows:
            lines.append(f"| {r.replace('true_', '')} | " + " | ".join(str(conf[c][r]) for c in cols) + " |")
    return "\n".join(lines) + "\n"
