"""Report figures (matplotlib, Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .db import read_df  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow) + neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SHADE = "#0b0b0b", "#52514e", "#e6e5e1", "#f1f0ec"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


def _observed(conn, run_id: int, code: str, t0, t1) -> pd.DataFrame:
    df = read_df(conn, """SELECT o.ts, o.x, o.y, o.z, o.f, ft.d_f
                          FROM readings_observed(%s) o
                          LEFT JOIN features ft ON ft.run_id = %s AND ft.iaga_code = o.iaga_code
                                               AND ft.ts = o.ts
                          WHERE o.iaga_code = %s AND o.ts BETWEEN %s AND %s ORDER BY o.ts""",
                 (run_id, run_id, code, t0, t1))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts")


def _dates(ax):
    loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _rug(ax, ts, y, color, label):
    if len(ts):
        ax.plot(ts, np.full(len(ts), y), "|", color=color, markersize=7, label=label)


def fault_examples(conn, run_id: int, faults: pd.DataFrame, det: pd.DataFrame, path: Path) -> None:
    """One column pair per fault type: affected channel (top) and dF (bottom)."""
    picks = faults.sort_values("t_start").groupby("fault_type").head(1)
    if picks.empty:
        return
    n = len(picks)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(11, 2.6 * rows), squeeze=False,
                             gridspec_kw={"height_ratios": [1.4, 1] * rows})
    for i, f in enumerate(picks.itertuples()):
        r, c = 2 * (i // cols), i % cols
        top, bot = axes[r, c], axes[r + 1, c]
        pad = pd.Timedelta(minutes=max(90, int(0.6 * (f.t_end - f.t_start).total_seconds() / 60)))
        obs = _observed(conn, run_id, f.iaga_code, f.t_start - pad, f.t_end + pad)
        chans = ["x", "y", "z"] if f.channel == "xyz" else ["f"] if f.channel == "f" else [f.channel]
        for j, ch in enumerate(chans):
            s = obs[ch]
            top.plot(s.index, s - s.iloc[: max(1, len(s) // 10)].median(), color=SERIES[j],
                     label=ch.upper())
        top.set_ylabel("nT (rel.)")
        top.set_title(f"{f.fault_type} · {f.iaga_code} {f.channel.upper()} · "
                      f"mag {f.magnitude:.3g}", loc="left", color=INK, fontsize=9)
        if len(chans) > 1:
            top.legend(loc="upper right", ncol=3, fontsize=8)
        bot.plot(obs.index, obs["d_f"], color=INK_2, linewidth=1.0)
        bot.set_ylabel("ΔF (nT)")
        d = det[(det["iaga_code"] == f.iaga_code) & det["ts"].between(obs.index.min(), obs.index.max())]
        ylo = np.nanmin(obs["d_f"]) if obs["d_f"].notna().any() else 0
        yhi = np.nanmax(obs["d_f"]) if obs["d_f"].notna().any() else 1
        span = max(yhi - ylo, 0.5)
        _rug(bot, d.loc[d["method"] == "scalar_glrt", "ts"], ylo - 0.15 * span, SERIES[1], "scalar GLRT")
        _rug(bot, d.loc[d["method"] == "network_glrt", "ts"], ylo - 0.30 * span, SERIES[2], "network GLRT")
        for ax in (top, bot):
            ax.axvspan(f.t_start, f.t_end, color=SHADE, zorder=0)
            ax.tick_params(axis="x", labelsize=7)
            _dates(ax)
    for j in range(n, rows * cols):
        axes[2 * (j // cols), j % cols].axis("off")
        axes[2 * (j // cols) + 1, j % cols].axis("off")
    fig.suptitle("Injected faults (shaded): affected channel, then ΔF = |B| − F with alarm minutes",
                 x=0.01, y=0.998, ha="left", color=INK)
    fig.legend(handles=[Patch(color=SHADE, label="fault interval"),
                        Line2D([], [], color=SERIES[1], marker="|", ls="", ms=9, label="scalar GLRT alarm"),
                        Line2D([], [], color=SERIES[2], marker="|", ls="", ms=9, label="network GLRT alarm")],
               loc="upper left", bbox_to_anchor=(0.005, 0.982), ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def storm_overview(conn, run_id: int, det: pd.DataFrame, path: Path,
                   faults: pd.DataFrame | None = None) -> None:
    kp = read_df(conn, """SELECT k.ts_start, k.kp FROM kp_index k
                          WHERE k.ts_start >= (SELECT min(ts) FROM features WHERE run_id = %s)
                          ORDER BY k.kp DESC, k.ts_start LIMIT 1""", (run_id,))
    if kp.empty:
        return
    peak = pd.Timestamp(kp["ts_start"].iloc[0]).tz_convert("UTC")
    t0, t1 = peak - pd.Timedelta(hours=18), peak + pd.Timedelta(hours=30)
    df = read_df(conn, """SELECT iaga_code, ts, dh, z_dh FROM features
                          WHERE run_id = %s AND ts BETWEEN %s AND %s ORDER BY iaga_code, ts""",
                 (run_id, t0, t1))
    if df.empty:
        return
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["iaga_code"] = df["iaga_code"].str.strip()
    codes = sorted(df["iaga_code"].unique())
    fig, axes = plt.subplots(len(codes), 1, figsize=(11, 1.6 * len(codes) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, code, color in zip(axes, codes, SERIES * 2):
        g = df[df["iaga_code"] == code]
        ax.plot(g["ts"], g["dh"], color=color)
        ax.set_ylabel(f"{code}\nΔH (nT)")
        d = det[(det["iaga_code"] == code) & det["ts"].between(t0, t1)]
        top = np.nanmax(g["dh"]) if g["dh"].notna().any() else 1
        _rug(ax, d.loc[d["method"] == "naive_z", "ts"], top * 1.08, INK_2, "naive z alarm")
        cl = d[d["method"] == "classifier"]
        _rug(ax, cl.loc[cl["label"] == 1, "ts"], top * 1.18, SERIES[1], "classifier: sensor fault")
        _rug(ax, cl.loc[cl["label"] == 2, "ts"], top * 1.28, SERIES[0], "classifier: geomagnetic")
        if faults is not None:
            for fr in faults[(faults["iaga_code"] == code) & (faults["t_end"] >= t0)
                             & (faults["t_start"] <= t1)].itertuples():
                ax.axvspan(fr.t_start, fr.t_end, color=SHADE, zorder=0)
                ax.text(fr.t_start, top * 0.9, f" {fr.fault_type} {fr.channel.upper()}",
                        fontsize=7, color=INK_2, va="top")
        _dates(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles + [Patch(color=SHADE, label="injected fault")],
               loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=4, fontsize=8)
    fig.suptitle(f"Largest storm in the record (peak Kp {kp['kp'].iloc[0]:.1f}, "
                 f"{peak:%Y-%m-%d %H:%M} UT): horizontal disturbance per station",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def recall_by_type(per_fault: pd.DataFrame, path: Path) -> None:
    if per_fault.empty:
        return
    methods = ["scalar_glrt", "network_glrt", "classifier", "iforest"]
    tab = (per_fault[per_fault["method"].isin(methods)]
           .groupby(["fault_type", "method"])["detected"].mean().unstack("method")
           .reindex(columns=methods))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(tab))
    w = 0.2
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1.5) * w, tab[m].fillna(0), width=w * 0.9, color=SERIES[i], label=m)
    ax.set_xticks(x, tab.index, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("event recall")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.18))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
