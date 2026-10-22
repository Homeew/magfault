"""Command-line entry point: `magfault <command>`."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from . import config


def _date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="magfault", description=__doc__)
    p.add_argument("--dsn", default=None, help="PostgreSQL DSN (default: $MAGFAULT_DSN)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("initdb", help="create / update schema and functions")
    s.add_argument("--drop", action="store_true", help="drop all tables first")

    s = sub.add_parser("download", help="fetch INTERMAGNET 1-minute data (IAGA-2002)")
    s.add_argument("--stations", default=",".join(config.DEFAULT_STATIONS))
    s.add_argument("--start", type=_date, required=True)
    s.add_argument("--end", type=_date, required=True)
    s.add_argument("--out", type=Path, default=Path("data/raw"))
    s.add_argument("--force", action="store_true")
    s.add_argument("--chunk-days", type=int, default=16, help="max days per request")

    s = sub.add_parser("kp", help="fetch GFZ Kp index (JSON)")
    s.add_argument("--start", type=_date, required=True)
    s.add_argument("--end", type=_date, required=True)
    s.add_argument("--out", type=Path, default=Path("data/raw/kp"))
    s.add_argument("--force", action="store_true")
    s.add_argument("--from-file", type=Path, default=None,
                   help="use a Kp file downloaded by hand (GFZ JSON or Kp_ap_since_1932.txt)")

    s = sub.add_parser("synth", help="write a synthetic 4-station network + Kp (offline demo)")
    s.add_argument("--days", type=int, default=90)
    s.add_argument("--start", default="2024-01-01")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", type=Path, default=Path("data/synth"))

    s = sub.add_parser("load", help="load IAGA-2002 files and Kp JSON found under a directory")
    s.add_argument("path", type=Path)

    s = sub.add_parser("inject", help="create a fault run (ground-truth corruption)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--calibration-days", type=int, default=14)
    s.add_argument("--per-type", type=int, default=3, help="faults per type per station")

    for name, help_ in (("features", "build SQL features for a run"),
                        ("detect", "run all detectors for a run"),
                        ("evaluate", "score detectors and write reports/run_<id>/")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--run", type=int, default=None, help="fault run id (default: latest)")
        if name == "detect":
            s.add_argument("--pfa", type=float, default=1e-6)
            s.add_argument("--window", type=int, default=15)
            s.add_argument("--thresholds", choices=["guarded", "analytic"], default="guarded")
        if name == "evaluate":
            s.add_argument("--out", type=Path, default=None, help="report directory")

    args = p.parse_args(argv)

    if args.cmd == "download":
        from .download import download_intermagnet
        stations = [c.strip().upper() for c in args.stations.split(",") if c.strip()]
        files = download_intermagnet(stations, args.start, args.end, args.out, args.force,
                                     max_days=args.chunk_days)
        print(f"{len(files)} new files under {args.out}")
        return
    if args.cmd == "kp":
        from .download import download_kp
        download_kp(args.start, args.end, args.out, args.force, from_file=args.from_file)
        return
    if args.cmd == "synth":
        from .synth import write_network
        print(write_network(args.out, days=args.days, start=args.start, seed=args.seed))
        return

    from . import db
    conn = db.connect(args.dsn)
    try:
        if args.cmd == "initdb":
            db.init_schema(conn, drop=args.drop)
            print("schema ready")
        elif args.cmd == "load":
            from .load import load_directory
            print(load_directory(conn, args.path))
        elif args.cmd == "inject":
            from .pipeline import inject
            inject(conn, seed=args.seed, calibration_days=args.calibration_days,
                   per_type=args.per_type)
        else:
            from . import pipeline
            from .detectors import DetectConfig
            run_id = args.run or db.latest_run(conn)
            if args.cmd == "features":
                pipeline.build_features(conn, run_id)
            elif args.cmd == "detect":
                pipeline.detect(conn, run_id, DetectConfig(pfa=args.pfa, window=args.window,
                                                           threshold_mode=args.thresholds))
            elif args.cmd == "evaluate":
                pipeline.run_evaluation(conn, run_id, out_dir=args.out)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
