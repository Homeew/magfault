"""Idempotent loading of IAGA-2002 files and Kp JSON into PostgreSQL.

Each file is hashed; a file already recorded in ingest_files is skipped. Rows go
through a temp staging table and are upserted, so a corrected re-publication of
the same month replaces only the values that changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from . import iaga2002
from .db import copy_rows

IAGA_SUFFIXES = {".min", ".iaga2002", ".iaga"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover(root: Path) -> tuple[list[Path], list[Path]]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    iaga = [p for p in files if p.suffix.lower() in IAGA_SUFFIXES]
    kp = [p for p in files if p.suffix.lower() == ".json" and p.name.lower().startswith("kp")]
    return iaga, kp


def upsert_station(conn: psycopg.Connection, f: iaga2002.IagaFile, source: str) -> None:
    conn.execute(
        """
        INSERT INTO stations (iaga_code, name, latitude, longitude, elevation_m, reported, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (iaga_code) DO UPDATE
           SET name = EXCLUDED.name, latitude = EXCLUDED.latitude,
               longitude = EXCLUDED.longitude, elevation_m = EXCLUDED.elevation_m,
               reported = EXCLUDED.reported
        """,
        (f.code, f.meta.get("Station Name"), f.latitude, f.longitude, f.elevation,
         f.reported, source),
    )


def load_iaga_file(conn: psycopg.Connection, path: Path) -> int:
    """Load one file in a single transaction. Returns rows upserted (0 if skipped)."""
    digest = sha256(path)
    if conn.execute("SELECT 1 FROM ingest_files WHERE sha256 = %s", (digest,)).fetchone():
        return 0

    f = iaga2002.read(path)
    df = f.data
    if df.empty:
        return 0
    source = "synthetic" if "synthetic" in f.meta.get("Data Type", "").lower() else "INTERMAGNET"

    with conn.transaction():
        upsert_station(conn, f, source)
        for month in pd.date_range(df.index.min().floor("D"), df.index.max(), freq="MS").union(
                [df.index.min()]):
            conn.execute("SELECT ensure_month_partition(%s)", (month.to_pydatetime(),))

        n_missing = int(df[["x", "y", "z"]].isna().any(axis=1).sum())
        ingest_id = conn.execute(
            """INSERT INTO ingest_files (iaga_code, file_name, sha256, first_ts, last_ts,
                                         n_rows, n_missing)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING ingest_id""",
            (f.code, path.name, digest, df.index.min().to_pydatetime(),
             df.index.max().to_pydatetime(), len(df), n_missing),
        ).fetchone()[0]

        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS readings_stage "
            "(LIKE readings INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
        )
        conn.execute("TRUNCATE readings_stage")
        vals = df[["x", "y", "z", "f"]].to_numpy(dtype=float)
        rows = (
            (f.code, t.to_pydatetime(), *[None if np.isnan(v) else float(v) for v in row], ingest_id)
            for t, row in zip(df.index, vals)
        )
        copy_rows(conn, "readings_stage", ["iaga_code", "ts", "x", "y", "z", "f", "ingest_id"], rows)
        cur = conn.execute(
            """
            INSERT INTO readings AS r (iaga_code, ts, x, y, z, f, ingest_id)
            SELECT iaga_code, ts, x, y, z, f, ingest_id FROM readings_stage
            ON CONFLICT (iaga_code, ts) DO UPDATE
               SET x = EXCLUDED.x, y = EXCLUDED.y, z = EXCLUDED.z, f = EXCLUDED.f,
                   ingest_id = EXCLUDED.ingest_id
             WHERE (r.x, r.y, r.z, r.f) IS DISTINCT FROM (EXCLUDED.x, EXCLUDED.y, EXCLUDED.z, EXCLUDED.f)
            """
        )
        n = cur.rowcount
    conn.commit()
    return n


def load_kp_file(conn: psycopg.Connection, path: Path) -> int:
    payload = json.loads(Path(path).read_text())
    times = payload["datetime"]
    kps = payload["Kp"]
    status = payload.get("status") or [None] * len(times)
    with conn.transaction():
        n = 0
        for t, kp, st in zip(times, kps, status):
            if kp is None:
                continue
            conn.execute(
                """INSERT INTO kp_index (ts_start, kp, status) VALUES (%s, %s, %s)
                   ON CONFLICT (ts_start) DO UPDATE SET kp = EXCLUDED.kp, status = EXCLUDED.status
                   WHERE kp_index.kp IS DISTINCT FROM EXCLUDED.kp
                      OR kp_index.status IS DISTINCT FROM EXCLUDED.status""",
                (t, float(kp), st),
            )
            n += 1
    return n


def load_directory(conn: psycopg.Connection, root: Path) -> dict[str, int]:
    iaga_files, kp_files = discover(root)
    stats = {"files_seen": len(iaga_files), "files_loaded": 0, "rows_upserted": 0, "kp_rows": 0}
    for p in iaga_files:
        n = load_iaga_file(conn, p)
        if n:
            stats["files_loaded"] += 1
            stats["rows_upserted"] += n
            print(f"loaded {p.name}: {n} rows")
    for p in kp_files:
        stats["kp_rows"] += load_kp_file(conn, p)
    conn.execute("ANALYZE readings")
    conn.commit()
    return stats
