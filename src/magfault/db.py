"""Thin PostgreSQL helpers (psycopg 3)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from . import config


def connect(dsn: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(dsn or config.dsn())
    conn.execute("SET TIME ZONE 'UTC'")
    conn.commit()
    return conn


def run_sql_file(conn: psycopg.Connection, path: Path) -> None:
    conn.execute(Path(path).read_text())


def init_schema(conn: psycopg.Connection, drop: bool = False) -> None:
    if drop:
        conn.execute(
            "DROP TABLE IF EXISTS detections, features, fault_samples, injected_faults, "
            "fault_runs, kp_index, readings, ingest_files, stations CASCADE"
        )
    for name in ("00_schema.sql", "10_functions.sql"):
        run_sql_file(conn, config.SQL_DIR / name)
    conn.commit()


def read_df(conn: psycopg.Connection, sql: str, params: Sequence | dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    # NULLs in float columns arrive as None -> object dtype; coerce those back to float.
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: v is None or isinstance(v, float)).all():
            df[c] = df[c].astype(float)
    return df


def copy_rows(conn: psycopg.Connection, table: str, columns: Sequence[str],
              rows: Iterable[Sequence]) -> int:
    """COPY rows into table. NaN floats are written as NULL."""
    n = 0
    cols = ", ".join(columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({cols}) FROM STDIN") as cp:
            for row in rows:
                cp.write_row([None if isinstance(v, float) and np.isnan(v) else v for v in row])
                n += 1
    return n


def latest_run(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT max(run_id) FROM fault_runs").fetchone()
    if row is None or row[0] is None:
        raise SystemExit("No fault run found. Run `magfault inject` first.")
    return int(row[0])
