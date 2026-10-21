"""End-to-end test against a real PostgreSQL. Skipped when no database is reachable.

Uses its own schema-free database objects by dropping and recreating the tables,
so point MAGFAULT_TEST_DSN at a scratch database (CI uses a service container).
"""

import json
import os

import pytest

from magfault import db, load, pipeline, synth
from magfault.detectors import DetectConfig

DSN = os.environ.get("MAGFAULT_TEST_DSN")

pytestmark = pytest.mark.skipif(not DSN, reason="set MAGFAULT_TEST_DSN to run DB tests")


@pytest.fixture(scope="module")
def conn():
    c = db.connect(DSN)
    db.init_schema(c, drop=True)
    yield c
    c.close()


def test_end_to_end(conn, tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("synth")
    info = synth.write_network(data_dir, days=30, seed=11)
    assert info["files"] >= 4

    first = load.load_directory(conn, data_dir)
    assert first["rows_upserted"] == 4 * 30 * 1440
    second = load.load_directory(conn, data_dir)                  # idempotent reload
    assert second["files_loaded"] == 0 and second["rows_upserted"] == 0
    n = conn.execute("SELECT count(*) FROM readings").fetchone()[0]
    assert n == 4 * 30 * 1440

    run_id = pipeline.inject(conn, seed=5, calibration_days=10, per_type=1)
    clean_rows = conn.execute("SELECT count(*) FROM readings_observed(NULL)").fetchone()[0]
    assert clean_rows == n
    n_faulty = conn.execute(
        "SELECT count(*) FROM fault_samples WHERE run_id = %s", (run_id,)).fetchone()[0]
    assert n_faulty > 0

    assert pipeline.build_features(conn, run_id) == n
    counts = pipeline.detect(conn, run_id, DetectConfig())
    assert set(counts) == set(pipeline.METHODS)

    out = tmp_path_factory.mktemp("report")
    res = pipeline.run_evaluation(conn, run_id, out_dir=out)
    assert (out / "metrics.json").exists() and (out / "summary.md").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    clf = metrics["methods"]["classifier"]["event"]
    assert clf["n_faults"] == res["methods"]["classifier"]["event"]["n_faults"] > 0
    # The physics-aware classifier must beat the naive detector on storm confusion.
    assert (metrics["methods"]["classifier"]["storm_minutes_flagged"] or 0) <= \
           (metrics["methods"]["naive_z"]["storm_minutes_flagged"] or 0)
