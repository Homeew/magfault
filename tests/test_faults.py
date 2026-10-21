import numpy as np
import pandas as pd
import pytest

from magfault import faults


def _segment(n=120):
    idx = pd.date_range("2024-02-01", periods=n, freq="min", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": 15000 + rng.normal(0, 1, n), "y": 3000 + rng.normal(0, 1, n),
        "z": 52000 + rng.normal(0, 1, n), "f": 54200.0,
    }, index=idx)


def _fault(kind, channel="x", mag=5.0, **params):
    return faults.Fault("TST", kind, channel, pd.Timestamp("2024-02-01", tz="UTC"), 120, mag,
                        {"sign": 1.0, **params})


def test_tilt_preserves_magnitude():
    seg = _segment()
    bad = faults.apply_fault(seg, _fault("tilt", "xyz", 3.0, axis=[0.6, 0.8, 0.0]),
                             np.random.default_rng(1))
    mag_before = np.linalg.norm(seg[["x", "y", "z"]].to_numpy(), axis=1)
    mag_after = np.linalg.norm(bad[["x", "y", "z"]].to_numpy(), axis=1)
    assert np.max(np.abs(mag_after - mag_before)) < 0.1          # rounding only
    assert np.max(np.abs(bad["z"] - seg["z"])) > 5               # but components moved


def test_offset_and_drift():
    seg = _segment()
    off = faults.apply_fault(seg, _fault("offset", "y", 4.0), np.random.default_rng(1))
    assert (off["y"] - seg["y"]).mean() == pytest.approx(4.0, abs=0.06)
    dr = faults.apply_fault(seg, _fault("drift", "z", 10.0), np.random.default_rng(1))
    delta = (dr["z"] - seg["z"]).to_numpy()
    assert delta[0] == pytest.approx(0.0, abs=0.06) and delta[-1] == pytest.approx(10.0, abs=0.06)


def test_stuck_holds_first_value():
    seg = _segment()
    bad = faults.apply_fault(seg, _fault("stuck", "x", 0.0), np.random.default_rng(1))
    assert bad["x"].nunique() == 1
    assert bad["y"].equals(seg["y"].round(1))


def test_scalar_offset_only_touches_f():
    seg = _segment()
    bad = faults.apply_fault(seg, _fault("scalar_offset", "f", 2.0), np.random.default_rng(1))
    assert (bad["f"] - seg["f"]).abs().max() == pytest.approx(2.0)
    assert np.allclose(bad["x"], seg["x"].round(1))


def test_schedule_is_reproducible_non_overlapping_and_after_calibration():
    idx = pd.date_range("2024-01-01", periods=60 * 1440, freq="min", tz="UTC")
    coverage = {"AAA": idx, "BBB": idx}
    cal_end = idx[0] + pd.Timedelta(days=14)
    s1 = faults.make_schedule(coverage, cal_end, per_type=2, seed=3)
    s2 = faults.make_schedule(coverage, cal_end, per_type=2, seed=3)
    assert [f.to_record() for f in s1] == [f.to_record() for f in s2]
    assert len(s1) == 2 * 2 * len(faults.FAULT_TYPES)
    for code in coverage:
        mine = sorted((f for f in s1 if f.iaga_code == code), key=lambda f: f.t_start)
        assert all(f.t_start > cal_end for f in mine)
        for a, b in zip(mine, mine[1:]):
            assert b.t_start > a.t_end
