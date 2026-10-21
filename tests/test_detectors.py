import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter
from scipy.stats import chi2

from magfault import detectors


def ar1(n, phi, sigma, seed=0):
    e = np.random.default_rng(seed).normal(0, sigma, n)
    return lfilter([1.0], [1.0, -phi], e)


def test_ar1_whitener_recovers_parameters():
    x = 3.0 + ar1(50_000, 0.7, 0.2)
    w = detectors.AR1Whitener.fit(x)
    assert w.phi == pytest.approx(0.7, abs=0.02)
    assert w.sigma == pytest.approx(0.2, rel=0.05)
    e = w.transform(x)
    assert np.nanstd(e) == pytest.approx(1.0, rel=0.05)


@pytest.mark.parametrize("dof", [1, 3])
def test_glrt_false_alarm_rate_matches_chi2(dof):
    e = np.random.default_rng(1).normal(size=(200_000, dof))
    t_mean, t_var = detectors.window_glrt(e if dof > 1 else e[:, 0], 15)
    thr = chi2.isf(1e-2, dof)
    assert np.nanmean(t_mean > thr) == pytest.approx(1e-2, rel=0.2)
    # The variance statistic is only asymptotically chi2; check it is in the right range.
    assert 2e-3 < np.nanmean(t_var > thr) < 3e-2


def test_glrt_detects_mean_shift():
    e = np.random.default_rng(2).normal(size=5000)
    e[3000:3100] += 1.5
    t_mean, _ = detectors.window_glrt(e, 15)
    thr = chi2.isf(1e-6, 1)
    assert np.nanmax(t_mean[3010:3100]) > thr
    assert np.nanmax(t_mean[:2900]) < thr


def test_siegmund_threshold_monotone_and_reasonable():
    h1 = detectors.siegmund_h(0.5, 1e4)
    h2 = detectors.siegmund_h(0.5, 1e6)
    assert 3 < h1 < h2 < 15


def test_cusum_detects_small_persistent_shift_and_recovers():
    e = np.random.default_rng(3).normal(size=6000)
    e[2000:3000] += 0.8
    h = detectors.siegmund_h(0.5, 1e6)
    s = detectors.cusum_two_sided(e, 0.5, cap=3 * h)
    first = np.argmax(s > h)
    assert 2000 <= first < 2100
    assert s[3000 + int(3 * h / 0.3)] < h                        # recovers after the fault


def _network(n=20_000, seed=4):
    rng = np.random.default_rng(seed)
    common = np.stack([ar1(n, 0.99, 5.0, seed + k) for k in range(3)], axis=1)
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    cols = {}
    for j, code in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        gain = 0.8 + 0.1 * j
        local = rng.normal(0, 0.3, (n, 3))
        for k, c in enumerate(["dx", "dy", "dz"]):
            cols[(code, c)] = gain * common[:, k] + local[:, k]
    return pd.DataFrame(cols, index=idx)


def test_network_glrt_flags_local_fault_and_isolates_it():
    wide = _network()
    cfg = detectors.DetectConfig(pfa=1e-6, window=15)
    model = detectors.NetworkGLRT(cfg).fit(wide.iloc[:10_000])
    faulty = wide.copy()
    faulty.loc[faulty.index[15_000:15_300], ("BBB", "dy")] += 3.0     # Y offset: invisible to dF
    stat = model.isolate(model.residual_stat(faulty))
    assert (stat["BBB"].iloc[15_010:15_300] > 1).mean() > 0.9
    for other in ("AAA", "CCC", "DDD"):
        assert (stat[other].iloc[15_000:15_300] > 1).mean() < 0.05
    quiet = stat.iloc[10_000:14_900]
    assert (quiet > 1).to_numpy().mean() < 1e-3


def test_classifier_logic():
    idx = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    feats = pd.DataFrame({"z_dh": [0, 10, 10, 10], "net_z_dh": [0, 10, 0, 10]}, index=idx)
    scalar = pd.Series([0, 0, 0, 2], index=idx)
    network = pd.Series([0, 2, 2, 0], index=idx)
    lab = detectors.classify(scalar, network, feats, detectors.DetectConfig())
    # quiet; shared disturbance -> geomagnetic; local residual -> fault; dF alarm -> fault
    assert lab.tolist() == [0, 2, 1, 1]


def test_guard_scale():
    cfg = detectors.DetectConfig(threshold_mode="guarded", empirical_rate=0.01)
    s = pd.Series(np.linspace(0, 3, 1001))
    assert detectors.guard_scale(s, cfg) == pytest.approx(2.97, abs=0.01)
    assert detectors.guard_scale(pd.Series(np.zeros(100)), cfg) == 1.0
    cfg_a = detectors.DetectConfig(threshold_mode="analytic")
    assert detectors.guard_scale(s, cfg_a) == 1.0
