import numpy as np
import pandas as pd

from magfault import evaluate


def _setup():
    idx = pd.date_range("2024-01-01", periods=600, freq="min", tz="UTC")
    minutes = pd.DataFrame({"iaga_code": "AAA", "ts": idx, "kp": 1.0})
    minutes.loc[400:459, "kp"] = 6.0                                   # one storm hour
    faults = pd.DataFrame([
        {"fault_id": 1, "iaga_code": "AAA", "fault_type": "offset", "channel": "x",
         "t_start": idx[100], "t_end": idx[149], "magnitude": 3.0},
        {"fault_id": 2, "iaga_code": "AAA", "fault_type": "spike", "channel": "z",
         "t_start": idx[300], "t_end": idx[300], "magnitude": 20.0},
    ])
    return minutes, faults


def test_truth_labels_and_dont_care():
    minutes, faults = _setup()
    lab = evaluate.truth_labels(minutes, faults, tol_min=10)
    assert (lab.loc[100:149, "truth"] == 1).all()
    assert (lab.loc[400:459, "truth"] == 2).all()
    assert lab.loc[150:159, "dont_care"].all() and not lab.loc[160, "dont_care"]


def test_event_metrics_delay_and_false_alarms():
    minutes, faults = _setup()
    lab = evaluate.truth_labels(minutes, faults, tol_min=10)
    alarm = pd.Series(False, index=lab.index)
    alarm.loc[105:120] = True        # detects fault 1 with 5 min delay
    alarm.loc[410:415] = True        # alarm during the storm -> storm false alarm
    alarm.loc[500:502] = True        # quiet false alarm
    res = evaluate.evaluate_method(alarm, lab, faults, tol_min=10)
    ev = res["event"]
    assert ev["n_faults"] == 2 and ev["recall"] == 0.5
    assert ev["median_delay_min"] == 5
    assert ev["false_alarms_storm"] == 1 and ev["false_alarms_quiet"] == 1
    assert res["storm_minutes_flagged"] == 6 / 60
    # Point adjustment rewards the partial hit with the whole segment.
    assert res["point_adjusted"]["recall"] > res["pointwise"]["recall"]


def test_confusion_matrix_shape():
    minutes, faults = _setup()
    lab = evaluate.truth_labels(minutes, faults, tol_min=10)
    pred = pd.Series(np.zeros(len(lab), dtype=int))
    pred.loc[100:149] = 1
    m = evaluate.confusion(pred, lab)
    assert m.shape == (3, 3)
    assert m.loc["true_sensor_fault", "pred_sensor_fault"] == 50
    assert m.to_numpy().sum() == (~lab["dont_care"]).sum()
