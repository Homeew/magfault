import numpy as np
import pandas as pd
import pytest

from magfault import iaga2002

# Header + data in the layout served by the INTERMAGNET GIN (values from a real
# Conrad Observatory file; the 4th column is labelled G but carries the full field).
GIN_SAMPLE = """\
 Format                 IAGA-2002                                    |
 Source of Data         Zentralanstalt fur Meteorologie und Geodynami|
 Station Name           Conrad Observatory                           |
 IAGA Code              WIC                                          |
 Geodetic Latitude      47.930                                       |
 Geodetic Longitude     015.865                                      |
 Elevation              1087                                         |
 Reported               XYZG                                         |
 Sensor Orientation     HDZF                                         |
 Data Type              Definitive                                   |
 # D-conversion factor  10000                                        |
DATE       TIME         DOY     WICX      WICY      WICZ      WICG   |
2021-03-10 00:00:00.000 069     21010.00   1697.70  43982.70  48772.74
2021-03-10 00:01:00.000 069     21009.90   1697.70  43982.70  48772.70
2021-03-10 00:02:00.000 069     99999.00  99999.00  99999.00  99999.00
2021-03-10 00:03:00.000 069     21009.60   1697.70  43982.70  48772.57
"""


def test_parse_gin_sample():
    f = iaga2002.parse(GIN_SAMPLE)
    assert f.code == "WIC"
    assert f.reported == "XYZG"
    assert f.latitude == pytest.approx(47.93)
    assert f.longitude == pytest.approx(15.865)
    assert len(f.data) == 4
    assert str(f.data.index.tz) == "UTC"
    assert f.data.iloc[0]["x"] == pytest.approx(21010.0)
    assert f.data.iloc[2].isna().all()                  # 99999 sentinel -> NaN
    # Full-field values under the G label are kept as F (median > 1000 nT).
    assert f.data.iloc[0]["f"] == pytest.approx(48772.74)
    assert f.comments and "D-conversion" in f.comments[0]


def test_g_column_as_delta_f():
    text = GIN_SAMPLE.replace("48772.74", "0.17").replace("48772.70", "0.15").replace(
        "48772.57", "0.12")
    f = iaga2002.parse(text)
    row = f.data.iloc[0]
    total = np.sqrt(row.x**2 + row.y**2 + row.z**2)
    assert row.f == pytest.approx(total - 0.17)        # G = F_vector - F_scalar


def test_hdz_rotation():
    text = """\
 Format                 IAGA-2002                                    |
 IAGA Code              ABC                                          |
 Reported               HDZF                                         |
DATE       TIME         DOY     ABCH      ABCD      ABCZ      ABCF   |
2024-01-01 00:00:00.000 001     20000.00    600.00  45000.00  49244.29
"""
    f = iaga2002.parse(text)
    d = np.deg2rad(10.0)                                 # 600 arcmin
    assert f.data.iloc[0]["x"] == pytest.approx(20000 * np.cos(d))
    assert f.data.iloc[0]["y"] == pytest.approx(20000 * np.sin(d))


def test_write_read_roundtrip(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    df = pd.DataFrame({"x": [1.0, 2, 3, np.nan, 5], "y": 2.0, "z": 3.0, "f": 4.0}, index=idx)
    path = tmp_path / "TST.min"
    iaga2002.write(path, "TST", df, {"Geodetic Latitude": "50.000"})
    back = iaga2002.read(path)
    assert back.code == "TST"
    assert back.latitude == pytest.approx(50.0)
    pd.testing.assert_frame_equal(back.data, df, check_freq=False, check_names=False)


def test_rejects_non_iaga():
    with pytest.raises(ValueError):
        iaga2002.parse("<html>error</html>")
